#!/usr/bin/python2.7

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
from loguru import logger
from tqdm import tqdm
from focalloss import FocalLoss
sys.path.append('include/')
from utils import write_str_to_file


class MS_TCN2(nn.Module):
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes):
        super(MS_TCN2, self).__init__()
        self.PG = Prediction_Generation(num_layers_PG, num_f_maps, dim, num_classes)
        self.Rs = nn.ModuleList([copy.deepcopy(Refinement(num_layers_R, num_f_maps, num_classes, num_classes)) for s in range(num_R)])

    def forward(self, x):
        out = self.PG(x)
        outputs = out.unsqueeze(0)
        for R in self.Rs:
            out = R(F.softmax(out, dim=1))
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)

        return outputs

class Prediction_Generation(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Prediction_Generation, self).__init__()

        self.num_layers = num_layers

        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)

        self.conv_dilated_1 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2**(num_layers-1-i), dilation=2**(num_layers-1-i))
            for i in range(num_layers)
        ))

        self.conv_dilated_2 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2**i, dilation=2**i)
            for i in range(num_layers)
        ))

        self.conv_fusion = nn.ModuleList((
             nn.Conv1d(2*num_f_maps, num_f_maps, 1)
             for i in range(num_layers)

            ))


        self.dropout = nn.Dropout()
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        f = self.conv_1x1_in(x)

        for i in range(self.num_layers):
            f_in = f
            f = self.conv_fusion[i](torch.cat([self.conv_dilated_1[i](f), self.conv_dilated_2[i](f)], 1))
            f = F.relu(f)
            f = self.dropout(f)
            f = f + f_in

        out = self.conv_out(f)

        return out

class Refinement(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Refinement, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2**i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out
    
class MS_TCN(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MS_TCN, self).__init__()
        self.stage1 = SS_TCN(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SS_TCN(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SS_TCN(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SS_TCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class Trainer:
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes, dataset, split, loss_mse, loss_dice, loss_focal, weights, w_coeff, adaptive_mse, window_mse, device):
        self.model = MS_TCN2(num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes)
        self.num_classes = num_classes
        self.loss_mse = loss_mse         #lambda parameter for weighting of MSE and CEL loss 
        self.loss_dice = loss_dice     #dice loss parameter
        self.loss_focal_param = loss_focal # focal loss parameter, gamma=0 is normal cross_entropy
        self.weights = None
        if weights is not None: 
            self.weights = torch.Tensor(weights).to(device)
            self.weights = self.weights * (1 - w_coeff) + w_coeff / self.weights.shape[0]
        self.ce = nn.CrossEntropyLoss(ignore_index=-100, weight=self.weights)
        self.mse = nn.MSELoss(reduction='none')
        self.fl = FocalLoss(gamma=self.loss_focal_param, alpha=self.weights) if loss_focal > 0.0 else self.ce
        self.mse_window = window_mse
        self.adaptive_mse = adaptive_mse

        logger.add('logs/' + dataset + "_" + split + "_{time}.log")
        logger.add(sys.stdout, colorize=True, format="{message}")
        logger.info(f"Starting training with the paramters: num_layers_PG={num_layers_PG}, num_layers_R={num_layers_R}, num_R={num_R}, num_f_maps={num_f_maps}, fdims={dim}, num_classes={num_classes}, loss_mse={loss_mse}, loss_dice={loss_dice}, loss_focal={loss_focal}, weights={self.weights}, w_coeff={w_coeff}, adaptive_mse={self.adaptive_mse}, window_mse={self.mse_window}")

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.6 * num_epochs), int(0.9 * num_epochs)], gamma=0.3)
        smooth_vec = torch.from_numpy(np.concatenate([np.geomspace(0.1, 1.0, self.mse_window//2)[::-1], np.geomspace(0.1, 1.0, self.mse_window//2)])).to(device)
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            while batch_gen.has_next():
                batch_input, batch_target, mask = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                predictions = self.model(batch_input)
                loss = 0
                acc_mse, acc_ce = 0.0, 0.0
                for p in predictions: #p.shape (classes x num_frames)
                    #loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += self.fl(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1)) # cross-entropy loss or focal loss
                    acc_ce += self.fl(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    mse_values = torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16) * mask[:, :, 1:] #max = 16
                    mse_values = torch.mean(mse_values, dim=1)
                    if self.adaptive_mse: 
                        class_changes_tensor = torch.diff(batch_target.view(-1)) # get class changes that should not be mse punished
                        class_changes_tensor = torch.nonzero(class_changes_tensor).squeeze()
                        for elm in class_changes_tensor:
                            if elm <= self.mse_window or batch_target.view(-1).shape[0] - elm <= self.mse_window or not self.adaptive_mse: 
                                continue # Exclude beginning/end gt
                            mse_values[0, elm-self.mse_window//2:elm+self.mse_window//2] = mse_values[0, elm:elm+smooth_vec.shape[0]] * smooth_vec
                    loss += self.loss_mse * torch.mean(mse_values)
                    loss += self.loss_dice * self.calc_dice_loss(p, batch_target.view(-1), softmax=True)
                    acc_mse += self.loss_mse * torch.mean(mse_values)
                #print(f"Loss mse: {acc_mse}, loss ce {acc_ce}")

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            last_lr = scheduler.get_last_lr()
            scheduler.step()
            batch_gen.reset()
            if last_lr != scheduler.get_last_lr(): 
                logger.info(f"Reduced learning rate from {last_lr} to {scheduler.get_last_lr()}")
            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            logger.info("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                               float(correct)/total))

    def predict(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in tqdm(list_of_vids):
                #print vid
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions = self.model(input_x)
                #print("Predictions: ", predictions[-1].data.shape)
                max_values, predicted = torch.max(predictions[-1].data, 1)
                #predicted_prob = F.softmax(predictions[-1], dim=1).data.squeeze()
                predicted = predicted.squeeze()
                max_values = max_values.squeeze()
                #print(predicted_prob.shape, F.softmax(predicted_prob, dim=1))
                
                # map predicted index to action
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
                #print(recognition.shape, recognition_prob.mean(), recognition_prob.min(), recognition_prob.max())
                
                # write predictions to file
                f_name = vid.split('/')[-1].split('.')[0]
                write_str_to_file(results_dir + "/" + f_name, "### Frame level recognition: ###\n" + ' '.join(recognition)) 
                np.save(results_dir + "/" + f_name + ".npy", predictions[-1].squeeze().cpu().detach().numpy())


    def calc_dice_loss(self, logits, targets, softmax=None, smooth=1e-6):
        probabilities = logits
        if softmax is not None:
            probabilities = nn.Softmax(dim=1)(logits)[0]

        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes)
        # print(targets_one_hot.shape) # Convert from NHWC to NCHW
        # print(targets_one_hot.shape, probabilities.shape)
        targets_one_hot = targets_one_hot.permute(1, 0)
        # Multiply one-hot encoded ground truth labels with the probabilities to get the prredicted probability for the actual class.
        intersection = (targets_one_hot * probabilities).sum()
        mod_a = intersection.sum()
        mod_b = targets.numel()
        dice_coefficient = 2. * intersection / (mod_a + mod_b + smooth)
        dice_loss = -dice_coefficient.log()
        return dice_loss

