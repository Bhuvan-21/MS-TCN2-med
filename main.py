#!/usr/bin/python2.7

import os
import argparse
import torch
from datetime import datetime
import random
import pandas as pd
import numpy as np
from model import Trainer
from batch_gen import BatchGenerator



def create_results_table(args, results_dir):
    metrics = ['acc', 'edit', 'IoU', 'mean_roc', 'mean_pr_auc', 'f1@0.10', 'f1@0.25', 'f1@0.50', 'sen@0.10', 'spec@0.10', 'sen@0.25', 'spec@0.25', 'sen@0.50', 'spec@0.50']
    cols = ['timestamp', 'results_dir', 'N']
    
    if not os.path.exists('./results.xlsx'): 
        cols.extend(list(args.keys()))
        cols.extend(metrics)
        df = pd.DataFrame()
    else:
        df = pd.read_excel('./results.xlsx')
    
    n_samples = sum(1 for _ in open(f'./data/{args["dataset"]}/splits/train.split{args["split"]}.bundle', 'rb'))
    new_row = {'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'results_dir': results_dir, 'N': n_samples} 
    new_row.update(args)
    new_row.update({k: 0.0 for k in metrics})
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_excel('./results.xlsx', index=False)


def setup_device(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    return torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', default='train')
    parser.add_argument('--dataset', default="gtea")
    parser.add_argument('--split', default='1')
    parser.add_argument('--features_dim', default='2048', type=int)
    parser.add_argument('--bz', default='1', type=int)
    parser.add_argument('--lr', default='0.0005', type=float)
    parser.add_argument('--num_f_maps', default='64', type=int)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--num_layers_PG', type=int)
    parser.add_argument('--num_layers_R', type=int)
    parser.add_argument('--num_R', type=int)
    parser.add_argument('--loss_mse', default=0.15, type=float)
    parser.add_argument('--loss_dice', default=0.0, type=float)
    parser.add_argument('--loss_focal', default=0.0, type=float)
    parser.add_argument('--weights', default=None)
    return parser.parse_args()


def create_directories(dataset, split):
    model_dir = f"./models/{dataset}/split_{split}"
    results_dir = f"./results/{dataset}/split_{split}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return model_dir, results_dir


def load_actions(mapping_file):
    with open(mapping_file, 'r') as file_ptr:
        actions = file_ptr.read().split('\n')[:-1]
    actions_dict = {a.split()[1]: int(a.split()[0]) for a in actions}
    return actions_dict


def load_weights(weight_file, action_dict, inverse=True):
    """
    Weight file should be in the format: class_name weight
    Action dict has the format: class_id: class_name
    
    returns a list of weights in the same order as the action_dict
    """
    if not os.path.exists(weight_file): return None
    with open(weight_file, 'r') as file_ptr:
        weight_strs = file_ptr.readlines()
        
    weights = []
    weight_dict = {}
    for weight_str in weight_strs:
        weight_str = weight_str.strip('\n')
        weight_dict[weight_str.split()[0]] = float(weight_str.split()[1])  
    for key in action_dict.keys():
        weights.append(weight_dict[key])
    
    return 1.0 / np.array(weights) if inverse else np.array(weights)


def run(args):
    device = setup_device(1538574472)
    
    num_epochs = args.num_epochs
    features_dim = args.features_dim
    bz = args.bz
    lr = args.lr
    num_layers_PG = args.num_layers_PG
    num_layers_R = args.num_layers_R
    num_R = args.num_R
    num_f_maps = args.num_f_maps

    # use the full temporal resolution @ 15fps
    sample_rate = 1
    # sample input features @ 15fps instead of 30 fps for 50salads, and up-sample the output to 30 fps
    if args.dataset == "50salads":
        sample_rate = 2

    vid_list_file = "./data/"+args.dataset+"/splits/train.split"+args.split+".bundle"
    vid_list_file_tst = "./data/"+args.dataset+"/splits/test.split"+args.split+".bundle"
    features_path = "./data/"+args.dataset+"/features/"
    gt_path = "./data/"+args.dataset+"/groundTruth/"

    actions_dict = load_actions(f"./data/{args.dataset}/mapping.txt")
    weights = load_weights(f"./data/{args.dataset}/weights.txt", actions_dict) if args.weights else None

    model_dir, results_dir = create_directories(args.dataset, args.split)

    num_classes = len(actions_dict)
    trainer = Trainer(num_layers_PG, num_layers_R, num_R, num_f_maps, features_dim, num_classes, args.dataset, args.split, args.loss_mse, args.loss_dice, args.loss_focal, weights, device)
    if args.action == "train":
        batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
        batch_gen.read_data(vid_list_file)
        trainer.train(model_dir, batch_gen, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device)

    if args.action == "predict":
        trainer.predict(model_dir, results_dir, features_path, vid_list_file_tst, num_epochs, actions_dict, device, sample_rate)
        create_results_table(vars(args), results_dir)


if __name__ == '__main__':
    arguments = parse_arguments()
    print("## Parsed arguments:")
    for arg, value in vars(arguments).items():
        print(f"> {arg}: {value}")
    run(arguments)
