#!/usr/bin/python2.7
# adapted from: https://github.com/colincsl/TemporalConvolutionalNetworks/blob/master/code/metrics.py

import numpy as np
import argparse
from sklearn import metrics


def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content


def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float64)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)

    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def prediction_scores(recognized, ground_truth, scores, action_dict):
    rocs = []
    action_dict = dict((v,k) for k,v in action_dict.items()) 
    for c in range(scores.shape[0]):
        class_score = scores[c, :]
        cur_class = action_dict[c]
        class_gt = [1 if cur_class == ground_truth[i] else 0 for i in range(0, len(ground_truth))]
        # roc = metrics.roc_auc_score(class_gt, class_score.tolist()[:-1])
        rocs.append((class_gt, class_score.tolist()[:-1]))
        #print(class_score.shape, len(class_gt), roc) 
    return rocs


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default="gtea")
    parser.add_argument('--split', default='1')

    args = parser.parse_args()

    ground_truth_path = "./data/"+args.dataset+"/groundTruth/"
    recog_path = "./results/"+args.dataset+"/split_"+args.split+"/"
    scores_path = "./results/"+args.dataset+"/split_"+args.split+"/"
    file_list = "./data/"+args.dataset+"/splits/test.split"+args.split+".bundle"
    mapping_file = "./data/"+args.dataset+"/mapping.txt"
    
    list_of_videos = read_file(file_list).split('\n')[:-1]
   
    # Load class mappings 
    actions = read_file(mapping_file).split('\n')[:-1]
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

    correct = 0
    total = 0
    edit = 0

    class_scores = [([], []) for i in range(0, len(actions_dict))]

    for vid in list_of_videos:
        #print(f"Working on {vid}...")
        gt_file = ground_truth_path + vid
        gt_content = read_file(gt_file).split('\n')[0:-1]

        recog_file = recog_path + vid.split('.')[0]
        recog_content = read_file(recog_file).split('\n')[1].split()
        recog_scores = np.load(scores_path + vid.split('.')[0] + ".npy")
        
        # count correct predictions
        for i in range(len(gt_content)):
            total += 1
            if gt_content[i] == recog_content[i]:
                correct += 1

        edit += edit_score(recog_content, gt_content)

        # get 2x2 table for different overlap values
        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1

        # accumulate ground_truth/scores for ROC_AUC calculation per class
        scores = prediction_scores(recog_content, gt_content, recog_scores, actions_dict)
        for i in range(len(class_scores)):
            class_scores[i][0].extend(scores[i][0])
            class_scores[i][1].extend(scores[i][1])

    print("Acc: %.4f" % (100*float(correct)/total))
    print('Edit: %.4f' % ((1.0*edit)/len(list_of_videos)))
    acc = (100*float(correct)/total)
    edit = ((1.0*edit)/len(list_of_videos))
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s]+fp[s])
        recall = tp[s] / float(tp[s]+fn[s])
        sensitivity = tp[s] / float(tp[s]+fn[s])
        specificity = (total - tp[s] - fn[s] - fp[s]) / float(total - tp[s] - fn[s]) 

        f1 = 2.0 * (precision*recall) / (precision+recall)

        f1 = np.nan_to_num(f1)*100
        print('F1@%0.2f: %.4f' % (overlap[s], f1))
        print(f'Sensitivity@{overlap[s]:.2f}: {sensitivity:.4f}')
        print(f'Specificity@{overlap[s]:.2f}: {specificity:.4f}')

    roc_aucs = []
    pr_aucs = []
    for i in range(len(class_scores)):
        roc_aucs.append(metrics.roc_auc_score(class_scores[i][0],class_scores[i][1]))
        pr_aucs.append(metrics.average_precision_score(class_scores[i][0], class_scores[i][1])) 

    print("Average ROC AUC score over all classes: ", np.mean(roc_aucs))
    print("Average PR AUC score over all classes: ", np.mean(pr_aucs))
    print("ROC_AUC per class: ", list(zip(actions_dict.keys(), roc_aucs)))
    print("PR_AUC per class: ", list(zip(actions_dict.keys(), pr_aucs)))


if __name__ == '__main__':
    main()
