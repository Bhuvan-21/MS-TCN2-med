import sys
import numpy as np
import pandas as pd
import argparse
from sklearn import metrics
sys.path.append('include/')
from utils import read_file, load_action_map, get_labels_start_end_time, plot_graphs_for_dataset, plot_confusion_matrix


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
    pairs = []
    action_dict = dict((v,k) for k,v in action_dict.items()) 
    for c in range(scores.shape[0]):
        class_score = scores[c, :]
        cur_class = action_dict[c]
        class_gt = [1 if cur_class == ground_truth[i] else 0 for i in range(0, class_score.shape[0])]
        # roc = metrics.roc_auc_score(class_gt, class_score.tolist()[:-1])
        pairs.append((class_gt, class_score.tolist()[:]))
        #print(class_score.shape, len(class_gt), roc) 
        if len(class_gt) != class_score.shape[0]: print('Missmatch: ', cur_class, len(class_gt), class_score.shape)
    return pairs


def write_result_to_table(df, name, result):
    df.loc[df.index[-1], name] = result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="gtea")
    parser.add_argument('--split', default='1')
    parser.add_argument('--sample_rate', default=1, type=int)
    args = parser.parse_args()

    ground_truth_path = "./data/"+args.dataset+"/groundTruth/"
    recog_path = "./results/"+args.dataset+"/split_"+args.split+"/"
    scores_path = "./results/"+args.dataset+"/split_"+args.split+"/"
    file_list = "./data/"+args.dataset+"/splits/test.split"+args.split+".bundle"
    mapping_file = "./data/"+args.dataset+"/mapping.txt"
    
    list_of_videos = read_file(file_list).split('\n')[:-1]
    results_df = pd.read_excel('./results.xlsx')
   
    # Load class mappings 
    actions_dict = load_action_map(mapping_file)

    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

    correct = 0
    total = 0
    edit = 0
    IoU = 0

    class_scores = [([], []) for i in range(0, len(actions_dict))]
    accumulated_gt, accumulated_predictions = [], []

    for vid in list_of_videos:
        #print(f"Working on {vid}...")
        gt_file = ground_truth_path + vid
        gt_content = read_file(gt_file).split('\n')
        if gt_content[-1] == '': gt_content = gt_content[:-1]

        recog_file = recog_path + vid.split('.')[0]
        recog_content = read_file(recog_file).split('\n')[1].split()
        recog_scores = np.load(scores_path + vid.split('.')[0] + ".npy")
        
        if len(recog_content) > len(gt_content):
            gt_content = gt_content[::args.sample_rate]
            recog_content = recog_content[::args.sample_rate]
        elif len(gt_content) > len(recog_content):
            gt_content = gt_content[:len(recog_content)]
        
        # count correct predictions
        for i in range(len(gt_content)):
            total += 1
            if i > len(recog_content) - 1: 
                continue
            if gt_content[i] == recog_content[i]:
                correct += 1

        edit += edit_score(recog_content, gt_content, bg_class=[""])
        IoU += metrics.jaccard_score(gt_content, recog_content, average='micro')
        
        # Accumlate results for later metrics
        accumulated_gt += gt_content
        accumulated_predictions += recog_content

        # get 2x2 table for different overlap values
        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s], bg_class=[""])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1

        # accumulate ground_truth/probabilites for ROC_AUC calculation per class
        scores = prediction_scores(recog_content, gt_content, recog_scores, actions_dict)
        for i in range(len(class_scores)):
            class_scores[i][0].extend(scores[i][0])
            class_scores[i][1].extend(scores[i][1])
            #print(len(class_scores[i][0]), len(class_scores[i][1]))

    acc = (100*float(correct)/total)
    edit = ((1.0*edit)/len(list_of_videos))
    IoU = 100 * IoU / len(list_of_videos)
    print(f'Acc: {acc:.4f}')
    print(f'Edit: {edit:.4f}')
    print(f'IoU: {IoU:.4f}')
    
    write_result_to_table(results_df, 'acc', acc)
    write_result_to_table(results_df, 'edit', edit)
    write_result_to_table(results_df, 'IoU', IoU)

    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s]+fp[s])
        recall = tp[s] / float(tp[s]+fn[s])
        f1 = 2.0 * (precision*recall) / (precision+recall)
        f1 = np.nan_to_num(f1)*100
        print(f'F1@{overlap[s]:.2f}: {f1:.2f}')
        write_result_to_table(results_df, f'f1@{overlap[s]:.2f}', f1)

    roc_aucs = []
    pr_aucs = []
    for i in range(len(class_scores)):
        try: 
            roc_aucs.append(metrics.roc_auc_score(class_scores[i][0],class_scores[i][1]))
            pr_aucs.append(metrics.average_precision_score(class_scores[i][0], class_scores[i][1])) 
        except: 
            print('AUC error on class: ', i)
            roc_aucs.append(np.mean(roc_aucs))
            pr_aucs.append(np.mean(pr_aucs))

    print(f"Average ROC AUC score over all classes: {np.mean(roc_aucs) * 100.0:.2f}")
    print(f"Average PR AUC score over all classes: {np.mean(pr_aucs) * 100.0:.2f}")
    print("ROC_AUC per class: ", list(zip(actions_dict.keys(), map(lambda d: round(d, 4), roc_aucs))))
    print("PR_AUC per class: ", list(zip(actions_dict.keys(), map(lambda d: round(d, 4), pr_aucs))))
    
    write_result_to_table(results_df, 'mean_roc', np.mean(roc_aucs) * 100.0)
    write_result_to_table(results_df, 'mean_pr_auc', np.mean(pr_aucs) * 100.0 )

    for i, key in enumerate(actions_dict.keys()): 
        write_result_to_table(results_df, key+'_roc', roc_aucs[i])
        write_result_to_table(results_df, key+'_pr', pr_aucs[i])

    results_df.to_excel('./results.xlsx', index=False)
    output_dir = f"./results/{args.dataset}/figures.split{args.split}/"
    plot_graphs_for_dataset(args.dataset, args.split, output_dir)
    plot_confusion_matrix(accumulated_gt, accumulated_predictions, actions_dict, output_dir, normalized='true')
    plot_confusion_matrix(accumulated_gt, accumulated_predictions, actions_dict, output_dir, normalized=None)
    #print(metrics.confusion_matrix(accumulated_gt, accumulated_predictions, labels=list(actions_dict.keys())))


if __name__ == '__main__':
    main()
