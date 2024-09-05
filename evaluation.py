import sys
import numpy as np
import pandas as pd
import argparse
from sklearn import metrics
from scipy.special import softmax
from sklearn.preprocessing import OneHotEncoder
sys.path.append('include/')
from utils import read_file, load_action_map, plot_graphs_for_dataset, plot_confusion_matrix, edit_score, collaps_confusion_matrix, prepare_results, acc_phase


def remove_columns(probabilites, actions_dict, gt_content, axis=0):
    idx_to_remove = []
    for action in actions_dict.keys():
        if action not in set(gt_content):
            idx_to_remove.append(actions_dict[action])
    probabilites = np.delete(probabilites, idx_to_remove, axis=axis)
    return probabilites

def insert_ground_truth(actions_dict, gt_content, axis=0):
    idx_to_insert = []
    for action in actions_dict.keys():
        if action not in set(gt_content):
            idx_to_insert.append(actions_dict[action])
    for i, idx in enumerate(idx_to_insert): 
        gt_content[-i-1] = idx
    if len(idx_to_insert) > 0:
        print("Had to insert index", idx_to_insert)
    return gt_content


def bootstrap_metric(results, metric, alpha=0.5, num_samples=500, seed=0, action_dict=None):
    rng = np.random.RandomState(seed=seed)
    idx = np.arange(len(results['labels']))
    accumulation_metric = []
    encoder = OneHotEncoder(sparse_output=False)

    for _ in range(num_samples):
        pred_idx = rng.choice(idx, size=idx.shape[0], replace=True)
        boot_label = [label for j, label in enumerate(results['labels']) if j in pred_idx]
        boot_pred = [pred for j, pred in enumerate(results['predictions']) if j in pred_idx]
        boot_prob = [probs for j, probs in enumerate(results['probs']) if j in pred_idx]
        if action_dict != None:
            boot_label = [insert_ground_truth(action_dict, label, axis=0) for label in boot_label]

        if metric == metrics.roc_auc_score or metric == metrics.average_precision_score:
            y_true = encoder.fit_transform(np.concatenate(boot_label).reshape(-1, 1)).T
            test_mean = metric(y_true.ravel(), np.concatenate(boot_prob, axis=1).ravel(), average='macro')
        elif metric == edit_score or metric == acc_phase:
            scores = [metric(pred[:-10], label[:-10], bg_class=[0]) for pred, label in zip(boot_pred, boot_label)]
            test_mean = np.mean(scores) / 100
        else:
            boot_label = [label[:-10] for label in boot_label]
            boot_pred = [pred[:-10] for pred in boot_pred]
            test_mean = metric(np.concatenate(boot_label), np.concatenate(boot_pred))
        
        accumulation_metric.append(test_mean)
    
    bootstrap_mean = np.mean(accumulation_metric)
    ci_lower = np.percentile(accumulation_metric, alpha/2.0)
    ci_upper = np.percentile(accumulation_metric, 100-alpha/2.0)
    return bootstrap_mean, (ci_lower, ci_upper)


def write_result_to_table(df, name, result):
    df.loc[df.index[-1], name] = result


def evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--split', required=True)
    args = parser.parse_args()
    
    ground_truth_path = f"./data/{args.dataset}/groundTruth/"
    predictions_path = f"./results/{args.dataset}/split_{args.split}/"
    output_dir = f"./results/{args.dataset}/figures.split{args.split}/"
    results_df = pd.read_excel('./results.xlsx')
    
    list_of_videos = read_file(f"./data/{args.dataset}/splits/test.split{args.split}.bundle").split('\n')[:-1]
    actions_dict = load_action_map(f"./data/{args.dataset}/mapping.txt") # load actions mapping
    action_dict_inv = dict((v,k) for k,v in actions_dict.items())
    
    accumulated_gt, accumulated_predictions = [], []
    accumulated_probs = np.empty((0, len(actions_dict)))
    tp, tn, fp, fn = 0, 0, 0, 0
    scores = {
        "boot_accuracy": 0,
        "boot_acc_phase": 0,
        "boot_edit": [],
        "f1_micro": [],
        # "IoU": [],
        "boot_roc_auc": (0, (0, 0)),
        "boot_pr_auc": (0, (0, 0)),
        "sensitivity": [], # Recall
        "specificity": []
    }
    
    for video in list_of_videos:
        gt_file = ground_truth_path + video
        gt_content = read_file(gt_file).split('\n')
        if gt_content[-1] == '': gt_content = gt_content[:-1] # Load video ground truth
        
        predictions = read_file(predictions_path + video.split('.')[0]).split('\n')[1].split()
        probabilites = np.load(predictions_path + video.split('.')[0] + ".npy")
        probabilites = softmax(probabilites.T, axis=1)
        accumulated_probs = np.concatenate((accumulated_probs, probabilites), axis=0)
        probabilites = remove_columns(probabilites, actions_dict, gt_content)
        
        gt_content = [actions_dict[action] for action in gt_content] # transform actions names to integers
        predictions = [actions_dict[action] for action in predictions]
        #scores['edit'].append(edit_score(predictions, gt_content, bg_class=[0]))
        
        accumulated_gt += gt_content # Accumlate results for later processing
        accumulated_predictions += predictions
        tp, tn, fp, fn = tuple(map(sum, zip((tp, tn, fp, fn), collaps_confusion_matrix(gt_content, predictions))))

    def calc_sen_spec(gt, preds, metric='sens'):
            tp, tn, fp, fn = collaps_confusion_matrix(gt, preds)
            return tp / (tp + fn) if metric == 'sens' else tn / (tn + fp)

    results = prepare_results(ground_truth_path, predictions_path, actions_dict, sample_rate=5)
    scores['boot_accuracy'] = bootstrap_metric(results, metrics.accuracy_score)
    scores['boot_acc_phase'] = bootstrap_metric(results, acc_phase)
    # scores['IoU'] = metrics.jaccard_score(accumulated_gt, accumulated_predictions, average='macro')
    scores['f1_micro'] = bootstrap_metric(results, lambda a, b: metrics.f1_score(a, b, average='macro'))
    scores['boot_roc_auc'] = bootstrap_metric(results, metrics.roc_auc_score, action_dict={v:v for k,v in actions_dict.items()})
    scores['boot_pr_auc'] = bootstrap_metric(results, metrics.average_precision_score, action_dict={v:v for k,v in actions_dict.items()})
    scores['sensitivity'] = bootstrap_metric(results, lambda a, b: calc_sen_spec(a, b))
    scores['specificity'] = bootstrap_metric(results, lambda a, b: calc_sen_spec(a, b, metric='spec'))
    # scores['edit'] = np.mean(scores['edit']) / 100
    scores['boot_edit'] = bootstrap_metric(results, edit_score)
    
    print(f"# Average scores for {args.dataset}, split {args.split}:")
    for k, v in scores.items(): # Print and write scores
        if type(v) == tuple:
            print(f"- {k}: {v[0] * 100:.2f} ({v[1][0]:.4f} - {v[1][1]:.4f})")
            write_result_to_table(results_df, k, v[0] * 100)
            write_result_to_table(results_df, k + '_lci', v[1][0])
            write_result_to_table(results_df, k + '_uci', v[1][1])
        else: 
            print(f"- {k}: {v * 100:.2f}")
            write_result_to_table(results_df, k, v[0] * 100)

    # Plot confusion matrix and graphs
    #plot_graphs_for_dataset(args.dataset, args.split, output_dir)
    #plot_confusion_matrix(accumulated_gt, accumulated_predictions, actions_dict, output_dir, normalized='true')
    results_df.to_excel('./results.xlsx', index=False)


if __name__ == '__main__':
    evaluation()
