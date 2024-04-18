import os
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from matplotlib import colormaps
from sklearn import metrics


def write_str_to_file(filename, content):
    with open(filename, 'w') as f:
        f.write(content)

        
def load_action_map(mapping_file):
    actions = read_file(mapping_file).split('\n')[:-1]
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
    
    return actions_dict

        
def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content


def get_labels_start_end_time(frame_wise_labels, bg_class=[""]):
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


def survey(results, category_names, colors, ax, fig_size=(25, 3)):
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data))
    
    for i, (colname, color) in enumerate(zip(category_names, colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh([""], widths, left=starts, height=0.75, label=colname, color=color)

        r, g, b = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        ax.bar_label(rects, labels=[colname], label_type='center', color=text_color, rotation=90 if fig_size[1] >=3 else 0 )
    return ax



def get_labels(dataset_name):
    if "cataract101" in dataset_name:
        labels = ['action_start', 'incision', 'viscous_agent_injection', 'rhexis', 'hydrodissection', 'phacoemulsificiation', 'irrigation_aspiration', 
                  'capsule_polishing', 'lens_implant_settingup', 'viscous_agent_removal', 'tonifying_antibiotics', 'action_end']
    else:
        labels = ['background', 'main_incision_entry', 'cautery', 'peritomy', 'tunnel_suture', 'hydroprocedure', 'conjunctival_cautery', 'tunnel', 
                  'nucleus_prolapse', 'OVD_IOL_insertion', 'sideport', 'scleral_groove', 'OVD_injection', 'cortical_wash', 'OVD_wash', 'stromal_hydration', 
                  'nucleus_delivery', 'incision', 'capsulorrhexis', 'AB_injection_and_wash']
    return labels


def get_colormap(labels, cm_name='viridis'):
    colormap = colormaps[cm_name]
    colors = colormap(np.linspace(0, 1, len(labels)))
    color_mapping = {labels[i]: colors[i, :][:3] for i in range(len(labels))}
    return color_mapping


def plot_graphs_for_dataset(dataset_name, split, output_dir):
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    files = os.listdir(f"./results/{dataset_name}/split_{split}/")
    files.sort()
    cases = files[::2]    
    
    labels = get_labels(dataset_name)
    color_mapping = get_colormap(labels)
    
    print('Plotting results: ', end='')
    for case in cases:
        print(f'{case}', end=', ' if case != cases[-1] else '\n')
        pred_path = f"./results/{dataset_name}/split_{split}/{case}"
        gt_path = f"./data/{dataset_name}/groundTruth/{case}.txt"

        predictions = read_file(pred_path).split('\n')[1].split()
        if predictions[-1] == '': predictions = predictions[:-1]
        ground_truth = read_file(gt_path).split('\n')
        if ground_truth[-1] == '': ground_truth = ground_truth[:-1]
        
        labels_gt, start_gt, end_gt = get_labels_start_end_time(ground_truth)
        labels_pred, start_pred, end_pred = get_labels_start_end_time(predictions)

        data_gt = {"Ground Truth": [end_gt[i] - start_gt[i] for i, _ in enumerate(start_gt)]}
        data_pred = {"Prediction": [end_pred[i] - start_pred[i] for i, _ in enumerate(start_pred)]}
        
        fig, axs = plt.subplots(2, 1, figsize=(25, 6), sharex=True, sharey=True, layout="tight")
        axs[0].axis('off')
        axs[1].axis('off')

        colors_gt = [color_mapping[elm] for elm in labels_gt]
        ax = survey(data_gt, labels_gt, colors_gt, axs[0])
        colors_pred = [color_mapping[elm] for elm in labels_pred]
        ax2 = survey(data_pred, labels_pred, colors_pred, axs[1])
        
        plt.show()
        fig.savefig(output_dir + case + ".png")
        plt.close()

        
def plot_confusion_matrix(ground_truth, predictions, actions_dict, output_dir, normalized='true'):
    action_labels = list(actions_dict.keys())
    mat = metrics.confusion_matrix(ground_truth, predictions, labels=action_labels)
    fig, ax = plt.subplots(figsize=(16,16))
    suffix = 'Normalized' if normalized is not None else 'Default'
    form = ".2%" if normalized is not None else "d"
    print(f"Plotting confusion matrix, plot-mode: {suffix.lower()}...")
    
    display = metrics.ConfusionMatrixDisplay.from_predictions(ground_truth, predictions, labels=action_labels, 
                                                              normalize=normalized, display_labels=action_labels)
    display.plot(ax=ax, cmap="viridis", values_format=form, colorbar=False)
    # ax.set_title(f"Confusion Matrix - {suffix}")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40)
    ax.set_yticklabels(['\n_'.join(l.get_text().split('_')) for l in ax.get_yticklabels()], rotation=0)
    plt.rcParams.update({'font.size': 14})
    plt.show()
    fig.savefig(os.path.join(output_dir, f"confusion_matrix_{suffix.lower()}.png"))
    plt.close()


# dataset = "sics73_rgb"
# plot_graphs_for_dataset(dataset, 0, f"./results/{dataset}/figures/")
