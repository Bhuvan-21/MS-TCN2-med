import click
import pandas as pd
import numpy as np
import os
from os.path import join
from sklearn.model_selection import KFold
import random


@click.command()
@click.option('--input_path', help='Path to input folder (groundTruth_folder)')
@click.option('--output_path', help='Path where splits should be saved')
@click.option('--k', default=4, help='Number of splits to create')
@click.option('--testset/--no_testset', default=True, is_flag=True, help='Flag for creating additional test set')
def run(input_path, output_path, k, testset=False):
    """Create k splits for the ground truth files with training/validation splits. 
    Also creates an additional validation split for testing """

    random.seed(42)
    k_folds = kf = KFold(n_splits=k, shuffle=True, random_state=42)

    gt_files = os.listdir(input_path)
    random.shuffle(gt_files)

    test_size = len(gt_files) // (k+1) if testset else 0
    train_val_size = len(gt_files) - test_size

    test_split = gt_files[:test_size]
    gt_files = gt_files[test_size:]
    #gt_files = pd.DataFrame(data=gt_files, columns=['cases'])
    
    for i, (train_index, test_index) in enumerate(kf.split(gt_files)):
        print(f"Fold {i}:")
        print(f"  Train: {train_index}")
        print(f"  Validation:  {test_index}")

        write_split('train', i, train_index, gt_files, output_path)
        write_split('test', i, test_index, gt_files, output_path)

    print("Test split length: ", len(test_split))
    print("Train/val split length: ", len(gt_files))
    #print(gt_files.cases.tolist())
    print(test_split)

    if testset:
        write_split('train', 9, np.concatenate([train_index, test_index]), gt_files, output_path)
        write_split('test', 9, list(range(len(test_split))), test_split, output_path)

def write_split(set_name, split, index, files, path):
    if not os.path.exists(path): os.mkdir(path)
    if not os.path.exists(join(path, 'splits/')): os.mkdir(join(path, 'splits/'))

    file_name = join(path, 'splits/', set_name + '.split' + str(split) + '.bundle')
    with open(file_name, 'w') as file:
        for idx in index:
            file.write(files[idx] +'\n') 
    


if __name__ == '__main__':
    run()
