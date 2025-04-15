import numpy as np
import click
import os 
from os.path import join


@click.command()
@click.option('--input_path', help='Path to feature folder that has to be prepared for training')
@click.option('--gt_path', help='Path to corresponding ground truth folder')
@click.option('--check', help='Only check dimensions', default=False, is_flag=True)
@click.option('--transpose', help='Transpose found np arrays', default=False, is_flag=True)
def run(input_path, gt_path, check, transpose):
    features = os.listdir(input_path)

    for path in features:
        feat = np.load(join(input_path, path))
        if transpose:
            print(f'Transposing {path} now with shape {feat.shape}')
            new_feature = np.transpose(feat)
            np.save(join(input_path, path), new_feature)
            continue

        # load corresponding gt
        ground_truth_path = join(gt_path, os.path.splitext(path)[0] + '.txt')
        if not os.path.exists(ground_truth_path):
            print(f'GT file {ground_truth_path} does not exist, skipping...')
            continue
        ground_truth = open(ground_truth_path, 'r').readlines()
        gt_len = len(ground_truth)

        difference = gt_len - feat.shape[1]
        if feat.shape[0] > feat.shape[1] and feat.shape[0] != 1024:
            print('Incorrect feature shape, is the matrix transposed yet? ', feat.shape)
            break
        if gt_len == feat.shape[1]:
            print(f'Skip smaple {path}, dimensions already equal gt/dim {gt_len}/{feat.shape}')
            continue
        
        print(f'Comparision {path}: gt_len {gt_len}, feat_shape {feat.shape}, diff {difference}') 

        if difference < 0:
            print(f'Feature {path} is longer than gt, removing {abs(difference)} frames')
            new_feature = feat[:, :gt_len, ...]
            np.save(join(input_path, path), new_feature)
        elif difference > 0:
            raise ValueError(f'Feature {path} is shorter than gt, please check your data!')


if __name__ == "__main__": 
    run()
