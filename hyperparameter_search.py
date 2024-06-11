"""
Run hyperparamter search on k folds. Trained on different splits, evaluated on the same
TODO: Test on all folds and calc average metrics
"""

import itertools
import subprocess 
from os.path import join
import click
from random import shuffle


@click.command()
@click.option('--dataset', help='Name of the dataset (name as in ./data folder)')
@click.option('--tries', '-k', default=10, help='Number of permuations to try (random search)')
@click.option('--num_epochs', default=100, help='Number of training epochs to run for each fold')
@click.option('--features_dim', default=2048, help='Feature dimension of the dataset (for i3d 1024 rgb only, 2048 for flow/rgb)')
@click.option('--device', default='cuda:0', help='Device to run on')
def run(dataset, tries, num_epochs, features_dim, device):
    """ Run training and evaluation on multiple folds, defined in the ./data/<name> folder"""

    base_dir = './'
    main_path = join(base_dir, 'main.py')
    eval_path = join(base_dir, 'eval.py')
    eval2_path = join(base_dir, 'evaluation.py')

    hp_space = {
        'lambda': [0.25, 0.30, 0.35, 0.40],
        'num_R': [3, 4, 5],
        'layers_PG': [12, 13],
        'layers_R': [11, 12, 13],
        'amse': [20] 
    }
    keys, values = zip(*hp_space.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    shuffle(permutations)

    print(f'Testing the following hyperparameter {len(permutations)} combinations, e.g.: ', permutations[0])

    for i, perm in enumerate(permutations[:tries]):
        cur_fold = 0 # i % folds
        print(f'Running permutation #{i} {perm} on dataset {dataset}_split{cur_fold}')
        # Train model on split
        subprocess.run(['python', main_path, 
                      '--action=train', 
                      f'--dataset={dataset}', 
                      f'--split={cur_fold}', 
                      f'--num_epochs={num_epochs}',
                      f'--features_dim={features_dim}',
                      f'--loss_mse={perm["lambda"]}', 
                      f'--num_R={perm["num_R"]}',
                      f'--num_layers_PG={perm["layers_PG"]}', 
                      f'--num_layers_R={perm["layers_R"]}',
                      f'--device={device}', 
                      '--weights=None',
                      f'--window_mse={perm["amse"]}',
                      '--adaptive_mse'
                      ], shell=False, check=False)

        #Run predictions on test set
        evaluation_split = 0
        subprocess.run(['python', main_path,
                      '--action=predict', 
                      f'--dataset={dataset}', 
                      f'--split={evaluation_split}', 
                      f'--num_epochs={num_epochs}',
                      f'--features_dim={features_dim}',
                      f'--loss_mse={perm["lambda"]}', 
                      f'--num_R={perm["num_R"]}',
                      f'--num_layers_PG={perm["layers_PG"]}', 
                      f'--num_layers_R={perm["layers_R"]}',
                      f'--device={device}',
                      '--weights=None',
                      f'--window_mse={perm["amse"]}',
                      '--adaptive_mse'
                      ], shell=False, check=False)
        #Evaluate predictions
        subprocess.run(['python', eval_path, f'--dataset={dataset}', f'--split={evaluation_split}'], shell=False, check=False)
        subprocess.run(['python', eval2_path, f'--dataset={dataset}', f'--split={evaluation_split}'], shell=False, check=False)


if __name__ == '__main__':
    run(None, None, None, None)
