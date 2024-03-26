"""
Run hyperparamter search on k folds. Trained on different splits, evaluated on the same
TODO: Test on all folds and calc average metrics
"""

import itertools
import subprocess 
from os.path import join
import click


@click.command()
@click.option('--dataset', help='Name of the dataset (name as in ./data folder)')
@click.option('--folds', '-k', default=4, help='Number of folds to run')
@click.option('--num_epochs', default=100, help='Number of training epochs to run for each fold')
@click.option('--features_dim', default=2048, help='Feature dimension of the dataset (for i3d 1024 rgb only, 2048 for flow/rgb)')
def run(dataset, folds, num_epochs, features_dim):
    """ Run training and evaluation on multiple folds, defined in the ./data/<name> folder"""

    base_dir = './'
    main_path = join(base_dir, 'main.py')
    eval_path = join(base_dir, 'eval.py')

    hp_space = {
        'lambda': [0.1, 0.15, 0.25],
        'refinement': [0, 1, 3]
    }
    keys, values = zip(*hp_space.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print('Testing the following hyperparameter {len(permutations)} combinations: ', permutations)

    for i, perm in enumerate(permutations):
        cur_fold = i % folds
        print(f'Running permutation {perm} on dataset {dataset}_{cur_fold}')
        # Train model on split
        subprocess.run(['python', main_path, 
                      '--action=train', 
                      f'--dataset={dataset}', 
                      f'--split={cur_fold}', 
                      f'--num_epochs={num_epochs}',
                      f'--features_dim={features_dim}',
                      f'--loss_lambda={perm["lambda"]}',
                      f'--num_R={perm["refinement"]}',
                      '--num_layers_PG=11', '--num_layers_R=10'
                      ], shell=False, check=False)

        #Run predictions on test set
        evaluation_split = 0
        subprocess.run(['python', main_path,
                      '--action=predict', 
                      f'--dataset={dataset}', 
                      f'--split={evaluation_split}', 
                      f'--num_epochs={num_epochs}',
                      f'--features_dim={features_dim}',
                      f'--loss_lambda={perm["lambda"]}',
                      f'--num_R={perm["refinement"]}',
                      '--num_layers_PG=11', '--num_layers_R=10', '--num_R=3'
                      ], shell=False, check=False)
        #Evaluate predictions
        subprocess.run(['python', eval_path, f'--dataset={dataset}', f'--split={evaluation_split}'], shell=False, check=False)


if __name__ == '__main__':
    run(None, None, None, None)
