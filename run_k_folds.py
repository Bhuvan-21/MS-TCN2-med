import subprocess 
from os.path import join
import click
import pandas as pd


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

    for k in range(folds):
        print(f'Running fold {k} on dataset {dataset}')
        # Train model on split
        subprocess.run(['python', main_path, 
                      '--action=train', 
                      f'--dataset={dataset}', 
                      f'--split={k}', 
                      f'--num_epochs={num_epochs}',
                      f'--features_dim={features_dim}',
                      '--num_layers_PG=13', '--num_layers_R=13', '--num_R=4', '--loss_mse=0.35', '--adaptive_mse'
                      ], shell=False, check=False)

        #Run predictions on test set
        subprocess.run(['python', main_path, 
                      '--action=predict', 
                      f'--dataset={dataset}', 
                      f'--split={k}', 
                      f'--num_epochs={num_epochs}',
                      f'--features_dim={features_dim}',
                      '--num_layers_PG=13', '--num_layers_R=13', '--num_R=4', '--loss_mse=0.35', '--adaptive_mse'
                      ], shell=False, check=False)

        #Evaluate predictions
        subprocess.run(['python', eval_path, f'--dataset={dataset}', f'--split={k}'], shell=False, check=False)


def print_last_result(path):
    df = pd.read_excel(join(path, 'results.xlsx'))
    print(df.tail(1))


if __name__ == '__main__':
    run(None, None, None, None)
