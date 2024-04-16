#!/usr/bin/env python

import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Evaluate model trained on a dataset from a certain epoch")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset path")
    parser.add_argument("--split", type=str, required=True, help="Split parameter")
    parser.add_argument("--num_epochs", type=int, required=True, help="Number of epochs")
    parser.add_argument("--features_dim", type=int, required=True, help="Features dimension")
    parser.add_argument("--num_layers_PG", type=int, default=11, help="Number of layers for PG")
    parser.add_argument("--num_layers_R", type=int, default=10, help="Number of layers for R")
    parser.add_argument("--num_R", type=int, default=3, help="Number of R")
    parser.add_argument("--loss_mse", type=float, default=0.15, help="Loss mse lambda param")
    parser.add_argument("--loss_dice", type=float, default=0.0, help="Loss dice param")
    parser.add_argument("--loss_focal", type=float, default=0.0, help="Focal loss, gamma parameter")
    parser.add_argument("--weights", type=str, default=None, help="Weights path")

    args = parser.parse_args()

    # Execute main.py
    main_command = ["python", "main.py",
                    "--action", "predict",
                    "--dataset", args.dataset,
                    "--split", args.split,
                    "--num_epochs", str(args.num_epochs),
                    "--num_layers_PG", str(args.num_layers_PG),
                    "--num_layers_R", str(args.num_layers_R),
                    "--num_R", str(args.num_R),
                    "--features_dim", str(args.features_dim),
                    "--loss_mse", str(args.loss_mse),
                    "--los_focal", str(args.loss_focal),
                    "--loss_dice", str(args.loss_dice),
                    "--weights", args.weights]
    subprocess.run(main_command, check=True)

    # Execute eval.py
    eval_command = ["python", "eval.py",
                    "--dataset", args.dataset,
                    "--split", args.split]
    subprocess.run(eval_command, check=True)


if __name__ == "__main__":
    main()
