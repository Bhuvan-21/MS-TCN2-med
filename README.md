# MS-TCN2 for surgery recordings
 Aim of this project is adapting the below-cited action segementation architecture for surgery phase recognition in medical recordings and specifically for cataract surgery:
 - based on the great publication and code from the publication MS-TCN++: Multi-Stage Temporal Convolutional Network for Action Segmentation (TPAMI 2020)
 - [MS-TCN++: Multi-Stage Temporal Convolutional Network for Action Segmentation](https://arxiv.org/pdf/2006.09220.pdf).
   
## Datasets 
- Adaption of the Cataract 101 dataset for this architecture
- SOON Release of the first public Small-Incision Cataract Surgery (SICS) dataset with 105 annotated videos
- Preprocessing and evaluation tools for both datasets

## Folder structure
- data/: here lives the training data of this project
- data/<dataset_name>/features/: i3d features of the used dataset
- data/<dataset_name>/groundTruth/: framewise annotation for the given dataset
- data/<dataset_name>/splits/: train/validation splits for the given dataset
- data/<dataset_name>/mapping.txt: integeter to phase name mapping (0-N)
- data/<dataset_name>/vides/: raw videos files (optional, features are used for training/validation

# Important files
- run.sh: Script to run train and prediction cycles and calculate point and bootstrap metrics
- model.py: Here you find the code for the MS-TCN++ model
- evaluation.py: bootstraped validation/test metrics (f1, acc, sens, spec, roc_acu, pr_auc)
- eval.py: point metrics like f1, acc, sens, spec, roc_acu, pr_auc (legacy)

## Based on:
```BibTeX
@article{li2020ms,
   author={Shi-Jie Li and Yazan AbuFarha and Yun Liu and Ming-Ming Cheng and Juergen Gall},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
    title={MS-TCN++: Multi-Stage Temporal Convolutional Network for Action Segmentation}, 
    year={2020},
    volume={},
    number={},
    pages={1-1},
    doi={10.1109/TPAMI.2020.3021756},
}

@inproceedings{farha2019ms,
  title={Ms-tcn: Multi-stage temporal convolutional network for action segmentation},
  author={Farha, Yazan Abu and Gall, Jurgen},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3575--3584},
  year={2019}
}

```
