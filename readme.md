# Transformer4SED

Implementations of "[MAT-SED: A Masked Audio Transformer with Masked-Reconstruction Based Pre-training for Sound Event Detection](https://www.isca-archive.org/interspeech_2024/cai24_interspeech.html)" (accepted by Interspeech 2024).

## Introduction
MAT-SED (**M**asked **A**udio **T**ransformer for **S**ound **E**vent **D**etection) is a pure Transformer-based SED model.
-  MAT-SED comprising two main components: the encoder network (green) and the context network (yellow), both of which are based on Transformer structures.
- The Transformer structures lack some of the inductive biases inherent to RNNs, such as sequentiality, which makes the Transformer-based context networks do not generalize well when trained on insufficient data. To address this problem, we use the **masked-reconstruction task** to pre-train the context network in the self-supervised manner.
<div align="center"><img src="./archive/img/structure.png" width=60%></div>

- During the semi-supervised fine-tuning stage, we propose a novel strategy 
 termed **global-local feature fusion strategy**,  as depicted in the below figure, to combine both local and global characteristics.
<div align="center"><img src="./archive/img/sliding-windows.png" width=60%></div>

## Runing
``` shell
cd  ./exps/mat-sed/base
./train.sh
```

## Results
DESED validation set
| Post Processing                   | PSDS1 | PSDS2 |
| --------------------------------- | ----- | ----- |
| Median Filter                     | 0.587 | 0.792 |
| Max Filter                        | 0.090 | 0.896 |
| Event Bounding Boxes <br />(SEBB) | 0.602 | -     |


## Others
- It is worth noting that our code is implemented using pytorch, refactored from the DCASE official pytorch-lighting baseline. If you feel that the baseline implementation with pytorch-lighting is not flexible, then developing on the basis of this code may be a good choice.

-  Multi-GPU training via `nn.DataParallel()` is supported.
- Welcome to ask in the issues if any problems is encountered during code reproduction.

## Citation
```
@inproceedings{cai24_interspeech,
  title     = {MAT-SED: A Masked Audio Transformer with Masked-Reconstruction Based Pre-training for Sound Event Detection},
  author    = {Pengfei Cai and Yan Song and Kang Li and Haoyu Song and Ian McLoughlin},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {557--561},
  doi       = {10.21437/Interspeech.2024-714},
  issn      = {2958-1796},
}
```
