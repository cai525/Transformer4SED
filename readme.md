# Transformer4SED

Implementations of "[MAT-SED: A Masked Audio Transformer with Masked-Reconstruction Based Pre-training for Sound Event Detection](https://arxiv.org/abs/2408.08673)" (accepted by Interspeech 2024).

## Runing
``` shell
cd /home/cpf/code/open/Transformer4SED/exps/mat-sed/base
./train.sh
```
## Results

| Post Processing                   | PSDS1 | PSDS2 |
| --------------------------------- | ----- | ----- |
| Median Filter                     | 0.587 | 0.792 |
| Max Filter                        | 0.090 | 0.896 |
| Event Bounding Boxes <br />(SEBB) | 0.602 | -     |


## Others
- It is worth noting that our code is implemented using pytorch, based on the DCASE official pytorch-lighting baseline. If you feel that the baseline implementation with pytorch-lighting is not flexible, then developing on the basis of this code is a good choice.

-  Multi-GPU training via `nn.DataParallel()` is supported.
- Welcome to ask in the issues if any problems is encountered during code reproduction.
