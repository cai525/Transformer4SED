# Transformer4SED
Transformer4SED is a  repository which aims to collect Transformer-based sound event detection (SED) algorithms. 

## Features

- Implemented using pytorch, refactored from the DCASE official pytorch-lighting baseline
- Kaldi style recipes;
- [TODO] Support for commonly used datasets in the sound event detection field, including DESED, MAESTRO, audioset-strong, etc.



## recipes
### [MAT-SED (interspeech 2024)](./docs/mat-sed/readme.md)

MAT-SED (**M**asked **A**udio **T**ransformer for **S**ound **E**vent **D**etection) is a pure Transformer-based SED model with masked-reconstruction-based pre-training.

<div align="center"><img src="./docs/mat-sed/img/structure.png" width=40%></div>

### [PMAM (ICASSP 2025)](./docs/pmam/readme.md)

**P**rototype based **M**asked **A**udio **M**odel (PMAM) is a self-supervised representation learning algorithm designed for frame-level audio tasks like sound event detection, to better exploit unlabeled data. 
<div align="center"><img src="./docs/pmam/img/pmam.png" width=60%></div>
