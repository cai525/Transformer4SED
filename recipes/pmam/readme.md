# PMAM
[![PWC](https://img.shields.io/badge/arXiv-2409.17656-brightgreen)](https://arxiv.org/abs/2409.17656)
Implementations of "[Prototype based Masked Audio Model for Self-Supervised Learning of Sound Event Detection](https://arxiv.org/abs/2409.17656)" (submitted to ICASSP2025).


## Overview
**P**rototype based **M**asked **A**udio **M**odel (PMAM) is a self-supervised representation learning algorithm designed for frame-level audio tasks like sound event detection, to better exploit unlabeled data. 
<br/>
<div align="center"><img src="../../archive/img/pmam.png" width=80%></div>
<br/>

- Semantically rich frame-level pseudo labels are constructed from a Gaussian mixture model (GMM) based <ins>prototypical distribution modeling</ins>;
- The pseudo labels supervise the learning of a Transformer-based <ins>masked audio model</ins>;
-  The prototypical distribution modeling and the masked audio model training
 are performed iteratively to enhance the quality of pseudo labels, similar to E-step and M-step in the <ins>expectation-maximization (EM) algorithm</ins>;
- A novel <ins>  prototype-wise binary cross-entropy loss</ins> is employed instead of the widely used InfoNCE loss, to provide independent loss contributions from different prototypes;

## Citation
```
@misc{cai2024prototypebasedmaskedaudio,
      title={Prototype based Masked Audio Model for Self-Supervised Learning of Sound Event Detection}, 
      author={Pengfei Cai and Yan Song and Nan Jiang and Qing Gu and Ian McLoughlin},
      year={2024},
      eprint={2409.17656},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2409.17656}, 
}
```