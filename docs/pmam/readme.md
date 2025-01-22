# PMAM
[![PWC](https://img.shields.io/badge/arXiv-2409.17656-brightgreen)](https://arxiv.org/abs/2409.17656)
Implementations of "[Prototype based Masked Audio Model for Self-Supervised Learning of Sound Event Detection](https://arxiv.org/abs/2409.17656)" (accepted by to ICASSP2025). 

## Overview
**P**rototype based **M**asked **A**udio **M**odel (PMAM) is a self-supervised representation learning algorithm designed for frame-level audio tasks like sound event detection, to better exploit unlabeled data. 
<br/>
<div align="center"><img src="./img/pmam.png" width=80%></div>
<br/>

- Semantically rich frame-level pseudo labels are constructed from a Gaussian mixture model (GMM) based <ins>prototypical distribution modeling</ins>;
- The pseudo labels supervise the learning of a Transformer-based <ins>masked audio model</ins>;
-  The prototypical distribution modeling and the masked audio model training are performed iteratively to enhance the quality of pseudo labels;
- A novel <ins>  prototype-wise binary cross-entropy loss</ins> is employed instead of the widely used InfoNCE loss, to provide independent loss contributions from different prototypes;

## Demo
Tip: Turn on the sound when watching the demo.

https://github.com/user-attachments/assets/a23c3713-88b9-4603-9929-16a184c2c292

https://github.com/user-attachments/assets/cce21d30-7214-4277-8142-64a90debe33b

https://github.com/user-attachments/assets/f2904264-59ef-4b1a-9034-a9ae3a9d386e

https://github.com/user-attachments/assets/d9f5892f-2b64-4be8-b90d-20f22a692fc5

https://github.com/user-attachments/assets/ca9d2c20-8012-4950-a947-5ea773680a30


> The demos are made by [this script](../..//src/utils/visualization/gen_video_demo.py).

## Running
1. Install required libraries.
```shell
pip install -r requirements.txt
```

2. Use the global replacement function, which is supported by most IDEs, to replace `ROOT-PATH` with your custom root path of the project. And the dataset paths in the configuration files also need to be replaced with your custom dataset paths.

3. Download the pretrained PaSST model weight, if you have not downloaded it before.
```shell
wget -P ./pretrained_model  https://github.com/kkoutini/PaSST/releases/download/v0.0.1-audioset/passt-s-f128-p16-s10-ap.476-swa.pt
``` 

4. Download the pseudo-labels and GMM model required for running. We have uploaded the pseudo-labels and GMM model needed for the second iteration to Hugging Face.
``` shell
cd ./exps/pmam
mkdir run
cd run
wget https://huggingface.co/CPF2/PMAM/resolve/main/meta.gz
wget https://huggingface.co/CPF2/PMAM/resolve/main/tokenizer.gz
tar -zxvf meta.gz
tar -zxvf tokenizer.gz
```

5. Model training : self-supervised training + semi-supervised finetuning.
``` shell
./train.sh
```

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
