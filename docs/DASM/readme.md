<h1 align="center">Detect Any Sound: Open-Vocabulary Sound Event Detection with Multi-Modal Queries</h1>


<div align='center'>
  <span class="author-block">Pengfei Cai<sup>1</sup>, Yan Song<sup>1</sup>, Qing Gu<sup>1</sup>, Nan Jiang<sup>1</sup>, Haoyu Song<sup>2</sup>, Ian McLoughlin<sup>2</sup></span>
</div>

<div align='center'>
  <span class="author-block"><sup>1</sup><strong> University of
Science and Technology of China, China</strong>,</span>
 
 <span><sup>2</sup><strong> ICT Cluster, Singapore Institute of Technology, Singapore
 </strong></span>
</div>

<div align='center'>
<p>
    <a href="https://cai525.github.io/Transformer4SED/demo_page/DASM/index.html" target="_blank"><img src="https://img.shields.io/badge/Project-Demo_page-green" alt="demo homepage"></a>
    <a href="http://arxiv.org/abs/2507.16343" target="_blank"><img src="https://img.shields.io/badge/arXiv-2507.16343-red" alt="demo homepage"></a>
    <a href="https://huggingface.co/CPF2/detect_any_sound/tree/main" target="_blank"><img src="https://img.shields.io/badge/huggingface-model_weights-yellow" alt="huggingface"></a>
<p>
</div>


Implementations of "Detect Any Sound: Open-Vocabulary Sound Event Detection with Multi-Modal Queries" (accepted by to MM 2025).

<div align='center'><img src='../demo_page/DASM/data/main.png' width=90%></div>

## Inference
1. Download model weights from huggingface
```shell
cd ./pretrained_model
git lfs install
git clone https://huggingface.co/CPF2/detect_any_sound
cd ../..
```
2. Set dependencies: DASM depends on MGA-CLAP to extract event queries.
```shell
cd third_parties
git clone git@github.com:Ming-er/MGA-CLAP.git
```


3. The inference script is located at [./recipes/audioset_strong/detect_any_sound/detect_any_sound.ipynb](recipes/audioset_strong/detect_any_sound/detect_any_sound.ipynb).


## Setting of the AS-partial expriment
Read [../../meta/audioset_strong/readme.md](../../meta/audioset_strong/readme.md) for detail.

## TODO
- [x] Deploy the web demo;  
- [x] Release model and inference code;  
- [ ] Release training code and configurations;  
