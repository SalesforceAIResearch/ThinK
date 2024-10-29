
<div align="center">
  <a href="https://github.com/SalesforceAIResearch/ThinK/tree/main"><img width="400px" height="auto" src="./images/logo_think.png"></a>
</div>
<br/>

<div align="center">

  ![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)
  [![License](https://img.shields.io/badge/License-Apache-green.svg)]()
 [![GitHub star chart](https://img.shields.io/github/stars/SalesforceAIResearch/ThinK?style=social)](https://star-history.com/#SalesforceAIResearch/ThinK)

</div>
<p align="center">
  <a href="https://arxiv.org/abs/2407.21018">Paper</a> |
  <a href="https://github.com/SalesforceAIResearch/ThinK/tree/main?tab=readme-ov-file#Installation">Installation</a> |
  <a href="https://github.com/SalesforceAIResearch/ThinK/tree/main?tab=readme-ov-file#Eviction">Eviction</a> |
  <a href="https://github.com/SalesforceAIResearch/ThinK/tree/main?tab=readme-ov-file#Quantization">Quantization</a>
</p>

We provide three implementations. `ThinK_eager` contains the code for eager attention, `ThinK_flash` utilizes FlashAttention and `TinK_KIVI` which intergrates with KV quantization. Please note that the current implementations may not be fully optimized, and we are actively working on improving their efficiency. We use LongBench to evaluate the performance.

## ✅ TODO

- [ ] Support More Models
- [ ] Support Multi-GPUs
- [ ] Optimize Efficiency

# Installation
Step 1: Clone this repository

Step 2: Setup Environments
```shell
conda create -n think python=3.10
conda activate think
pip install -r requirements.txt
```

# Evaluation
## Eviction
Evaluate on LongBench: You can first modify the hyperparameters in `scripts/scripts_longBench/eval.sh`(e.g., pruning_ratio)

```shell
cd ThinK_flash
sh ./scripts/scripts_longBench/eval.sh
```

Results:
```shell
sh ./scripts/scripts_longBench/metrics.sh
```

## Quantization
```shell
cd ThinK_kivi
```
Set up the environments as per the instructions from KIVI, adding an additional argument, `pruning_ratio`. Currently, only LLaMA-2 is supported.

# Notes
Users need to make their own assessment regarding any obligations or responsibilities under the corresponding licenses or terms and conditions pertaining to the original datasets and data. This repository is being released for research purposes only.

# Citation
```markdown
@article{xu2024think,
  title={ThinK: Thinner Key Cache by Query-Driven Pruning},
  author={Xu, Yuhui and Jie, Zhanming and Dong, Hanze and Wang, Lei and Lu, Xudong and Zhou, Aojun and Saha, Amrita and Xiong, Caiming and Sahoo, Doyen},
  journal={arXiv preprint arXiv:2407.21018},
  year={2024}
}
```

## Acknowledgement
This repo builds on the [SnapKV](https://github.com/FasterDecoding/SnapKV), [PyramidKV](https://github.com/Zefan-Cai/PyramidKV/tree/main?tab=readme-ov-file),
[KIVI](https://github.com/jy-yuan/KIVI/tree/main) repos.