# ThinK: Thinner Key Cache by Query-Driven Pruning
<p align="center">
  <a href="https://arxiv.org/abs/2407.21018">Paper</a>
</p>

We provide two kinds of implementation. `ThinK_eager` contains the code for eager attention while `ThinK_flash` includes the FlashAttention. Note that the current implementation may not efficient enough.

## ✅ TODO

- [ ] Support More Kinds of Models
- [ ] Support Multi-GPUs
- [ ] Optimize Efficiency

## Installation
Step 1: Clone this repository

Step 2: Setup Environments
```shell
conda create -n think python=3.10
conda activate think
pip install -r requirements.txt
```

## Getting Started
Evaluate on LongBench: You can first modify the hyperparameters in `scripts/scripts_longBench/eval.sh`(e.g., pruning_ratio)

```shell
cd ThinK_flash
sh ./scripts/scripts_longBench/eval.sh
```

Results:
```shell
sh ./scripts/scripts_longBench/metrics.sh
```

## Experiments ThinK+KIVI
```shell
cd ThinK_kivi
```
Set up environments following the instrunctions of KIVI. We add one more argument `pruning_ratio`. Currently, only LLaMA-2 is supported.

## Citation
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