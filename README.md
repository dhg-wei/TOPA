# TOPA
TOPA: Extend Large Language Models for Video Understanding via Text-Only Pre-Alignment

[arXiv](https://www.arxiv.org/pdf/2405.13911)

The code, pre-trained model, and data will be released in **June**.

## TODO:

Data

Pretrained models

## Train & Eval
### Text-only Pre-alignment
```
./scripts/pretrain/llama2_7b.sh
```
### Zero-shot inference
```
./scripts/eval/zeroshot_eval_egos.sh
./scripts/eval/zeroshot_eval_nextqa.sh
./scripts/eval/zeroshot_eval_star.sh
./scripts/eval/zeroshot_eval_tvqa.sh
```
### Evaluate on MVBench
[mvbench.ipynb](demos/mvbench.ipynb)

### Evaluate on video captioning benchmarks
[MSRVTT](demos/Eval_Cap_MSRVTT.ipynb)

[VATEX](demos/Eval_Cap_VATEX.ipynb)

## Acknowledgements
This repo benefits from [Flipped-VQA](https://github.com/mlvlab/Flipped-VQA), [LLaMA-Adapter](https://github.com/OpenGVLab/LLaMA-Adapter), [MVBench](https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/MVBENCH.md), [LLama2](https://github.com/meta-llama/llama) and [LLama3](https://github.com/meta-llama/llama3).


## Citations

```
@article{li2024topa,
  title={TOPA: Extend Large Language Models for Video Understanding via Text-Only Pre-Alignment},
  author={Li, Wei and Fan, Hehe and Wong, Yongkang and Kankanhalli, Mohan and Yang, Yi},
  journal={arXiv preprint arXiv:2405.13911},
  year={2024}
}
```