# miipher
This repository provides an unofficial implementation of speech restoration model Miipher.
Miipher is originally proposed by Koizumi et. al. [arxiv](https://arxiv.org/abs/2303.01664)
Please note that the model provided in this repository doesn't represent the performance of the original model proposed by Koizumi et. al. as this implementation differs in many ways from the paper.

# Installation
Install with pip. The installation is confirmed on Python 3.10.11
```python
pip install git+https://github.com/CShulby/miipher
```

# Pretrained model
The pretrained model is trained on [LibriTTS-R](http://www.openslr.org/141/) and [JVS corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus),
and provided in **CC-BY-NC-2.0 license**.

# Inference in Batch 

```
python run_miipher.py
```

You can also run in parallel on CPU by running the following script and passing a list of the wav files (note they should have corresponding transcriptions in the same folder):

```
python run_miipher_parallel.py --wav_list wav_list
```

If you are still hungry for more you can run the same way using full GPU inference:

```
python run_miipher_gpu.py --wav_list wav_list
```

Tests on an RTX 4090 showed a difference of 3.5x real time with the parallel CPU script vs. 30x real time on GPU

# Differences from the original paper
| | [original paper](https://arxiv.org/abs/2303.01664) | This repo |
|---|---|---|
| Clean speech dataset | proprietary | [LibriTTS-R](http://www.openslr.org/141/) and [JVS corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus) |
| Noise dataset |  TAU Urban Audio-Visual Scenes 2021 dataset | TAU Urban Audio-Visual Scenes 2021 dataset and Slakh2100 |
| Speech SSL model | [W2v-BERT XL](https://arxiv.org/abs/2108.06209) | [WavLM-large](https://arxiv.org/abs/2110.13900) |
| Language SSL model | [PnG BERT](https://arxiv.org/abs/2103.15060) | [XPhoneBERT](https://github.com/VinAIResearch/XPhoneBERT) |
| Feature cleaner building block | [DF-Conformer](https://arxiv.org/abs/2106.15813) | [Conformer](https://arxiv.org/abs/2005.08100) |
| Vocoder | [WaveFit]https://arxiv.org/abs/2210.01029) | [HiFi-GAN](https://arxiv.org/abs/2010.05646) |
| X-Vector model | Streaming Conformer-based speaker encoding model | [speechbrain/spkrec-xvect-voxceleb](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb) |

# LICENSE
Code in this repo: MIT License

Weights on huggingface: CC-BY-NC-2.0 license
