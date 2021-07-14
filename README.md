# Miscellaneous-Self-Attention-Mechanisms-in-Transformers

> We survey and implement several new attention mechanisms for transformer-based models, which are proposed to accelerate training.

**The algorithms we implement can be catagorized into four types:**
- Sparse Attention
- LSH
- Routing Transformer
- Synthesizer


## Table of Contents
 - [Introduction](#introduction)
 - [Example](#example)
 - [Citations](#citations)
 
## Introduction 
[![INSERT YOUR GRAPHIC HERE](https://imgur.com/So7ZcF1.png)]()

In recent years, there are a great number of algorithsms proposed, trying to improve the efficiency of self-attention in Transformers. Some of them have already been realized on NLP or CV tasks, while the others are merely examined with either mathematical theories or some simple experiments. Here we implement all of them; especially, we make the I/O of our attention algorithms compatible with those of HuggingFace, making them easy-to-use. Illustrations are shown above. Following, we elaborate these four kinds attentions briefly.

### Sparse Attention
 - There are two types of attention masks proposed in the original work: **Strided** and **Fixed**.
 - Here half of the attention heads (Light Cyan) attend to the local information, while the other heads (Blue) attend to the global information.

### LSH
 - We implement four kinds of LSH/ALSH algorithms.
 - The default number of attended keys is 32.

### Routing Transformer
 - Here, Routing Transformer use K-means to determine the keys to be attended to.

### Synthesizer
 - There are two different types of Synthesizer being proposed: **Dense** and **Random**.
 - In our experiments, this algorithms achieve the best performance (With some of our proposed initialization techiniques).
 - For our **Random** attention module, we make the attention weights become trainable model parameters. And we initialize the weights of 7 of the attention heads with handcrafted patterns before pretraining. [![INSERT YOUR GRAPHIC HERE](https://i.imgur.com/C0x3YQZ.png)]()
 - After pretraining, they become [![INSERT YOUR GRAPHIC HERE](https://i.imgur.com/ZeUzRl5.png)


### Experiment result
All these modules can be incorporated into [Mockingjay](https://github.com/andi611/Mockingjay-Speech-Representation) and pretrained and finetuned on [Librispeech](https://www.openslr.org/12). Please refer to [our paper](https://arxiv.org/pdf/2006.05174.pdf) for more details of the experiment results.

## Example
See the examples in `example.py`.


## Citations
```bibtex
@article{
    title={Generating long sequences with sparse transformers},
    author={Child, Rewon and Gray, Scott and Radford, Alec and Sutskever, Ilya},
    year={2019},
    journal={arXiv preprint arXiv:1904.10509},
    year={2019}
}

@article{
    title={Efficient Content-Based Sparse Attention with Routing Transformers},
    author={Roy, Aurko and Saffar, Mohammad and Vaswani, Ashish and Grangier, David},
    year={2020},
    journal={arXiv preprint arXiv:2003.05997},
    year={2020}
}

@inproceedings{10.1145/3219819.3219971,
  title={Accurate and fast asymmetric locality-sensitive hashing scheme for maximum inner product search},
  author={Huang, Qiang and Ma, Guihong and Feng, Jianlin and Fang, Qiong, and Tung, Anthony KH},
  booktitle={The 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining},
  pages={1561–-1570},
  year={2018},
  organization={ACM SIGKDD}
}

@inproceedings{raganato-etal-2020-fixed,
    title = {Fixed Encoder Self-Attention Patterns in Transformer-Based Machine Translation},
    author = {Raganato, Alessandro  and Scherrer, Yves  and Tiedemann, J{\"o}rg},
    booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2020},
    year = {2020},
    publisher = {Association for Computational Linguistics},
    pages = {556--568}
}

@article{DBLP:journals/corr/abs-2005-00743,
  title     = {Synthesizer: Rethinking Self-Attention in Transformer Models},
  author    = {Yi, Tay and Dara, Bahri and Donald, Metzler and Da{-}Cheng, Juan and Zhe, Zhao and Che, Zheng},
  journal   = {CoRR},
  volume    = {abs/2005.00743},
  year      = {2020},
  url       = {https://arxiv.org/abs/2005.00743},
  archivePrefix = {arXiv},
  eprint    = {2005.00743},
}


@article{
    title={Input-independent Attention Weights Are Expressive Enough: A Study of Attention in Self-supervised Audio Transformers},
    author={Wu, Tsung-Han and Hsieh, Chun-Chen and Chen, Yen-Hao and Chi, Po-Han and Lee, Hung-yi},
    year={2020},
    journal={arXiv preprint arXiv:2006.05174},
    year={2020}
}
```

