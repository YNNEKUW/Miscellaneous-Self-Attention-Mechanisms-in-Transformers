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

@article{
    title={Input-independent Attention Weights Are Expressive Enough: A Study of Attention in Self-supervised Audio Transformers},
    author={Wu, Tsung-Han and Hsieh, Chun-Chen and Chen, Yen-Hao and Chi, Po-Han and Lee, Hung-yi},
    year={2020},
    journal={arXiv preprint arXiv:2006.05174},
    year={2020}
}
```

