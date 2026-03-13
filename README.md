# Word2Vec in Pure NumPy

This repository contains a from-scratch implementation of **Word2Vec** in **pure NumPy**, without using PyTorch, TensorFlow, or other ML frameworks.

The implemented model is **Skip-gram with Negative Sampling (SGNS)**. The focus of the project is the manual implementation of the core training loop, including forward pass, loss computation, gradient computation, and parameter updates.

This project was developed as part of an internship-style technical task.

## Features

- Skip-gram with Negative Sampling
- Pure NumPy implementation
- Vocabulary building and token-to-index encoding
- Frequent-word subsampling
- Negative sampling with unigram distribution raised to the power `0.75`
- SGD-based embedding training
- Cosine similarity for nearest words
- Basic word analogy evaluation

## Files

- `train.py`
- `data/holmes.txt`

## How to Run

Install NumPy:

```bash
pip install numpy
```

Run Training:

```bash
python train.py
```

Epoch 30/30 finished. Average loss: 2.3577

Most similar to 'holmes':
  mr              0.6763
  sherlock        0.6617
  basket          0.5663
  scotland        0.5638
  blandly         0.5604

Most similar to 'sherlock':
  holmes          0.6617
  friend          0.5624
  called          0.5590
  chair           0.5391
  greeting        0.5147

Most similar to 'case':
  clearing        0.5569
  matter          0.5467
  forced          0.5384
  conclusions     0.5350
  steps           0.5328

Most similar to 'house':
  manor           0.6268
  copper          0.5396
  beeches         0.5330
  loving          0.5290
  doran           0.5189

By default, the script uses data/holmes.txt. It can also be adapted to larger corpora such as text8.

## Reference

Mikolov et al., Efficient Estimation of Word Representations in Vector Space (2013)
Mikolov et al., Distributed Representations of Words and Phrases and their Compositionality (2013)
