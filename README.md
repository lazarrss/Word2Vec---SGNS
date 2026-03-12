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

 Run Training:

```bash
python train.py

By default, the script uses data/holmes.txt. It can also be adapted to larger corpora such as text8.

## Reference

Mikolov et al., Efficient Estimation of Word Representations in Vector Space (2013)
Mikolov et al., Distributed Representations of Words and Phrases and their Compositionality (2013)
