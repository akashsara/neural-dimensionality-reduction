# Compression of Word Embeddings via Neural Dimensionality Reduciton

Contributors: Akash Saravanan

This repository was created for the term project of CMPUT 651 - Deep Learning for NLP, a graduate course at the University of Alberta.

## Goal
---
A group of word embeddings offer 3 different avenues for compression. The first is to reduce the size of the vocabulary. The second is to reduce the precision (quantization) of the floating points. The final avenue is to reduce the dimensionality of the word embeddings. In this work, we explore different approaches perform the latter approach. Specifically, we examing how we can use neural networks to perform this task.

Prior work on the same area tends to focus on learning reduced task-specific embeddings which can then be used for specific downstream tasks. However, our goal is to learn reduced embeddings that are task agnostic. The primary motivation behind this is that the former approach would require N different models to learn N different embeddings for N different tasks. In our case, a single model would learn a single embedding for N different tasks. Since in most cases, the embeddings are finetuned while training on the downstream task, it makes more sense to learn a single general purpose embedding.

## Data
---
Due to size constraints, we do not store the datasets in the repository. However, we provide links to each:

GloVe: https://nlp.stanford.edu/projects/glove/

Word Analogy: https://github.com/nicholas-leonard/word2vec/blob/master/questions-words.txt

CoNLL-2003: https://deepai.org/dataset/conll-2003-english

AG News Corpus: https://www.kaggle.com/amananandrai/ag-news-classification-dataset

## Usage
---
The `notebooks` directory contains all the code used for training all models (both for creating embeddings as well as evaluating on a downstream task). Most of these notebooks are setup for use in both a local environment and a Google Colab environment. 

The `autoencoder.ipynb`, `autoencoder_knowledge_distillation.ipynb`, `supervised.ipynb` and `pca.ipynb` notebooks are all used to create smaller embeddings. 

THe `NER.ipynb` and `classification.ipynb` notebooks are used for evaluation. To compute word analogy scores, please run `compute_word_analogy_scores.py` in the `utils` folder. 

For exact reproduction of our results, please clip the word embeddings using the `clip_vectors.py` script in the `utils` folder and do not change the random seed set in the notebooks.

Note that you will need to either set up a new environment or have an existing environment that contains all the packages mentioned in `requirements.txt`