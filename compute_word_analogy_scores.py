import numpy as np
from gensim.scripts import glove2word2vec
from gensim.models.keyedvectors import Word2VecKeyedVectors
import os

# Reproducibility
seed = 42
np.random.seed(seed)

### USER INPUT ###
experiment_name = "pca_v2"
test_data_path = "data/datasets/questions-words.txt"
input_file_format = ".glove.6B.300d.txt"
### END USER INPUT ###

input_file = f"data/embeddings/trained/{experiment_name}{input_file_format}"
converted_file = f"data/embeddings/trained/{experiment_name}.txt"

# Convert GloVe vectors into a format usable by gensim
_ = glove2word2vec.glove2word2vec(input_file, converted_file)
print(f"Converted to Gensim Format: {_}")

# Load converted vectors
glove = Word2VecKeyedVectors.load_word2vec_format(converted_file)

# Evaluate & Print
print("Evaluating...")
results = glove.evaluate_word_analogies(test_data_path, restrict_vocab=400000)

syn_correct = 0
syn_incorrect = 0
sem_correct = 0
sem_incorrect = 0
for item in results[1]:
    correct = len(item['correct'])
    incorrect = len(item['incorrect'])
    if item['section'] == "Total accuracy":
        print(f"{item['section']}: {correct * 100/(correct+incorrect)}")
    elif item['section'][:4] == 'gram':
        sem_correct += correct
        sem_incorrect += incorrect
    else:
        syn_correct += correct
        syn_incorrect += incorrect
print(f"Semantic Accuracy: {sem_correct * 100/(sem_correct+sem_incorrect)}")
print(f"Syntactic Accuracy: {syn_correct * 100/(syn_correct+syn_incorrect)}")

# Remove converted vectors file
os.remove(converted_file)