import numpy as np
from gensim.scripts import glove2word2vec
from gensim.models.keyedvectors import Word2VecKeyedVectors
import os
import sys


def get_gensim_vectors(input_file, converted_file):
    # Convert GloVe vectors into a format usable by gensim
    _ = glove2word2vec.glove2word2vec(input_file, converted_file)
    # Load converted vectors
    return Word2VecKeyedVectors.load_word2vec_format(converted_file)


def print_results(results):
    syn_correct = 0
    syn_incorrect = 0
    sem_correct = 0
    sem_incorrect = 0
    for item in results[1]:
        correct = len(item["correct"])
        incorrect = len(item["incorrect"])
        if item["section"] == "Total accuracy":
            print(f"{item['section']}: {correct * 100/(correct+incorrect)}")
        elif item["section"][:4] == "gram":
            sem_correct += correct
            sem_incorrect += incorrect
        else:
            syn_correct += correct
            syn_incorrect += incorrect
    print(f"Semantic Accuracy: {sem_correct*100/(sem_correct+sem_incorrect)}")
    print(f"Syntactic Accuracy: {syn_correct*100/(syn_correct+syn_incorrect)}")


# Reproducibility
seed = 42
np.random.seed(seed)

### USER INPUT ###
path_to_dir_or_file = sys.argv[1]
mode = sys.argv[2]
test_data_path = "data/datasets/questions-words.txt"
### END USER INPUT ###

if mode == "bulk":
    print("Running in BULK mode.")
    list_of_files = os.listdir(path_to_dir_or_file)
    for input_file in list_of_files:
        print("\n")
        print(input_file)
        input_file = os.path.join(path_to_dir_or_file, input_file)
        converted_file = input_file + "2"
        glove = get_gensim_vectors(input_file, converted_file)

        # Evaluate
        results = glove.evaluate_word_analogies(test_data_path, restrict_vocab=400000)

        # Print Results
        print_results(results)

        # Remove converted vectors file
        os.remove(converted_file)
else:
    converted_file = path_to_dir_or_file + "2"
    glove = get_gensim_vectors(path_to_dir_or_file, converted_file)

    # Evaluate
    print("Evaluating...")
    results = glove.evaluate_word_analogies(test_data_path, restrict_vocab=400000)

    # Print Results
    print_results(results)

    # Remove converted vectors file
    os.remove(converted_file)