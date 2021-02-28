import numpy as np
import gensim
import re

seed = 42
np.random.seed(seed)
embedding_path = "data/embeddings/glove.6B.300d.txt"
output_path = "data/embeddings/glove.clipped.6B.300d.txt"

clipped = []
with open(embedding_path, "r", encoding='utf-8') as fp:
    for line in fp:
        line = line.split()
        word = line[0]
        vector = np.asarray(line[1:], 'float32').clip(min=-1.0, max=1.0)
        vector = re.split('\s+', str(vector)[1:-1].strip().replace('\n', ' '))
        final = " ".join([word] + vector)
        clipped.append(final)

with open(output_path, "w", encoding='utf-8') as fp:
    fp.write("\n".join(clipped))