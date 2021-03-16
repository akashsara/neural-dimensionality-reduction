import numpy as np
import gensim
import re
import sys

embedding_path = sys.argv[1] # Source Embedding
output_path = sys.argv[2] # Save Location

seed = 42
np.random.seed(seed)

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