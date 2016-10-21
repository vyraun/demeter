import numpy as np
import matplotlib.pyplot as plt

def interpolate(data):
    chunks = np.array_split(data, int(len(data) / 25))
    chunk_means = [np.mean(chunk) for chunk in chunks]
    chunk_indexes = [i * 25 for i in range(1, len(chunks) + 1)]
    return chunk_indexes, chunk_means

def make_plots(keys, data):

    for line in data:
        x, y = interpolate(line)
        plt.plot(x, y)

    plt.legend(keys, loc='upper left')
    plt.show()
