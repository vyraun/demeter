import numpy as np
import matplotlib.pyplot as plt

def plot_batch_stats(self, plot_dir, name):
    chunks = np.array_split(self.batch_total_eps_rewards, int(len(self.batch_total_eps_rewards) / 25))
    chunk_means = [np.mean(chunk) for chunk in chunks]
    chunk_indexes = [i * 25 for i in range(1, len(chunks) + 1)]

    plt.plot(chunk_indexes, chunk_means)
    plt.legend(['total episode rewards'])

    plt.savefig(plot_dir + "/" + name)
