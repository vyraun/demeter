from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import numpy as np

from lib import plots


plot_dir = "../tmp/pg/"

# cartpole
cartpole_names = ['actor_critic_cartpole.npz',
         'reinforce_baseline_cartpole.npz',
         'reinforce_cartpole.npz']
# acrobot
acrobot_names = ['actor_critic_acrobot.npz',
        'reinforce_acrobot.npz',
        'reinforce_baseline_acrobot.npz']

def plot_total_rewards(names, suffix, keys=None):

    stats = [np.load(plot_dir + n)["stats"][0].total_rewards for n in names]
    to_replace = "_" + suffix + ".npz"
    keys = [k.replace(to_replace, '') for k in names] if keys is None else keys

    plots.make_plots(keys, stats)

plot_total_rewards(cartpole_names, 'cartpole')
#plot_total_rewards([cartpole_names[-1]], 'cartpole', ['total episode reward'])

#plot_total_rewards([acrobot_names[1]], 'acrobot', ['total episode reward'])
#plot_total_rewards(acrobot_names, 'acrobot')
