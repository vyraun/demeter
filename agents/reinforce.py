from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import tensorflow as tf

from . import base_agent

class REINFORCE(base_agent.BaseAgent):
    def __init__(self, sess,
                       state_dim,
                       num_actions,
                       summary_writer,
                       summary_every,
                       action_policy):

        super(REINFORCE, self).__init__(sess, state_dim, num_actions, summary_writer, summary_every)

        # agent specific
        self.action_policy = action_policy

        # placeholders
        self.actions = tf.placeholder(tf.int32, [None, 1], "actions")
        self.returns = tf.placeholder(tf.float32, [None, 1], "returns")

    def train(self, traj):

        summary = self.action_policy.train(traj.states, traj.actions, traj.returns, self.action_policy.summary_op)

        # bookkeeping
        self.train_iter += 1
        self.write_scalar_summaries([summary])
        if self.action_policy.annealer is not None:
            self.action_policy.annealer.anneal(self.train_iter)
