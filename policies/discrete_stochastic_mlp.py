from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import tensorflow as tf
import numpy as np

from networks.nn import FullyConnectedNN

class DiscreteStochasticMLPPolicy(object):

    def __init__(self, network_name,
                       sess,
                       optimizer,
                       hidden_layers,
                       num_inputs,
                       num_actions,
                       annealer = None):

        # tf
        self.sess = sess
        self.optimizer = optimizer

        # env
        self.num_actions = num_actions

        # anneal
        self.annealer = annealer

        # placeholders
        self.actions = tf.placeholder(tf.int32, [None, 1], "actions")
        self.targets = tf.placeholder(tf.float32, [None, 1], "targets")

        # network
        self.network = FullyConnectedNN(
            name = network_name,
            sess = sess, 
            optimizer = optimizer,
            hidden_layers = "",
            num_inputs = num_inputs,
            num_outputs = num_actions)

        # construct pi_theta(a|s)
        self.actions_log_prob = tf.squeeze(tf.nn.log_softmax(self.network.logits))
        self.actions_mask = tf.squeeze(tf.one_hot(indices = self.actions, depth = num_actions))
        self.picked_actions_log_prob = tf.reduce_sum(self.actions_log_prob * self.actions_mask, reduction_indices = 1)

        # construct policy loss and train op
        self.standardized = tf.squeeze(self.targets)
        self.loss = -tf.reduce_sum(self.picked_actions_log_prob * self.standardized)
        self.train_op = self.optimizer.minimize(self.loss)

        # summary
        self.summary_op = tf.scalar_summary("policy_loss", self.loss)

    def softmax(self,x):
        e_x = np.exp(x - np.max(x))
        out = e_x / e_x.sum()
        return out

    def sample_action(self, observation):
        """
        Samples an action from \pi_\theta(a|s)

        tf ops are eliminated on purpose here since this is a hot code path and 
        were optimizing for CPU usage...or maybe tf.multinomial is just slow in general.

        Using TF ops:
        sample_action_op = tf.squeeze(tf.nn.softmax(self.network.logits))
        action = tf.multinomial(sample_action_op)
        """
        if self.annealer is None or not self.annealer.is_explore_step():
            action_probs = self.network.sess.run(
                self.network.logits, 
                {self.network.observations: [observation]}
            )[0]
            action = np.random.choice(np.arange(len(action_probs)), p = self.softmax(action_probs))

            return action
        else:
            # take random action
            return np.random.randint(0, self.num_actions)

    def train(self, states, actions, targets, summary_op):
        _, summary_str = self.sess.run([
            self.train_op, 
            summary_op
        ], {
            self.network.observations: states,
            self.targets: targets,
            self.actions: actions
        })

        return summary_str
