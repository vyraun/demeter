from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import tensorflow as tf

from networks.nn import FullyConnectedNN

class MLPValueEstimator(object):
    def __init__(self, network_name,
                       sess,
                       optimizer,
                       hidden_layers,
                       num_inputs):

        # tf
        self.sess = sess
        self.optimizer = optimizer

        # placeholders
        self.targets = tf.placeholder(tf.float32, [None, 1], "targets")

        # network
        self.network = FullyConnectedNN(
            name = network_name,
            sess = sess, 
            optimizer = optimizer,
            hidden_layers = hidden_layers,
            num_inputs = num_inputs,
            num_outputs = 1)

        # construct value loss and train op
        self.loss = tf.reduce_mean(tf.square(self.targets - self.network.logits))
        self.train_op = self.optimizer.minimize(self.loss)

        # summary
        self.summary_op = tf.summary_op = tf.scalar_summary("value_loss", self.loss)

    def predict(self, observations):
        '''
        Do forward prop and return the values of the output layer for a batch
        of observation vectors.
        '''

        return self.sess.run(self.network.logits, {
            self.network.observations: observations
        })

    def train(self, states, targets, summary_op):
        _, summary_str = self.sess.run([
            self.train_op, 
            summary_op
        ], {
            self.network.observations: states,
            self.targets: targets
        })

        return summary_str

