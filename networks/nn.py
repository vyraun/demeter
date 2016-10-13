import tensorflow as tf

import tensorflow.contrib.slim as slim

class FullyConnectedNN(object):
    """
    A generic neural network. 
    
    Requires a separate loss and train op to update weights (left up to the agent).
    """
    def __init__(self,
                 name,
                 sess,
                 optimizer,
                 hidden_layers,
                 num_inputs,
                 num_outputs):

        # tf
        self.sess = sess
        self.optimizer = optimizer

        with tf.variable_scope(name):

            self.observations = tf.placeholder(shape=[None, num_inputs], dtype=tf.float32)
            flat_input_state = slim.flatten(self.observations, scope='flat')

            if hidden_layers == "":
                self.logits = slim.fully_connected(
                    inputs=flat_input_state,
                    num_outputs=num_outputs,
                    activation_fn=None
                    #weights_initializer=tf.zeros_initializer
                    )
            else:
                final_hidden = self.hidden_layers_starting_at(flat_input_state, hidden_layers)
                self.logits = slim.fully_connected(inputs=final_hidden,
                                num_outputs=num_outputs,
                                activation_fn=None)

    def hidden_layers_starting_at(self, layer, config):
        layer_sizes = map(int, config.split(","))
        assert len(layer_sizes) > 0
        for i, size in enumerate(layer_sizes):
            layer = slim.fully_connected(scope="h%d" % i,
                              inputs=layer,
                              num_outputs=size,
                              weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                              #weights_initializer=tf.zeros_initializer,
                              activation_fn=tf.nn.relu)
        return layer
