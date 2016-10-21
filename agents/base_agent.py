import tensorflow as tf

class BaseAgent(object):
    """
    Common class variables and methods used by all agents
    """
    def __init__(self, sess,
                       state_dim,
                       num_actions,
                       summary_writer=None,
                       summary_every=100):

        # tf
        self.sess = sess

        # env
        self.state_dim = state_dim
        self.num_actions = num_actions

        # bookkeeping
        self.train_iter = 0

        # summary
        self.summary_writer = summary_writer

        if summary_writer is not None:
            self.summary_writer.add_graph(self.sess.graph)
            self.summary_every = summary_every
        else:
            self.summary_op = tf.no_op()

    def train(self):
        """
        Left up to the individual agent...
        """

        raise NotImplementedError()

    def write_scalar_summaries(self, summaries):
        calculate_summaries = self.train_iter % self.summary_every == 0 and self.summary_writer is not None

        # emit summaries
        if calculate_summaries:
            for summary in summaries:
                self.summary_writer.add_summary(summary, self.train_iter)
