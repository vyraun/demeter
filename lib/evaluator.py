import tensorflow as tf

from global_stats import GlobalStats

class BaseEvaluator(object):
    def __init__(self, agent,
                       sampler,
                       batch_size,
                       max_steps,
                       num_batches):

        self.agent = agent
        self.sampler = sampler

        self.batch_size = batch_size
        self.max_steps = max_steps
        self.num_batches = num_batches

    def run_avg(self, n):
        stats = []
        for _ in range(0, n):
            run_stats = self.run()
            stats.append(run_stats)

        return GlobalStats.merge(stats)

    def run(self):
        global_stats = GlobalStats()

        tf.initialize_all_variables().run()

        for i_batch in xrange(self.num_batches):
            batch_stats, traj = self.sampler.sample(i_batch, self.batch_size, self.max_steps)

            global_stats.add_batch_stats(batch_stats)
            global_stats.print_lookback_summary(100)

            self.agent.train(traj)

        return global_stats
