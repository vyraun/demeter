Demeter
-------

Demeter is a python library for reinforcement learning. It is a testbed for fast experimentation and research with reinforcement learning algorithms, ranging from derivative-free models to deep actor-critic models.

It is currently supports the following algorithms:

* Derivative-free
  * Cross-entropy

* Policy Gradients
  * REINFORCE
  * REINFORCE with baseline
  * Vanilla Actor Critic

It is __environment agnostic__, but contains example code for:

* OpenAI gym 

Demeter is built on top of TensorFlow. It enables features such as computation graphs, automatic differentiation, and visualization with TensorBoard

## Components

Demeter is built with modularity, reusability, and code encapsulation in mind. A typical RL algorithm using Demeter is composed of one or more of the following pieces:

### Agents

The highest level of abstraction. Encapsulates the provided policy and (optionally) baseline. Performs any algorithm specific data exchange between policy and baselines in the case of actor-critic models.

### Policies

Responsible for performing rollouts via `policy.sample_action` and model updates via `policy.train`. Can be backed by a neural network or linear regressor.

### Networks

A collection of networks and linear regressors used in policies and baselines. The final layer output is accessed by `network.logits`.

### Baselines 

Value estimators used in value estimation or actor-critic based algorithms. Responsible for returning value estimation prediction via `value_estimator.predict`

## Example

```python
# init gym
env = gym.make(args.env_name)

# env vars
state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

with tf.Session() as sess:

    annealer = LinearAnnealer(
            init_exp = args.anneal_init,
            final_exp = args.anneal_final,
            anneal_steps = args.anneal_steps)

    policy = DiscreteStochasticMLPPolicy(
                network_name = "action-network",
                sess = sess, 
                optimizer = tf.train.AdamOptimizer(args.policy_learning_rate),
                hidden_layers = args.hidden_sizes,
                num_inputs = state_dim,
                num_actions = num_actions,
                annealer = annealer)

    writer = tf.train.SummaryWriter("/tmp/{}".format(args.experiment_name))

    agent = REINFORCE(
        sess = sess,
        state_dim = state_dim,
        num_actions = num_actions,
        summary_writer = writer,
        summary_every = 100,
        action_policy = policy)

    tf.initialize_all_variables().run()

    stats = Stats(args.batch_size, args.max_steps)

    for i_batch in xrange(args.num_batches):
        traj = BatchTrajectory()
        
        for i_eps in xrange(args.batch_size):
            state = env.reset()

            for t in xrange(args.max_steps):
                action = policy.sample_action(state)

                next_state, reward, is_terminal, info = env.step(action)

                norm_reward = -10 if is_terminal else 0.1
                traj.store_step(state.tolist(), action, norm_reward)
                stats.store_reward(reward)

                state = next_state
    
                if is_terminal: break
            
            # discounts the rewards over a single episode
            eps_rewards = traj.rewards[-t-1:]
            traj.calc_and_store_discounted_returns(eps_rewards, args.discount) 

            stats.mark_eps_finished(i_batch, i_eps)

        agent.train(traj)

    # plotting
    stats.plot_batch_stats(args.plot_dir, args.experiment_name)
```
