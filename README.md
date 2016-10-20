Demeter
-------

Demeter is a python library for reinforcement learning. It is a testbed for fast experimentation and research with reinforcement learning algorithms, ranging from derivative-free models to deep actor-critic models.

It is currently supports the following algorithms:

+ Derivative-free
  + Cross-entropy
+ Policy Gradients
  + REINFORCE
  + REINFORCE with baseline
  + Vanilla Actor Critic

It is __environment agnostic__, but contains example code for:

+ OpenAI gym 
  + Classic control

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

    policy = DiscreteStochasticMLPPolicy(
        network_name = "action-network",
        sess = sess,
        optimizer = tf.train.AdamOptimizer(args.policy_learning_rate),
        hidden_layers = args.hidden_sizes,
        num_inputs = state_dim,
        num_actions = num_actions)

    writer = tf.train.SummaryWriter(output_path)

    agent = REINFORCE(
        sess = sess,
        state_dim = state_dim,
        num_actions = num_actions,
        summary_writer = writer,
        summary_every = 100,
        action_policy = policy)

    sampler = BatchSampler(
        env = env,
        policy = policy,
        norm_reward = lambda x: 5.0 if x else -0.1,
        discount = args.discount)

    evaluator = BaseEvaluator(
        agent = agent,
        sampler = sampler,
        batch_size = args.batch_size,
        max_steps = args.max_steps,
        num_batches = args.num_batches)
    
    # do 5 runs and save the averaged results
    averaged_stats = evaluator.run_avg(5)
    np.savez(output_path, stats=averaged_stats)
```    


