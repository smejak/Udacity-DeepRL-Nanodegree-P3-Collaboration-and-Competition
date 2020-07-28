# Project 3 of the Udacity Deep Reinforcement Learning Nanodegree
## Learning Algorithm
### DDPG
The learning algorithm used in this project is the Deep Deterministic Policy Gradient (DDPG), first introduced in the paper [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf). 
DDPG is an actor-critic method, which makes use of two neural networks for approximating and evaluating an optimal policy. The actor learns to deterministically approximate the optimal policy and the critic learns to evaluate the optimal action-value function by using the best action according to the actor. DDPG can be thought of as a type of DQN which is optimized for continuous action spaces.
### Experience Replay
This task involves multiple agents learning to cooperate with each other, therefore, to make learning more effective, both agents sample their experiences from a shared replay buffer. This leads to better data efficiency as a single experience can be used multiple times and also results in better learning, because correlation from consecutive timesteps is diminished.
### Soft Updates
As in the DQN algorithm, DDPG uses a local and a target network (for both the actor and the critic), however, the target networks are updated using the so called soft updates, which have been shown to lead to faster learning convergence. At every timestep, a small percentage of the local networks' weights are passed to the target networks.
### Batch Normalization
As described in the [paper](https://arxiv.org/pdf/1509.02971.pdf), to avoid the issue that different dimensions of the state space may have different units and scales, I adopted the batch normalization technique by adding batch normalization layers to both actor-critic neural networks.
### Ornstein-Uhlenbeck process
The Ornstein-Uhlenbeck process was used to add noise to the action space for better exploration efficiency.
![](/algo.png)
## Implementation
The implementation of the DDPG algorithm can be found in __ddpg_agent.py__. The neural network architectures for the actor and the critic can be found in __model.py__. The networks were constructed using PyTorch. Both networks have two fully connected layers of 128 units and two layers nn.LayerNorm() for batch normalization. The input to the networks is the state size, which for this environment is 24 nodes. See __Tennis.ipynb__ for more details.
```
Number of agents: 2
Size of each action: 2
There are 2 agents. Each observes a state with length: 24
The state for the first agent looks like: [ 0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.         -6.65278625 -1.5
 -0.          0.          6.83172083  6.         -0.          0.        ]
```
The output of the actor network has 2 nodes corresponding to the action size and for the critic network 1 node.
### Hyperparameters
The hyperparameters for the DDPG agent can be found in __ddpg_agent.py__.
Parameters used for the training function are:
* maximum number of timesteps per episode: 1000
## Results
```
Episode 100	Average Score: -0.00
Episode 200	Average Score: 0.02
Episode 300	Average Score: -0.00
Episode 400	Average Score: 0.02
Episode 500	Average Score: 0.03
Episode 600	Average Score: 0.07
Episode 700	Average Score: 0.06
Episode 800	Average Score: 0.11
Episode 900	Average Score: 0.45
Environment solved in 989 episodes! Average Score: 0.5026500076707453
```
![](/graph.png)
The results show that the environment has been successfully solved as the average score over 100 consecutive episodes for both agents reached +0.5.
## Ideas for Future Work
I started this project by implementing my solution to the previous project on continuous control and found it worked very well for this task as well, which is reasonable, considering the agents were learning a cooperative task instead of a competitive one. In a future implementation, I could try to add prioritized experience replay, which would make the replay buffer sampling more efficient as there are still experiences saved and sampled which provide little to no learning value. Fruthermore, I could try to experiment with different actor critic methods such as A2C, A3C or D4PG adjusted for multi-agent RL. Lastly, I definitely want to try to implement the MADDPG algorithm from this [paper](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf) on competitive environments, which should be more difficult to solve, given the unstable behavior of multiple competing agents.
