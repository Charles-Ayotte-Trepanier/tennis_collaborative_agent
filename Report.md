# Report

## Learning algorithm
For the project, the learning algorithm used to train the agent is based on the work of Shangtong Zhang (https://github.com/ShangtongZhang/DeepRL.)
The approach Shangtong Zhang took in developing agents makes them agnostic to the specific environment they might be used for. 

Due to the continuous nature of the action-space, the specific algorithm used was similar, to MADDPG.
We took the DDPG algorithm from Shangtong Zhang and modified it for our collaborative problem:
- A single agent is trained. This agent controls both players.
- Replay buffer stores not only the state/next state, but also the state and next state from our 'colleague' (the other agent with whom we are playing tennis.)
- For each sample, the reward is the sum of reward of both agents.
- At each time step, the replay buffer actually add 2 samples (one from each player.)
- The critic has access to the state/next state of the other agent - not the actor.
- Besides these slight differences, the algorithm is very similar to the DDPG one we successfully used in the second project of the nanodegree.

Our own implementation of the MADDPG agent can be found in the notebook Tennis.ipynb .

From Shangtong Zhang's code, we this agent leveraged:
- DeterministicActorCriticNet: A convenient wrapper over the actor and critic networks required for DDPG.
- UniformReplay: Replay buffer implementation.
- OrnsteinUhlenbeckProcess: Implementation of the Ornstein-Uhlenbeck process, a solution for exploration in continuous action space.

The following configurations/hyper-parameters were used:
- Reward discounting factor: 0.99
- Soft update parameter: 5e-3
- Ornstein-Uhlenbeck process: standard deviation of 0.1
- Actor network: Fully connected network with layers [50, 50, 50, 50, 50, 20] and relu gates (and tanh activation to predict the action.)
- Critic network: Fully connected network with layers [100, 50, 50, 50, 50, 50, 20] and relu gates (linear activation)
- Optimizer (for both nerworks): Adam with learning rate 1e-3 
- Replay buffer: memory size of 1e6 and batch size of 128.
- Updates frequency: every 4 time steps.
- Warm up: Wait 1e3 time steps before starting updates.

   
## Solving the environment
The average reward between episodes 3100 and 3200 is 0.61 . We can then say that the environment
was solved after episode 3100.

We wanted to solve the environment 'perfectly' and continued training until we reached an average score of 2 (4200 episodes.)

## Plot of rewards
![image info](./tennis_scores.png)

## Ideas for future work

1. There are definitely opportunities of improving the agent through hyper-parameters tuning (including the architecture of the actor/critic networks in the MADDPG agent.)
2. It took many episodes to solve the problem, and quite a while to get our average score of 2 (since episodes take a while to run when the agent is good.) Prioritized replay could possibly help speeding up convergence.
3. We built our own MADDPG-ish algorithm. It would be interesting to compare with the actual MADDPG algorithm.