# drlnd-continuous-control
This repo is for the Continuous Control project of Udacity Deep Reinforcement
Learning Nanodegree.

## About the task

In this project the task is
[Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher)
from Unity ml-agent project. The agent controls a double-jointed arm and moves
its hand to a target range and keep within it. The target range can move slowly.

The observation contains the moving status (position, rotation, velocity and
angular velocity) of the arm and the target range. The action contains 4 torque
applied to the two joints of the arm.

Positive reward is granted per step if the hand is located in the current target
range. The benchmark mean reward is 30, in other words, the task is considered
to be solved if the agent achieves an average reward of >30 over the 100
episodes.

## About this solution

The dependencies of the projects in this Nanodegree are listed in this
[document](https://github.com/udacity/deep-reinforcement-learning#dependencies)
from Udacity. As for this specific project,
[here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control)
is the repository containing the instruction of additional set-ups and the
starter code. We don't need to actually install Unity since the enviornment used
in this project is already provided with a pre-built Unity app.

All the actual implementation of the project is located in the `src` directory,
including the nerual net built with `pytorch`, the DDPG agent and some other
utilities related to training and logging. The results and analysis is in
[Report.ipynb](./Report.ipynb).

There is also a testing script [test_agents.py](./test_agents.py) to train the
agent on [Pendulum-v0](https://github.com/openai/gym/wiki/Pendulum-v0) task from
OpenAI Gym. I used this script to verify the implementation. With the hyper
parameters configured in the script the DDPG agent achieves 100-episode average
reward of >-250 after 800 training episodes.
