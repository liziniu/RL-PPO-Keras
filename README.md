# RL-PPO-Keras

This is an implementation of proximal policy optimization(PPO) algorithm with Keras.

## Usage

Start an experiment:

``python run_exp.py``

## Code

* ``python run_exp.py``

	Create environment and agent. Agent interacts and learns.

* ``memory.py``

   Memory includes FIFO and Prioritized Memory.

* ``ppo.py``

    Build Agent with PPO algorithm. Define actor and critic network and how to train.


