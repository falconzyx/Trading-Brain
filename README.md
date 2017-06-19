Trading Brain is a framework example for implementing and testing trading strategies.
It is composed of mainly three components communicating through APIs:
- Brain
- Memory
- Agent

This library can be used to test agents with the Trading-Gym.

## Architecture

![](https://docs.google.com/drawings/d/1bL6Gl1hJh0PqsvHh0T5dcqcVrrxP7taN7UUL7a07zEk/pub?w=700)

## Installation

Install packages in requirements.txt file

## Roll out your own `Agent`

To create your own agent, it must inherit from the `Agent` base class which can be found at 'tbrn/base/agent.py'. It consists of three basic methods that need to be overridden in order to implement your own logic:
- `act`: returns the action chosen by the agent.
- `observe`: returns a real value (can be the loss in the case of a `DQNAgent` for instance). This method is where the learning logic of the agent is located. Can be blank for dummy agents.
- `end`: any logic at the end of an episode.

## Examples

One example can be found in `examples/`

- Simple agent (`examples/dqn_agent.py`)


*Copyright © 2017 RKR Epsilon UK Ltd. All rights reserved.*
