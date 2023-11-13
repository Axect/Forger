# Forger - Reinforcement Learning Library in Rust

## Introduction

Forger is a Reinforcement Learning (RL) library in Rust, offering a robust and efficient framework for implementing RL algorithms. It features a modular design with components for agents, environments, policies, and utilities, facilitating easy experimentation and development of RL models.

## Features

- **Modular Components**: Includes agents, environments, and policies as separate modules.
- **Efficient and Safe**: Built in Rust, ensuring high performance and safety.
- **Customizable Environments**: Provides a framework to create and manage different RL environments.
- **Flexible Agent Implementations**: Supports various agent strategies and learning algorithms.
- **Extensible Policy Framework**: Allows for the implementation of diverse action selection policies.

## Modules

1. **Policy (`policy`)**: 
   - Defines the interface for action selection policies.
   - Includes an implementation of Epsilon Greedy (with Decay) Policy.

2. **Agent (`agent`)**: 
   - Outlines the structure for RL agents.
   - Implements Value Iteration - Every Visit Monte Carlo (`VEveryVisitMC`) and Q-Learning - Every Visit Monte Carlo (`QEveryVisitMC`).

3. **Environment (`env`)**: 
   - Provides the `Env` trait to define RL environments.
   - Contains `LineWorld`, a simple linear world environment for experimentation.

4. **Prelude (`prelude`)**: 
   - Exports commonly used items from the `env`, `agent`, and `policy` modules for convenient access.

## Getting Started

### Prerequisites

- Rust Programming Environment

### Installation

In your project directory, run the following command:
```shell
cargo add forger
```

### Basic Usage

[Include a basic usage example that demonstrates creating an environment, an agent, and utilizing a policy to interact within the environment.]

## Examples

1. [**Monte Carlo with Epsilon Decay in `LineWorld`**](./examples/lineworld_mc_edecay.rs):
   - Demonstrates the use of the Q-Learning Every Visit Monte Carlo (`QEveryVisitMC`) agent with an Epsilon Greedy Policy (with decay) in the `LineWorld` environment.
   - Illustrates the process of running multiple episodes, selecting actions, updating the agent, and decaying the epsilon value over time.
   - Includes data gathering and exporting functionality, showcasing practical aspects of RL experiments.


## Contributing

Contributions to Forger are welcome! If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.

## License

Forger is licensed under the MIT License or the Apache 2.0 License.
