# Forger - Reinforcement Learning Library in Rust

<img align="center" src="https://github.com/Axect/Forger/blob/master/forger_logo.png" width=40%>

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

```rust
use forger::prelude::*;
use forger::env::lineworld::{LineWorld, LineWorldAction};

pub type S = usize;             // State
pub type A = LineWorldAction;   // Action
pub type P = EGreedyPolicy<A>;  // Policy
pub type E = LineWorld;         // Environment

fn main() {
    let env = LineWorld::new(
        5,      // number of states
        1,      // initial state
        4,      // goal state
        vec![0] // terminal states
    );

    let mut agent = QEveryVisitMC::<S, A, P, E>::new(0.9); // Q-learning (Everyvisit MC, gamma = 0.9)
    let mut policy = EGreedyPolicy::new(0.5, 0.95);        // Epsilon Greedy Policy (epsilon = 0.5, decay = 0.95)

    for _ in 0 .. 200 {
        let mut episode = vec![];
        let mut state = env.get_init_state();

        loop {
            let action = agent.select_action(&state, &mut policy, &env);
            let (next_state, reward) = env.transition(&state, &action);
            episode.push((state, action.unwrap(), reward));
            match next_state {
                Some(s) => state = s,
                None => break,
            }
        }

        agent.update(&episode);
        policy.decay_epsilon();
    }
}
```

## Examples

1. [**Monte Carlo with Epsilon Decay in `LineWorld`**](./examples/lineworld_mc_edecay.rs):

   - Demonstrates the use of the Q-Learning Every Visit Monte Carlo (`QEveryVisitMC`) agent with an Epsilon Greedy Policy (with decay) in the `LineWorld` environment.
   - Illustrates the process of running multiple episodes, selecting actions, updating the agent, and decaying the epsilon value over time.
   - Updates the agent after each episode.

2. [**TD0 with Epsilon Decay in `GridWorld`**](./examples/gridworld_td0_edecay.rs):

   - Demonstrates the use of the TD0 (`TD0`) agent with an Epsilon Greedy Policy (with decay) in the `GridWorld` environment.
   - Illustrates the process of running multiple episodes, selecting actions, updating the agent, and decaying the epsilon value over time.
   - Updates the agent every steps in each episode.

## Contributing

Contributions to Forger are welcome! If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.

## License

Forger is licensed under the MIT License or the Apache 2.0 License.
