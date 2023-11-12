use std::{collections::HashMap, hash::Hash};
use peroxide::fuga::*;

pub trait Environment<S, A> {
    fn transition(&self, state: &S, action: &Option<A>) -> (Option<S>, f64);
    fn is_terminal(&self, state: &S) -> bool;
    fn is_goal(&self, state: &S) -> bool;
    fn available_actions(&self, state: &S) -> Vec<A>;
}

pub trait Agent<S, A, P: Policy<A>, E: Environment<S, A>> {
    type Information;
    fn select_action(&self, state: &S, policy: &mut P, env: &E) -> Option<A>;
    fn update(&mut self, info: &Self::Information);
    fn get_value(&self, state: &S) -> f64;
    fn get_action_value(&self, state: &S, action: &A) -> f64;
}

pub trait Policy<A> {
    fn select_action(&mut self, action_rewards: &[(A, f64)]) -> Option<A>;
}

// ┌──────────────────────────────────────────────────────────┐
//  Value Iteration - Every Visit MC
// └──────────────────────────────────────────────────────────┘
pub struct VEveryVisitMC<S, A, P: Policy<A>, E: Environment<S, A>> {
    pub value_function: HashMap<S, f64>,
    gamma: f64,
    _action_type: std::marker::PhantomData<A>,
    _policy_type: std::marker::PhantomData<P>,
    _env_type: std::marker::PhantomData<E>,
}

impl <S: Hash + Eq + Copy, A: Clone, P: Policy<A>, E: Environment<S, A>> VEveryVisitMC<S, A, P, E> {
    pub fn new(gamma: f64) -> Self {
        Self {
            value_function: HashMap::new(),
            gamma,
            _action_type: std::marker::PhantomData,
            _policy_type: std::marker::PhantomData,
            _env_type: std::marker::PhantomData,
        }
    }

    pub fn update_value(&mut self, state: &S, value: f64) {
        self.value_function.insert(*state, value);
    }
}

impl<S: Hash + Eq + Copy, A: Clone, P: Policy<A>, E: Environment<S, A>> Agent<S, A, P, E> for VEveryVisitMC<S, A, P, E> {
    // Information = Episode
    type Information = Vec<(S, f64)>;

    fn get_action_value(&self, _state: &S, _action: &A) -> f64 {
        unimplemented!()
    }

    fn get_value(&self, state: &S) -> f64 {
        *self.value_function.get(state).unwrap_or(&0.0)
    }


    fn select_action(&self, state: &S, policy: &mut P, env: &E) -> Option<A> {
        let actions = env.available_actions(state);
        let candidates = actions.iter().filter_map(|a| {
            let (s, _) = env.transition(state, &Some(a.clone()));
            if let Some(s) = s {
                let v = self.get_value(&s);
                Some((a.clone(), v))
            } else {
                None
            }
        }).collect::<Vec<_>>();

        policy.select_action(&candidates)
    }

    #[allow(non_snake_case)]
    fn update(&mut self, info: &Self::Information) {
        if info.is_empty() {
            panic!("Empty episode!")
        }

        // Backward update for cumulative discounted return
        let R: Vec<f64> = info
            .iter()
            .rev()
            .scan(0.0, |acc, (_, r)| {
                *acc = *acc * self.gamma + r;
                Some(*acc)
            })
            .collect();

        // Forward update for value function
        info
            .iter()
            .zip(R)
            .enumerate()
            .for_each(|(t, ((s, _), r))| {
                let v = self.get_value(s);
                let alpha = 5.0 / (t + 5) as f64;
                self.update_value(s, v + alpha * (r - v));
            });
    }

}

// ┌──────────────────────────────────────────────────────────┐
//  Epsilon Greedy Policy
// └──────────────────────────────────────────────────────────┘
pub struct EGreedyPolicy<A> {
    epsilon: f64,
    random: bool,
    _action_type: std::marker::PhantomData<A>,
}

impl <A: Clone> EGreedyPolicy<A> {
    pub fn new(epsilon: f64) -> Self {
        Self {
            epsilon,
            random: true,
            _action_type: std::marker::PhantomData,
        }
    }
}

impl <A: Clone> Policy<A> for EGreedyPolicy<A> {
    fn select_action(&mut self, action_rewards: &[(A, f64)]) -> Option<A> {
        if action_rewards.is_empty() {
            return None;
        }

        let u = Uniform(0f64, 1f64);
        let sample = u.sample(1)[0];

        if sample < self.epsilon && self.random {
            let mut rng = thread_rng();
            Some(action_rewards.choose(&mut rng).unwrap().0.clone())
        } else {
            let mut max_reward = action_rewards[0].1;
            let mut max_actions = vec![];

            for (a, r) in action_rewards.iter() {
                if *r > max_reward {
                    max_reward = *r;
                    max_actions = vec![a.clone()];
                } else if *r == max_reward {
                    max_actions.push(a.clone());
                }
            }

            let mut rng = thread_rng();
            Some(max_actions.choose(&mut rng).unwrap().clone())
        }
    }
}

// ┌──────────────────────────────────────────────────────────┐
//  Line World
// └──────────────────────────────────────────────────────────┘
#[derive(Debug, Clone)]
pub struct LineWorld {
    num_rows: usize,
    init_state: usize,
    goal_state: usize,
    terminal_state: Vec<usize>,
}

impl LineWorld {
    pub fn new(num_rows: usize, init_state: usize, goal_state: usize, terminal_state: Vec<usize>) -> Self {
        Self {
            num_rows,
            init_state,
            goal_state,
            terminal_state,
        }
    }

    pub fn get_init_state(&self) -> usize {
        self.init_state
    }

    pub fn get_goal_state(&self) -> usize {
        self.goal_state
    }

    pub fn get_terminal_state(&self) -> &Vec<usize> {
        &self.terminal_state
    }
}

#[derive(Debug, Copy, Clone)]
pub enum LineWorldAction {
    Up,
    Down,
}

impl Environment<usize, LineWorldAction> for LineWorld {
    fn is_terminal(&self, state: &usize) -> bool {
        self.terminal_state.contains(state)
    }

    fn is_goal(&self, state: &usize) -> bool {
        *state == self.goal_state
    }

    fn transition(&self, state: &usize, action: &Option<LineWorldAction>) -> (Option<usize>, f64) {
        if self.is_terminal(state) {
            (None, -1.0)
        } else if self.is_goal(state) {
            (None, 1.0)
        } else {
            let action = action.as_ref().unwrap();
            match action {
                LineWorldAction::Up => (Some(*state + 1), 0.0),
                LineWorldAction::Down => (Some(*state - 1), 0.0),
            }
        }
    }

    fn available_actions(&self, state: &usize) -> Vec<LineWorldAction> {
        match state {
            0 => vec![LineWorldAction::Up],
            r if *r == self.num_rows - 1 => vec![LineWorldAction::Down],
            _ => vec![LineWorldAction::Up, LineWorldAction::Down],
        }
    }
}
