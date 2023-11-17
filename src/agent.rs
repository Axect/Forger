use std::{collections::HashMap, hash::Hash};
use peroxide::fuga::*;

use crate::policy::Policy;
use crate::env::Env;

pub trait Agent<S, A, P: Policy<A>, E: Env<S, A>> {
    type Information;
    fn select_action(&self, state: &S, policy: &mut P, env: &E) -> Option<A>;
    fn update(&mut self, info: &Self::Information);
    fn get_value(&self, state: &S) -> f64;
    fn get_action_value(&self, state: &S, action: &A) -> f64;
}

// ┌──────────────────────────────────────────────────────────┐
//  Value Iteration - Every Visit MC
// └──────────────────────────────────────────────────────────┘
pub struct VEveryVisitMC<S, A, P: Policy<A>, E: Env<S, A>> {
    pub value_function: HashMap<S, f64>,
    gamma: f64,
    _action_type: std::marker::PhantomData<A>,
    _policy_type: std::marker::PhantomData<P>,
    _env_type: std::marker::PhantomData<E>,
}

impl<S: Hash + Eq + Copy, A: Clone, P: Policy<A>, E: Env<S, A>> VEveryVisitMC<S, A, P, E> {
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

impl<S: Hash + Eq + Copy, A: Clone, P: Policy<A>, E: Env<S, A>> Agent<S, A, P, E>
    for VEveryVisitMC<S, A, P, E>
{
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
        let candidates = actions
            .iter()
            .filter_map(|a| {
                let (s, _) = env.transition(state, &Some(a.clone()));
                if let Some(s) = s {
                    let v = self.get_value(&s);
                    Some((a.clone(), v))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

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
        info.iter().zip(R).enumerate().for_each(|(t, ((s, _), r))| {
            let v = self.get_value(s);
            let alpha = 1.0 / (t + 1) as f64;
            self.update_value(s, v + alpha * (r - v));
        });
    }
}

// ┌──────────────────────────────────────────────────────────┐
//  Q-Learning - Every Visit MC
// └──────────────────────────────────────────────────────────┘
pub struct QEveryVisitMC<S, A, P: Policy<A>, E: Env<S, A>> {
    pub q_table: HashMap<(S, A), f64>,
    gamma: f64,
    _action_type: std::marker::PhantomData<A>,
    _policy_type: std::marker::PhantomData<P>,
    _env_type: std::marker::PhantomData<E>,
}

impl<S: Hash + Eq + Copy, A: Hash + Eq + Copy, P: Policy<A>, E: Env<S, A>>
    QEveryVisitMC<S, A, P, E>
{
    pub fn new(gamma: f64) -> Self {
        Self {
            q_table: HashMap::new(),
            gamma,
            _action_type: std::marker::PhantomData,
            _policy_type: std::marker::PhantomData,
            _env_type: std::marker::PhantomData,
        }
    }

    pub fn update_value(&mut self, state: &S, action: &A, value: f64) {
        self.q_table.insert((*state, *action), value);
    }
}

impl<S: Hash + Eq + Copy, A: Hash + Eq + Copy, P: Policy<A>, E: Env<S, A>> Agent<S, A, P, E>
    for QEveryVisitMC<S, A, P, E>
{
    // Information = Episode
    type Information = Vec<(S, A, f64)>;

    fn get_action_value(&self, state: &S, action: &A) -> f64 {
        *self.q_table.get(&(*state, *action)).unwrap_or(&0.0)
    }

    fn get_value(&self, _state: &S) -> f64 {
        unimplemented!()
    }

    fn select_action(&self, state: &S, policy: &mut P, env: &E) -> Option<A> {
        let actions = env.available_actions(state);
        let candidates = actions
            .iter()
            .map(|a| (*a, self.get_action_value(state, a)))
            .collect::<Vec<_>>();

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
            .scan(0.0, |acc, (_, _, r)| {
                *acc = *acc * self.gamma + r;
                Some(*acc)
            })
            .collect();

        // Forward update for value function
        info.iter()
            .zip(R)
            .enumerate()
            .for_each(|(t, ((s, a, _), r))| {
                let v = self.get_action_value(s, a);
                let alpha = 1.0 / (t + 1) as f64;
                self.update_value(s, a, v + alpha * (r - v));
            })
    }
}

// ┌──────────────────────────────────────────────────────────┐
//  Q-Learning - TD0
// └──────────────────────────────────────────────────────────┘
pub struct QTD0<S, A, P: Policy<A>, E: Env<S, A>> {
    pub q_table: HashMap<(S, A), f64>,
    pub gamma: f64,
    t: usize,
    c: f64,
    eta: f64,
    _action_type: std::marker::PhantomData<A>,
    _policy_type: std::marker::PhantomData<P>,
    _env_type: std::marker::PhantomData<E>,
}

impl<S: Hash + Eq + Copy, A: Hash + Eq + Copy, P: Policy<A>, E: Env<S, A>> QTD0<S, A, P, E> {
    pub fn new(gamma: f64, c: f64, eta: f64) -> Self {
        Self {
            q_table: HashMap::new(),
            gamma,
            t: 0,
            c,
            eta,
            _action_type: std::marker::PhantomData,
            _policy_type: std::marker::PhantomData,
            _env_type: std::marker::PhantomData,
        }
    }

    pub fn update_value(&mut self, state: &S, action: &A, value: f64) {
        self.q_table.insert((*state, *action), value);
    }

    pub fn add_value(&mut self, state: &S, action: &A, value: f64) {
        let old = *self.q_table.get(&(*state, *action)).unwrap_or(&0.0);
        self.q_table.insert((*state, *action), old + value);
    }

    pub fn increment_count(&mut self) {
        self.t += 1;
    }

    pub fn reset_count(&mut self) {
        self.t = 0;
    }

    pub fn get_alpha(&self) -> f64 {
        self.c * (self.t as f64 + 1f64).powf(-self.eta)
    }
}

impl<S: Hash + Eq + Copy, A: Hash + Eq + Copy, P: Policy<A>, E: Env<S, A>> Agent<S, A, P, E>
    for QTD0<S, A, P, E>
{
    // Information = Step
    type Information = (S, A, f64, Option<S>, Vec<A>);

    fn get_action_value(&self, state: &S, action: &A) -> f64 {
        *self.q_table.get(&(*state, *action)).unwrap_or(&0.0)
    }

    fn get_value(&self, _state: &S) -> f64 {
        unimplemented!()
    }

    fn select_action(&self, state: &S, policy: &mut P, env: &E) -> Option<A> {
        let actions = env.available_actions(state);
        let candidates = actions
            .iter()
            .map(|a| (*a, self.get_action_value(state, a)))
            .collect::<Vec<_>>();

        policy.select_action(&candidates)
    }

    #[allow(non_snake_case)]
    fn update(&mut self, info: &Self::Information) {
        let (s, a, r, s_next, a_pool) = info;

        let delta = if let Some(s_next) = s_next {
            let mut Q_next_max = std::f64::MIN;
            let mut a_next = vec![];
            for a_prime in a_pool {
                let q = self.get_action_value(s_next, a_prime);
                if q > Q_next_max {
                    Q_next_max = q;
                    a_next = vec![*a_prime];
                } else if q == Q_next_max {
                    a_next.push(*a_prime);
                }
            }
            let a_next = a_next.choose(&mut thread_rng()).unwrap();
            r + self.gamma * self.get_action_value(s_next, a_next)
        } else {
            r - self.get_action_value(s, a)
        };

        let alpha = self.get_alpha();
        self.add_value(s, a, delta * alpha);
        self.increment_count();
    }
}

pub struct QTD0Min<S, A, P: Policy<A>, E: Env<S, A>> {
    pub q_table: HashMap<(S, A), f64>,
    pub gamma: f64,
    t: usize,
    c: f64,
    eta: f64,
    _action_type: std::marker::PhantomData<A>,
    _policy_type: std::marker::PhantomData<P>,
    _env_type: std::marker::PhantomData<E>,
}

impl<S: Hash + Eq + Copy, A: Hash + Eq + Copy, P: Policy<A>, E: Env<S, A>> QTD0Min<S, A, P, E> {
    pub fn new(gamma: f64, c: f64, eta: f64) -> Self {
        Self {
            q_table: HashMap::new(),
            gamma,
            t: 0,
            c,
            eta,
            _action_type: std::marker::PhantomData,
            _policy_type: std::marker::PhantomData,
            _env_type: std::marker::PhantomData,
        }
    }

    pub fn update_value(&mut self, state: &S, action: &A, value: f64) {
        self.q_table.insert((*state, *action), value);
    }

    pub fn add_value(&mut self, state: &S, action: &A, value: f64) {
        let old = *self.q_table.get(&(*state, *action)).unwrap_or(&0.0);
        self.q_table.insert((*state, *action), old + value);
    }

    pub fn increment_count(&mut self) {
        self.t += 1;
    }

    pub fn reset_count(&mut self) {
        self.t = 0;
    }

    pub fn get_alpha(&self) -> f64 {
        self.c * (self.t as f64 + 1f64).powf(-self.eta)
    }
}

impl<S: Hash + Eq + Copy, A: Hash + Eq + Copy, P: Policy<A>, E: Env<S, A>> Agent<S, A, P, E>
    for QTD0Min<S, A, P, E>
{
    // Information = Step
    type Information = (S, A, f64, Option<S>, Vec<A>);

    fn get_action_value(&self, state: &S, action: &A) -> f64 {
        *self.q_table.get(&(*state, *action)).unwrap_or(&0.0)
    }

    fn get_value(&self, _state: &S) -> f64 {
        unimplemented!()
    }

    fn select_action(&self, state: &S, policy: &mut P, env: &E) -> Option<A> {
        let actions = env.available_actions(state);
        let candidates = actions
            .iter()
            .map(|a| (*a, self.get_action_value(state, a)))
            .collect::<Vec<_>>();

        policy.select_action(&candidates)
    }

    #[allow(non_snake_case)]
    fn update(&mut self, info: &Self::Information) {
        let (s, a, r, s_next, a_pool) = info;

        let delta = if let Some(s_next) = s_next {
            let mut Q_next_min = std::f64::MAX;
            let mut a_next = vec![];
            for a_prime in a_pool {
                let q = self.get_action_value(s_next, a_prime);
                if q < Q_next_min {
                    Q_next_min = q;
                    a_next = vec![*a_prime];
                } else if q == Q_next_min {
                    a_next.push(*a_prime);
                }
            }
            let a_next = a_next.choose(&mut thread_rng()).unwrap();
            r + self.gamma * self.get_action_value(s_next, a_next)
        } else {
            r - self.get_action_value(s, a)
        };

        let alpha = self.get_alpha();
        self.add_value(s, a, delta * alpha);
        self.increment_count();
    }
}
