use std::{collections::HashMap, hash::Hash};

pub trait Environment<S, A> {
    fn transition(&self, state: &S, action: &A) -> (Option<S>, f64);
    fn is_terminal(&self, state: &S) -> bool;
    fn is_goal(&self, state: &S) -> bool;
    fn available_actions(&self, state: &S) -> Vec<A>;
}

pub trait Agent<S, A, E: Environment<S, A>> {
    type Information;
    fn select_action(&self, state: &S, env: &E) -> A;
    fn update(&mut self, info: &Self::Information);
    fn get_value(&self, state: &S) -> f64;
    fn get_action_value(&self, state: &S, action: &A) -> f64;
}

// TODO: Implement agent with value iteration (MC)
pub struct VEveryVisitMC<S, A, E: Environment<S, A>> {
    value_function: HashMap<S, f64>,
    _action_type: std::marker::PhantomData<A>,
    _env_type: std::marker::PhantomData<E>,
}

impl<S: Hash + Eq, A, E: Environment<S, A>> Agent<S, A, E> for VEveryVisitMC<S, A, E> {
    // Information = Episode
    type Information = Vec<(S, f64)>;

    fn get_action_value(&self, state: &S, action: &A) -> f64 {
        unimplemented!()
    }

    fn get_value(&self, state: &S) -> f64 {
        self.value_function.get(state).unwrap_or(&0.0).clone()
    }

    fn select_action(&self, state: &S, env: &E) -> A {
        let actions = env.available_actions(state);
    }

    fn update(&mut self, info: &Self::Information) {
        todo!()
    }
}

// TODO: One column grid world
// NOTE: Use value iteration (MC & TD0) to solve the problem
// NOTE: Valid row : 0 ~ num_rows - 1
#[derive(Debug, Clone)]
pub struct LineWorld {
    num_rows: usize,
    init_state: usize,
    goal_state: usize,
    terminal_state: Vec<usize>,
}

impl LineWorld {
    pub fn new(num_rows: usize, init_state: usize, goal_state: usize) -> Self {
        Self {
            num_rows,
            init_state,
            goal_state,
            terminal_state: vec![],
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

    fn transition(&self, state: &usize, action: &LineWorldAction) -> (Option<usize>, f64) {
        if self.is_terminal(state) {
            (None, -1.0)
        } else if self.is_goal(state) {
            (None, 1.0)
        } else {
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
