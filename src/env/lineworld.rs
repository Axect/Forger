use crate::env::Env;

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
    pub fn new(
        num_rows: usize,
        init_state: usize,
        goal_state: usize,
        terminal_state: Vec<usize>,
    ) -> Self {
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

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum LineWorldAction {
    Up,
    Down,
}

impl Env<usize, LineWorldAction> for LineWorld {
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
