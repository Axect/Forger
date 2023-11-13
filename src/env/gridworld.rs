use crate::env::Env;

// ┌──────────────────────────────────────────────────────────┐
//  Grid World
// └──────────────────────────────────────────────────────────┘
#[derive(Debug, Clone)]
pub struct GridWorld {
    num_x: usize,
    num_y: usize,
    init_state: (usize, usize),
    goal_state: (usize, usize),
    terminal_state: Vec<(usize, usize)>,
}

impl GridWorld {
    pub fn new(
        num_x: usize,
        num_y: usize,
        init_state: (usize, usize),
        goal_state: (usize, usize),
        terminal_state: Vec<(usize, usize)>,
    ) -> Self {
        Self {
            num_x,
            num_y,
            init_state,
            goal_state,
            terminal_state,
        }
    }

    pub fn get_init_state(&self) -> (usize, usize) {
        self.init_state
    }

    pub fn get_goal_state(&self) -> (usize, usize) {
        self.goal_state
    }

    pub fn get_terminal_state(&self) -> &Vec<(usize, usize)> {
        &self.terminal_state
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum GridWorldAction {
    Up,
    Down,
    Left,
    Right,
}

impl Env<(usize, usize), GridWorldAction> for GridWorld {
    fn is_terminal(&self, state: &(usize, usize)) -> bool {
        self.terminal_state.contains(state)
    }

    fn is_goal(&self, state: &(usize, usize)) -> bool {
        *state == self.goal_state
    }

    fn transition(
        &self,
        state: &(usize, usize),
        action: &Option<GridWorldAction>,
    ) -> (Option<(usize, usize)>, f64) {
        if self.is_terminal(state) {
            (None, -1.0)
        } else if self.is_goal(state) {
            (None, 1.0)
        } else {
            let action = action.as_ref().unwrap();
            match action {
                GridWorldAction::Up => (Some((state.0, state.1 + 1)), 0.0),
                GridWorldAction::Down => (Some((state.0, state.1 - 1)), 0.0),
                GridWorldAction::Left => (Some((state.0 - 1, state.1)), 0.0),
                GridWorldAction::Right => (Some((state.0 + 1, state.1)), 0.0),
            }
        }
    }

    fn available_actions(&self, state: &(usize, usize)) -> Vec<GridWorldAction> {
        match state {
            (0, 0) => vec![GridWorldAction::Right, GridWorldAction::Up],
            (0, y) if *y == self.num_y - 1 => vec![GridWorldAction::Right, GridWorldAction::Down],
            (0, _) => vec![GridWorldAction::Right, GridWorldAction::Up, GridWorldAction::Down],
            (x, y) if *x == self.num_x - 1 && *y == self.num_y - 1 => {
                vec![GridWorldAction::Left, GridWorldAction::Down]
            }
            (_, y) if *y == self.num_y - 1 => {
                vec![
                    GridWorldAction::Left,
                    GridWorldAction::Right,
                    GridWorldAction::Down,
                ]
            }
            (x, 0) if *x == self.num_x - 1 => {
                vec![GridWorldAction::Left, GridWorldAction::Up]
            }
            (x, _) if *x == self.num_x - 1 => {
                vec![
                    GridWorldAction::Left,
                    GridWorldAction::Up,
                    GridWorldAction::Down,
                ]
            }
            (_, 0) => vec![GridWorldAction::Left, GridWorldAction::Right, GridWorldAction::Up],
            _ => vec![
                GridWorldAction::Left,
                GridWorldAction::Right,
                GridWorldAction::Up,
                GridWorldAction::Down,
            ]
        }
    }
}
