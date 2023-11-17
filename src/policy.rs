use peroxide::fuga::*;

pub trait Policy<A> {
    fn select_action(&mut self, action_rewards: &[(A, f64)]) -> Option<A>;
}

// ┌──────────────────────────────────────────────────────────┐
//  Epsilon Greedy (with Decay) Policy                                                            
// └──────────────────────────────────────────────────────────┘
pub struct EGreedyPolicy<A> {
    epsilon: f64,
    decay: f64,
    random: bool,
    _action_type: std::marker::PhantomData<A>,
}

impl<A: Clone> EGreedyPolicy<A> {
    pub fn new(epsilon: f64, decay: f64) -> Self {
        Self {
            epsilon,
            decay,
            random: true,
            _action_type: std::marker::PhantomData,
        }
    }

    pub fn decay_epsilon(&mut self) {
        self.epsilon *= self.decay;
    }

    pub fn eval(&mut self) {
        self.random = false;
    }
}

impl<A: Clone> Policy<A> for EGreedyPolicy<A> {
    fn select_action(&mut self, action_rewards: &[(A, f64)]) -> Option<A> {
        if action_rewards.is_empty() {
            return None;
        }

        let epsilon = self.epsilon;

        let u = Uniform(0f64, 1f64);
        let sample = u.sample(1)[0];

        if sample < epsilon && self.random {
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

pub struct EGreedyPolicyMin<A> {
    epsilon: f64,
    decay: f64,
    random: bool,
    _action_type: std::marker::PhantomData<A>,
}

impl<A: Clone> EGreedyPolicyMin<A> {
    pub fn new(epsilon: f64, decay: f64) -> Self {
        Self {
            epsilon,
            decay,
            random: true,
            _action_type: std::marker::PhantomData,
        }
    }

    pub fn decay_epsilon(&mut self) {
        self.epsilon *= self.decay;
    }

    pub fn eval(&mut self) {
        self.random = false;
    }
}

impl<A: Clone> Policy<A> for EGreedyPolicyMin<A> {
    fn select_action(&mut self, action_rewards: &[(A, f64)]) -> Option<A> {
        if action_rewards.is_empty() {
            return None;
        }

        let epsilon = self.epsilon;

        let u = Uniform(0f64, 1f64);
        let sample = u.sample(1)[0];

        if sample < epsilon && self.random {
            let mut rng = thread_rng();
            Some(action_rewards.choose(&mut rng).unwrap().0.clone())
        } else {
            let mut min_reward = action_rewards[0].1;
            let mut min_actions = vec![];

            for (a, r) in action_rewards.iter() {
                if *r < min_reward {
                    min_reward = *r;
                    min_actions = vec![a.clone()];
                } else if *r == min_reward {
                    min_actions.push(a.clone());
                }
            }

            let mut rng = thread_rng();
            Some(min_actions.choose(&mut rng).unwrap().clone())
        }
    }
}

