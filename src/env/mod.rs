pub mod lineworld;

pub trait Env<S, A> {
    fn transition(&self, state: &S, action: &Option<A>) -> (Option<S>, f64);
    fn is_terminal(&self, state: &S) -> bool;
    fn is_goal(&self, state: &S) -> bool;
    fn available_actions(&self, state: &S) -> Vec<A>;
}
