use peroxide::fuga::*;
use rlai2::{Environment, Agent, EGreedyPolicy, VEveryVisitMC, LineWorld, LineWorldAction};

pub type S = usize;
pub type A = LineWorldAction;
pub type P = EGreedyPolicy<A>;
pub type E = LineWorld;

fn main() {
    let env = LineWorld::new(6, 1, 5, vec![0]);
    let mut agent = VEveryVisitMC::<S, A, P, E>::new(0.9);
    let mut policy = EGreedyPolicy::<A>::new(0.1);

    let mut history = Vec::new();
    for _ in 0 .. 500 {
        let mut episode = vec![];
        let mut state = env.get_init_state();

        loop {
            let action = agent.select_action(&state, &mut policy, &env);
            let (next_state, reward) = env.transition(&state, &action);
            episode.push((state, reward)); 
            match next_state {
                Some(next_state) => state = next_state,
                None => break,
            }
        }

        agent.update(&episode);
        history.push(episode);
    }

    let history_len_vec = history.iter().map(|episode| episode.len()).collect::<Vec<_>>();
    history_len_vec.print();
    println!("Value Function: {:?}", agent.value_function);
}
