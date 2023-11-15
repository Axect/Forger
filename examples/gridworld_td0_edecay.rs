use peroxide::fuga::*;
use forger::prelude::*;
use forger::env::gridworld::{GridWorld, GridWorldAction};

pub type S = (usize, usize);
pub type A = GridWorldAction;
pub type P = EGreedyPolicy<A>;
pub type E = GridWorld;

fn main() {
    let env = GridWorld::new(4, 4, (0, 0), (3, 3), vec![(1, 3), (3, 1)]);
    let mut agent = QTD0::<S, A, P, E>::new(0.95);
    let mut policy = EGreedyPolicy::<A>::new(0.9, 0.9);

    let mut history = Vec::new();
    for _ in 0..100 {
        agent.reset_count();
        let mut episode = vec![];
        let mut state = env.get_init_state();

        loop {
            let action = agent.select_action(&state, &mut policy, &env);
            let (next_state, reward) = env.transition(&state, &action);
            match next_state {
                Some(next_state) => {
                    let step = (state, action.unwrap(), reward, Some(next_state),
        env.available_actions(&next_state));
                    agent.update(&step);
                    episode.push((state, action.unwrap()));
                    state = next_state
                }
                None => {
                    let step = (state, action.unwrap(), reward, None, vec![]);
                    agent.update(&step);
                    episode.push((state, action.unwrap()));
                    break;
                }
            }
        }

        agent.q_table.iter_mut().for_each(|(_, v)| *v = (*v / (2f64 - agent.gamma)).tanh());
        history.push(episode);
        policy.decay_epsilon();
    }

    let history_len_vec = history
        .iter()
        .map(|episode| episode.len() as u64)
        .collect::<Vec<_>>();
    history_len_vec.print();

    // Sort Q table via key
    let mut q_table = agent.q_table.iter().collect::<Vec<_>>();
    q_table.sort_by(|a, b| a.0 .0.cmp(&b.0 .0));

    println!("Q Table: {:#?}", q_table);

    // Evaluate
    policy.eval();
    agent.reset_count();
    let mut episode = vec![];
    let mut state = env.get_init_state();

    loop {
        let action = agent.select_action(&state, &mut policy, &env);
        let (next_state, _) = env.transition(&state, &action);
        match next_state {
            Some(next_state) => {
                episode.push((state, action.unwrap()));
                state = next_state
            }
            None => {
                episode.push((state, action.unwrap()));
                break;
            }
        }
    }

    println!("Episode: {:?}", episode);


    // Write parquet
    let mut df = DataFrame::new(vec![]);
    df.push("len", Series::new(history_len_vec));
    df.print();
    df.write_parquet("data/gridworld_td0_edecay.parquet", CompressionOptions::Uncompressed).unwrap();
}
