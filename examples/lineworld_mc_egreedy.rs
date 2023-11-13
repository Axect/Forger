use peroxide::fuga::*;
use forger::prelude::*;
use forger::env::lineworld::{LineWorld, LineWorldAction};

pub type S = usize;
pub type A = LineWorldAction;
pub type P = EGreedyPolicy<A>;
pub type E = LineWorld;

fn main() {
    let env = LineWorld::new(10, 1, 9, vec![0]);
    let mut agent = QEveryVisitMC::<S, A, P, E>::new(0.9);
    let mut policy = EGreedyPolicy::<A>::new(0.1, 1.0);

    let mut history = Vec::new();
    for _ in 0..200 {
        let mut episode = vec![];
        let mut state = env.get_init_state();

        loop {
            let action = agent.select_action(&state, &mut policy, &env);
            let (next_state, reward) = env.transition(&state, &action);
            episode.push((state, action.unwrap(), reward));
            match next_state {
                Some(next_state) => state = next_state,
                None => break,
            }
        }

        agent.update(&episode);
        history.push(episode);
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

    // Write parquet
    let mut df = DataFrame::new(vec![]);
    df.push("len", Series::new(history_len_vec));
    df.print();
    df.write_parquet("data/lineworld_mc_egreedy.parquet", CompressionOptions::Uncompressed).unwrap();
}
