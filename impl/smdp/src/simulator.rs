use csv::Writer;
use num_traits::{pow, Float};
use rand::{rngs::StdRng, seq::SliceRandom, Rng};
use serde::Serialize;
use std::fmt::Debug;
use std::hash::Hash;
use std::iter::Sum;
use std::{fmt::Display, fs::File};

use crate::learning::{BatchQ, UpdateFn};
use crate::telemetry::{MemoryInfo, RewardInfo};

#[derive(Clone, Debug)]
pub struct LearningParameters {
    pub num_episodes: usize,
    pub max_episode_len: usize,
    pub update_fn: UpdateFn,
}

pub struct ExplorationParameters {
    pub epsilon: f64,
    pub prefer_unseen: bool,
}

pub struct MemoryParameters {
    pub memory_capacity: Option<usize>,
    pub memory_penalty_hard_limit: f64,
    pub memory_penalty_continuous: f64,
}

pub struct EvaluationParameters {
    pub interval: usize,
    pub num_episodes: usize,
    pub max_episode_len: usize,
}

pub trait SMDP<S, A, R>: Clone
where
    S: Eq + Hash + Clone + Serialize + Debug,
    A: Eq + Hash + Clone + Display + Serialize,
    R: Eq + Hash + Clone + Float + Sum + Display + Serialize,
{
    fn get_rng(&mut self) -> &mut StdRng;
    fn get_tmp_rng(&mut self) -> &mut StdRng;
    fn seed_tmp_rng(&mut self, seed: u64);
    fn initialize(&mut self, state_id: usize, use_tmp_rng: bool);
    fn available_actions(&self) -> Vec<A>; // must be a function of local state representation
    fn get_subjective(&self) -> S;
    fn memory_load(&self) -> usize;
    fn is_terminal(&self) -> bool;
    fn tidy_up(&self);
    fn apply(&mut self, action: &A, use_tmp_rng: bool) -> Vec<(bool, R)>;
    fn step(&mut self, action: &A, use_tmp_rng: bool) -> (S, A, S, Vec<(bool, R)>) {
        let s1_subj = self.get_subjective();
        let rewards = self.apply(action, use_tmp_rng);
        let s2_subj = self.get_subjective();
        (s1_subj, action.clone(), s2_subj, rewards)
    }

    fn evaluate(
        &mut self,
        batch_q: &BatchQ<S, A, R>,
        evaluation_parameters: &EvaluationParameters,
        training_episodes_offset: usize,
        memory_capacity: Option<usize>,
        discount_factor: R,
    ) -> (Vec<RewardInfo<R>>, Vec<MemoryInfo>) {
        let mut reward_infos: Vec<RewardInfo<R>> = Vec::new();
        let mut memory_infos: Vec<MemoryInfo> = Vec::new();

        for episode in 0..evaluation_parameters.num_episodes {
            let mut reward_info: RewardInfo<R> = RewardInfo::default();
            let mut memory_info: MemoryInfo = MemoryInfo::default();

            self.initialize(training_episodes_offset + episode, true);
            memory_info.log_memory(self.memory_load());

            for step in 0..evaluation_parameters.max_episode_len {
                if self.is_terminal() {
                    break;
                } else if memory_capacity.is_some_and(|c| self.memory_load() > c) {
                    memory_info.log_die();
                    break;
                } else {
                    let subjective_state = self.get_subjective();
                    let available_actions = self.available_actions();
                    let action = batch_q
                        .pi(&subjective_state, &available_actions)
                        .as_slice()
                        .choose(self.get_tmp_rng())
                        .unwrap_or_else(|| {
                            panic!("ERROR: No allowed action in state {subjective_state:?}!")
                        })
                        .clone();
                    let (_s1, _a, _s2, rewards) = self.step(&action, true);
                    let accumulated_discount_factor = pow(discount_factor, step);
                    reward_info.add(step + 1, accumulated_discount_factor, rewards);
                    memory_info.log_memory(self.memory_load());
                }
            }

            reward_infos.push(reward_info);
            memory_infos.push(memory_info);
        }
        (reward_infos, memory_infos)
    }

    fn save_policy_if_improvement(
        batch_q: &BatchQ<S, A, R>,
        curr_value: &R,
        best_value: &mut R,
        filename: &String,
        episode_index: usize,
    ) {
        if curr_value > best_value {
            *best_value = *curr_value;
            let policy_filename = format!("{}.policy_{}.json", filename, episode_index);
            batch_q.save_to(&policy_filename);
        }
    }

    fn log_evaluation_results(
        batch_q: &BatchQ<S, A, R>,
        reward_infos: &[RewardInfo<R>],
        memory_infos: &[MemoryInfo],
        stats_writer: &mut Writer<File>,
        episode_index: usize,
    ) {
        let num_trans = batch_q.num_transitions_in_memory();
        let num_states = batch_q.num_states_in_q_table();

        let avg_discounted_reward = RewardInfo::avg_discounted_reward(reward_infos);
        let avg_total_reward = RewardInfo::avg_total_reward(reward_infos);

        // simple console output
        println!(
                "  Episode {}: Unique transitions in memory: {}. States in Q-table: {}. Mean discounted reward (on evaluation samples) = {}.",
                episode_index,
                num_trans,
                num_states,
                avg_discounted_reward
            );

        // detailed file output
        stats_writer
            .write_record(
                [
                    &[
                        episode_index.to_string(),
                        num_trans.to_string(),
                        num_states.to_string(),
                        avg_discounted_reward.to_string(),
                        avg_total_reward.to_string(),
                    ],
                    RewardInfo::produce_record(reward_infos).as_slice(),
                    &MemoryInfo::produce_record(memory_infos),
                ]
                .concat(),
            )
            .unwrap();
        let _ = stats_writer.flush();
    }

    #[allow(clippy::too_many_arguments)]
    fn learn(
        &mut self,
        discount_factor: R,
        learning_parameters: LearningParameters,
        exploration_parameters: ExplorationParameters,
        memory_parameters: MemoryParameters,
        evaluation_parameters: EvaluationParameters,
        reward_names: &[String],
        output_filename: String,
    ) -> BatchQ<S, A, R> {
        println!("Starting Q-learning!");

        let memory_penalty = R::from(memory_parameters.memory_penalty_hard_limit).unwrap();
        let memory_penalty_continuous =
            R::from(memory_parameters.memory_penalty_continuous).unwrap();

        let mut batch_q = BatchQ::new(learning_parameters.update_fn, discount_factor);
        let base_evaluation_seed: u64 = self.get_rng().gen();
        self.seed_tmp_rng(base_evaluation_seed);

        // we will log policy performance in a csv file
        let stats_filename = format!("{}.stats.csv", output_filename);
        let mut stats_writer = csv::Writer::from_path(stats_filename).unwrap();

        let mut csv_headers = vec![
            "episode".to_string(),
            "observed_transitions".to_string(),
            "states_in_qtable".to_string(),
            "reward_d".to_string(),
            "reward_t".to_string(),
        ];
        for r in reward_names {
            csv_headers.extend([
                format!("{r}_rate"),
                format!("{r}_count"),
                format!("{r}_first"),
                format!("{r}_reward_d"),
                format!("{r}_reward_t"),
            ]);
        }
        csv_headers.extend([
            "mem_load".to_string(),
            "mem_die_rate".to_string(),
            "mem_die_step".to_string(),
            "episode_length".to_string(),
        ]);

        stats_writer.write_record(&csv_headers).unwrap();

        // and store all policies where performance increases from the previously best policy
        let mut best_policy_value = R::min_value();

        // before the first episode, we will evaluate the empty (= uniform random) policy
        let (reward_infos, memory_infos) = self.evaluate(
            &batch_q,
            &evaluation_parameters,
            learning_parameters.num_episodes,
            memory_parameters.memory_capacity,
            discount_factor,
        );
        Self::log_evaluation_results(&batch_q, &reward_infos, &memory_infos, &mut stats_writer, 0);

        for ep in 1..=learning_parameters.num_episodes {
            // perform simulation and store the transitions
            self.initialize(ep, false);
            for _step in 1..=learning_parameters.max_episode_len {
                if self.is_terminal()
                    || memory_parameters
                        .memory_capacity
                        .is_some_and(|c| self.memory_load() > c)
                {
                    break;
                } else {
                    let subjective_state = self.get_subjective();
                    let available_actions = self.available_actions();
                    let action = if self.get_rng().gen_bool(exploration_parameters.epsilon) {
                        available_actions
                            .as_slice()
                            .choose(self.get_rng())
                            .unwrap()
                            .clone()
                    } else {
                        let mut best_actions = batch_q.pi(&subjective_state, &available_actions);
                        if exploration_parameters.prefer_unseen {
                            let unseen_subset: Vec<A> = best_actions
                                .iter()
                                .filter(|&a| !batch_q.explored(&subjective_state, a))
                                .cloned()
                                .collect();
                            if !unseen_subset.is_empty() {
                                best_actions = unseen_subset;
                            }
                        }
                        best_actions
                            .as_slice()
                            .choose(self.get_rng())
                            .unwrap()
                            .clone()
                    };
                    let (state1, action, state2, rewards) = self.step(&action, false);
                    let memory_load = self.memory_load();
                    let mut reward = rewards
                        .iter()
                        .map(|(_, r)| *r)
                        .reduce(R::add)
                        .unwrap_or(R::zero());
                    reward = reward - (R::from(memory_load).unwrap() * memory_penalty_continuous);
                    if let Some(capacity) = memory_parameters.memory_capacity {
                        if memory_load > capacity {
                            reward = reward - memory_penalty;
                        }
                    }
                    let available_actions_2 = self.available_actions();

                    batch_q.add_transition(
                        state1,
                        action,
                        state2,
                        reward,
                        available_actions,
                        available_actions_2,
                    );
                }
            }
            // update the policy
            batch_q.q_update();

            // evaluate the policy
            self.seed_tmp_rng(base_evaluation_seed + ep as u64);
            if ep % evaluation_parameters.interval == 0 {
                let (reward_infos, memory_infos) = self.evaluate(
                    &batch_q,
                    &evaluation_parameters,
                    learning_parameters.num_episodes,
                    memory_parameters.memory_capacity,
                    discount_factor,
                );
                Self::save_policy_if_improvement(
                    &batch_q,
                    &RewardInfo::avg_discounted_reward(&reward_infos),
                    &mut best_policy_value,
                    &output_filename,
                    ep,
                );
                Self::log_evaluation_results(
                    &batch_q,
                    &reward_infos,
                    &memory_infos,
                    &mut stats_writer,
                    ep,
                );

                self.tidy_up();
            }
        }
        batch_q
    }
}
