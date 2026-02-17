use clap::Parser;
use epirl::bbmdp::{BbMdp, ProblemInput};
use indexmap::IndexSet;
use ordered_float::OrderedFloat;
use smdp::{
    learning::UpdateFn,
    simulator::{
        EvaluationParameters, ExplorationParameters, LearningParameters, MemoryParameters,
    },
    SMDP,
};
use std::{borrow::BorrowMut, collections::HashSet, fs};

#[derive(clap::ValueEnum, Debug, Clone, Default, serde::Deserialize, serde::Serialize)]
enum ParsableUpdateFn {
    /// Minimize (assume worst-case transition for a given action).
    Min,
    /// Use the mean (assume all transitions are equally probable)
    Mean,
    /// Use the expected value based on the observed transition probabilities.
    #[default]
    Expected,
}

impl ParsableUpdateFn {
    fn update_fn(&self) -> UpdateFn {
        match self {
            ParsableUpdateFn::Min => UpdateFn::Min,
            ParsableUpdateFn::Mean => UpdateFn::Mean,
            ParsableUpdateFn::Expected => UpdateFn::Expected,
        }
    }
}

impl std::fmt::Display for ParsableUpdateFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParsableUpdateFn::Min => write!(f, "min"),
            ParsableUpdateFn::Mean => write!(f, "mean"),
            ParsableUpdateFn::Expected => write!(f, "expected"),
        }
    }
}

#[derive(Parser, Debug, serde::Deserialize, serde::Serialize)]
#[command(version, about, long_about = None)]
/// Reinforcement Learning with Belief Bases
struct Args {
    /// File containing JSON specification for Belief Base MDP.
    #[arg()]
    input: String,

    /// Random seed.
    #[arg(short = 's', long, default_value_t = 0)]
    seed: u64,

    /// Discount factor.
    #[arg(short = 'd', long, default_value_t = 0.95)]
    discount_factor: f64,

    /// Number of training episodes.
    #[arg(short = 'n', long, default_value_t = 100000)]
    num_episodes: usize,

    /// Maximum length per training episode.
    #[arg(short = 'l', long, default_value_t = 1000)]
    max_episode_len: usize,

    /// Function for Q-Update.
    #[arg(short='f', long, default_value_t=ParsableUpdateFn::Expected)]
    update_fn: ParsableUpdateFn,

    /// Exploration parameter (epsilon-greedy).
    #[arg(short = 'e', long, default_value_t = 0.1)]
    epsilon: f64,

    /// Prefer actions for which no transition exist if all other actions are equally good or worse.
    #[arg(short = 'u', long, default_value_t = false)]
    prefer_unseen: bool,

    /// Number of entries that the agent can maintain in their belief base at the same time (hard limit).
    #[arg(short = 'c', long)]
    memory_capacity: Option<usize>,

    /// Penalty the learning agent receives when the hard limit is exceeded (in addition to the run ending).
    #[arg(short = 'p', long, default_value_t = 1.)]
    memory_penalty_hard_limit: f64,

    /// Continuous penalty the learning agent receives for each belief base entry.
    #[arg(short = 'P', long, default_value_t = 0.)]
    memory_penalty_continuous: f64,

    /// Interval at which the policy is evaluated.
    #[arg(short = 'I', long, default_value_t = 1)]
    evaluation_interval: usize,

    /// Number of episodes on which the Q-function is evaluated for each evaluation.
    #[arg(short = 'N', long, default_value_t = 100)]
    evaluation_num_episodes: usize,

    /// Maximum length per evaluation episode.
    #[arg(short = 'L', long, default_value_t = 100)]
    evaluation_max_episode_len: usize,

    /// Filename for experimental output files (without extension).
    #[arg(short, long, default_value_t = String::from("out"))]
    output: String,

    /// Disable actions from the JSON specification (comma-separated list).
    #[arg(long, value_delimiter = ',')]
    disable_actions: Vec<String>,

    /// Disable shields from the JSON specification (comma-separated list).
    #[arg(long, value_delimiter = ',')]
    disable_shields: Vec<String>,

    /// Disable rewards from the JSON specification (comma-separated list).
    #[arg(long, value_delimiter = ',')]
    disable_rewards: Vec<String>,

    /// Disable common ground.
    #[arg(long, default_value_t = false)]
    disable_common_ground: bool,
}

fn main() {
    // argument parsing
    let args = Args::parse();

    // load problem from json
    let input_filename = &args.input;
    let input =
        fs::read_to_string(input_filename).expect("Something went wrong reading the input file");
    let mut task: ProblemInput = serde_json::from_str(&input).unwrap();

    // save info file
    let output_fileprefix = &args.output;
    let info_filename = format!("{output_fileprefix}.info.json");
    let csv_filename = format!("{output_fileprefix}.stats.csv");
    let info = serde_json::to_string_pretty(&args).unwrap();
    let _ = fs::write(&info_filename, info);

    println!("Learning on specification from {input_filename} (output: {info_filename}, {csv_filename}).");

    let learning_parameters = LearningParameters {
        num_episodes: args.num_episodes,
        max_episode_len: args.max_episode_len,
        update_fn: args.update_fn.update_fn(),
    };

    let exploration_parameters = ExplorationParameters {
        epsilon: args.epsilon,
        prefer_unseen: args.prefer_unseen,
    };

    let memory_parameters = MemoryParameters {
        memory_capacity: args.memory_capacity,
        memory_penalty_hard_limit: args.memory_penalty_hard_limit,
        memory_penalty_continuous: args.memory_penalty_continuous,
    };

    let evaluation_parameters = EvaluationParameters {
        interval: args.evaluation_interval,
        num_episodes: args.evaluation_num_episodes,
        max_episode_len: args.evaluation_max_episode_len,
    };

    // remove all disabled actions
    let disabled_actions: HashSet<String> = args.disable_actions.into_iter().collect();
    for disabled_action in disabled_actions.iter() {
        task.actions
            .shift_remove(disabled_action)
            .unwrap_or_else(|| panic!("ERROR: Action {disabled_action} does not exist!"));
    }
    // and print remaining ones
    let enabled_actions: Vec<String> = task.actions.keys().cloned().collect();
    println!("  Enabled actions: {enabled_actions:?}.");

    // remove all disabled shields from all remaining actions
    let disabled_shields: HashSet<String> = args.disable_shields.into_iter().collect();
    for disabled_shield in disabled_shields.iter() {
        for action in task.actions.values_mut() {
            if let Some(shields) = action.shields.borrow_mut() {
                let _ = shields.remove(disabled_shield);
            }
        }
    }
    // and print remaining ones
    let mut enabled_shields: IndexSet<String> = IndexSet::new();
    for action in task.actions.values() {
        if let Some(shields) = &action.shields {
            enabled_shields.extend(shields.keys().cloned());
        }
    }
    let mut enabled_shields: Vec<String> = enabled_shields.into_iter().collect();
    enabled_shields.sort();
    println!("  Enabled shields: {enabled_shields:?}.");

    // remove all disabled rewards
    let disabled_rewards: HashSet<String> = args.disable_rewards.into_iter().collect();
    for disabled_reward in disabled_rewards.iter() {
        task.rewards
            .shift_remove(disabled_reward)
            .unwrap_or_else(|| panic!("ERROR: Reward {disabled_reward} does not exist!"));
    }
    // and print remaining ones
    let enabled_rewards: Vec<String> = task.rewards.keys().cloned().collect();
    println!("  Enabled rewards: {enabled_rewards:?}.");

    // disable common ground
    if args.disable_common_ground {
        task.common_ground = None;
    }

    // generate the mdp
    let mut bbmdp = BbMdp::new(
        task.clone(),
        learning_parameters.num_episodes + evaluation_parameters.num_episodes,
        args.seed,
    );

    // learn (and write )
    bbmdp.learn(
        OrderedFloat(args.discount_factor),
        learning_parameters,
        exploration_parameters,
        memory_parameters,
        evaluation_parameters,
        &enabled_rewards,
        args.output,
    );
}
