use num_traits::Float;
use std::{fmt::Display, iter::Sum};

pub struct RewardInfo<R>
where
    R: Sum + Float + Display,
{
    count_obtained: Vec<usize>,
    first_obtained: Vec<usize>,
    total_reward: Vec<R>,
    discounted_reward: Vec<R>,
}

impl<R> Default for RewardInfo<R>
where
    R: Sum + Float + Display,
{
    fn default() -> Self {
        RewardInfo {
            count_obtained: Vec::new(),
            first_obtained: Vec::new(),
            total_reward: Vec::new(),
            discounted_reward: Vec::new(),
        }
    }
}

impl<R> RewardInfo<R>
where
    R: Sum + Float + Display,
{
    pub fn add(&mut self, step: usize, accumulated_discount_factor: R, rewards: Vec<(bool, R)>) {
        for (i, (reward_obtained, reward_val)) in rewards.iter().enumerate() {
            if self.total_reward.len() <= i {
                self.count_obtained.push(0);
                self.first_obtained.push(0);
                self.total_reward.push(R::zero());
                self.discounted_reward.push(R::zero());
            };

            if *reward_obtained {
                if self.count_obtained[i] == 0 {
                    self.first_obtained[i] = step;
                }
                self.count_obtained[i] += 1;
            }
            self.discounted_reward[i] =
                self.discounted_reward[i] + accumulated_discount_factor * *reward_val;
            self.total_reward[i] = self.total_reward[i] + *reward_val;
        }
    }

    fn total_reward(&self) -> R {
        self.total_reward
            .iter()
            .cloned()
            .reduce(R::add)
            .unwrap_or(R::zero())
    }

    fn discounted_reward(&self) -> R {
        self.discounted_reward
            .iter()
            .cloned()
            .reduce(R::add)
            .unwrap_or(R::zero())
    }

    pub fn avg_discounted_reward(reward_infos: &[Self]) -> R {
        let n = R::from(reward_infos.len()).unwrap();
        reward_infos
            .iter()
            .map(|ri| ri.discounted_reward())
            .reduce(R::add)
            .unwrap_or(R::zero())
            / n
    }

    pub fn avg_total_reward(reward_infos: &[Self]) -> R {
        let n = R::from(reward_infos.len()).unwrap();
        reward_infos
            .iter()
            .map(|ri| ri.total_reward())
            .reduce(R::add)
            .unwrap_or(R::zero())
            / n
    }

    pub fn produce_record(reward_infos: &[Self]) -> Vec<String> {
        let n_episodes = reward_infos.len();
        assert!(n_episodes > 0);
        let n_rewards = reward_infos[0].total_reward.len();

        let mut result = Vec::new();

        for i in 0..n_rewards {
            let obtained = reward_infos.iter().filter(|ri| ri.count_obtained[i] > 0);
            let num_obtained = obtained.clone().count();
            let rate_obtained: f64 = (num_obtained as f64) / (n_episodes as f64);
            let avg_first_obtained: f64 = (obtained.map(|ri| ri.first_obtained[i]).sum::<usize>()
                as f64)
                / (num_obtained as f64);
            let avg_count_obtained: f64 = (reward_infos
                .iter()
                .map(|ri| ri.count_obtained[i])
                .sum::<usize>() as f64)
                / (n_episodes as f64);
            let avg_discounted_reward: R = reward_infos
                .iter()
                .map(|ri| ri.discounted_reward[i])
                .sum::<R>()
                / R::from(n_episodes).unwrap();
            let avg_total_reward: R = reward_infos.iter().map(|ri| ri.total_reward[i]).sum::<R>()
                / R::from(n_episodes).unwrap();

            result.push(rate_obtained.to_string());
            result.push(avg_count_obtained.to_string());
            result.push(avg_first_obtained.to_string());
            result.push(avg_discounted_reward.to_string());
            result.push(avg_total_reward.to_string());
        }
        result
    }
}

#[derive(Default)]
pub struct MemoryInfo {
    count_continuous: usize,
    died_of_exhaustion: bool,
    steps: usize,
}

impl MemoryInfo {
    pub fn log_die(&mut self) {
        self.died_of_exhaustion = true;
    }

    pub fn log_memory(&mut self, memory: usize) {
        self.steps += 1;
        self.count_continuous += memory;
    }

    pub fn produce_record(memory_infos: &[Self]) -> Vec<String> {
        let n_episodes = memory_infos.len();
        let mut result = Vec::new();

        let died = memory_infos.iter().filter(|mi| mi.died_of_exhaustion);
        let num_died = died.clone().count();

        let rate_died: f64 = (num_died as f64) / (n_episodes as f64);
        let avg_step_died: f64 =
            (died.map(|mi| mi.steps).sum::<usize>() as f64) / (num_died as f64);
        let avg_memory_load: f64 = memory_infos
            .iter()
            .map(|mi| (mi.count_continuous as f64) / (mi.steps as f64))
            .sum::<f64>()
            / (n_episodes as f64);
        let avg_total_steps: f64 =
            (memory_infos.iter().map(|mi| mi.steps).sum::<usize>() as f64) / (n_episodes as f64);

        result.push(avg_memory_load.to_string());
        result.push(rate_died.to_string());
        result.push(avg_step_died.to_string());
        result.push(avg_total_steps.to_string());

        result
    }
}
