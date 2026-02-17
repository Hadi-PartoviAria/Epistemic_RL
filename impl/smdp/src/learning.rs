use indexmap::IndexMap;
use num_traits::Float;
use serde::Serialize;
use std::hash::Hash;
use std::iter::Sum;

#[derive(Clone, Debug, Serialize)]
pub enum UpdateFn {
    Min,
    Mean,
    Expected,
}

#[derive(Clone, Debug, Serialize)]
pub struct BatchQ<S, A, R>
where
    S: Eq + Hash + Clone + Serialize,
    A: Eq + Hash + Clone + Serialize,
    R: Eq + Hash + Clone + Float + Sum + Serialize,
{
    update_fn: UpdateFn,
    discount_factor: R,
    available_actions: IndexMap<S, Vec<A>>,
    transitions: IndexMap<(S, A), IndexMap<(S, R), usize>>,
    q_table: IndexMap<S, IndexMap<A, R>>,
}

impl<S, A, R> BatchQ<S, A, R>
where
    S: Eq + Hash + Clone + Serialize,
    A: Eq + Hash + Clone + Serialize,
    R: Eq + Hash + Clone + Float + Sum + Serialize,
{
    pub fn new(update_fn: UpdateFn, discount_factor: R) -> Self {
        BatchQ {
            update_fn,
            discount_factor,
            available_actions: IndexMap::new(),
            transitions: IndexMap::new(),
            q_table: IndexMap::new(),
        }
    }

    /// Save to file.
    pub fn save_to(&self, _filename: &str) {
        // TODO: implement this
    }

    // TODO: Load from file.

    /// Return the size (number of states) of the Q-table.
    pub fn num_states_in_q_table(&self) -> usize {
        self.q_table.len()
    }

    /// Return the number of unique transitions in the transition memory.
    pub fn num_transitions_in_memory(&self) -> usize {
        self.transitions.values().map(|x| x.len()).sum()
    }

    /// Add a transition to the transition memory.
    pub fn add_transition(
        &mut self,
        state: S,
        action: A,
        state2: S,
        reward: R,
        available_s1: Vec<A>,
        available_s2: Vec<A>,
    ) {
        // first, we memorize the set of allowed actions in each state
        if !self.available_actions.contains_key(&state) {
            self.available_actions.insert(state.clone(), available_s1);
        } else {
            assert!(
                self.available_actions[&state] == available_s1,
                "Modeling Error: The same state can have different sets of allowed actions!"
            )
        }

        if !self.available_actions.contains_key(&state2) {
            self.available_actions.insert(state2.clone(), available_s2);
        } else {
            assert!(
                self.available_actions[&state2] == available_s2,
                "Modeling Error: The same state can have different sets of allowed actions!"
            )
        }

        let state_action_pair = (state, action);
        let outcome_reward_pair = (state2, reward);
        // add state-action pair into transition table if it was not present before
        if !self.transitions.contains_key(&state_action_pair) {
            self.transitions
                .insert(state_action_pair.clone(), IndexMap::new());
        }
        // enter outcome (resulting state & reward) if it was not present before
        let state_action_outcomes = self.transitions.get_mut(&state_action_pair).unwrap();
        if !state_action_outcomes.contains_key(&outcome_reward_pair) {
            state_action_outcomes.insert(outcome_reward_pair.clone(), 0);
        }
        // update counter for that particular outcome given the state-action pair
        *state_action_outcomes.get_mut(&outcome_reward_pair).unwrap() += 1;
    }

    /// Check whether a state-action pair is already present in the transition table.
    pub fn explored(&self, state: &S, action: &A) -> bool {
        self.transitions
            .contains_key(&(state.clone(), action.clone()))
    }

    /// Return the Q-value of a state-action pair.
    pub fn q(&self, state: &S, action: &A) -> R {
        match self.q_table.get(state) {
            Some(action_q_values) => *action_q_values.get(action).unwrap_or(&R::zero()),
            None => R::zero(),
        }
    }

    /// Return the value of a state.
    pub fn v(&self, state: &S) -> R {
        if let Some(available_actions) = self.available_actions.get(state) {
            available_actions
                .iter()
                .map(|action| self.q(state, action))
                .reduce(R::max)
                .unwrap_or(R::zero())
        } else {
            R::zero()
        }
    }

    // Return the set of actions maximizing Q(s,a).
    pub fn pi(&self, state: &S, avilable_actions: &[A]) -> Vec<A> {
        let value = self.v(state);
        if let Some(available_actions) = self.available_actions.get(state) {
            available_actions
                .iter()
                .filter(|a| self.q(state, a) == value)
                .cloned()
                .collect()
        } else {
            avilable_actions.to_vec()
        }
    }

    /// Perform a Q-value update episode (iterating over the whole transition table).
    pub fn q_update(&mut self) {
        let mut q_table_new: IndexMap<S, IndexMap<A, R>> = IndexMap::new();

        // For each state-action pair (s,a)
        for ((state, action), outcome) in self.transitions.iter() {
            // and each observed transition s --a--> s' with reward r,
            let outcome_values: Vec<(R, usize)> = outcome
                .iter()
                .map(|((state2, reward), count)| {
                    (
                        // we compute r + discount_factor * max_a Q(s',a)
                        *reward + self.discount_factor * self.v(state2),
                        // and recall how often we have observed that particular outcome.
                        *count,
                    )
                })
                .collect();

            // We aggregate this information to obtain the updated value for Q(s,a)
            let new_q_value = match self.update_fn {
                // by pessimistic value,
                UpdateFn::Min => outcome_values
                    .iter()
                    .map(|(q, _)| *q)
                    .reduce(R::min)
                    .unwrap(),
                // by average value (assuming all outcomes are equally likely), or
                UpdateFn::Mean => {
                    let sum: R = R::sum(outcome_values.iter().map(|(q, _)| *q));
                    let count = R::from(outcome_values.len()).unwrap();
                    sum / count
                }
                // by expected value (assuming the probability of each outcome corresponds to its observed frequency).
                UpdateFn::Expected => {
                    let weighted_sum = R::sum(
                        outcome_values
                            .iter()
                            .map(|(q, n)| *q * R::from(*n).unwrap()),
                    );
                    let count_all: R =
                        R::sum(outcome_values.iter().map(|(_, n)| R::from(*n).unwrap()));
                    weighted_sum / count_all
                }
            };

            // We enter the new value for Q(s,a) into our new Q-table.
            if !q_table_new.contains_key(state) {
                q_table_new.insert(state.clone(), IndexMap::new());
            }
            q_table_new
                .get_mut(state)
                .unwrap()
                .insert(action.clone(), new_q_value);
        }

        self.q_table = q_table_new;
    }
}
