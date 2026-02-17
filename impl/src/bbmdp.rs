use bbases::dynamics::{Action, State};
use bbases::language::{AtomOrBelief, FormulaOrTopOrBot, L0Formula};
use bbases::utils::satsolving::EvaluationContext;
use frozenset::{Freeze, FrozenSet};
use indexmap::IndexMap;
use indexmap::IndexSet;
use ordered_float::OrderedFloat;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use smdp::simulator::SMDP;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::rc::Rc;
use std::path::Path;

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct InitialStateConstraints {
    fluents: Vec<AtomOrBelief>,
    constraint: L0Formula,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct RewardSpecification {
    pre: Option<FormulaOrTopOrBot>,
    actions: Option<IndexSet<String>>,
    post: Option<FormulaOrTopOrBot>,
    reward: f64,
}

impl RewardSpecification {
    pub fn compute_reward(
        &self,
        old_state: &State,
        action: &String,
        new_state: &State,
        context: &EvaluationContext,
    ) -> (bool, f64) {
        let pre_ok = self
            .pre
            .as_ref()
            .map_or(true, |pre| old_state.eval(pre, context));
        let action_ok = self
            .actions
            .as_ref()
            .map_or(true, |actions| actions.contains(action));
        let post_ok = self
            .post
            .as_ref()
            .map_or(true, |post| new_state.eval(post, context));
        if pre_ok && action_ok && post_ok {
            (true, self.reward)
        } else {
            (false, 0.)
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct ProblemInput {
    pub common_ground: Option<L0Formula>,
    pub initial_states: InitialStateConstraints,
    pub actions: IndexMap<String, Action>,
    pub terminal: FormulaOrTopOrBot,
    pub rewards: IndexMap<String, Vec<RewardSpecification>>,
}

type StateRewardsTuple = (Rc<State>, Vec<(bool, OrderedFloat<f64>)>);

#[derive(Clone, Debug)]
pub struct BbMdp {
    rng: rand::rngs::StdRng,
    tmp_rng: rand::rngs::StdRng,
    initial_state_samples: Vec<Rc<State>>,
    actions: IndexMap<String, Action>,
    state: Rc<State>,
    context: EvaluationContext,
    terminal_state_formula: FormulaOrTopOrBot,
    rewards: IndexMap<String, Vec<RewardSpecification>>,
    successor_memory: HashMap<String, HashMap<Rc<State>, StateRewardsTuple>>,
}

fn generate_initial_state_samples(
    constraints: &InitialStateConstraints,
    num_samples: usize,
    rng: &mut rand::rngs::StdRng,
) -> Vec<Rc<State>> {
    // NOTE: We prefer SPUR for (approximately) uniform sampling on large CNFs.
    // For small toy examples, we provide a fallback that enumerates all models.

    let tmp_val: u64 = rand::thread_rng().gen();
    let tmp_filename = format!("tmp{}.dimacs", tmp_val);
    let tmp_infilename = format!("samples_tmp{}.txt", tmp_val);

    let mut variables: IndexMap<AtomOrBelief, usize> = IndexMap::new();
    let mut variables2: Vec<AtomOrBelief> = Vec::new();
    for v in constraints.fluents.iter() {
        variables2.push(v.clone());
        variables.insert(v.clone(), variables.len());
    }
    let cnf = constraints.constraint.to_cnf();
    let mut cnf2: Vec<Vec<i64>> = Vec::new();
    for clause in cnf.iter() {
        let mut clause2: Vec<i64> = Vec::new();
        for x in clause.iter() {
            let v = variables[&x.abs()] + 1;
            let v2: i64 = (x.sign() as i64) * (v as i64);
            clause2.push(v2);
        }
        cnf2.push(clause2);
    }

    let mut dimacs = File::create(&tmp_filename).expect("Cannot create file");
    writeln!(&mut dimacs, "p cnf {} {}", variables.len(), cnf2.len())
        .expect("Cannot write to file");
    for clause2 in cnf2.iter() {
        let x = clause2
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>()
            .join(" ");
        writeln!(&mut dimacs, "{} 0", x).expect("Cannot write to file");
    }
    dimacs.flush().expect("Cannot write to file");

    // Try SPUR if available.
    let spur_path = Path::new("./spur");
    let mut used_spur = false;
    if spur_path.exists() {
        println!(
            "Using SPUR to uniformly draw {} initial state samples (with duplicates).",
            num_samples
        );
        let seed = rng.gen_range(0..=i16::MAX);
        let spur_result = std::process::Command::new("./spur")
            .arg("-cnf")
            .arg(&tmp_filename)
            .arg("-seed")
            .arg(seed.to_string())
            .arg("-s")
            .arg(num_samples.to_string())
            .output();
        used_spur = spur_result.is_ok();
    }

    if used_spur && Path::new(&tmp_infilename).exists() {
        let mut unique_states = IndexMap::new();
        let mut unique_count = IndexMap::new();
        let mut with_duplicates = Vec::new();
        let mut started = false;
        for line in std::fs::read_to_string(&tmp_infilename).unwrap().lines() {
            if line == "#START_SAMPLES" {
                started = true;
                continue;
            } else if line == "#END_SAMPLES" {
                break;
            } else if line.contains("tot_num_models") {
                println!("Number of all possible initial states: {}", &line[15..]);
                continue;
            } else if !started {
                continue;
            } else {
                let id_sep = line.find(',').unwrap();
                let num: usize = line[0..id_sep].parse().unwrap();
                let state = &line[id_sep + 1..];
                for _ in 1..=num {
                    // randomly determine unspecified variable (which may be either true or false)
                    let full_state: String = state
                        .chars()
                        .map(|c| {
                            if c == '*' {
                                if rng.gen_bool(0.5) {
                                    '1'
                                } else {
                                    '0'
                                }
                            } else {
                                c
                            }
                        })
                        .collect();

                    // create the state and push it to result vector
                    let mut true_vars = IndexSet::new();
                    for (i, c) in full_state.chars().enumerate() {
                        if c == '1' {
                            true_vars.insert(variables2[i].clone());
                        }
                    }
                    let s = State::new(true_vars.iter().cloned().collect());
                    if !unique_states.contains_key(&full_state) {
                        unique_states.insert(full_state.clone(), s.clone());
                        unique_count.insert(full_state.clone(), 0);
                    }
                    *unique_count.get_mut(&full_state).unwrap() += 1;

                    with_duplicates.push(s);
                }
            }
        }
        let _ = std::fs::remove_file(tmp_filename);
        let _ = std::fs::remove_file(tmp_infilename);

        println!("The following unique states where sampled:");
        for (i, (str, state)) in unique_states.iter().enumerate() {
            let count = unique_count.get(str).unwrap();
            println!("  {}. {} ({} copies)", i, state, count);
        }

        with_duplicates.shuffle(rng);
        return with_duplicates;
    }

    // Fallback: enumerate all satisfying assignments (small problems only).
    println!(
        "WARNING: ./spur not found or failed; falling back to model enumeration (may be slow for large tasks)."
    );

    // Convert CNF to i32 literals.
    let cnf_i32: Vec<Vec<i32>> = cnf2
        .iter()
        .map(|clause| clause.iter().map(|x| *x as i32).collect())
        .collect();
    let num_vars = variables.len() as i32;

    // Enumerate models by repeatedly solving with blocking clauses.
    let mut blocking: Vec<Vec<i32>> = Vec::new();
    let mut all_models: Vec<Vec<bool>> = Vec::new();
    let model_cap: usize = 100_000;
    loop {
        if all_models.len() >= model_cap {
            panic!(
                "Fallback model enumeration hit cap ({} models). Install SPUR for large tasks.",
                model_cap
            );
        }

        let mut solver = cat_solver::Solver::default();
        if num_vars > 0 {
            solver.reserve(num_vars);
        }
        for clause in cnf_i32.iter() {
            solver.add_clause(clause.clone());
        }
        for clause in blocking.iter() {
            solver.add_clause(clause.clone());
        }

        match solver.solve() {
            Some(true) => {
                let mut assignment: Vec<bool> = Vec::with_capacity(num_vars as usize);
                for v in 1..=num_vars {
                    assignment.push(solver.value(v).unwrap_or(false));
                }

                // Add blocking clause for this exact model.
                let mut block_clause: Vec<i32> = Vec::with_capacity(num_vars as usize);
                for (i, val) in assignment.iter().enumerate() {
                    let lit = (i as i32) + 1;
                    // Flip at least one variable.
                    block_clause.push(if *val { -lit } else { lit });
                }
                blocking.push(block_clause);
                all_models.push(assignment);
            }
            Some(false) => break,
            None => panic!(
                "SAT solver aborted during fallback enumeration. Install SPUR for robust sampling."
            ),
        }
    }

    if all_models.is_empty() {
        panic!("No initial states satisfy the constraint formula.");
    }
    println!(
        "Fallback enumeration found {} possible initial states.",
        all_models.len()
    );

    // Sample with replacement uniformly from the enumerated models.
    let mut with_duplicates: Vec<Rc<State>> = Vec::with_capacity(num_samples);
    for _ in 0..num_samples {
        let model = all_models.choose(rng).unwrap();
        let mut true_vars = IndexSet::new();
        for (i, is_true) in model.iter().enumerate() {
            if *is_true {
                true_vars.insert(variables2[i].clone());
            }
        }
        with_duplicates.push(State::new(true_vars.iter().cloned().collect()));
    }

    let _ = std::fs::remove_file(tmp_filename);
    let _ = std::fs::remove_file(tmp_infilename);
    with_duplicates.shuffle(rng);
    with_duplicates
}

impl BbMdp {
    pub fn new(problem: ProblemInput, num_samples: usize, seed: u64) -> BbMdp {
        let rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut tmp_rng = rand::rngs::StdRng::seed_from_u64(seed + 1);
        let initial_state_samples =
            generate_initial_state_samples(&problem.initial_states, num_samples, &mut tmp_rng);
        let context =
            EvaluationContext::new(&problem.initial_states.fluents, &problem.common_ground);
        BbMdp {
            rng,
            tmp_rng,
            initial_state_samples,
            actions: problem.actions,
            state: State::dummy(),
            context,
            terminal_state_formula: problem.terminal,
            rewards: problem.rewards,
            successor_memory: HashMap::new(),
        }
    }
}

impl SMDP<FrozenSet<L0Formula>, String, OrderedFloat<f64>> for BbMdp {
    fn initialize(&mut self, state_id: usize, _use_tmp_rng: bool) {
        let mut state_id = state_id;
        if state_id >= self.initial_state_samples.len() {
            println!("WARNING: Sample id {} is out of bounds.", state_id);
            state_id %= self.initial_state_samples.len();
        }
        self.state = self.initial_state_samples.get(state_id).unwrap().clone();
        //println!("    True State: {}", self.state);
    }

    fn get_subjective(&self) -> FrozenSet<L0Formula> {
        // Belief-based POMDP interface: the agent observes only the localized belief base
        // (i.e., formulas φ such that △φ is present in the underlying augmented state).
        self.state.localize(&None).freeze()
    }

    fn memory_load(&self) -> usize {
        self.get_subjective().len()
    }

    fn apply(&mut self, action: &String, _use_tmp_rng: bool) -> Vec<(bool, OrderedFloat<f64>)> {
        // if we have computed the transition before
        if let Some(action_successors) = self.successor_memory.get(action) {
            if let Some((succ, reward)) = action_successors.get(&self.state) {
                self.state = succ.clone();
                return reward.clone();
            }
        } else {
            self.successor_memory.insert(action.clone(), HashMap::new());
        }

        // otherwise compute the transition
        let old_state = self.state.clone();
        let action_spec = self.actions.get(action).unwrap();
        self.state = self.state.apply(action_spec, &self.context);
        // TODO: split up terminal condition and (additive) reward formulas
        let rewards: Vec<(bool, OrderedFloat<f64>)> = self
            .rewards
            .iter()
            .map(|(_name, specs)| {
                let mut r_ok = false;
                let mut r_val = 0.0;
                for spec in specs.iter() {
                    let (spec_ok, spec_val) =
                        spec.compute_reward(&old_state, action, &self.state, &self.context);
                    r_ok |= spec_ok;
                    r_val += spec_val;
                }
                (r_ok, OrderedFloat::<f64>::from(r_val))
            })
            .collect();

        self.successor_memory
            .get_mut(action)
            .unwrap()
            .insert(old_state, (self.state.clone(), rewards.clone()));
        rewards
    }

    fn available_actions(&self) -> Vec<String> {
        self.actions
            .iter()
            .filter(|(_, action)| match &action.shields {
                Some(shields) => shields
                    .iter()
                    .all(|(_, shield)| self.state.eval(shield, &self.context)),
                None => true,
            })
            .map(|(name, _)| name.clone())
            .collect()
    }

    fn is_terminal(&self) -> bool {
        self.state.eval(&self.terminal_state_formula, &self.context)
    }

    fn get_rng(&mut self) -> &mut rand::prelude::StdRng {
        &mut self.rng
    }

    fn get_tmp_rng(&mut self) -> &mut rand::prelude::StdRng {
        &mut self.tmp_rng
    }

    fn seed_tmp_rng(&mut self, seed: u64) {
        self.tmp_rng = rand::rngs::StdRng::seed_from_u64(seed);
    }

    fn tidy_up(&self) {
        State::tidy_up()
    }
}
