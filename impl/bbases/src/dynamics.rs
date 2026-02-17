use crate::language::{AtomOrBelief, Base, FormulaOrTopOrBot, L0Formula, Language};
use crate::utils::satsolving::{BBaseCNF, EvaluationContext};
use frozenset::{Freeze, FrozenSet};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::ops::Deref;
use std::rc::Rc;

#[derive(Clone, Debug)]
pub struct State {
    pub fluents: FrozenSet<AtomOrBelief>,
    pub formula_memory: RefCell<HashMap<FormulaOrTopOrBot, bool>>,
    pub localize_memory: RefCell<HashMap<Option<Rc<String>>, HashSet<L0Formula>>>,
    pub sat_repr_memory: RefCell<HashMap<Option<Rc<String>>, BBaseCNF>>,
}

impl Hash for State {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.fluents.hash(state);
    }
}

impl PartialEq for State {
    fn eq(&self, other: &Self) -> bool {
        self.fluents == other.fluents
    }
}

impl Eq for State {}

thread_local!(static STATES: RefCell<HashMap<FrozenSet<AtomOrBelief>, Rc<State>>> = RefCell::new(HashMap::new()));

impl State {
    pub fn new(fluents: FrozenSet<AtomOrBelief>) -> Rc<Self> {
        STATES.with_borrow_mut(|states| {
            if let Some(state) = states.get(&fluents) {
                state.clone()
            } else {
                let state = Rc::new(State {
                    fluents: fluents.clone(),
                    formula_memory: RefCell::new(HashMap::new()),
                    sat_repr_memory: RefCell::new(HashMap::new()),
                    localize_memory: RefCell::new(HashMap::new()),
                });
                states.insert(fluents, state.clone());
                state.clone()
            }
        })
    }

    pub fn dummy() -> Rc<Self> {
        State::new(FrozenSet::new())
    }

    pub fn tidy_up() {
        // freeing all state memo tables (TODO: more selective)
        STATES.with_borrow_mut(|states| {
            for s in states.values() {
                drop(s.localize_memory.replace(HashMap::new()));
                drop(s.sat_repr_memory.replace(HashMap::new()));
                drop(s.formula_memory.replace(HashMap::new()));
            }
        });
    }

    pub fn localize(&self, agent: &Option<Rc<String>>) -> HashSet<L0Formula> {
        let mut memory = self.localize_memory.borrow_mut();
        memory
            .entry(agent.clone())
            .or_insert_with(|| self.localize2(agent))
            .clone()
    }

    pub fn localize2(&self, agent: &Option<Rc<String>>) -> HashSet<L0Formula> {
        let mut result: HashSet<L0Formula> = HashSet::new();
        for belief in self.fluents.iter() {
            match &**belief {
                Base::Triangle(a, phi) => {
                    if *a == *agent {
                        let without_triangle: L0Formula = phi.clone().try_into().unwrap();
                        result.insert(without_triangle);
                    }
                }
                Base::Atom(_) => (),
                _ => panic!(),
            };
        }
        result
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct Action {
    pub effects: HashMap<AtomOrBelief, FormulaOrTopOrBot>,
    pub shields: Option<HashMap<String, FormulaOrTopOrBot>>,
}

impl State {
    pub fn eval(&self, phi: &FormulaOrTopOrBot, context: &EvaluationContext) -> bool {
        let mut memory = self.formula_memory.borrow_mut();
        *memory
            .entry(phi.clone())
            .or_insert_with(|| self.eval2(phi, context))
    }

    fn eval2(&self, phi: &FormulaOrTopOrBot, context: &EvaluationContext) -> bool {
        match phi.deref() {
            Base::Top => true,
            Base::Bot => false,
            Base::Atom(_) | Base::Triangle(_, _) => self
                .fluents
                .contains(&phi.to_owned().try_convert().unwrap()),
            Base::Not(phi) => !self.eval2(&phi.to_owned().try_into().unwrap(), context),
            Base::And(phi, psi) => {
                self.eval2(&phi.to_owned().try_into().unwrap(), context)
                    && self.eval2(&psi.to_owned().try_into().unwrap(), context)
            }
            Base::Or(phi, psi) => {
                self.eval2(&phi.to_owned().try_into().unwrap(), context)
                    || self.eval2(&psi.to_owned().try_into().unwrap(), context)
            }
            Base::Square(agt, phi) => {
                let to_evaluate: L0Formula = phi
                    .clone()
                    .try_into()
                    .unwrap_or_else(|_| panic!("Evaluating nested boxes is not supported yet."));
                let mut memory = self.sat_repr_memory.borrow_mut();
                let bbase: &BBaseCNF = memory
                    .entry(agt.clone())
                    .or_insert_with(|| context.base_description(self.localize(agt), agt));
                context.evaluate(bbase, &to_evaluate)
            }
            Base::Diamond(agt, phi) => {
                let as_square = Base::not(Base::square(agt.clone(), Base::not(phi.clone())));
                let as_square: FormulaOrTopOrBot = as_square.try_into().unwrap();
                self.eval(&as_square, context)
            }
        }
    }

    pub fn apply(&self, action: &Action, context: &EvaluationContext) -> Rc<State> {
        let mut fluents: HashSet<AtomOrBelief> = self.fluents.iter().cloned().collect();

        for (atom_or_belief, condition) in action.effects.iter() {
            if self.eval(condition, context) {
                fluents.insert(atom_or_belief.clone());
            } else {
                fluents.remove(atom_or_belief);
            }
        }

        State::new(fluents.freeze())
    }
}

impl std::fmt::Display for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let as_str = self
            .fluents
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>();
        write!(f, "[{}]", as_str.join(", "))
    }
}
