use crate::language::{AtomOrBelief, Base, L0Formula};

use std::{
    collections::{HashMap, HashSet},
    ops::{Deref, Not},
    rc::Rc,
};

#[derive(Clone, Debug)]
pub struct EvaluationContext {
    variables: HashMap<Base, i32>,
    common_ground: FormulaCNF,
}

pub type BBaseCNF = Vec<Vec<i32>>;
pub type FormulaCNF = Vec<Vec<i32>>;

fn to_literal(literal: &Base, variables: &HashMap<Base, i32>) -> i32 {
    match literal {
        Base::Atom(_) | Base::Triangle(_, _) => {
            let var = variables
                .get(literal)
                .unwrap_or_else(|| panic!("ERROR: {literal} is unknown."));
            *var
        }
        Base::Not(inner) => {
            let inner = to_literal(inner, variables);
            assert!(inner > 0);
            -inner
        }
        _ => panic!("ERROR: {literal} is not an atom or belief literal!"),
    }
}

fn to_sat_instance(formula: &L0Formula, variables: &HashMap<Base, i32>) -> FormulaCNF {
    formula
        .to_cnf()
        .iter()
        .map(|clause| {
            clause
                .iter()
                .map(|lit| to_literal(lit.deref(), variables))
                .collect()
        })
        .collect()
}

impl EvaluationContext {
    pub fn new(fluents: &[AtomOrBelief], common_ground: &Option<L0Formula>) -> Self {
        // create a variable from each fluent
        let mut variables = HashMap::new();
        for (i, v) in fluents.iter().enumerate() {
            let vi = (i + 1) as i32;
            variables.insert(v.deref().clone(), vi);
        }

        // now build a sat instance from the cnf of the common ground
        let common_ground = match common_ground {
            Some(common_ground) => to_sat_instance(common_ground, &variables),
            None => Vec::new(),
        };

        EvaluationContext {
            variables,
            common_ground,
        }
    }

    pub fn base_description(
        &self,
        bbase: HashSet<L0Formula>,
        agent: &Option<Rc<String>>,
    ) -> BBaseCNF {
        let mut result = Vec::new();

        // direct beliefs from the belief base
        for formula in bbase.iter() {
            let as_sat = to_sat_instance(formula, &self.variables);
            result.extend(as_sat);
        }

        // what if belief base contains triangle_agent phi but not phi?
        // -> invalid belief base, ignore this case for now

        // add positive/negative belief literals depending on whether the believed formula occurs in belief base
        for (var, vi) in self.variables.iter() {
            if let Base::Triangle(a, belief) = var {
                let belief: L0Formula = belief.clone().try_into().unwrap();
                if a == agent {
                    if bbase.contains(&belief) {
                        result.push(vec![*vi]);
                    } else {
                        result.push(vec![-vi]);
                    }
                }
            }
        }
        result
    }

    pub fn evaluate(&self, base: &BBaseCNF, formula: &L0Formula) -> bool {
        let neg_formula: Base = formula.deref().clone().not();
        let neg_formula: L0Formula = neg_formula.try_into().unwrap();
        let neg_formula: FormulaCNF = to_sat_instance(&neg_formula, &self.variables);

        let mut solver = cat_solver::Solver::default();

        for clause in self.common_ground.iter() {
            solver.add_clause(clause.clone());
        }

        for clause in base.iter() {
            solver.add_clause(clause.clone());
        }

        for clause in neg_formula.into_iter() {
            solver.add_clause(clause);
        }

        !solver.solve().unwrap()
    }
}
