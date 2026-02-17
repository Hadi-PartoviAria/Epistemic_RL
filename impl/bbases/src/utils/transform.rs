use std::{collections::HashSet, rc::Rc};

use crate::language::{AtomOrBeliefLiteral, Base};

impl Base {
    pub fn try_to_nnf(&self) -> Result<Base, String> {
        match self {
            Base::Top => Err("Cannot convert Top to NNF".to_string()),
            Base::Bot => Err("Cannot convert Bot to NNF".to_string()),
            Base::Atom(_) | Base::Triangle(_, _) => Ok(self.clone()),
            Base::Square(agent, alpha) => {
                Ok(Base::Square(agent.clone(), Rc::new(alpha.try_to_nnf()?)))
            }
            Base::Diamond(agent, alpha) => {
                Ok(Base::Diamond(agent.clone(), Rc::new(alpha.try_to_nnf()?)))
            }
            Base::And(phi, psi) => Ok(Base::And(
                Rc::new(phi.try_to_nnf()?),
                Rc::new(psi.try_to_nnf()?),
            )),
            Base::Or(phi, psi) => Ok(Base::Or(
                Rc::new(phi.try_to_nnf()?),
                Rc::new(psi.try_to_nnf()?),
            )),
            Base::Not(phi) => match &**phi {
                Base::Top => Err("Cannot convert Top to NNF".to_string()),
                Base::Bot => Err("Cannot convert Bot to NNF".to_string()),
                Base::Atom(_) | Base::Triangle(_, _) => Ok(self.clone()),
                Base::Not(psi) => psi.try_to_nnf(),
                Base::And(phi, psi) => Ok(Base::Or(
                    Rc::new(Base::Not(phi.clone()).try_to_nnf()?),
                    Rc::new(Base::Not(psi.clone()).try_to_nnf()?),
                )),
                Base::Or(phi, psi) => Ok(Base::And(
                    Rc::new(Base::Not(phi.clone()).try_to_nnf()?),
                    Rc::new(Base::Not(psi.clone()).try_to_nnf()?),
                )),
                Base::Square(agent, alpha) => Ok(Base::Diamond(
                    agent.clone(),
                    Rc::new(Base::Not(alpha.clone()).try_to_nnf()?),
                )),
                Base::Diamond(agent, alpha) => Ok(Base::Square(
                    agent.clone(),
                    Rc::new(Base::Not(alpha.clone()).try_to_nnf()?),
                )),
            },
        }
    }

    pub fn to_lmod(&self) -> Result<Base, String> {
        match self {
            Base::Top => Err("Cannot convert Top to Mod".to_string()),
            Base::Bot => Err("Cannot convert Bot to Mod".to_string()),
            Base::Atom(_) => Ok(self.clone()),
            Base::And(phi, psi) => Ok(Base::And(Rc::new(phi.to_lmod()?), Rc::new(psi.to_lmod()?))),
            Base::Or(phi, psi) => Ok(Base::Or(Rc::new(phi.to_lmod()?), Rc::new(psi.to_lmod()?))),
            Base::Triangle(agent, alpha) => Ok(Base::And(
                Base::triangle(agent.clone(), alpha.clone()),
                Base::square(agent.clone(), alpha.clone()),
            )),
            Base::Square(agent, beta) => Ok(Base::Square(agent.clone(), beta.clone())),
            Base::Diamond(agent, beta) => Ok(Base::Square(agent.clone(), beta.clone())),
            Base::Not(phi) => match **phi {
                Base::Atom(_) | Base::Triangle(_, _) => Ok(self.clone()),
                _ => Err("Formula is not in NNF".to_string()),
            },
        }
    }

    /// Convert a L0 formula in NNF to CNF
    pub fn try_to_cnf(&self) -> Result<Vec<HashSet<AtomOrBeliefLiteral>>, &'static str> {
        match self {
            Base::Top => Err("T has no CNF"),
            Base::Bot => Err("⊥ has no CNF"),
            Base::Square(_, _) => Err("□ has no CNF"),
            Base::Diamond(_, _) => Err("◇ has no CNF"),
            Base::Atom(_) | Base::Triangle(_, _) => {
                let mut clause = HashSet::new();
                clause.insert(self.clone().try_into()?);
                Ok(vec![clause])
            }
            Base::Not(phi) => match &**phi {
                Base::Atom(_) | Base::Triangle(_, _) => {
                    let mut clause = HashSet::new();
                    clause.insert(self.clone().try_into()?);
                    Ok(vec![clause])
                }
                _ => Err("Formula is not in NNF"),
            },
            Base::And(phi, psi) => {
                let mut cnf1 = phi.try_to_cnf()?;
                let mut cnf2 = psi.try_to_cnf()?;
                cnf1.append(&mut cnf2);
                Ok(cnf1)
            }
            Base::Or(phi, psi) => {
                let cnf1 = phi.try_to_cnf()?;
                let cnf2 = psi.try_to_cnf()?;

                let mut result = Vec::new();
                for clause1 in cnf1.iter() {
                    for clause2 in cnf2.iter() {
                        let mut clause: HashSet<AtomOrBeliefLiteral> = HashSet::new();
                        clause.extend(clause1.clone().into_iter());
                        clause.extend(clause2.clone().into_iter());
                        result.push(clause);
                    }
                }
                Ok(result)
            }
        }
    }
}
