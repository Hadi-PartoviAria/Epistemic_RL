use std::{collections::HashSet, ops::Deref, rc::Rc};

use crate::language::{Base, Language};

use super::Atom;

#[derive(Eq, Hash, PartialEq, Debug, Clone, bbases_derive::Language)]
pub struct L0Formula {
    base: std::rc::Rc<Base>,
}

impl L0Formula {
    pub fn to_nnf(&self) -> L0Formula {
        L0Formula {
            base: Rc::new(self.base.try_to_nnf().unwrap()),
        }
    }

    pub fn to_cnf(&self) -> Vec<HashSet<AtomOrBeliefLiteral>> {
        self.to_nnf().base.try_to_cnf().unwrap()
    }
}

impl Language for L0Formula {
    fn check(phi: &Base) -> bool {
        match phi {
            Base::Atom(_) => true,
            Base::Not(phi) | Base::Triangle(_, phi) => L0Formula::check(phi),
            Base::And(phi, psi) | Base::Or(phi, psi) => {
                L0Formula::check(phi) && L0Formula::check(psi)
            }
            _ => false,
        }
    }
}

#[derive(Eq, Hash, PartialEq, Debug, Clone, bbases_derive::Language)]
pub struct Belief {
    base: std::rc::Rc<Base>,
}

impl Language for Belief {
    fn check(phi: &Base) -> bool {
        match phi {
            Base::Triangle(_, phi) => L0Formula::check(phi),
            _ => false,
        }
    }
}

#[derive(Eq, Hash, PartialEq, Debug, Clone, bbases_derive::Language)]
pub struct AtomOrBelief {
    base: std::rc::Rc<Base>,
}

impl Language for AtomOrBelief {
    fn check(phi: &Base) -> bool {
        match phi {
            Base::Atom(_) => true,
            Base::Triangle(_, phi) => L0Formula::check(phi),
            _ => false,
        }
    }
}

impl From<Belief> for AtomOrBelief {
    fn from(value: Belief) -> Self {
        AtomOrBelief { base: value.base }
    }
}

impl From<Atom> for AtomOrBelief {
    fn from(value: Atom) -> Self {
        AtomOrBelief {
            base: Rc::new(value.deref().clone()),
        } // TODO: make this nicer
    }
}

#[derive(Eq, Hash, PartialEq, Debug, Clone, bbases_derive::Language)]
pub struct AtomOrBeliefLiteral {
    base: std::rc::Rc<Base>,
}

impl Language for AtomOrBeliefLiteral {
    fn check(phi: &Base) -> bool {
        match phi {
            Base::Atom(_) => true,
            Base::Triangle(_, phi) => L0Formula::check(phi),
            Base::Not(phi) => AtomOrBelief::check(phi),
            _ => false,
        }
    }
}

impl AtomOrBeliefLiteral {
    pub fn abs(&self) -> AtomOrBelief {
        match &*self.base {
            Base::Atom(_) | Base::Triangle(_, _) => AtomOrBelief {
                base: self.base.clone(),
            },
            Base::Not(inner) => AtomOrBelief {
                base: inner.clone(),
            },
            _ => panic!(),
        }
    }

    pub fn sign(&self) -> i32 {
        match &*self.base {
            Base::Atom(_) | Base::Triangle(_, _) => 1,
            Base::Not(_) => -1,
            _ => panic!(),
        }
    }
}

impl From<Belief> for AtomOrBeliefLiteral {
    fn from(value: Belief) -> Self {
        AtomOrBeliefLiteral { base: value.base }
    }
}

impl From<Atom> for AtomOrBeliefLiteral {
    fn from(value: Atom) -> Self {
        AtomOrBeliefLiteral {
            base: Rc::new(value.deref().clone()),
        } // TODO: make this nicer
    }
}
