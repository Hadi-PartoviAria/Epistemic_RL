use crate::language::{Base, Language};

#[derive(Eq, Hash, PartialEq, Debug, Clone, bbases_derive::Language)]
pub struct PropFormula {
    base: std::rc::Rc<Base>,
}

impl Language for PropFormula {
    fn check(phi: &Base) -> bool {
        match phi {
            Base::Atom(_) => true,
            Base::Not(phi) => PropFormula::check(phi),
            Base::And(phi, psi) | Base::Or(phi, psi) => {
                PropFormula::check(phi) && PropFormula::check(psi)
            }
            _ => false,
        }
    }
}
