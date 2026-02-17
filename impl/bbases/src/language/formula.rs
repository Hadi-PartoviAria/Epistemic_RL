use crate::language::{Base, L0Formula, Language};

#[derive(Eq, Hash, PartialEq, Debug, Clone, bbases_derive::Language)]
pub struct Formula {
    base: std::rc::Rc<Base>,
}

impl Language for Formula {
    fn check(phi: &Base) -> bool {
        match phi {
            Base::Top | Base::Bot => false,
            Base::Atom(_) => true,
            Base::Triangle(_, phi) => L0Formula::check(phi),
            Base::Not(phi) | Base::Square(_, phi) | Base::Diamond(_, phi) => Formula::check(phi),
            Base::And(phi, psi) | Base::Or(phi, psi) => Formula::check(phi) && Formula::check(psi),
        }
    }
}

#[derive(Eq, Hash, PartialEq, Debug, Clone, bbases_derive::Language)]
pub struct FormulaOrTopOrBot {
    base: std::rc::Rc<Base>,
}

impl Language for FormulaOrTopOrBot {
    fn check(phi: &Base) -> bool {
        match phi {
            Base::Top => true,
            Base::Bot => true,
            _ => Formula::check(phi),
        }
    }
}
