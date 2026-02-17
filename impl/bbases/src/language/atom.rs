use crate::language::{Base, Language};

#[derive(Eq, Hash, PartialEq, Debug, Clone, bbases_derive::Language)]
pub struct Atom {
    base: std::rc::Rc<Base>,
}

impl Language for Atom {
    fn check(phi: &Base) -> bool {
        matches!(phi, Base::Atom(_))
    }
}

#[derive(Eq, Hash, PartialEq, Debug, Clone, bbases_derive::Language)]
pub struct Literal {
    base: std::rc::Rc<Base>,
}

impl Language for Literal {
    fn check(phi: &Base) -> bool {
        match phi {
            Base::Atom(_) => true,
            Base::Not(phi) => Atom::check(phi),
            _ => false,
        }
    }
}
