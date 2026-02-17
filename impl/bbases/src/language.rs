mod base;
pub(crate) use base::Base;

pub trait Language: Into<std::rc::Rc<Base>> + TryFrom<std::rc::Rc<Base>> {
    fn check(phi: &Base) -> bool;
    fn try_convert<T: Language>(self) -> Result<T, <T as TryFrom<std::rc::Rc<Base>>>::Error> {
        let base: std::rc::Rc<Base> = self.into();
        base.try_into()
    }
}

mod atom;
pub use atom::Atom;
pub use atom::Literal;

mod prop_formula;
pub use prop_formula::PropFormula;

mod lzero_formula;
pub use lzero_formula::AtomOrBelief;
pub use lzero_formula::AtomOrBeliefLiteral;
pub use lzero_formula::Belief;
pub use lzero_formula::L0Formula;

mod formula;
pub use formula::Formula;
pub use formula::FormulaOrTopOrBot;
