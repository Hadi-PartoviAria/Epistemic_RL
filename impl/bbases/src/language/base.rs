use std::rc::Rc;

#[derive(Eq, Hash, PartialEq, Debug, Clone)]
pub enum Base {
    Top,
    Bot,
    Atom(Rc<String>),
    Triangle(Option<Rc<String>>, Rc<Base>),
    Square(Option<Rc<String>>, Rc<Base>),
    Diamond(Option<Rc<String>>, Rc<Base>),
    Not(Rc<Base>),
    And(Rc<Base>, Rc<Base>),
    Or(Rc<Base>, Rc<Base>),
}

impl Base {
    pub fn atom(a: &str) -> Rc<Base> {
        Rc::new(Base::Atom(Rc::new(String::from(a))))
    }

    pub fn triangle(agent: Option<Rc<String>>, phi: Rc<Base>) -> Rc<Base> {
        Rc::new(Base::Triangle(agent, phi))
    }

    pub fn square(agent: Option<Rc<String>>, phi: Rc<Base>) -> Rc<Base> {
        Rc::new(Base::Square(agent, phi))
    }

    pub fn diamond(agent: Option<Rc<String>>, phi: Rc<Base>) -> Rc<Base> {
        Rc::new(Base::Diamond(agent, phi))
    }

    pub fn not(phi: Rc<Base>) -> Rc<Base> {
        Rc::new(Base::Not(phi))
    }

    pub fn and(phi: Rc<Base>, psi: Rc<Base>) -> Rc<Base> {
        Rc::new(Base::And(phi, psi))
    }

    pub fn or(phi: Rc<Base>, psi: Rc<Base>) -> Rc<Base> {
        Rc::new(Base::Or(phi, psi))
    }

    pub fn implies(phi: Rc<Base>, psi: Rc<Base>) -> Rc<Base> {
        Base::or(Base::not(phi), psi)
    }

    pub fn impliedby(phi: Rc<Base>, psi: Rc<Base>) -> Rc<Base> {
        Base::implies(psi, phi)
    }

    pub fn biimplication(phi: Rc<Base>, psi: Rc<Base>) -> Rc<Base> {
        let both_pos = Base::and(phi.clone(), psi.clone());
        let both_neg = Base::and(Base::not(phi), Base::not(psi));
        Base::or(both_pos, both_neg)
    }
}

impl std::ops::Not for Base {
    type Output = Base;

    fn not(self) -> Base {
        Base::Not(Rc::new(self))
    }
}

impl std::ops::BitAnd for Base {
    type Output = Base;

    fn bitand(self, rhs: Self) -> Self::Output {
        Base::And(Rc::new(self), Rc::new(rhs))
    }
}

impl std::ops::BitOr for Base {
    type Output = Base;

    fn bitor(self, rhs: Self) -> Self::Output {
        Base::Or(Rc::new(self), Rc::new(rhs))
    }
}
