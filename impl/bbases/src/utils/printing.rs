use serde::{Serialize, Serializer};
use std::fmt;

use crate::language::Base;

impl fmt::Display for Base {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Base::Top => write!(f, "T"),
            Base::Bot => write!(f, "⊥"),
            Base::Atom(a) => write!(f, "{a}"),
            Base::Triangle(agent, phi) => match &**phi {
                Base::Top
                | Base::Bot
                | Base::Atom(_)
                | Base::Triangle(_, _)
                | Base::Square(_, _)
                | Base::Diamond(_, _)
                | Base::Not(_) => match agent {
                    Some(name) => write!(f, "△{{{name}}}{phi}"),
                    None => write!(f, "△{phi}"),
                },
                Base::And(_, _) | Base::Or(_, _) => match agent {
                    Some(name) => write!(f, "△{{{name}}}({phi})"),
                    None => write!(f, "△({phi})"),
                },
            },
            Base::Square(agent, phi) => match &**phi {
                Base::Top
                | Base::Bot
                | Base::Atom(_)
                | Base::Triangle(_, _)
                | Base::Square(_, _)
                | Base::Diamond(_, _)
                | Base::Not(_) => match agent {
                    Some(name) => write!(f, "□{{{name}}}{phi}"),
                    None => write!(f, "□{phi}"),
                },
                Base::And(_, _) | Base::Or(_, _) => match agent {
                    Some(name) => write!(f, "□{{{name}}}({phi})"),
                    None => write!(f, "□({phi})"),
                },
            },
            Base::Diamond(agent, phi) => match &**phi {
                Base::Top
                | Base::Bot
                | Base::Atom(_)
                | Base::Triangle(_, _)
                | Base::Square(_, _)
                | Base::Diamond(_, _)
                | Base::Not(_) => match agent {
                    Some(name) => write!(f, "◇{{{name}}}{phi}"),
                    None => write!(f, "◇{phi}"),
                },
                Base::And(_, _) | Base::Or(_, _) => match agent {
                    Some(name) => write!(f, "◇{{{name}}}({phi})"),
                    None => write!(f, "◇{phi}"),
                },
            },
            Base::Not(phi) => match &**phi {
                Base::Top
                | Base::Bot
                | Base::Atom(_)
                | Base::Triangle(_, _)
                | Base::Square(_, _)
                | Base::Diamond(_, _)
                | Base::Not(_) => {
                    write!(f, "!{phi}")
                }
                Base::And(_, _) | Base::Or(_, _) => write!(f, "!({phi})"),
            },
            Base::And(phi, psi) => match &**phi {
                Base::Top
                | Base::Bot
                | Base::Atom(_)
                | Base::Triangle(_, _)
                | Base::Square(_, _)
                | Base::Diamond(_, _)
                | Base::Not(_) => match &**psi {
                    Base::Top
                    | Base::Bot
                    | Base::Atom(_)
                    | Base::Triangle(_, _)
                    | Base::Square(_, _)
                    | Base::Diamond(_, _)
                    | Base::Not(_) => {
                        write!(f, "{phi}&{psi}")
                    }
                    Base::And(_, _) | Base::Or(_, _) => write!(f, "{phi}&({psi})"),
                },
                Base::And(_, _) | Base::Or(_, _) => match &**psi {
                    Base::Top
                    | Base::Bot
                    | Base::Atom(_)
                    | Base::Triangle(_, _)
                    | Base::Square(_, _)
                    | Base::Diamond(_, _)
                    | Base::Not(_) => {
                        write!(f, "({phi})&{psi}")
                    }
                    Base::And(_, _) | Base::Or(_, _) => write!(f, "({phi})&({psi})"),
                },
            },
            Base::Or(phi, psi) => match &**phi {
                Base::Top
                | Base::Bot
                | Base::Atom(_)
                | Base::Triangle(_, _)
                | Base::Square(_, _)
                | Base::Diamond(_, _)
                | Base::Not(_) => match &**psi {
                    Base::Top
                    | Base::Bot
                    | Base::Atom(_)
                    | Base::Triangle(_, _)
                    | Base::Square(_, _)
                    | Base::Diamond(_, _)
                    | Base::Not(_) => {
                        write!(f, "{phi}|{psi}")
                    }
                    Base::And(_, _) | Base::Or(_, _) => write!(f, "{phi}|({psi})"),
                },
                Base::And(_, _) | Base::Or(_, _) => match &**psi {
                    Base::Top
                    | Base::Bot
                    | Base::Atom(_)
                    | Base::Triangle(_, _)
                    | Base::Square(_, _)
                    | Base::Diamond(_, _)
                    | Base::Not(_) => {
                        write!(f, "({phi})|{psi}")
                    }
                    Base::And(_, _) | Base::Or(_, _) => write!(f, "({phi})|({psi})"),
                },
            },
        }
    }
}

impl Serialize for Base {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}
