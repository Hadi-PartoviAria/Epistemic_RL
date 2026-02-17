use crate::language::Base;
use nom::bytes::complete::{take_while, take_while1};
use nom::character::complete::{char, one_of};
use nom::character::{is_alphabetic, is_alphanumeric};
use nom::error::{Error, ErrorKind};
use nom::sequence::{pair, tuple};
use nom::{Err, Finish, IResult};
use serde::{Deserialize, Deserializer};
use std::rc::Rc;
use std::str::FromStr;

fn is_ascii_lowercase(chr: char) -> bool {
    chr.is_ascii() && is_alphabetic(chr as u8) && chr.is_lowercase()
}

fn is_ascii_alphanumeric_or_underline(chr: char) -> bool {
    chr.is_ascii() && (is_alphanumeric(chr as u8) || chr == '_')
}

/// Nom parser for proposition identifiers ([a-z][a-zA-Z0-9_]+).
pub fn atom_identifier(input: &str) -> IResult<&str, String> {
    let (input, (c1, c2)) = pair(
        take_while1(is_ascii_lowercase),
        take_while(is_ascii_alphanumeric_or_underline),
    )(input)?;
    Ok((input, String::from(c1) + c2))
}

/// Nom parser for T.
fn top(input: &str) -> IResult<&str, ()> {
    let (input, _chr) = char('T')(input)?;
    Ok((input, ()))
}

/// Nom parser for ⊥.
fn bot(input: &str) -> IResult<&str, ()> {
    let (input, _chr) = char('⊥')(input)?;
    Ok((input, ()))
}

impl FromStr for Base {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match Self::parser(s).finish() {
            Ok((remaining, t)) => {
                if remaining.is_empty() {
                    Ok(t)
                } else {
                    Err("Malformed formula.")
                }
            }
            Err(Error { input: _, code: _ }) => Err("Malformed formula."),
        }
    }
}

impl Base {
    fn l_parser(input: &str) -> IResult<&str, Base> {
        if let Ok((input, _)) = top(input) {
            Ok((input, Base::Top))
        } else if let Ok((input, _)) = bot(input) {
            Ok((input, Base::Bot))
        } else if let Ok((input, a)) = atom_identifier(input) {
            Ok((input, Base::Atom(Rc::new(a))))
        } else if let Ok((input, (_, phi))) = pair(char('!'), Base::l_parser)(input) {
            Ok((input, !phi))
        } else if let Ok((input, (_, phi))) = pair(char('△'), Base::l_parser)(input) {
            Ok((input, Base::Triangle(None, Rc::new(phi))))
        } else if let Ok((input, (_, phi))) = pair(char('□'), Base::l_parser)(input) {
            Ok((input, Base::Square(None, Rc::new(phi))))
        } else if let Ok((input, (_, phi))) = pair(char('◇'), Base::l_parser)(input) {
            Ok((input, Base::Diamond(None, Rc::new(phi))))
        } else if let Ok((input, (_, _, agent, _, phi))) = tuple((
            char('△'),
            char('{'),
            atom_identifier,
            char('}'),
            Base::l_parser,
        ))(input)
        {
            Ok((input, Base::Triangle(Some(Rc::new(agent)), Rc::new(phi))))
        } else if let Ok((input, (_, _, agent, _, phi))) = tuple((
            char('□'),
            char('{'),
            atom_identifier,
            char('}'),
            Base::l_parser,
        ))(input)
        {
            Ok((input, Base::Square(Some(Rc::new(agent)), Rc::new(phi))))
        } else if let Ok((input, (_, _, agent, _, phi))) = tuple((
            char('◇'),
            char('{'),
            atom_identifier,
            char('}'),
            Base::l_parser,
        ))(input)
        {
            Ok((input, Base::Diamond(Some(Rc::new(agent)), Rc::new(phi))))
        } else if let Ok((input, (_, phi, _))) = tuple((char('('), Base::parser, char(')')))(input)
        {
            Ok((input, phi))
        } else {
            Err(Err::Failure(Error::new(input, ErrorKind::Fail)))
        }
    }

    fn parser(input: &str) -> IResult<&str, Base> {
        if let Ok((input, (phi, op, psi))) =
            tuple((Base::l_parser, one_of("&|><="), Base::parser))(input)
        {
            match op {
                '&' => Ok((input, phi & psi)),
                '|' => Ok((input, phi | psi)),
                '>' => Ok((input, !phi | psi)),
                '<' => Ok((input, phi | !psi)),
                '=' => {
                    let rc_phi = Rc::new(phi);
                    let rc_psi = Rc::new(psi);
                    Ok((
                        input,
                        Base::Or(
                            Base::and(rc_phi.clone(), rc_psi.clone()),
                            Base::and(Base::not(rc_phi), Base::not(rc_psi)),
                        ),
                    ))
                }
                _ => Err(Err::Failure(Error::new(input, ErrorKind::Fail))),
            }
        } else {
            Base::l_parser(input)
        }
    }
}

pub struct LangVisitor<T> {
    phantom: std::marker::PhantomData<T>,
}

impl<T> Default for LangVisitor<T> {
    fn default() -> Self {
        LangVisitor {
            phantom: std::marker::PhantomData,
        }
    }
}

impl<'de, T> serde::de::Visitor<'de> for LangVisitor<T>
where
    T: std::str::FromStr<Err = &'static str>,
{
    type Value = T;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a formula string")
    }

    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        let s = T::from_str(v);
        match s {
            Ok(s) => Ok(s),
            Result::Err(s) => Err(serde::de::Error::custom(s)),
        }
    }
}

impl<'de> Deserialize<'de> for Base {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_str(LangVisitor::<Base>::default())
    }
}
