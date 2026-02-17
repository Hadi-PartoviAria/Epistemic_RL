use bbases::language::Formula;
use std::str::FromStr;

fn main() {
    let formula = Formula::from_str("△(p|q)&!!!□(x>△y)").unwrap();
    let nnf_formula = formula.try_to_nnf().unwrap();
    let mod_formula = nnf_formula.to_lmod().unwrap();
    println!(
        "{}  -- nnf -->  {}  -- mod -->  {}",
        formula, nnf_formula, mod_formula
    )
}
