use bbases::language::FormulaOrTopOrBot;
use std::str::FromStr;

fn main() {
    // T⊥△□◇
    let formula = FormulaOrTopOrBot::from_str("T").unwrap();
    println!("{formula}");
    let formula = FormulaOrTopOrBot::from_str("□p|◇{alice}q&△{bob}r").unwrap();
    println!("{formula}");
}
