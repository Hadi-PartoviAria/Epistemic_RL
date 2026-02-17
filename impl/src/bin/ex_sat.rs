fn main() {
    let mut solver = kissat::Solver::new();
    let a = solver.var();
    let b = solver.var();
    solver.add2(a, !b);
    solver.add1(b);
    match solver.sat() {
        Some(solution) => println!("SAT: {:?} {:?}", solution.get(a), solution.get(b)),
        None => println!("UNSAT"),
    }
}
