This repository contains the source code for the AAMAS 2025 paper
**A Simple Integration of Epistemic Logic and Reinforcement Learning**
by *Thorsten Engesser, Thibaut Le Marre, Emiliano Lorini, François Schwarzentruber, and Bruno Zanuttini*.


# Prerequisites

* We have only tested our implementation on Arch and Ubuntu Linux. Windows will probably not work without manual changes.

* Make sure you have Rust installed on your system: <https://www.rust-lang.org/learn/get-started>.

* Our implementation uses the SPUR SAT sampler: <https://github.com/ZaydH/spur.git>. You will need to compile SPUR and put the `spur` executable in the `impl/` folder.

* To run the experiments in parallel, install the Slurm workload manager: <https://slurm.schedmd.com/>.


# Running the Experiments

* Make sure that the `impl/` folder contains a working `spur` executable.

* To build our executable, go to the `impl/` folder and run `cargo build --release`.

* To run the experiments, go to the `experiments/wumpus` folder and run `sh run_experiments.sh`.

* The raw output files of the experiments (which are already in the `experiments/wumpus` folder) will then be overwritten.

* You can recreate the plots with `sh plot_experiments.sh` (this requires Pandas and Seaborn to be installed).


# Miscellaneous

* If you are interested in our specification of the Wumpus world, you can find it in the `wumpus/` folder.

* It contains some additional actions and rewards that we did not use in our paper and that are disabled in the experiments.


# Toy Case Study: Chop Cup / Magic Cup (Belief-Based POMDP)

This repo also includes a tiny cups-and-balls inspired task under `magic_cup/`.
It is a **belief-based POMDP**: the underlying state contains the true ball location, but the learning agent only observes the observer's **belief base** (explicit beliefs), and rewards can depend on **implicit belief** (logical entailment under common ground).

Entry points:

* Task generator (JSON spec): `magic_cup/magic_cup.py`
* RL runner (consumes JSON): `impl/src/bin/rlbb.rs`
* Subjective observation = localized belief base: `impl/src/bbmdp.rs` (`get_subjective`)
* Epistemic entailment via SAT: `impl/bbases/src/dynamics.rs` + `impl/bbases/src/utils/satsolving.rs`
