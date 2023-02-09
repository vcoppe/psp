use std::{fs::File, io::BufReader, time::Duration};

use clap::Args;
use ddo::{FixedWidth, TimeBudget, NoDupFringe, MaxUB, ParBarrierSolverFc, Completion, Solver};

use crate::resolution::model::{Psp, PspRelax, PspRanking};
use crate::instance::PspInstance;

#[derive(Debug, Args)]
pub struct Solve {
    /// The path to the instance file
    #[clap(short, long)]
    pub instance: String,
    /// max number of nodes in a layeer
    #[clap(short, long, default_value="100")]
    pub width: usize,
    /// timeout
    #[clap(short, long, default_value="60")]
    pub timeout: u64,
    /// If present, the path where to write the output html
    #[clap(short, long)]
    pub output: Option<String>,
}

impl Solve {
    pub fn solve(&self) {
        let instance: PspInstance = serde_json::from_reader(BufReader::new(File::open(&self.instance).unwrap())).unwrap();

        let prev_demands = Psp::compute_prev_demands(&instance.demands);
        let rem_demands = Psp::compute_rem_demands(&instance.demands);
        
        let problem = Psp {
            n_items: instance.nb_types,
            horizon: instance.nb_periods,
            stocking: instance.stocking,
            changeover: instance.changeover,
            demands: instance.demands,
            prev_demands,
            rem_demands,
        };
        let relaxation = PspRelax::new(problem.clone());

        let width = FixedWidth(self.width);
        let cutoff = TimeBudget::new(Duration::from_secs(self.timeout));
        let ranking = PspRanking;
        let mut fringe = NoDupFringe::new(MaxUB::new(&ranking));

        let mut solver = ParBarrierSolverFc::new(&problem, &relaxation, &ranking, &width, &cutoff, &mut fringe);

        let Completion{best_value, is_exact} = solver.maximize();

        let best_value = best_value.map(|v| -v).unwrap_or(isize::MAX);
        println!("is exact {is_exact}");
        println!("best value {best_value}");

        let mut sol = String::new();
        solver.best_solution().unwrap()
            .iter().map(|d| d.value)
            .for_each(|v| sol.push_str(&format!("{v} ")));

        println!("solution: {sol}");
    }
}