use std::collections::HashMap;

use clustering::{kmeans, Elem};
use ddo::{Compression, Problem, Decision};

use super::model::{Psp, IDLE, PspState};

struct Item<'a> {
    id: usize,
    psp: &'a Psp,
}

impl<'a> Elem for Item<'a> {
    fn dimensions(&self) -> usize {
        self.psp.n_items + 1
    }

    fn at(&self, i: usize) -> f64 {
        if i < self.psp.n_items {
            self.psp.changeover[self.id][i] as f64
        } else {
            self.psp.stocking[self.id] as f64
        }
    }
}

pub struct PspCompression<'a> {
    pub problem: &'a Psp,
    pub meta_problem: Psp,
    pub membership: HashMap<isize, isize>,
    rem_demand_to_prev_demand: Vec<Vec<isize>>,
}

impl<'a> PspCompression<'a> {
    pub fn new(problem: &'a Psp, n_meta_items: usize) -> Self {
        let mut elems = vec![];
        for i in 0..problem.n_items {
            elems.push(Item {
                id: i,
                psp: problem,
            });
        }
        let clustering = kmeans(n_meta_items, Some(0), &elems, 1000);

        let stocking = Self::compute_meta_stocking(problem, &clustering.membership, n_meta_items);
        let changeover = Self::compute_meta_changeover(problem, &clustering.membership, n_meta_items);
        let demands = Self::compute_meta_demands(problem, &clustering.membership, n_meta_items);
        let prev_demands = Psp::compute_prev_demands(&demands);
        let rem_demands = Psp::compute_rem_demands(&demands);

        let meta_problem = Psp {
            n_items: n_meta_items,
            horizon: problem.horizon,
            stocking,
            changeover,
            demands,
            prev_demands,
            rem_demands,
        };

        let rem_demand_to_prev_demand = Self::compute_rem_demand_to_prev_demand(&meta_problem);

        let mut membership = HashMap::new();
        for (i, j) in clustering.membership.iter().enumerate() {
            membership.insert(i as isize, *j as isize);
        }
        membership.insert(IDLE, IDLE);

        PspCompression {
            problem,
            meta_problem,
            membership,
            rem_demand_to_prev_demand,
        }
    }

    fn compute_meta_demands(psp: &Psp, membership: &Vec<usize>, n_meta_items: usize) -> Vec<Vec<usize>> {
        let mut meta_demands = vec![vec![0; psp.horizon]; n_meta_items];

        for i in 0..n_meta_items {
            let mut cur = 0;
            for t in (0..psp.horizon).rev() {
                for j in 0..psp.n_items {
                    if membership[j] == i && psp.demands[j][t] > 0 {
                        cur += psp.demands[j][t];
                    }
                }

                if cur > 0 {
                    meta_demands[i][t] = 1;
                    cur -= 1;
                }
            }
        }

        meta_demands
    }

    fn compute_meta_stocking(psp: &Psp, membership: &Vec<usize>, n_meta_items: usize) -> Vec<usize> {
        let mut meta_stocking = vec![usize::MAX; n_meta_items];
        
        for (i, j) in membership.iter().copied().enumerate() {
            meta_stocking[j] = meta_stocking[j].min(psp.stocking[i]);
        }

        meta_stocking
    }

    fn compute_meta_changeover(psp: &Psp, membership: &Vec<usize>, n_meta_items: usize) -> Vec<Vec<usize>> {
        let mut meta_changeover = vec![vec![usize::MAX; n_meta_items]; n_meta_items];

        for (i, a) in membership.iter().copied().enumerate() {
            for (j, b) in membership.iter().copied().enumerate() {
                if a != b {
                    meta_changeover[a][b] = meta_changeover[a][b].min(psp.changeover[i][j]);
                }
            }
        }

        for i in 0..n_meta_items {
            meta_changeover[i][i] = 0;
        }

        meta_changeover
    }

    fn compute_rem_demand_to_prev_demand(meta_psp: &Psp) -> Vec<Vec<isize>> {
        let mut rem_demand_to_prev_demand = vec![vec![-1; meta_psp.horizon]; meta_psp.n_items];
        
        for i in 0..meta_psp.n_items {
            for (_, prev) in meta_psp.prev_demands[i].iter().copied().enumerate() {
                if prev >= 0 {
                    rem_demand_to_prev_demand[i][meta_psp.rem_demands[i][prev as usize] as usize] = prev;
                }
            }
        }

        rem_demand_to_prev_demand
    }
}

impl<'a> Compression for PspCompression<'a> {
    type State = PspState;

    fn get_compressed_problem(&self) -> &dyn Problem<State = Self::State> {
        &self.meta_problem
    }

    fn compress(&self, state: &PspState) -> PspState {
        let mut rem_demands = vec![0; self.meta_problem.n_items];
        for (i, prev_demand) in state.prev_demands.iter().enumerate() {
            if *prev_demand >= 0 {
                let j = *self.membership.get(&(i as isize)).unwrap();
                rem_demands[j as usize] += self.problem.rem_demands[i][*prev_demand as usize];
            }
        }
        let prev_demands = rem_demands.iter().enumerate().map(|(i,r)| self.rem_demand_to_prev_demand[i][*r as usize]).collect();
        let next = match state.next {
            IDLE => IDLE,
            item => *self.membership.get(&item).unwrap(),
        };
        
        PspState {
            time: state.time,
            next,
            prev_demands,
        }
    }

    fn decompress(&self, solution: &Vec<Decision>) -> Vec<Decision> {
        solution.clone()
    }
}