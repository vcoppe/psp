use std::{time::{SystemTime, UNIX_EPOCH}, fs::File, io::Write, collections::BTreeSet, ops::Bound::*};

use clap::Args;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;
use rand_distr::{Uniform, Normal, Distribution};

use crate::instance::PspInstance;

#[derive(Debug, Args)]
pub struct PspGenerator {
    /// An optional seed to kickstart the instance generation
    #[clap(short='s', long)]
    seed: Option<u128>,
    /// The number of item types that must be produced
    #[clap(short='n', long, default_value="10")]
    nb_types: usize,
    /// The number of clusters of similar item types
    #[clap(short='c', long, default_value="3")]
    nb_clusters: usize,
    /// The number of time periods
    #[clap(short='p', long, default_value="50")]
    nb_periods: usize,
    /// The number of demands normalized by the number of periods
    #[clap(short='d', long, default_value="0.95")]
    density: f64,
    /// The minimum stocking cost
    #[clap(long, default_value="100")]
    min_stocking: usize,
    /// The maximum stocking cost
    #[clap(long, default_value="10000")]
    max_stocking: usize,
    /// The std deviation of the stocking cost among a cluster
    #[clap(long, default_value="100")]
    stocking_std_dev: usize,
    /// The minimum changeover position used to generate the pairwise costs
    #[clap(long, default_value="100")]
    min_changeover_position: isize,
    /// The maximum changeover position used to generate the pairwise costs
    #[clap(long, default_value="10000")]
    max_changeover_position: isize,
    /// The std deviation of the changeover positions among a cluster
    #[clap(long, default_value="100")]
    changeover_position_std_dev: isize,
    /// Name of the file where to generate the psp instance
    #[clap(short, long)]
    output: Option<String>,
}

impl PspGenerator {

    pub fn generate(&mut self) {
        if self.min_stocking < self.stocking_std_dev {
            self.max_stocking += self.stocking_std_dev - self.min_stocking;
            self.min_stocking = self.stocking_std_dev;
        }

        let mut rng = self.rng();

        let mut nb_types_per_cluster = vec![self.nb_types / self.nb_clusters; self.nb_clusters];
        for i in 0..(self.nb_types % self.nb_clusters) {
            nb_types_per_cluster[i] += 1;
        }
        
        let stocking = self.generate_stocking_costs(&mut rng, &nb_types_per_cluster);
        let changeover = self.generate_changeover_costs(&mut rng, &nb_types_per_cluster);
        let demands = self.generate_demands(&mut rng);

        let instance = PspInstance {
            nb_types: self.nb_types,
            nb_periods: self.nb_periods,
            stocking,
            changeover,
            demands
        };

        let instance = serde_json::to_string_pretty(&instance).unwrap();

        if let Some(output) = self.output.as_ref() {
            File::create(output).unwrap().write_all(instance.as_bytes()).unwrap();
        } else {
            println!("{instance}");
        }
    }

    fn generate_stocking_costs(&self, rng: &mut impl Rng, nb_types_per_cluster: &Vec<usize>) -> Vec<usize> {
        let mut stocking_costs = vec![];

        let rand_centroid = Uniform::new_inclusive(self.min_stocking, self.max_stocking);
        for i in 0..self.nb_clusters {
            let centroid = rand_centroid.sample(rng);
            let rand_stocking = Normal::new(centroid as f64, self.stocking_std_dev as f64).expect("cannot create normal dist");

            for _ in 0..nb_types_per_cluster[i] {
                stocking_costs.push(rand_stocking.sample(rng).round() as usize);
            }
        }

        stocking_costs
    }

    fn generate_changeover_costs(&self, rng: &mut impl Rng, nb_types_per_cluster: &Vec<usize>) -> Vec<Vec<usize>> {
        let mut members = vec![vec![]; self.nb_clusters];
        let mut t = 0_usize;
        for (i, n) in nb_types_per_cluster.iter().copied().enumerate() {
            for _ in 0..n {
                members[i].push(t);
                t += 1;
            }
        }

        let mut transition_costs = vec![vec![0; self.nb_types]; self.nb_types];

        let rand_centroid = Uniform::new_inclusive(self.min_changeover_position, self.max_changeover_position);
        for a in 0..self.nb_clusters {
            let centroid_a = rand_centroid.sample(rng);

            let rand_position_a = Normal::new(centroid_a as f64, self.changeover_position_std_dev as f64).expect("cannot create normal dist");
            let positions_a = (0..nb_types_per_cluster[a]).map(|_| rand_position_a.sample(rng).round() as usize).collect::<Vec<usize>>();

            for b in 0..self.nb_clusters {
                if a == b {
                    for (i, ti) in members[a].iter().copied().enumerate() {
                        for (j, tj) in members[a].iter().copied().enumerate() {
                            transition_costs[ti][tj] = positions_a[i].abs_diff(positions_a[j]);
                        }
                    }
                } else {
                    let centroid_b = rand_centroid.sample(rng);
        
                    let rand_position_b = Normal::new(centroid_b as f64, self.changeover_position_std_dev as f64).expect("cannot create normal dist");
                    let positions_b = (0..nb_types_per_cluster[b]).map(|_| rand_position_b.sample(rng).round() as usize).collect::<Vec<usize>>();

                    for (i, ti) in members[a].iter().copied().enumerate() {
                        for (j, tj) in members[b].iter().copied().enumerate() {
                            transition_costs[ti][tj] = positions_a[i].abs_diff(positions_b[j]);
                        }
                    }
                }
            }
        }
        
        transition_costs
    }

    fn generate_demands(&self, rng: &mut impl Rng) -> Vec<Vec<usize>> {
        let mut feasibility_check = PspFeasibility::new(self.nb_periods);

        let mut demands = vec![vec![0; self.nb_periods]; self.nb_types];
        let nb_demands = (self.density * self.nb_periods as f64).round() as usize;
        let mut count = 0;

        let rand_type = Uniform::new(0, self.nb_types);

        while count < nb_demands {
            let rand_period = Uniform::new(feasibility_check.min(), self.nb_periods);
            let p = rand_period.sample(rng);
            let t = rand_type.sample(rng);
            if demands[t][p] == 0 {
                demands[t][p] = 1;
                feasibility_check.remove(p);
                count += 1;
            }
        }

        demands
    }

    fn rng(&self) -> impl Rng {
        let init = self.seed.unwrap_or_else(|| SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis());
        let mut seed = [0_u8; 32];
        seed.iter_mut().zip(init.to_be_bytes().into_iter()).for_each(|(s, i)| *s = i);
        seed.iter_mut().rev().zip(init.to_le_bytes().into_iter()).for_each(|(s, i)| *s = i);
        ChaChaRng::from_seed(seed)
    }

}

struct PspFeasibility {
    available: BTreeSet<usize>,
}

impl PspFeasibility {
    fn new(nb_periods: usize) -> Self {
        PspFeasibility {
            available: BTreeSet::from_iter(0..nb_periods)
        }
    }

    fn min(&self) -> usize {
        *self.available.first().unwrap()
    }

    fn remove(&mut self, period: usize) {
        let largest = *self.available.range((Unbounded, Included(period))).last().unwrap();
        self.available.remove(&largest);
    }
}