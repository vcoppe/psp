use clap::{Parser, Subcommand};
use generate::PspGenerator;
use resolution::Solve;

mod instance;
mod generate;
mod resolution;

#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct PspTools {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Generate(PspGenerator),
    Solve(Solve)
}

fn main() {
    let cli = PspTools::parse();
    match cli.command {
        Command::Generate(mut generate) => generate.generate(),
        Command::Solve(solve) => solve.solve()
    }
}
