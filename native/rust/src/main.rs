//! File: Provides the Rust command-line adapter for model/vector loading and prediction output.
//! Functions: run and main validate arguments and surface errors. Variables: paths, model, and cases
//! are invocation-local; exact declaration lines are indexed in docs/CODE_INDEX.md.

use std::env;
use std::path::Path;

use edgevision_native::{Model, load_vectors, write_predictions};

fn run(arguments: &[String]) -> Result<(), String> {
    if arguments.len() != 4 {
        return Err("usage: edgevision-rust MODEL.evm VECTORS.csv OUTPUT.csv".to_owned());
    }
    let model = Model::load(Path::new(&arguments[1]))?;
    let cases = load_vectors(Path::new(&arguments[2]))?;
    write_predictions(Path::new(&arguments[3]), &model, &cases)?;
    println!("wrote {} prediction rows", cases.len());
    Ok(())
}

fn main() {
    let arguments: Vec<String> = env::args().collect();
    if let Err(error) = run(&arguments) {
        eprintln!("edgevision-rust: {error}");
        std::process::exit(1);
    }
}
