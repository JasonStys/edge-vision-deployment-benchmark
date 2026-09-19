//! File: Parses bounded portable models and vectors, runs inference, and writes predictions.
//! Functions: Model::load/predict, load_vectors, and write_predictions form the public API.
//! Variables: fixed model dimensions and bounded vectors mirror the shared contract; exact lines are
//! indexed in docs/CODE_INDEX.md.

use std::fmt::Write as _;
use std::fs;
use std::path::Path;

pub const INPUT_SIZE: usize = 64;
pub const HIDDEN_ONE: usize = 24;
pub const HIDDEN_TWO: usize = 12;
pub const CLASS_COUNT: usize = 3;
const MAXIMUM_MODEL_BYTES: u64 = 1_000_000;
const MAXIMUM_VECTOR_ROWS: usize = 4_096;
const CLASS_NAMES: [&str; CLASS_COUNT] = ["vertical", "horizontal", "diagonal"];

/// One normalized input image and its stable cross-runtime identifier.
#[derive(Debug, Clone)]
pub struct VectorCase {
    pub identifier: String,
    pub label: String,
    pub features: [f32; INPUT_SIZE],
}

/// An immutable calibrated 64-24-12-3 ReLU network.
#[derive(Debug, Clone)]
pub struct Model {
    weights_one: Vec<f32>,
    bias_one: [f32; HIDDEN_ONE],
    weights_two: Vec<f32>,
    bias_two: [f32; HIDDEN_TWO],
    weights_three: Vec<f32>,
    bias_three: [f32; CLASS_COUNT],
    temperature: f32,
}

struct Tokens<'a> {
    values: std::str::SplitWhitespace<'a>,
}

impl<'a> Tokens<'a> {
    fn next(&mut self, field: &str) -> Result<&'a str, String> {
        self.values
            .next()
            .ok_or_else(|| format!("portable model ended before {field}"))
    }

    fn expect(&mut self, expected: &str) -> Result<(), String> {
        let actual = self.next(expected)?;
        if actual == expected {
            Ok(())
        } else {
            Err(format!("expected {expected}, received {actual}"))
        }
    }

    fn usize(&mut self, field: &str) -> Result<usize, String> {
        self.next(field)?
            .parse::<usize>()
            .map_err(|_| format!("{field} is not an unsigned integer"))
    }

    fn finite_f32(&mut self, field: &str) -> Result<f32, String> {
        let value = self
            .next(field)?
            .parse::<f32>()
            .map_err(|_| format!("{field} is not numeric"))?;
        if value.is_finite() {
            Ok(value)
        } else {
            Err(format!("{field} must be finite"))
        }
    }

    fn vector(&mut self, section: &str, expected: usize) -> Result<Vec<f32>, String> {
        self.expect(section)?;
        if self.usize("tensor count")? != expected {
            return Err(format!("{section} count does not match the runtime"));
        }
        (0..expected).map(|_| self.finite_f32(section)).collect()
    }
}

fn fixed_array<const SIZE: usize>(values: Vec<f32>, name: &str) -> Result<[f32; SIZE], String> {
    values
        .try_into()
        .map_err(|_| format!("{name} count does not match the runtime"))
}

impl Model {
    /// Load a size-limited, count-delimited EDGEVISION_MLP version 1 artifact.
    pub fn load(path: &Path) -> Result<Self, String> {
        let metadata = fs::metadata(path).map_err(|error| format!("model metadata: {error}"))?;
        if !metadata.is_file() || metadata.len() > MAXIMUM_MODEL_BYTES {
            return Err("model is not a regular bounded file".to_owned());
        }
        let content = fs::read_to_string(path).map_err(|error| format!("model read: {error}"))?;
        let mut tokens = Tokens {
            values: content.split_whitespace(),
        };
        tokens.expect("EDGEVISION_MLP")?;
        if tokens.usize("version")? != 1 {
            return Err("unsupported portable model version".to_owned());
        }
        tokens.expect("dims")?;
        let dimensions = [
            tokens.usize("input dimension")?,
            tokens.usize("hidden one dimension")?,
            tokens.usize("hidden two dimension")?,
            tokens.usize("class dimension")?,
        ];
        if dimensions != [INPUT_SIZE, HIDDEN_ONE, HIDDEN_TWO, CLASS_COUNT] {
            return Err("portable model dimensions do not match the runtime".to_owned());
        }
        tokens.expect("classes")?;
        for class_name in CLASS_NAMES {
            tokens.expect(class_name)?;
        }
        tokens.expect("temperature")?;
        let temperature = tokens.finite_f32("temperature")?;
        if !(0.05..=10.0).contains(&temperature) {
            return Err("calibration temperature is outside its safe range".to_owned());
        }
        let model = Self {
            weights_one: tokens.vector("weights_one", INPUT_SIZE * HIDDEN_ONE)?,
            bias_one: fixed_array(tokens.vector("bias_one", HIDDEN_ONE)?, "bias_one")?,
            weights_two: tokens.vector("weights_two", HIDDEN_ONE * HIDDEN_TWO)?,
            bias_two: fixed_array(tokens.vector("bias_two", HIDDEN_TWO)?, "bias_two")?,
            weights_three: tokens.vector("weights_three", HIDDEN_TWO * CLASS_COUNT)?,
            bias_three: fixed_array(tokens.vector("bias_three", CLASS_COUNT)?, "bias_three")?,
            temperature,
        };
        tokens.expect("end")?;
        if tokens.values.next().is_some() {
            return Err("portable model contains trailing data".to_owned());
        }
        Ok(model)
    }

    /// Infer calibrated probabilities in O(parameter_count) time and O(hidden_width) memory.
    pub fn predict(&self, features: &[f32; INPUT_SIZE]) -> Result<[f32; CLASS_COUNT], String> {
        if features
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err("features must contain finite normalized values".to_owned());
        }
        let mut hidden_one = [0.0_f32; HIDDEN_ONE];
        for (output, value) in hidden_one.iter_mut().enumerate() {
            let mut sum = self.bias_one[output];
            for (input, feature) in features.iter().enumerate() {
                sum += feature * self.weights_one[input * HIDDEN_ONE + output];
            }
            *value = sum.max(0.0);
        }
        let mut hidden_two = [0.0_f32; HIDDEN_TWO];
        for (output, value) in hidden_two.iter_mut().enumerate() {
            let mut sum = self.bias_two[output];
            for (input, hidden) in hidden_one.iter().enumerate() {
                sum += hidden * self.weights_two[input * HIDDEN_TWO + output];
            }
            *value = sum.max(0.0);
        }
        let mut logits = [0.0_f32; CLASS_COUNT];
        for (output, value) in logits.iter_mut().enumerate() {
            let mut sum = self.bias_three[output];
            for (input, hidden) in hidden_two.iter().enumerate() {
                sum += hidden * self.weights_three[input * CLASS_COUNT + output];
            }
            *value = sum / self.temperature;
        }
        let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probabilities = logits.map(|value| (value - maximum).exp());
        let denominator: f32 = probabilities.iter().sum();
        probabilities
            .iter_mut()
            .for_each(|value| *value /= denominator);
        Ok(probabilities)
    }
}

/// Parse a generated CSV with exact field counts, normalized features, and a bounded row count.
pub fn load_vectors(path: &Path) -> Result<Vec<VectorCase>, String> {
    let content = fs::read_to_string(path).map_err(|error| format!("vector read: {error}"))?;
    if content.len() > 5_000_000 {
        return Err("test-vector file exceeds its size limit".to_owned());
    }
    let mut lines = content.lines();
    let header = lines.next().ok_or("test-vector file has no header")?;
    let header_fields: Vec<&str> = header.split(',').collect();
    if header_fields.len() != INPUT_SIZE + 2 || header_fields[..2] != ["id", "label"] {
        return Err("test-vector header does not match the schema".to_owned());
    }
    let mut cases = Vec::new();
    for line in lines {
        if cases.len() >= MAXIMUM_VECTOR_ROWS {
            return Err("test-vector file exceeds its row limit".to_owned());
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != INPUT_SIZE + 2
            || fields[0].is_empty()
            || !CLASS_NAMES.contains(&fields[1])
        {
            return Err("test-vector row does not match the schema".to_owned());
        }
        let mut features = [0.0_f32; INPUT_SIZE];
        for (destination, source) in features.iter_mut().zip(&fields[2..]) {
            let value = source
                .parse::<f32>()
                .map_err(|_| "test-vector feature is not numeric".to_owned())?;
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err("test-vector feature is not normalized".to_owned());
            }
            *destination = value;
        }
        cases.push(VectorCase {
            identifier: fields[0].to_owned(),
            label: fields[1].to_owned(),
            features,
        });
    }
    if cases.is_empty() {
        return Err("test-vector file contains no rows".to_owned());
    }
    Ok(cases)
}

/// Write stable probability CSV output for the cross-runtime comparison script.
pub fn write_predictions(path: &Path, model: &Model, cases: &[VectorCase]) -> Result<(), String> {
    let mut output = String::from("id,vertical,horizontal,diagonal\n");
    for vector_case in cases {
        let probabilities = model.predict(&vector_case.features)?;
        writeln!(
            output,
            "{},{:.9},{:.9},{:.9}",
            vector_case.identifier, probabilities[0], probabilities[1], probabilities[2]
        )
        .map_err(|error| format!("prediction formatting: {error}"))?;
    }
    fs::write(path, output).map_err(|error| format!("prediction write: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generated_model_path() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("artifacts/model/compact-mlp.evm")
    }

    #[test]
    fn model_probabilities_are_normalized() {
        let model = Model::load(&generated_model_path()).expect("generated model should load");
        let probabilities = model
            .predict(&[0.0; INPUT_SIZE])
            .expect("normalized features should infer");
        let sum: f32 = probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(probabilities.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn invalid_features_are_rejected() {
        let model = Model::load(&generated_model_path()).expect("generated model should load");
        let mut features = [0.0; INPUT_SIZE];
        features[0] = f32::NAN;
        assert!(model.predict(&features).is_err());
    }

    #[test]
    fn generated_vectors_are_bounded() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("artifacts/test-vectors.csv");
        let vectors = load_vectors(&path).expect("generated vectors should load");
        assert!(!vectors.is_empty());
        assert!(vectors.len() <= MAXIMUM_VECTOR_ROWS);
    }
}
