/** File: Implements fail-closed parsing, dense ReLU inference, stable softmax, and strict CSV I/O.
 * Functions: expect, read_values, split, Model::load/predict, load_vectors, write_predictions.
 * Variables: tensors and fixed-size activation arrays remain bounded by model.hpp; exact lines are
 * indexed in docs/CODE_INDEX.md.
 */
#include "edgevision/model.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <ios>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace edgevision {
namespace {

constexpr std::array<std::string_view, kClassCount> kClassNames{
    "vertical", "horizontal", "diagonal"};

void expect(std::istream& input, const std::string_view expected) {
  std::string token;
  if (!(input >> token) || token != expected) {
    throw std::runtime_error("portable model marker mismatch: expected " +
                             std::string(expected));
  }
}

std::vector<float> read_values(std::istream& input,
                               const std::string_view section,
                               const std::size_t expected_count) {
  expect(input, section);
  std::size_t declared_count = 0;
  if (!(input >> declared_count) || declared_count != expected_count) {
    throw std::runtime_error("portable model tensor count mismatch: " +
                             std::string(section));
  }
  std::vector<float> values(expected_count);
  for (float& value : values) {
    if (!(input >> value) || !std::isfinite(value)) {
      throw std::runtime_error("portable model contains a non-finite tensor value");
    }
  }
  return values;
}

template <std::size_t Size>
std::array<float, Size> to_array(std::vector<float> values) {
  if (values.size() != Size) {
    throw std::runtime_error("portable model array conversion received the wrong size");
  }
  std::array<float, Size> output{};
  std::copy(values.begin(), values.end(), output.begin());
  return output;
}

std::vector<std::string> split(const std::string& text, const char separator) {
  std::vector<std::string> fields;
  std::stringstream stream(text);
  std::string field;
  while (std::getline(stream, field, separator)) {
    fields.push_back(field);
  }
  return fields;
}

float parse_feature(const std::string& text) {
  std::size_t consumed = 0;
  const float value = std::stof(text, &consumed);
  if (consumed != text.size() || !std::isfinite(value) || value < 0.0F || value > 1.0F) {
    throw std::runtime_error("test-vector feature is not a finite normalized number");
  }
  return value;
}

}  // namespace

Model Model::load(const std::filesystem::path& path) {
  if (!std::filesystem::is_regular_file(path) ||
      std::filesystem::file_size(path) > kMaximumModelBytes) {
    throw std::runtime_error("portable model is missing or exceeds its size limit");
  }
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("portable model could not be opened");
  }
  expect(input, "EDGEVISION_MLP");
  int version = 0;
  if (!(input >> version) || version != 1) {
    throw std::runtime_error("unsupported portable model version");
  }
  expect(input, "dims");
  std::array<std::size_t, 4> dimensions{};
  for (std::size_t& dimension : dimensions) {
    if (!(input >> dimension)) {
      throw std::runtime_error("portable model dimensions are incomplete");
    }
  }
  if (dimensions != std::array<std::size_t, 4>{kInputSize, kHiddenOne, kHiddenTwo,
                                               kClassCount}) {
    throw std::runtime_error("portable model dimensions do not match the runtime");
  }
  expect(input, "classes");
  for (const std::string_view expected_class : kClassNames) {
    expect(input, expected_class);
  }
  expect(input, "temperature");
  Model model;
  if (!(input >> model.temperature_) || !std::isfinite(model.temperature_) ||
      model.temperature_ < 0.05F || model.temperature_ > 10.0F) {
    throw std::runtime_error("portable model calibration temperature is unsafe");
  }
  model.weights_one_ = read_values(input, "weights_one", kInputSize * kHiddenOne);
  model.bias_one_ = to_array<kHiddenOne>(read_values(input, "bias_one", kHiddenOne));
  model.weights_two_ = read_values(input, "weights_two", kHiddenOne * kHiddenTwo);
  model.bias_two_ = to_array<kHiddenTwo>(read_values(input, "bias_two", kHiddenTwo));
  model.weights_three_ = read_values(input, "weights_three", kHiddenTwo * kClassCount);
  model.bias_three_ = to_array<kClassCount>(read_values(input, "bias_three", kClassCount));
  expect(input, "end");
  std::string trailing;
  if (input >> trailing) {
    throw std::runtime_error("portable model contains trailing data");
  }
  return model;
}

std::array<float, kClassCount> Model::predict(
    const std::array<float, kInputSize>& features) const {
  if (std::any_of(features.begin(), features.end(), [](const float value) {
        return !std::isfinite(value) || value < 0.0F || value > 1.0F;
      })) {
    throw std::invalid_argument("features must contain finite normalized values");
  }
  std::array<float, kHiddenOne> hidden_one{};
  for (std::size_t output = 0; output < kHiddenOne; ++output) {
    float sum = bias_one_[output];
    for (std::size_t input = 0; input < kInputSize; ++input) {
      sum += features[input] * weights_one_[input * kHiddenOne + output];
    }
    hidden_one[output] = std::max(sum, 0.0F);
  }
  std::array<float, kHiddenTwo> hidden_two{};
  for (std::size_t output = 0; output < kHiddenTwo; ++output) {
    float sum = bias_two_[output];
    for (std::size_t input = 0; input < kHiddenOne; ++input) {
      sum += hidden_one[input] * weights_two_[input * kHiddenTwo + output];
    }
    hidden_two[output] = std::max(sum, 0.0F);
  }
  std::array<float, kClassCount> logits{};
  for (std::size_t output = 0; output < kClassCount; ++output) {
    float sum = bias_three_[output];
    for (std::size_t input = 0; input < kHiddenTwo; ++input) {
      sum += hidden_two[input] * weights_three_[input * kClassCount + output];
    }
    logits[output] = sum / temperature_;
  }
  const float maximum = *std::max_element(logits.begin(), logits.end());
  std::array<float, kClassCount> probabilities{};
  float denominator = 0.0F;
  for (std::size_t index = 0; index < kClassCount; ++index) {
    probabilities[index] = std::exp(logits[index] - maximum);
    denominator += probabilities[index];
  }
  for (float& probability : probabilities) {
    probability /= denominator;
  }
  return probabilities;
}

std::vector<VectorCase> load_vectors(const std::filesystem::path& path) {
  if (!std::filesystem::is_regular_file(path) || std::filesystem::file_size(path) > 5'000'000) {
    throw std::runtime_error("test-vector file is missing or exceeds its size limit");
  }
  std::ifstream input(path);
  std::string line;
  if (!std::getline(input, line)) {
    throw std::runtime_error("test-vector file has no header");
  }
  const auto header = split(line, ',');
  if (header.size() != kInputSize + 2 || header[0] != "id" || header[1] != "label") {
    throw std::runtime_error("test-vector header does not match the schema");
  }
  std::vector<VectorCase> cases;
  while (std::getline(input, line)) {
    if (cases.size() >= kMaximumVectorRows) {
      throw std::runtime_error("test-vector file exceeds its row limit");
    }
    const auto fields = split(line, ',');
    if (fields.size() != kInputSize + 2 || fields[0].empty() ||
        std::find(kClassNames.begin(), kClassNames.end(), fields[1]) == kClassNames.end()) {
      throw std::runtime_error("test-vector row does not match the schema");
    }
    VectorCase vector_case;
    vector_case.identifier = fields[0];
    vector_case.label = fields[1];
    for (std::size_t index = 0; index < kInputSize; ++index) {
      vector_case.features[index] = parse_feature(fields[index + 2]);
    }
    cases.push_back(std::move(vector_case));
  }
  if (cases.empty()) {
    throw std::runtime_error("test-vector file contains no cases");
  }
  return cases;
}

void write_predictions(const std::filesystem::path& path,
                       const Model& model,
                       const std::vector<VectorCase>& cases) {
  std::ofstream output(path, std::ios::trunc);
  if (!output) {
    throw std::runtime_error("prediction output could not be opened");
  }
  output << "id,vertical,horizontal,diagonal\n" << std::setprecision(9);
  for (const VectorCase& vector_case : cases) {
    const auto probabilities = model.predict(vector_case.features);
    output << vector_case.identifier;
    for (const float probability : probabilities) {
      output << ',' << probability;
    }
    output << '\n';
  }
}

}  // namespace edgevision

