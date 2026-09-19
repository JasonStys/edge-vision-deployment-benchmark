/** File: Tests C++ artifact loading, normalized inference, and malformed-input rejection.
 * Functions: require and main implement a dependency-free test executable. Variables: model,
 * features, and probabilities remain fixed-size; exact lines are indexed in docs/CODE_INDEX.md.
 */
#include "edgevision/model.hpp"

#include <array>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

void require(const bool condition, const std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

}  // namespace

int main(const int argument_count, const char* const arguments[]) {
  try {
    require(argument_count == 2, "test requires the generated model path");
    const auto model = edgevision::Model::load(std::filesystem::path(arguments[1]));
    std::array<float, edgevision::kInputSize> features{};
    const auto probabilities = model.predict(features);
    float sum = 0.0F;
    for (const float probability : probabilities) {
      require(std::isfinite(probability) && probability >= 0.0F && probability <= 1.0F,
              "probability must be normalized");
      sum += probability;
    }
    require(std::abs(sum - 1.0F) < 1e-5F, "probabilities must sum to one");
    features[0] = -1.0F;
    bool rejected = false;
    try {
      static_cast<void>(model.predict(features));
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "negative input must be rejected");
    std::cout << "C++ model tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "C++ model test failure: " << error.what() << '\n';
    return 1;
  }
}

