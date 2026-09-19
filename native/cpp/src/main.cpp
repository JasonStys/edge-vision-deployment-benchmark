/** File: Provides the C++ repository-root adapter for portable inference and CSV output.
 * Functions: main loads fixed repository artifacts and reports exceptions. Variables: model,
 * cases, and allow-listed repository paths are limited to one invocation; exact lines are indexed.
 */
#include "edgevision/model.hpp"

#include <exception>
#include <filesystem>
#include <iostream>

namespace {

const std::filesystem::path kModelPath{"artifacts/model/compact-mlp.evm"};
const std::filesystem::path kVectorPath{"artifacts/test-vectors.csv"};
const std::filesystem::path kOutputPath{".runtime/cpp.csv"};

}  // namespace

int main() {
  try {
    const auto model = edgevision::Model::load(kModelPath);
    const auto cases = edgevision::load_vectors(kVectorPath);
    edgevision::write_predictions(kOutputPath, model, cases);
    std::cout << "wrote " << cases.size() << " prediction rows\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "edgevision_cpp: " << error.what() << '\n';
    return 1;
  }
}
