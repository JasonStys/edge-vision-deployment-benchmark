/** File: Provides the C++ command-line adapter for portable inference and CSV output.
 * Functions: main validates arguments, loads artifacts, and reports exceptions. Variables: model,
 * cases, and paths are limited to one invocation; exact lines are indexed in docs/CODE_INDEX.md.
 */
#include "edgevision/model.hpp"

#include <exception>
#include <filesystem>
#include <iostream>

int main(const int argument_count, const char* const arguments[]) {
  if (argument_count != 4) {
    std::cerr << "usage: edgevision_cpp MODEL.evm VECTORS.csv OUTPUT.csv\n";
    return 2;
  }
  try {
    const auto model = edgevision::Model::load(std::filesystem::path(arguments[1]));
    const auto cases = edgevision::load_vectors(std::filesystem::path(arguments[2]));
    edgevision::write_predictions(std::filesystem::path(arguments[3]), model, cases);
    std::cout << "wrote " << cases.size() << " prediction rows\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "edgevision_cpp: " << error.what() << '\n';
    return 1;
  }
}

