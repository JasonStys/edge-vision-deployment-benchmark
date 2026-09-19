/** File: Declares bounded portable-model parsing, inference, vector loading, and prediction output.
 * Functions: Model::load, Model::predict, load_vectors, and write_predictions form the public API.
 * Variables: fixed dimension constants and tensor vectors mirror the portable artifact; exact lines
 * are indexed in docs/CODE_INDEX.md.
 */
#ifndef EDGEVISION_MODEL_HPP
#define EDGEVISION_MODEL_HPP

#include <array>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace edgevision {

inline constexpr std::size_t kInputSize = 64;
inline constexpr std::size_t kHiddenOne = 24;
inline constexpr std::size_t kHiddenTwo = 12;
inline constexpr std::size_t kClassCount = 3;
inline constexpr std::size_t kMaximumModelBytes = 1'000'000;
inline constexpr std::size_t kMaximumVectorRows = 4'096;

struct VectorCase {
  std::string identifier;
  std::string label;
  std::array<float, kInputSize> features{};
};

class Model {
 public:
  /** Load and validate one inspectable EDGEVISION_MLP version 1 artifact. */
  static Model load(const std::filesystem::path& path);

  /** Infer calibrated probabilities for one normalized 8x8 image. */
  [[nodiscard]] std::array<float, kClassCount> predict(
      const std::array<float, kInputSize>& features) const;

 private:
  std::vector<float> weights_one_;
  std::array<float, kHiddenOne> bias_one_{};
  std::vector<float> weights_two_;
  std::array<float, kHiddenTwo> bias_two_{};
  std::vector<float> weights_three_;
  std::array<float, kClassCount> bias_three_{};
  float temperature_{1.0F};
};

/** Load a bounded generated CSV without accepting quoting or schema drift. */
[[nodiscard]] std::vector<VectorCase> load_vectors(const std::filesystem::path& path);

/** Write stable probability CSV output for cross-runtime comparison. */
void write_predictions(const std::filesystem::path& path,
                       const Model& model,
                       const std::vector<VectorCase>& cases);

}  // namespace edgevision

#endif

