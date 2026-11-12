#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <optional>
#include <span>
#include <vector>

namespace bruce::lhme::benchmark::statistics {

struct Summary {
  double minimum = 0.0;
  double median = 0.0;
  double maximum = 0.0;
  double mean = 0.0;
  double standard_deviation = 0.0;
};

[[nodiscard]] inline std::optional<Summary> summarize(
    std::span<const double> samples) {
  if (samples.empty()) return std::nullopt;

  std::vector<double> ordered{samples.begin(), samples.end()};
  std::sort(ordered.begin(), ordered.end());
  const auto total = std::accumulate(ordered.begin(), ordered.end(), 0.0);
  const auto mean = total / static_cast<double>(ordered.size());

  double squared_deviation_sum = 0.0;
  for (const auto value : ordered) {
    const auto delta = value - mean;
    squared_deviation_sum += delta * delta;
  }
  const auto divisor = ordered.size() > 1 ? static_cast<double>(ordered.size() - 1)
                                          : 1.0;
  const auto median_index = ordered.size() / 2U;
  const auto median = ordered.size() % 2U == 0U
      ? (ordered[median_index - 1U] + ordered[median_index]) / 2.0
      : ordered[median_index];
  return Summary{.minimum = ordered.front(),
                 .median = median,
                 .maximum = ordered.back(),
                 .mean = mean,
                 .standard_deviation =
                     std::sqrt(squared_deviation_sum / divisor)};
}

}  // namespace bruce::lhme::benchmark::statistics
