#include "test.hpp"

#include "benchmark_statistics.hpp"

#include <array>

TEST(benchmark_statistics_rejects_empty_samples) {
  const std::array<double, 0> samples{};
  CHECK(!bruce::lhme::benchmark::statistics::summarize(samples).has_value());
}

TEST(benchmark_statistics_summarizes_odd_sample_sets_deterministically) {
  const std::array<double, 5> samples{5.0, 1.0, 3.0, 2.0, 4.0};
  const auto summary = bruce::lhme::benchmark::statistics::summarize(samples);

  CHECK(summary.has_value());
  CHECK(summary->minimum == 1.0);
  CHECK(summary->median == 3.0);
  CHECK(summary->maximum == 5.0);
  CHECK(summary->mean == 3.0);
  CHECK(summary->standard_deviation > 1.58);
  CHECK(summary->standard_deviation < 1.59);
}
