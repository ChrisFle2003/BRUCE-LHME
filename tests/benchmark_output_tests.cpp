#include "test.hpp"

#include "benchmark_output.hpp"

#include <array>

TEST(raw_benchmark_csv_uses_stable_quoted_text_fields) {
  using bruce::lhme::benchmark::output::RawCsvRow;
  using bruce::lhme::benchmark::output::format_raw_csv_row;

  const auto row = format_raw_csv_row({.benchmark = "benchmark",
                                       .case_id = "case,with,comma",
                                       .metric = "elapsed",
                                       .iteration = 7,
                                       .unit = "us/request",
                                       .value = 1.25,
                                       .validation = "confirmed",
                                       .notes = "quote: \"ok\""});
  CHECK(row == "\"benchmark\",\"case,with,comma\",\"elapsed\",7,\"us/request\",1.250000000,"
               "\"confirmed\",\"quote: \"\"ok\"\"\"\n");
}

TEST(raw_benchmark_csv_header_has_a_fixed_schema) {
  using bruce::lhme::benchmark::output::raw_csv_header;

  CHECK(raw_csv_header() ==
        "benchmark,case_id,metric,iteration,unit,value,validation,notes\n");
}

TEST(raw_benchmark_summary_marks_single_sample_deviation_unavailable) {
  using bruce::lhme::benchmark::output::format_summary;

  const std::array<double, 1> sample{1.25};
  const auto summary = format_summary(sample, "ms/op");

  CHECK(summary.has_value());
  if (summary.has_value()) {
    CHECK(*summary == "median=1.250 ms/op; mean=1.250; min=1.250; max=1.250; "
                      "stddev=unavailable (n=1)");
  }
}

TEST(raw_benchmark_summary_uses_sample_standard_deviation) {
  using bruce::lhme::benchmark::output::format_summary;

  const std::array<double, 3> samples{1.0, 2.0, 3.0};
  const auto summary = format_summary(samples, "ms/op");

  CHECK(summary.has_value());
  if (summary.has_value()) {
    CHECK(*summary == "median=2.000 ms/op; mean=2.000; min=1.000; max=3.000; "
                      "stddev=1.000");
  }
}
