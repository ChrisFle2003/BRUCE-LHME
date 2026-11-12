#include "demo_report.hpp"

#include "benchmark_output.hpp"
#include "bruce/lhme/formula_override_snapshot.hpp"
#include "bruce/lhme/formula_override_store.hpp"
#include "table_storage.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using bruce::lhme::benchmark::table::HashValueTable;

constexpr std::uint64_t kVirtualAddressCount = 1'000'000;

struct Tier {
  std::string_view name;
  std::uint64_t override_count;
};

struct Measurement {
  bool success = false;
  std::string failure_reason;
  std::uint64_t override_count = 0;
  std::uint64_t snapshot_bytes = 0;
  double insert_milliseconds = 0.0;
  std::uint64_t computed_exact_evaluations = 0;
};

[[nodiscard]] std::filesystem::path snapshot_path(std::uint64_t override_count,
                                                   std::size_t iteration) {
  return std::filesystem::temp_directory_path() /
         ("bruce-hybrid-" + std::to_string(override_count) + "-" +
          std::to_string(iteration) + "-" +
          std::to_string(Clock::now().time_since_epoch().count()) + ".snapshot");
}

[[nodiscard]] Measurement measure_tier(std::uint64_t override_count,
                                       std::size_t iteration) {
  using namespace bruce::lhme;
  auto engine = Engine::create({});
  const auto binding = engine.register_formula(make_power_offset_spec(2, 100, 1));
  FormulaOverrideStore store{engine};

  const auto started = Clock::now();
  for (std::uint64_t ordinal = 0; ordinal < override_count; ++ordinal) {
    const FormulaAddress address{.index = static_cast<std::int64_t>(
        ordinal * kVirtualAddressCount / override_count)};
    if (store.set_override(binding, address,
                           BigInt{static_cast<std::int64_t>(ordinal % 97U + 1U)}) !=
        Status::Confirmed) {
      return {.failure_reason = "override insertion failed"};
    }
  }
  const auto inserted = std::chrono::duration<double, std::milli>(Clock::now() - started);

  if (override_count != 0) {
    const auto override = store.evaluate({.binding = binding,
                                          .address = {.index = 0},
                                          .mode = EvaluationMode::Exact,
                                          .budget = Budget::unlimited()});
    if (override.source != FormulaValueSource::Override ||
        override.result.status != Status::Confirmed) {
      return {.failure_reason = "override lookup failed"};
    }
  }
  if (override_count < kVirtualAddressCount) {
    const auto formula = store.evaluate(
        {.binding = binding,
         .address = {.index = static_cast<std::int64_t>(kVirtualAddressCount - 1)},
         .mode = EvaluationMode::Metric,
         .budget = Budget::unlimited()});
    if (formula.source != FormulaValueSource::Computed ||
        formula.result.status != Status::Confirmed) {
      return {.failure_reason = "formula lookup failed"};
    }
  }

  const auto path = snapshot_path(override_count, iteration);
  std::error_code error;
  if (std::filesystem::exists(path, error) || error) {
    return {.failure_reason = "unique snapshot path is unavailable"};
  }
  if (save_formula_override_snapshot(store, path) !=
      FormulaOverrideSnapshotStatus::Confirmed) {
    return {.failure_reason = "snapshot write failed"};
  }
  const auto snapshot_bytes = std::filesystem::file_size(path, error);
  std::filesystem::remove(path, error);
  if (error) return {.failure_reason = "snapshot cleanup failed"};

  const auto statistics = store.inspect();
  return {.success = statistics.override_count == override_count,
          .failure_reason = statistics.override_count == override_count
                                ? ""
                                : "stored override count differs from requested count",
          .override_count = statistics.override_count,
          .snapshot_bytes = snapshot_bytes,
          .insert_milliseconds = inserted.count(),
          .computed_exact_evaluations = statistics.computed_exact_evaluation_count};
}

[[nodiscard]] bool parse_positive(std::string_view text, std::size_t& value) {
  try {
    std::size_t consumed = 0U;
    const auto parsed = std::stoull(std::string{text}, &consumed);
    if (consumed != text.size() || parsed == 0U) return false;
    value = static_cast<std::size_t>(parsed);
    return true;
  } catch (...) {
    return false;
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  using namespace bruce::lhme;

  bool include_full_override_tier = false;
  std::size_t iterations = 3U;
  std::optional<std::filesystem::path> raw_path;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument{argv[index]};
    if (argument == "--full-overrides" && !include_full_override_tier) {
      include_full_override_tier = true;
    } else if (argument == "--quick") {
      iterations = 1U;
    } else if (argument == "--iterations" && index + 1 < argc) {
      if (!parse_positive(argv[++index], iterations)) {
        std::cerr << "--iterations requires a positive integer\n";
        return 2;
      }
    } else if (argument == "--csv" && index + 1 < argc && !raw_path.has_value()) {
      raw_path = std::filesystem::path{argv[++index]};
    } else {
      std::cerr << "usage: lhme_hybrid_storage_benchmark [--quick] [--iterations N] "
                   "[--full-overrides] [--csv FILE]\n";
      return 2;
    }
  }
  std::optional<benchmark::output::RawCsvWriter> raw_output;
  if (raw_path.has_value()) {
    raw_output.emplace(*raw_path);
    if (!raw_output->good()) {
      std::cerr << "--csv must name a new writable output file\n";
      return 2;
    }
  }

  demo::Report report;
  report.heading("FormulaRecipe + Override Storage Comparison");
  report.row("virtual address domain", std::to_string(kVirtualAddressCount));
  report.row("model", "FormulaRecipe base plus sparse BigInt overrides; no KEVIN grid");

  const auto hash = HashValueTable::make(kVirtualAddressCount);
  report.row("one-million hash table container bytes",
             std::to_string(hash.metrics().container_bytes));

  std::vector<Tier> tiers{{"0 percent", 0},
                          {"0.1 percent", 1'000},
                          {"1 percent", 10'000},
                          {"10 percent", 100'000}};
  if (include_full_override_tier) tiers.push_back({"100 percent", kVirtualAddressCount});

  std::optional<std::uint64_t> previous_snapshot_bytes;
  for (const auto tier : tiers) {
    const auto label = "FormulaWithOverrides " + std::string{tier.name};
    const auto warmup = measure_tier(tier.override_count, 0U);
    report.expect_true(label + " warm-up completed", warmup.success);

    std::vector<double> insertion_samples;
    std::vector<double> snapshot_samples;
    insertion_samples.reserve(iterations);
    snapshot_samples.reserve(iterations);
    std::optional<Measurement> representative;
    for (std::size_t iteration = 0U; iteration < iterations; ++iteration) {
      const auto measurement = measure_tier(tier.override_count, iteration + 1U);
      report.expect_true(label + " benchmark completed", measurement.success);
      if (!measurement.success) {
        report.row(label + " failure reason", measurement.failure_reason);
        continue;
      }
      report.expect_equal(label + " sample override count", tier.override_count,
                          measurement.override_count);
      report.expect_equal(label + " sample computed exact evaluations", std::uint64_t(0),
                          measurement.computed_exact_evaluations);
      if (!representative.has_value()) representative = measurement;
      report.expect_equal(label + " deterministic snapshot bytes",
                          representative->snapshot_bytes, measurement.snapshot_bytes);
      insertion_samples.push_back(measurement.insert_milliseconds);
      snapshot_samples.push_back(static_cast<double>(measurement.snapshot_bytes));
      if (raw_output.has_value() &&
          (!raw_output->write({.benchmark = "lhme_hybrid_storage_benchmark",
                               .case_id = "formula-overrides-" +
                                   std::to_string(tier.override_count),
                               .metric = "override_insert_elapsed",
                               .iteration = iteration + 1U,
                               .unit = "ms/insert-set",
                               .value = measurement.insert_milliseconds,
                               .validation = "confirmed",
                               .notes = "store construction and snapshot I/O excluded"}) ||
           !raw_output->write({.benchmark = "lhme_hybrid_storage_benchmark",
                               .case_id = "formula-overrides-" +
                                   std::to_string(tier.override_count),
                               .metric = "snapshot_size",
                               .iteration = iteration + 1U,
                               .unit = "bytes",
                               .value = static_cast<double>(measurement.snapshot_bytes),
                               .validation = "confirmed",
                               .notes = "ASCII snapshot bytes; not RAM"}))) {
        report.expect_true(label + " raw CSV write", false);
      }
    }
    const auto insertion_summary = benchmark::output::format_summary(
        insertion_samples, "ms/insert-set");
    const auto snapshot_summary = benchmark::output::format_summary(snapshot_samples, "bytes");
    report.expect_true(label + " insertion summary available", insertion_summary.has_value());
    report.expect_true(label + " snapshot summary available", snapshot_summary.has_value());
    if (!representative.has_value() || !insertion_summary.has_value() ||
        !snapshot_summary.has_value()) {
      continue;
    }
    report.row(label + " overrides", std::to_string(representative->override_count));
    report.row(label + " insert statistics", *insertion_summary);
    report.row(label + " snapshot size statistics", *snapshot_summary);
    report.expect_true(label + " snapshot exists", representative->snapshot_bytes > 0);
    if (previous_snapshot_bytes.has_value()) {
      report.expect_true(label + " snapshot grows with exceptions",
                         representative->snapshot_bytes > *previous_snapshot_bytes);
    }
    previous_snapshot_bytes = representative->snapshot_bytes;
  }

  if (raw_output.has_value()) {
    report.expect_true("hybrid storage raw CSV write", raw_output->good());
  }

  return report.finish();
}
