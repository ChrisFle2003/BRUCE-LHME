#include "demo_report.hpp"

#include "bruce/lhme/formula_override_snapshot.hpp"
#include "bruce/lhme/formula_override_store.hpp"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

namespace {

using namespace bruce::lhme;

std::string source_name(FormulaValueSource source) {
  return source == FormulaValueSource::Override ? "Override" : "Computed";
}

void expect_candidate(demo::Report& report,
                      const FormulaStreamingRangeResult& result,
                      std::size_t position, std::int64_t address,
                      const BigInt& value, FormulaValueSource source) {
  report.expect_true("candidate position exists",
                     position < result.retained_candidates.size());
  if (position >= result.retained_candidates.size()) return;
  const auto& candidate = result.retained_candidates[position];
  report.expect_equal("candidate address", address, candidate.address.index);
  report.expect_equal("candidate source", source, candidate.value.source);
  report.expect_equal("candidate exact value", value,
                      candidate.value.result.integer.value_or(BigInt{0}));
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc != 3 || std::string_view{argv[1]} != "--snapshot") {
    std::cerr << "usage: lhme_flagship_demo --snapshot FILE\n";
    return 2;
  }

  const std::filesystem::path snapshot_path{argv[2]};
  demo::Report report;
  report.heading("LHME Flagship: Virtual Rule Values with Sparse Overrides");

  auto engine = Engine::create({.worker_count = 1, .tile_size = 64});
  const auto formula = engine.register_formula(make_power_offset_spec(2, 100, 1));
  const FormulaBinding binding{formula};
  FormulaOverrideStore store{engine};

  report.row("rule", "PowerOffset(base=2, scale=100, version=1)");
  report.row("logical address domain", "0..1000000");
  report.expect_equal("set override 10", Status::Confirmed,
                      store.set_override(binding, {.index = 10}, BigInt{-40}));
  report.expect_equal("set override 777", Status::Confirmed,
                      store.set_override(binding, {.index = 777}, BigInt{17}));
  report.expect_equal("set override 999", Status::Confirmed,
                      store.set_override(binding, {.index = 999}, BigInt{0}));
  report.row("stored overrides", std::to_string(store.inspect().override_count));

  const auto metric = store.evaluate({.binding = binding,
                                      .address = {.index = 1'000'000},
                                      .mode = EvaluationMode::Metric,
                                      .budget = Budget::unlimited()});
  report.expect_equal("metric status", Status::Confirmed, metric.result.status);
  report.expect_equal("metric source", FormulaValueSource::Computed, metric.source);
  report.expect_true("metric avoids exact materialization",
                     !metric.result.integer.has_value());
  report.expect_equal("metric bit length", std::uint64_t(1'000'007),
                      metric.result.metric_bit_length.value_or(0));
  report.row("metric address", "1000000");
  report.row("metric source", source_name(metric.source));
  report.row("metric bit length",
             std::to_string(metric.result.metric_bit_length.value_or(0)));

  const auto override_point = store.evaluate({.binding = binding,
                                              .address = {.index = 10},
                                              .mode = EvaluationMode::Exact,
                                              .budget = Budget::unlimited()});
  const auto computed_point = store.evaluate({.binding = binding,
                                              .address = {.index = 11},
                                              .mode = EvaluationMode::Exact,
                                              .budget = Budget::unlimited()});
  report.expect_equal("override point source", FormulaValueSource::Override,
                      override_point.source);
  report.expect_equal("override point value", BigInt{-40},
                      override_point.result.integer.value_or(BigInt{0}));
  report.expect_equal("computed point source", FormulaValueSource::Computed,
                      computed_point.source);
  report.expect_equal("computed point value", BigInt{204700},
                      computed_point.result.integer.value_or(BigInt{0}));
  report.row("point 10", source_name(override_point.source) + " -40");
  report.row("point 11", source_name(computed_point.source) + " 204700");

  Budget stream_budget = Budget::unlimited();
  stream_budget.max_candidate_count = 1024;
  const StreamingOptions options{
      .worker_count = 1,
      .tile_size = 64,
      .retention_mode = RetentionMode::TopK,
      .max_retained_candidates = 4,
      .retention_memory_bytes = 4096,
      .filter = {.mode = FilterMode::ModularEquals,
                 .modulus = 17,
                 .expected_residue = 11}};
  const auto before_stream = store.inspect().computed_exact_evaluation_count;
  const auto stream = store.evaluate_streaming_range(
      {.binding = binding, .budget = stream_budget}, {0, 1023}, options);
  const auto after_stream = store.inspect().computed_exact_evaluation_count;

  report.expect_equal("stream status", Status::Confirmed, stream.status);
  report.expect_equal("stream checked", std::uint64_t(1024),
                      stream.summary.checked_count);
  report.expect_equal("stream filtered", std::uint64_t(896),
                      stream.filtered_out_count);
  report.expect_equal("stream exact", std::uint64_t(128),
                      stream.exact_evaluation_count);
  report.expect_equal("stream retained", std::size_t(4),
                      stream.retained_candidates.size());
  report.expect_equal("stream retention mode", RetentionMode::TopK,
                      stream.effective_retention_mode);
  report.expect_equal("computed exact delta", std::uint64_t(127),
                      after_stream - before_stream);
  expect_candidate(report, stream, 0, 2, BigInt{300}, FormulaValueSource::Computed);
  expect_candidate(report, stream, 1, 10, BigInt{-40}, FormulaValueSource::Override);
  expect_candidate(report, stream, 2, 18, BigInt{26214300},
                   FormulaValueSource::Computed);
  expect_candidate(report, stream, 3, 26, BigInt{6710886300LL},
                   FormulaValueSource::Computed);

  report.row("stream checked", std::to_string(stream.summary.checked_count));
  report.row("stream filtered", std::to_string(stream.filtered_out_count));
  report.row("stream exact", std::to_string(stream.exact_evaluation_count));
  report.row("stream retained", std::to_string(stream.retained_candidates.size()));
  report.row("computed exact delta", std::to_string(after_stream - before_stream));
  for (const auto& candidate : stream.retained_candidates) {
    const auto value = candidate.value.result.integer.value_or(BigInt{0});
    report.row("candidate " + std::to_string(candidate.address.index),
               source_name(candidate.value.source) + " " +
                   value.convert_to<std::string>());
  }

  const auto snapshot_status = save_formula_override_snapshot(store, snapshot_path);
  std::error_code snapshot_error;
  const auto snapshot_bytes = snapshot_status == FormulaOverrideSnapshotStatus::Confirmed
                                  ? std::filesystem::file_size(snapshot_path, snapshot_error)
                                  : 0U;
  report.expect_equal("snapshot save", FormulaOverrideSnapshotStatus::Confirmed,
                      snapshot_status);
  report.expect_true("snapshot size available", !snapshot_error && snapshot_bytes > 0);
  report.row("snapshot overrides", std::to_string(store.inspect().override_count));
  report.row("snapshot bytes", std::to_string(snapshot_bytes));

  auto restored_engine = Engine::create({});
  const auto restored_formula =
      restored_engine.register_formula(make_power_offset_spec(2, 100, 1));
  const FormulaBinding restored_binding{restored_formula};
  FormulaOverrideStore restored{restored_engine};
  const auto reload_status = load_formula_override_snapshot(restored, snapshot_path);
  const auto reloaded_override = restored.evaluate({.binding = restored_binding,
                                                    .address = {.index = 10},
                                                    .mode = EvaluationMode::Exact,
                                                    .budget = Budget::unlimited()});
  const auto reloaded_computed = restored.evaluate({.binding = restored_binding,
                                                    .address = {.index = 11},
                                                    .mode = EvaluationMode::Exact,
                                                    .budget = Budget::unlimited()});
  report.expect_equal("snapshot reload", FormulaOverrideSnapshotStatus::Confirmed,
                      reload_status);
  report.expect_equal("reloaded override count", std::size_t(3),
                      restored.inspect().override_count);
  report.expect_equal("reloaded override source", FormulaValueSource::Override,
                      reloaded_override.source);
  report.expect_equal("reloaded override value", BigInt{-40},
                      reloaded_override.result.integer.value_or(BigInt{0}));
  report.expect_equal("reloaded computed source", FormulaValueSource::Computed,
                      reloaded_computed.source);
  report.row("snapshot reload", source_name(reloaded_override.source) + " -40");

  const auto oversized = store.evaluate_streaming_range(
      {.binding = binding, .budget = stream_budget}, {0, 1024}, options);
  report.expect_equal("oversized range status", Status::BudgetExceeded,
                      oversized.status);
  report.expect_equal("oversized range exact work", std::uint64_t(0),
                      oversized.exact_evaluation_count);
  report.row("1025-address range status", demo::status_name(oversized.status));

  return report.finish();
}
