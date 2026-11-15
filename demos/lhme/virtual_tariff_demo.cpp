#include "demo_report.hpp"

#include "bruce/lhme/formula_override_snapshot.hpp"
#include "bruce/lhme/formula_override_store.hpp"
#include "bruce/lhme/kernel_sdk.hpp"
#include "bruce/lhme/trace.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace {

using namespace bruce::lhme;

class VirtualTariffKernel final : public TechniqueKernel {
 public:
  [[nodiscard]] TechniqueIdentity identity() const override {
    return {.id = "example.virtual_tariff", .version = 1};
  }

  [[nodiscard]] EvaluationResult evaluate(
      const TechniqueRequest& request) const override {
    if (cancellation_requested(request)) {
      return cancelled_technique_result(request.mode,
                                        "virtual tariff request cancelled");
    }
    if (request.identity != identity() || request.parameters != kParameters ||
        request.address.index < 0) {
      return invalid_technique_request_result(
          request.mode, "expected a virtual tariff request for the alpine region");
    }

    const auto cents = tariff_cents(request.address.index);
    const auto bits = bit_length(cents);
    switch (request.mode) {
      case EvaluationMode::Exact:
        if (bits > request.budget.max_bit_length ||
            value_bytes(bits) > request.budget.max_temporary_bytes ||
            request.budget.max_operations == 0) {
          return budget_exceeded(request.mode, bits);
        }
        return confirmed(request.mode, BigInt{cents});
      case EvaluationMode::Metric: {
        EvaluationResult result;
        result.status = Status::Confirmed;
        result.mode = request.mode;
        result.metric_bit_length = bits;
        result.trace = make_trace({});
        return result;
      }
      case EvaluationMode::Modular:
        if (!request.modulus.has_value() || *request.modulus <= 0) {
          return invalid_technique_request_result(request.mode,
                                                  "virtual tariff modulus is invalid");
        }
        return confirmed(request.mode, BigInt{cents} % *request.modulus);
      case EvaluationMode::Symbolic: {
        EvaluationResult result;
        result.status = Status::Confirmed;
        result.mode = request.mode;
        result.symbolic_expression =
            "tariff_cents(alpine_summer, zone=" +
            std::to_string(request.address.index) + ")";
        result.trace = make_trace({});
        return result;
      }
    }
    return invalid_technique_request_result(request.mode,
                                            "virtual tariff mode is invalid");
  }

  static constexpr std::string_view kParameters =
      "region=alpine;season=summer;currency=EUR";

 private:
  [[nodiscard]] static std::uint64_t tariff_cents(std::int64_t address) {
    // A deterministic 7.99–19.98 EUR virtual price field. The address is a
    // logical zone identifier, not a row in a materialized price table.
    const auto index = static_cast<std::uint64_t>(address);
    return 799U + ((index * 37U + 13U) % 1200U);
  }

  [[nodiscard]] static std::uint64_t bit_length(std::uint64_t value) {
    std::uint64_t bits = 0;
    while (value > 0) {
      ++bits;
      value >>= 1U;
    }
    return bits;
  }

  [[nodiscard]] static std::uint64_t value_bytes(std::uint64_t bits) {
    return bits / 8U + (bits % 8U == 0 ? 0U : 1U);
  }

  [[nodiscard]] static EvaluationResult confirmed(EvaluationMode mode,
                                                   BigInt value) {
    EvaluationResult result;
    result.status = Status::Confirmed;
    result.mode = mode;
    result.integer = std::move(value);
    result.trace = make_trace({});
    return result;
  }

  [[nodiscard]] static EvaluationResult budget_exceeded(EvaluationMode mode,
                                                         std::uint64_t bits) {
    EvaluationResult result;
    result.status = Status::BudgetExceeded;
    result.mode = mode;
    result.metric_bit_length = bits;
    result.trace = make_trace({{.scope = GateScope::Publish,
                                .status = result.status,
                                .reason = "virtual tariff exceeded request budget"}});
    return result;
  }
};

std::string source_name(FormulaValueSource source) {
  return source == FormulaValueSource::Override ? "Negotiated override"
                                                 : "Virtual tariff";
}

void expect_candidate(demo::Report& report,
                      const FormulaStreamingRangeResult& result,
                      std::size_t position, std::int64_t address,
                      const BigInt& cents, FormulaValueSource source) {
  report.expect_true("candidate position exists",
                     position < result.retained_candidates.size());
  if (position >= result.retained_candidates.size()) return;
  const auto& candidate = result.retained_candidates[position];
  report.expect_equal("candidate address", address, candidate.address.index);
  report.expect_equal("candidate source", source, candidate.value.source);
  report.expect_equal("candidate tariff cents", cents,
                      candidate.value.result.integer.value_or(BigInt{0}));
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc != 3 || std::string_view{argv[1]} != "--snapshot") {
    std::cerr << "usage: lhme_virtual_tariff_demo --snapshot FILE\n";
    return 2;
  }

  const std::filesystem::path snapshot_path{argv[2]};
  demo::Report report;
  report.heading("LHME Virtual Tariff Field: Local Pricing Corrections");

  auto engine = Engine::create({.cache_max_entries = 16, .cache_max_bytes = 4096});
  const auto kernel = std::make_shared<VirtualTariffKernel>();
  report.expect_true("register virtual tariff kernel", engine.register_technique(kernel));
  const TechniqueBinding binding{
      .identity = kernel->identity(),
      .canonical_parameters = std::string{VirtualTariffKernel::kParameters}};
  FormulaOverrideStore store{
      engine,
      {.max_override_count = 8,
       .max_override_storage_bytes = 4096,
       .max_override_value_bytes = 8}};

  report.row("logical tariff zones", "0..10000000");
  report.row("virtual tariff rule", "7.99–19.98 EUR, deterministic by zone");
  report.expect_equal("set negotiated price for zone 10", Status::Confirmed,
                      store.set_override(binding, {.index = 10}, BigInt{1299}));
  report.expect_equal("set negotiated price for zone 777", Status::Confirmed,
                      store.set_override(binding, {.index = 777}, BigInt{1500}));
  report.expect_equal("set negotiated price for zone 999", Status::Confirmed,
                      store.set_override(binding, {.index = 999}, BigInt{999}));
  report.row("stored local corrections",
             std::to_string(store.inspect().override_count));

  const auto metric = store.evaluate({.binding = binding,
                                      .address = {.index = 10'000'000},
                                      .mode = EvaluationMode::Metric,
                                      .budget = Budget::unlimited()});
  report.expect_equal("metric status", Status::Confirmed, metric.result.status);
  report.expect_equal("metric source", FormulaValueSource::Computed, metric.source);
  report.expect_true("metric avoids exact price materialization",
                     !metric.result.integer.has_value());
  report.expect_equal("metric price bit length", std::uint64_t(11),
                      metric.result.metric_bit_length.value_or(0));
  report.row("metric zone", "10000000");
  report.row("metric tariff bit length",
             std::to_string(metric.result.metric_bit_length.value_or(0)));

  const auto negotiated = store.evaluate({.binding = binding,
                                          .address = {.index = 10},
                                          .mode = EvaluationMode::Exact,
                                          .budget = Budget::unlimited()});
  const auto generated = store.evaluate({.binding = binding,
                                         .address = {.index = 11},
                                         .mode = EvaluationMode::Exact,
                                         .budget = Budget::unlimited()});
  report.expect_equal("negotiated source", FormulaValueSource::Override,
                      negotiated.source);
  report.expect_equal("negotiated tariff cents", BigInt{1299},
                      negotiated.result.integer.value_or(BigInt{0}));
  report.expect_equal("generated source", FormulaValueSource::Computed,
                      generated.source);
  report.expect_equal("generated tariff cents", BigInt{1219},
                      generated.result.integer.value_or(BigInt{0}));
  report.row("zone 10", "Negotiated override 12.99 EUR");
  report.row("zone 11", "Virtual tariff 12.19 EUR");

  Budget stream_budget = Budget::unlimited();
  stream_budget.max_candidate_count = 10'000;
  const StreamingOptions options{
      .worker_count = 1,
      .tile_size = 64,
      .retention_mode = RetentionMode::TopK,
      .max_retained_candidates = 4,
      .retention_memory_bytes = 4096,
      .filter = {.mode = FilterMode::ModularEquals,
                 .modulus = 100,
                 .expected_residue = 99}};
  const auto before_stream = store.inspect().computed_exact_evaluation_count;
  const auto stream = store.evaluate_streaming_range(
      {.binding = binding, .budget = stream_budget}, {0, 9'999}, options);
  const auto after_stream = store.inspect().computed_exact_evaluation_count;

  report.expect_equal("stream status", Status::Confirmed, stream.status);
  report.expect_equal("stream checked", std::uint64_t(10'000),
                      stream.summary.checked_count);
  report.expect_equal("stream filtered", std::uint64_t(9'898),
                      stream.filtered_out_count);
  report.expect_equal("stream exact", std::uint64_t(102),
                      stream.exact_evaluation_count);
  report.expect_equal("stream retained", std::size_t(4),
                      stream.retained_candidates.size());
  report.expect_equal("stream retention mode", RetentionMode::TopK,
                      stream.effective_retention_mode);
  report.expect_equal("computed exact delta", std::uint64_t(100),
                      after_stream - before_stream);
  expect_candidate(report, stream, 0, 10, BigInt{1299}, FormulaValueSource::Override);
  expect_candidate(report, stream, 1, 51, BigInt{1499}, FormulaValueSource::Computed);
  expect_candidate(report, stream, 2, 151, BigInt{1599}, FormulaValueSource::Computed);
  expect_candidate(report, stream, 3, 251, BigInt{1699}, FormulaValueSource::Computed);

  report.row("stream checked", std::to_string(stream.summary.checked_count));
  report.row("prices ending in .99", std::to_string(stream.exact_evaluation_count));
  report.row("retained price offers",
             std::to_string(stream.retained_candidates.size()));
  report.row("computed exact prices", std::to_string(after_stream - before_stream));
  for (const auto& candidate : stream.retained_candidates) {
    const auto cents = candidate.value.result.integer.value_or(BigInt{0});
    report.row("zone " + std::to_string(candidate.address.index),
               source_name(candidate.value.source) + " " +
                   cents.convert_to<std::string>() + " cents");
  }

  const auto snapshot_status = save_formula_override_snapshot(store, snapshot_path);
  std::error_code snapshot_error;
  const auto snapshot_bytes = snapshot_status == FormulaOverrideSnapshotStatus::Confirmed
                                  ? std::filesystem::file_size(snapshot_path, snapshot_error)
                                  : 0U;
  report.expect_equal("snapshot save", FormulaOverrideSnapshotStatus::Confirmed,
                      snapshot_status);
  report.expect_true("snapshot size available", !snapshot_error && snapshot_bytes > 0);
  report.row("snapshot corrections", std::to_string(store.inspect().override_count));
  report.row("snapshot bytes", std::to_string(snapshot_bytes));

  auto restored_engine = Engine::create({});
  const auto restored_kernel = std::make_shared<VirtualTariffKernel>();
  report.expect_true("register restored virtual tariff kernel",
                     restored_engine.register_technique(restored_kernel));
  FormulaOverrideStore restored{
      restored_engine,
      {.max_override_count = 8,
       .max_override_storage_bytes = 4096,
       .max_override_value_bytes = 8}};
  const auto reload_status = load_formula_override_snapshot(restored, snapshot_path);
  const auto reloaded = restored.evaluate({.binding = binding,
                                           .address = {.index = 10},
                                           .mode = EvaluationMode::Exact,
                                           .budget = Budget::unlimited()});
  report.expect_equal("snapshot reload", FormulaOverrideSnapshotStatus::Confirmed,
                      reload_status);
  report.expect_equal("reloaded correction count", std::size_t(3),
                      restored.inspect().override_count);
  report.expect_equal("reloaded source", FormulaValueSource::Override,
                      reloaded.source);
  report.expect_equal("reloaded tariff cents", BigInt{1299},
                      reloaded.result.integer.value_or(BigInt{0}));
  report.row("snapshot reload", "Negotiated override 12.99 EUR");

  const auto oversized = store.evaluate_streaming_range(
      {.binding = binding, .budget = stream_budget}, {0, 10'000}, options);
  report.expect_equal("oversized range status", Status::BudgetExceeded,
                      oversized.status);
  report.expect_equal("oversized range exact work", std::uint64_t(0),
                      oversized.exact_evaluation_count);
  report.row("10001-zone query status", demo::status_name(oversized.status));

  return report.finish();
}
