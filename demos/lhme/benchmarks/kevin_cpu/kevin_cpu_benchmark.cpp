#include "demo_report.hpp"

#include "benchmark_output.hpp"
#include "bruce/lhme/domain_projection.hpp"
#include "bruce/lhme/kevin_filter.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <future>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::uint32_t kTileCount = 64;
constexpr std::array<std::uint32_t, 4> kWorkerCounts{1, 2, 4, 8};

struct FilterMeasurement {
  bruce::lhme::TileReductionSummary summary;
  double elapsed_microseconds = 0.0;
};

class EngineFormulaEvaluator final : public bruce::lhme::FormulaEvaluator {
 public:
  EngineFormulaEvaluator(bruce::lhme::Engine& engine, bruce::lhme::FormulaSpec spec)
      : engine_(engine), spec_(std::move(spec)) {}

  bruce::lhme::EvaluationResult evaluate(
      bruce::lhme::FormulaAddress address, bruce::lhme::EvaluationMode mode,
      bruce::lhme::Budget budget,
      std::optional<bruce::lhme::BigInt> modulus) override {
    return engine_.evaluate({.spec = spec_, .address = address, .mode = mode,
                             .budget = budget, .modulus = std::move(modulus)});
  }

 private:
  bruce::lhme::Engine& engine_;
  bruce::lhme::FormulaSpec spec_;
};

void add_tile(bruce::lhme::TileReductionSummary& total,
              const bruce::lhme::TileSummary& tile) {
  if (tile.status != bruce::lhme::Status::Confirmed) total.status = tile.status;
  total.checked += tile.checked;
  total.modular_survivors += tile.modular_survivors;
  total.metric_survivors += tile.metric_survivors;
  total.final_survivors += tile.final_survivors;
  total.candidates.insert(total.candidates.end(), tile.candidates.begin(),
                          tile.candidates.end());
}

[[nodiscard]] std::vector<bruce::lhme::GridAddress> make_parent_tiles() {
  using namespace bruce::lhme;
  std::vector<GridAddress> parents;
  parents.reserve(kTileCount);
  for (std::uint8_t y = 0; y < 8; ++y) {
    for (std::uint8_t x = 0; x < 8; ++x) {
      const std::vector<GridStep> steps{{.x = x, .y = y}};
      parents.push_back(*GridAddress::from_steps(steps));
    }
  }
  return parents;
}

[[nodiscard]] FilterMeasurement measure_filter_once(
    std::uint32_t worker_count, const std::vector<bruce::lhme::GridAddress>& parents) {
  using namespace bruce::lhme;
  auto engine = Engine::create({});
  const auto formula = engine.register_formula(make_power_offset_spec(2, 100, 1));
  EngineFormulaEvaluator evaluator{engine, formula};
  const GridDomainSpec domain{{.id = "bruce.benchmark.cpu", .version = 1}, 2};
  DepthPathProjection projection;
  const DomainEvaluationContext context{.domain = domain, .projection = &projection,
                                        .evaluator = evaluator};
  TileReductionSummary summary;
  const auto started = Clock::now();
  summary = {.tile_count = parents.size()};
  for (std::size_t next = 0; next < parents.size();) {
    std::vector<std::future<TileSummary>> batch;
    const auto batch_size = std::min<std::size_t>(worker_count, parents.size() - next);
    batch.reserve(batch_size);
    for (std::size_t slot = 0; slot < batch_size; ++slot) {
      const auto parent = parents[next++];
      batch.push_back(std::async(std::launch::async, [&context, parent] {
        return filter_tile(context, parent, /*modulus=*/17, /*residue=*/11,
                           /*minimum bits=*/7, /*maximum bits=*/5'000);
      }));
    }
    for (auto& future : batch) add_tile(summary, future.get());
  }
  const auto elapsed = std::chrono::duration<double, std::micro>(Clock::now() - started);
  return {.summary = std::move(summary),
          .elapsed_microseconds = elapsed.count()};
}

[[nodiscard]] bool valid_measurement(const FilterMeasurement& measurement,
                                     const std::vector<bruce::lhme::GridAddress>& parents) {
  using namespace bruce::lhme;
  return measurement.summary.status == Status::Confirmed &&
         measurement.summary.tile_count == parents.size() &&
         measurement.summary.checked == static_cast<std::uint64_t>(kTileCount) * 64U &&
         measurement.summary.final_survivors == 512U &&
         measurement.summary.exact_evaluations == 0U;
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

  std::size_t iterations = 11U;
  std::optional<std::filesystem::path> raw_path;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument{argv[index]};
    if (argument == "--quick") {
      iterations = 3U;
    } else if (argument == "--iterations" && index + 1 < argc) {
      if (!parse_positive(argv[++index], iterations)) {
        std::cerr << "--iterations requires a positive integer\n";
        return 2;
      }
    } else if (argument == "--csv" && index + 1 < argc && !raw_path.has_value()) {
      raw_path = std::filesystem::path{argv[++index]};
    } else {
      std::cerr << "usage: lhme_kevin_cpu_benchmark [--quick] [--iterations N] [--csv FILE]\n";
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
  report.heading("KEVIN Domain Projection and Filter Overhead");
  report.row("comparison rule",
             "KEVIN supplies only 8x8 domain coordinates and explicit projection/filter work; FormulaRecipe owns values");
  report.row("timing rule",
             "one unscored warm-up plus repeated reductions; setup excluded; bounded asynchronous CPU batches");

  const auto parents = make_parent_tiles();
  std::optional<FilterMeasurement> reference;
  for (const auto worker_count : kWorkerCounts) {
    const auto label = "KEVIN " + std::to_string(worker_count) + " worker";
    const auto warmup = measure_filter_once(worker_count, parents);
    report.expect_true(label + " warm-up valid", valid_measurement(warmup, parents));

    std::vector<double> samples;
    samples.reserve(iterations);
    std::optional<FilterMeasurement> representative;
    for (std::size_t iteration = 0U; iteration < iterations; ++iteration) {
      const auto measurement = measure_filter_once(worker_count, parents);
      report.expect_true(label + " timed reduction valid", valid_measurement(measurement, parents));
      if (!valid_measurement(measurement, parents)) continue;
      report.expect_true(label + " timed deterministic addresses",
                         measurement.summary.candidates == warmup.summary.candidates);
      if (measurement.summary.candidates != warmup.summary.candidates) continue;
      if (!representative.has_value()) representative = measurement;
      samples.push_back(measurement.elapsed_microseconds);
      if (raw_output.has_value() && !raw_output->write({
            .benchmark = "lhme_kevin_cpu_benchmark",
            .case_id = "modern-domain-filter-workers-" + std::to_string(worker_count),
            .metric = "filter_elapsed",
            .iteration = iteration + 1U,
            .unit = "us/reduction",
            .value = measurement.elapsed_microseconds,
            .validation = "confirmed",
            .notes = "64 tiles; 4096 checked cells; zero Exact materializations; std::async batches",
          })) {
        report.expect_true(label + " raw CSV write", false);
      }
    }
    const auto summary = benchmark::output::format_summary(samples, "us/reduction");
    report.expect_true(label + " timing summary available", summary.has_value());
    if (summary.has_value()) report.row(label + " elapsed statistics", *summary);
    if (!representative.has_value()) continue;
    report.row(label + " checked virtual cells",
               std::to_string(representative->summary.checked));
    report.row(label + " survivors", std::to_string(representative->summary.final_survivors));
    report.row(label + " exact evaluations",
               std::to_string(representative->summary.exact_evaluations));
    if (!reference.has_value()) {
      reference = representative;
    } else {
      report.expect_equal(label + " deterministic survivors",
                          reference->summary.final_survivors,
                          representative->summary.final_survivors);
      report.expect_true(label + " deterministic addresses",
                         reference->summary.candidates == representative->summary.candidates);
    }
  }

  if (raw_output.has_value()) {
    report.expect_true("KEVIN raw CSV write", raw_output->good());
  }

  return report.finish();
}
