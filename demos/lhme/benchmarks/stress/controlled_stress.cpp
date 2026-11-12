#include "demo_report.hpp"

#include "benchmark_output.hpp"
#include "isolated_runner.hpp"
#include "stress_scenarios.hpp"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#if defined(__linux__)
#include <cerrno>
#include <unistd.h>
#endif

namespace {

using bruce::lhme::RetentionMode;
using bruce::lhme::Status;
using bruce::lhme::demo::Report;
using bruce::lhme::demo::stress::ChildReport;
using bruce::lhme::demo::stress::IsolatedRunResult;
using bruce::lhme::demo::stress::Scenario;

#if defined(__linux__)
[[nodiscard]] bool write_child_report(const ChildReport& report) {
  const auto* input = reinterpret_cast<const unsigned char*>(&report);
  std::size_t remaining = sizeof(report);
  while (remaining > 0) {
    const auto written = write(STDOUT_FILENO, input, remaining);
    if (written < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    if (written == 0) return false;
    input += written;
    remaining -= static_cast<std::size_t>(written);
  }
  return true;
}

[[nodiscard]] int run_isolated_child(Scenario scenario) {
  const auto child_report = bruce::lhme::demo::stress::execute_scenario(scenario);
  if (!bruce::lhme::demo::stress::valid_child_report(child_report) ||
      child_report.status != static_cast<std::uint32_t>(Status::Confirmed)) {
    return 2;
  }
  return write_child_report(child_report) ? 0 : 3;
}
#endif

void report_million_expectations(Report& report, std::string_view name,
                                 const ChildReport& child_report) {
  const auto prefix = std::string{name} + " ";
  report.expect_equal(prefix + "status",
                      static_cast<std::uint32_t>(Status::Confirmed),
                      child_report.status);
  report.expect_equal(prefix + "checked", std::uint64_t(1'000'001),
                      child_report.checked_count);
  report.expect_equal(prefix + "exact", std::uint64_t(0),
                      child_report.exact_evaluation_count);
  report.expect_equal(prefix + "filtered", std::uint64_t(1'000'001),
                      child_report.filtered_out_count);
  report.expect_equal(prefix + "retained", std::uint64_t(0),
                      child_report.retained_count);
  report.expect_equal(prefix + "retention mode",
                      static_cast<std::uint32_t>(RetentionMode::SummaryOnly),
                      child_report.retention_mode);
}

void report_adaptive_expectations(Report& report, const ChildReport& child_report) {
  report.expect_equal("adaptive status", static_cast<std::uint32_t>(Status::Confirmed),
                      child_report.status);
  report.expect_equal("adaptive checked", std::uint64_t(10'001),
                      child_report.checked_count);
  report.expect_equal("adaptive exact", std::uint64_t(1'250),
                      child_report.exact_evaluation_count);
  report.expect_equal("adaptive filtered", std::uint64_t(8'751),
                      child_report.filtered_out_count);
  report.expect_equal("adaptive retained", std::uint64_t(3),
                      child_report.retained_count);
  report.expect_equal("adaptive effective mode",
                      static_cast<std::uint32_t>(RetentionMode::TopK),
                      child_report.retention_mode);
}

[[nodiscard]] bool report_repeated_isolated_runs(
    Report& report, Scenario scenario, std::size_t iterations,
    bruce::lhme::benchmark::output::RawCsvWriter* raw_output) {
  const auto name = bruce::lhme::demo::stress::scenario_name(scenario);
  const auto warmup = bruce::lhme::demo::stress::run_isolated("/proc/self/exe", scenario);
  report.expect_true(std::string{name} + " warm-up available", warmup.available);
  report.expect_true(std::string{name} + " warm-up valid", warmup.success);

  std::vector<double> elapsed_samples;
  std::vector<double> peak_samples;
  elapsed_samples.reserve(iterations);
  peak_samples.reserve(iterations);
  std::optional<IsolatedRunResult> representative;
  for (std::size_t iteration = 0U; iteration < iterations; ++iteration) {
    const auto run = bruce::lhme::demo::stress::run_isolated("/proc/self/exe", scenario);
    report.expect_true(std::string{name} + " isolated measurement available", run.available);
    report.expect_true(std::string{name} + " isolated child valid", run.success);
    if (!run.success) continue;
    if (!representative.has_value()) representative = run;
    report.expect_equal(std::string{name} + " deterministic checked",
                        representative->report.checked_count, run.report.checked_count);
    report.expect_equal(std::string{name} + " deterministic exact evaluations",
                        representative->report.exact_evaluation_count,
                        run.report.exact_evaluation_count);
    report.expect_equal(std::string{name} + " deterministic retained count",
                        representative->report.retained_count, run.report.retained_count);
    elapsed_samples.push_back(static_cast<double>(run.report.elapsed_milliseconds));
    peak_samples.push_back(static_cast<double>(run.report.peak_bytes));
    if (raw_output != nullptr &&
        (!raw_output->write({.benchmark = "lhme_controlled_stress",
                             .case_id = name,
                             .metric = "elapsed",
                             .iteration = iteration + 1U,
                             .unit = "ms/scenario",
                             .value = elapsed_samples.back(),
                             .validation = "confirmed",
                             .notes = "isolated Linux child; deterministic scenario counters"}) ||
         !raw_output->write({.benchmark = "lhme_controlled_stress",
                             .case_id = name,
                             .metric = "peak_rss",
                             .iteration = iteration + 1U,
                             .unit = "bytes",
                             .value = peak_samples.back(),
                             .validation = "confirmed",
                             .notes = "isolated Linux child high-water mark"}))) {
      return false;
    }
  }
  const auto elapsed_summary = bruce::lhme::benchmark::output::format_summary(
      elapsed_samples, "ms/scenario");
  const auto peak_summary = bruce::lhme::benchmark::output::format_summary(
      peak_samples, "bytes");
  report.expect_true(std::string{name} + " elapsed summary available",
                     elapsed_summary.has_value());
  report.expect_true(std::string{name} + " peak summary available", peak_summary.has_value());
  if (!representative.has_value() || !elapsed_summary.has_value() ||
      !peak_summary.has_value()) {
    return false;
  }
  report.row(std::string{name} + " elapsed statistics", *elapsed_summary);
  report.row(std::string{name} + " isolated process peak RAM statistics", *peak_summary);
  report.expect_true(std::string{name} + " positive peak",
                     representative->report.peak_bytes > 0);

  if (scenario == Scenario::AdaptiveHits) {
    report_adaptive_expectations(report, representative->report);
  } else {
    report_million_expectations(report, name, representative->report);
  }
  return true;
}

void report_safe_stop(Report& report) {
  auto request = bruce::lhme::demo::stress::exact_request();
  request.cancellation_signal = std::make_shared<std::atomic_bool>(true);
  const auto cancelled = bruce::lhme::evaluate_streaming_range(
      request, {0, 1'000'000}, bruce::lhme::demo::stress::million_scan_options(4));
  report.expect_equal("safe stop status", Status::Cancelled, cancelled.status);
  report.expect_equal("safe stop checked", std::uint64_t(0),
                      cancelled.summary.checked_count);
  report.expect_equal("safe stop exact", std::uint64_t(0),
                      cancelled.exact_evaluation_count);
  report.expect_equal("safe stop filtered", std::uint64_t(0),
                      cancelled.filtered_out_count);
  report.expect_true("safe stop retains nothing", cancelled.retained_candidates.empty());
}

[[nodiscard]] int run_parent_controller(
    std::size_t iterations, bruce::lhme::benchmark::output::RawCsvWriter* raw_output) {
  Report report;
  report.heading("LHME Isolated CPU and Bounded-Retention Stress Test");

#if defined(__linux__)
  report.expect_true("million-scan-1 repeated measurements",
                     report_repeated_isolated_runs(report, Scenario::MillionScanOne,
                                                    iterations, raw_output));
  report.expect_true("million-scan-8 repeated measurements",
                     report_repeated_isolated_runs(report, Scenario::MillionScanEight,
                                                    iterations, raw_output));
  report.expect_true("adaptive-hits repeated measurements",
                     report_repeated_isolated_runs(report, Scenario::AdaptiveHits,
                                                    iterations, raw_output));
#else
  report.row("isolated process peak RAM bytes", "unavailable (Linux only)");
#endif

  report_safe_stop(report);
  if (raw_output != nullptr) {
    report.expect_true("stress raw CSV write", raw_output->good());
  }
  return report.finish();
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
  if (argc == 3 && std::string_view{argv[1]} == "--isolated-child") {
    const auto scenario = bruce::lhme::demo::stress::parse_scenario(argv[2]);
    if (!scenario.has_value()) return 1;
#if defined(__linux__)
    return run_isolated_child(*scenario);
#else
    return 1;
#endif
  }
  std::size_t iterations = 3U;
  std::optional<std::filesystem::path> raw_path;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument{argv[index]};
    if (argument == "--quick") {
      iterations = 1U;
    } else if (argument == "--iterations" && index + 1 < argc) {
      if (!parse_positive(argv[++index], iterations)) {
        std::cerr << "--iterations requires a positive integer\n";
        return 2;
      }
    } else if (argument == "--csv" && index + 1 < argc && !raw_path.has_value()) {
      raw_path = std::filesystem::path{argv[++index]};
    } else {
      std::cerr << "usage: lhme_controlled_stress [--quick] [--iterations N] [--csv FILE]\n";
      return 2;
    }
  }
  std::optional<bruce::lhme::benchmark::output::RawCsvWriter> raw_output;
  if (raw_path.has_value()) {
    raw_output.emplace(*raw_path);
    if (!raw_output->good()) {
      std::cerr << "--csv must name a new writable output file\n";
      return 2;
    }
  }
  return run_parent_controller(iterations, raw_output ? &*raw_output : nullptr);
}
