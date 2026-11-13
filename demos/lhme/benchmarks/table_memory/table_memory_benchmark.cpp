#include "demo_report.hpp"

#include "benchmark_output.hpp"
#include "kevin_sparse_table.hpp"
#include "process_memory.hpp"
#include "table_storage.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#if defined(__linux__)
#include <cerrno>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

using bruce::lhme::benchmark::table::DenseValueTable;
using bruce::lhme::benchmark::table::FormulaRecipeTable;
using bruce::lhme::benchmark::table::HashValueTable;
using bruce::lhme::benchmark::table::FormulaOverrideTable;
using bruce::lhme::benchmark::table::SortedPairTable;
using bruce::lhme::benchmark::table::StorageMetrics;
using bruce::lhme::benchmark::table::synthetic_value_at;

constexpr std::uint64_t kRecordCount = 1'000'000;
constexpr std::uint32_t kReportMagic = 0x54424c45U;  // "TBLE"
constexpr std::uint32_t kReportVersion = 1;

enum class TableKind : std::uint32_t {
  DenseValues = 1,
  SortedPairs = 2,
  HashTable = 3,
  FormulaOnly = 4,
  FormulaWithOverridesZero = 5,
  FormulaWithOverridesOnePercent = 6,
  FormulaWithOverridesTenPercent = 7,
  FormulaWithOverridesFull = 8,
};

struct ChildMeasurement {
  std::uint32_t magic = kReportMagic;
  std::uint32_t version = kReportVersion;
  std::uint32_t kind = 0;
  std::uint32_t status = 1;
  std::uint64_t record_count = 0;
  std::uint64_t value_payload_bytes = 0;
  std::uint64_t address_metadata_bytes = 0;
  std::uint64_t container_bytes = 0;
  std::uint64_t override_count = 0;
  std::int64_t elapsed_milliseconds = 0;
  std::uint64_t peak_bytes = 0;
};

static_assert(std::is_trivially_copyable_v<ChildMeasurement>);

struct IsolatedMeasurement {
  bool available = false;
  bool success = false;
  ChildMeasurement result{};
};

[[nodiscard]] constexpr std::string_view kind_name(TableKind kind) {
  switch (kind) {
    case TableKind::DenseValues:
      return "dense-vector";
    case TableKind::SortedPairs:
      return "sorted-pairs";
    case TableKind::HashTable:
      return "hash-table";
    case TableKind::FormulaOnly:
      return "formula-only";
    case TableKind::FormulaWithOverridesZero:
      return "formula-with-overrides-0-percent";
    case TableKind::FormulaWithOverridesOnePercent:
      return "formula-with-overrides-1-percent";
    case TableKind::FormulaWithOverridesTenPercent:
      return "formula-with-overrides-10-percent";
    case TableKind::FormulaWithOverridesFull:
      return "formula-with-overrides-100-percent";
  }
  return "unknown";
}

[[nodiscard]] constexpr std::optional<TableKind> parse_kind(std::string_view name) {
  if (name == "dense-vector") return TableKind::DenseValues;
  if (name == "sorted-pairs") return TableKind::SortedPairs;
  if (name == "hash-table") return TableKind::HashTable;
  if (name == "formula-only") return TableKind::FormulaOnly;
  if (name == "formula-with-overrides-0-percent") return TableKind::FormulaWithOverridesZero;
  if (name == "formula-with-overrides-1-percent") return TableKind::FormulaWithOverridesOnePercent;
  if (name == "formula-with-overrides-10-percent") return TableKind::FormulaWithOverridesTenPercent;
  if (name == "formula-with-overrides-100-percent") return TableKind::FormulaWithOverridesFull;
  return std::nullopt;
}

[[nodiscard]] constexpr bool known_kind(std::uint32_t kind) {
  return kind >= static_cast<std::uint32_t>(TableKind::DenseValues) &&
         kind <= static_cast<std::uint32_t>(TableKind::FormulaWithOverridesFull);
}

[[nodiscard]] bool valid_measurement(const ChildMeasurement& result) {
  return result.magic == kReportMagic && result.version == kReportVersion &&
         known_kind(result.kind) && result.status == 0 &&
         result.record_count == kRecordCount && result.peak_bytes > 0;
}

struct BuiltTable {
  StorageMetrics metrics;
  std::uint64_t override_count = 0;
};

[[nodiscard]] BuiltTable build_and_check(TableKind kind, bool& correct_lookup) {
  constexpr std::uint64_t probe_address = 654'321;
  const auto expected = synthetic_value_at(probe_address);
  switch (kind) {
    case TableKind::DenseValues: {
      const auto table = DenseValueTable::make(kRecordCount);
      correct_lookup = table.value_at(probe_address) == expected;
      return {.metrics = table.metrics()};
    }
    case TableKind::SortedPairs: {
      const auto table = SortedPairTable::make(kRecordCount);
      correct_lookup = table.value_at(probe_address) == expected;
      return {.metrics = table.metrics()};
    }
    case TableKind::HashTable: {
      const auto table = HashValueTable::make(kRecordCount);
      correct_lookup = table.value_at(probe_address) == expected;
      return {.metrics = table.metrics()};
    }
    case TableKind::FormulaOnly: {
      const auto table = FormulaRecipeTable::make(kRecordCount);
      correct_lookup = table.contains(probe_address) &&
                       table.symbolic_value_at(probe_address) ==
                           std::optional<std::string>{"100*(2^654321-1)"};
      return {.metrics = table.metrics()};
    }
    case TableKind::FormulaWithOverridesZero:
    case TableKind::FormulaWithOverridesOnePercent:
    case TableKind::FormulaWithOverridesTenPercent:
    case TableKind::FormulaWithOverridesFull: {
      const auto overrides = kind == TableKind::FormulaWithOverridesZero ? 0ULL
          : kind == TableKind::FormulaWithOverridesOnePercent ? 10'000ULL
          : kind == TableKind::FormulaWithOverridesTenPercent ? 100'000ULL
                                                       : kRecordCount;
      auto table = FormulaOverrideTable::make(kRecordCount, overrides);
      const auto first_source = table.source_at(0);
      const auto last_source = table.source_at(kRecordCount - 1);
      const auto expects_overrides = overrides == kRecordCount;
      correct_lookup = table.contains(probe_address) &&
                       table.override_count() == overrides &&
                       first_source == (overrides == 0
                                           ? FormulaOverrideTable::ValueSource::Formula
                                           : FormulaOverrideTable::ValueSource::Override) &&
                       last_source == (expects_overrides
                                           ? FormulaOverrideTable::ValueSource::Override
                                           : FormulaOverrideTable::ValueSource::Formula);
      return {.metrics = {.record_count = kRecordCount,
                          .value_payload_bytes = 0,
                          .address_metadata_bytes =
                              overrides * static_cast<std::uint64_t>(sizeof(std::uint64_t)),
                          .container_bytes = sizeof(table)},
              .override_count = overrides};
    }
  }
  correct_lookup = false;
  return {};
}

[[nodiscard]] ChildMeasurement measure_table(TableKind kind) {
  const auto started = std::chrono::steady_clock::now();
  bool correct_lookup = false;
  const auto built = build_and_check(kind, correct_lookup);
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - started);
  const auto peak = bruce::lhme::demo::stress::process_peak_memory();

  return {.kind = static_cast<std::uint32_t>(kind),
          .status = correct_lookup ? 0U : 1U,
          .record_count = built.metrics.record_count,
          .value_payload_bytes = built.metrics.value_payload_bytes,
          .address_metadata_bytes = built.metrics.address_metadata_bytes,
          .container_bytes = built.metrics.container_bytes,
          .override_count = built.override_count,
          .elapsed_milliseconds = elapsed.count(),
          .peak_bytes = peak.available ? peak.bytes : 0};
}

#if defined(__linux__)
[[nodiscard]] bool write_all(const void* source, std::size_t size) {
  const auto* bytes = reinterpret_cast<const unsigned char*>(source);
  while (size > 0) {
    const auto written = write(STDOUT_FILENO, bytes, size);
    if (written < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    if (written == 0) return false;
    bytes += written;
    size -= static_cast<std::size_t>(written);
  }
  return true;
}

[[nodiscard]] bool read_all(int file_descriptor, void* destination, std::size_t size) {
  auto* bytes = reinterpret_cast<unsigned char*>(destination);
  while (size > 0) {
    const auto received = read(file_descriptor, bytes, size);
    if (received == 0) return false;
    if (received < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    bytes += received;
    size -= static_cast<std::size_t>(received);
  }
  return true;
}

[[nodiscard]] bool no_extra_bytes(int file_descriptor) {
  unsigned char byte = 0;
  while (true) {
    const auto received = read(file_descriptor, &byte, sizeof(byte));
    if (received == 0) return true;
    if (received < 0 && errno == EINTR) continue;
    return false;
  }
}

[[nodiscard]] bool normal_success(pid_t child) {
  int wait_status = 0;
  while (waitpid(child, &wait_status, 0) < 0) {
    if (errno != EINTR) return false;
  }
  return WIFEXITED(wait_status) && WEXITSTATUS(wait_status) == 0;
}

[[nodiscard]] int run_child(TableKind kind) {
  const auto measurement = measure_table(kind);
  return valid_measurement(measurement) && write_all(&measurement, sizeof(measurement))
             ? 0
             : 1;
}
#endif

[[nodiscard]] IsolatedMeasurement run_isolated(TableKind kind) {
#if defined(__linux__)
  IsolatedMeasurement output{.available = true};
  int pipe_descriptors[2]{};
  if (pipe(pipe_descriptors) != 0) return output;

  const auto child = fork();
  if (child < 0) {
    close(pipe_descriptors[0]);
    close(pipe_descriptors[1]);
    return output;
  }
  if (child == 0) {
    close(pipe_descriptors[0]);
    if (dup2(pipe_descriptors[1], STDOUT_FILENO) < 0) _exit(126);
    close(pipe_descriptors[1]);
    const auto name = kind_name(kind);
    char* const arguments[] = {const_cast<char*>("/proc/self/exe"),
                               const_cast<char*>("--isolated-table-child"),
                               const_cast<char*>(name.data()), nullptr};
    execv("/proc/self/exe", arguments);
    _exit(127);
  }

  close(pipe_descriptors[1]);
  const bool got_record = read_all(pipe_descriptors[0], &output.result,
                                   sizeof(output.result));
  const bool only_record = got_record && no_extra_bytes(pipe_descriptors[0]);
  close(pipe_descriptors[0]);
  output.success = got_record && only_record && normal_success(child) &&
                   valid_measurement(output.result) &&
                   output.result.kind == static_cast<std::uint32_t>(kind);
  return output;
#else
  static_cast<void>(kind);
  return {};
#endif
}

[[nodiscard]] bool display_measurements(
    bruce::lhme::demo::Report& report, TableKind kind,
    const std::vector<IsolatedMeasurement>& measurements,
    bruce::lhme::benchmark::output::RawCsvWriter* raw_output) {
  const auto name = kind_name(kind);
  if (measurements.empty()) return false;
  const auto& first = measurements.front();
  report.expect_true(std::string{name} + " measurement available", first.available);
  report.expect_true(std::string{name} + " child report valid", first.success);
  if (!first.success) return false;

  std::vector<double> build_samples;
  std::vector<double> peak_samples;
  build_samples.reserve(measurements.size());
  peak_samples.reserve(measurements.size());
  for (std::size_t index = 0U; index < measurements.size(); ++index) {
    const auto& measurement = measurements[index];
    report.expect_true(std::string{name} + " sample child report valid", measurement.success);
    if (!measurement.success) return false;
    report.expect_equal(std::string{name} + " sample record count",
                        first.result.record_count, measurement.result.record_count);
    report.expect_equal(std::string{name} + " sample container bytes",
                        first.result.container_bytes, measurement.result.container_bytes);
    build_samples.push_back(static_cast<double>(measurement.result.elapsed_milliseconds));
    peak_samples.push_back(static_cast<double>(measurement.result.peak_bytes));
    if (raw_output != nullptr &&
        (!raw_output->write({.benchmark = "lhme_table_memory_benchmark",
                             .case_id = name,
                             .metric = "build_elapsed",
                             .iteration = index + 1U,
                             .unit = "ms/build",
                             .value = build_samples.back(),
                             .validation = "confirmed",
                             .notes = "isolated Linux child; one million virtual addresses"}) ||
         !raw_output->write({.benchmark = "lhme_table_memory_benchmark",
                             .case_id = name,
                             .metric = "peak_rss",
                             .iteration = index + 1U,
                             .unit = "bytes",
                             .value = peak_samples.back(),
                             .validation = "confirmed",
                             .notes = "isolated Linux child high-water mark"}))) {
      return false;
    }
  }

  const auto build_summary = bruce::lhme::benchmark::output::format_summary(
      build_samples, "ms/build");
  const auto peak_summary = bruce::lhme::benchmark::output::format_summary(
      peak_samples, "bytes");
  if (!build_summary.has_value() || !peak_summary.has_value()) return false;

  report.row(std::string{name} + " records",
             std::to_string(first.result.record_count));
  report.row(std::string{name} + " value payload bytes",
             std::to_string(first.result.value_payload_bytes));
  report.row(std::string{name} + " address/key metadata bytes",
             std::to_string(first.result.address_metadata_bytes));
  report.row(std::string{name} + " container bytes",
             std::to_string(first.result.container_bytes));
  report.row(std::string{name} + " explicit overrides",
             std::to_string(first.result.override_count));
  report.row(std::string{name} + " build statistics", *build_summary);
  report.row(std::string{name} + " isolated process peak RAM statistics", *peak_summary);
  report.expect_true(std::string{name} + " positive peak", first.result.peak_bytes > 0);
  return true;
}

[[nodiscard]] int run_parent(std::size_t iterations,
                              bruce::lhme::benchmark::output::RawCsvWriter* raw_output) {
  bruce::lhme::demo::Report report;
  report.heading("LHME Million-Value Table Memory Comparison");
  report.row("value model",
             "materialized tables use deterministic UInt64 payloads; BRUCE stores h(x)=100*(2^x-1) as a FormulaRecipe");
  report.row("process peak RAM", "measured in isolated Linux child processes");
  report.row("container bytes",
             "analytic lower bounds; allocator and runtime overhead are visible only in process peak");

#if defined(__linux__)
  constexpr std::array<TableKind, 8> kinds{TableKind::DenseValues,
                                            TableKind::SortedPairs,
                                            TableKind::HashTable,
                                            TableKind::FormulaOnly,
                                            TableKind::FormulaWithOverridesZero,
                                            TableKind::FormulaWithOverridesOnePercent,
                                            TableKind::FormulaWithOverridesTenPercent,
                                            TableKind::FormulaWithOverridesFull};
  std::array<IsolatedMeasurement, kinds.size()> measurements{};
  for (std::size_t index = 0; index < kinds.size(); ++index) {
    std::vector<IsolatedMeasurement> samples;
    samples.reserve(iterations);
    for (std::size_t iteration = 0U; iteration < iterations; ++iteration) {
      samples.push_back(run_isolated(kinds[index]));
    }
    if (!samples.empty()) measurements[index] = samples.front();
    report.expect_true(std::string{kind_name(kinds[index])} + " repeated measurements valid",
                       display_measurements(report, kinds[index], samples, raw_output));
  }
  if (measurements[0].success && measurements[1].success &&
      measurements[3].success) {
    report.expect_equal("dense vector payload", std::uint64_t(8'000'000),
                        measurements[0].result.value_payload_bytes);
    report.expect_equal("sorted pairs value payload", std::uint64_t(8'000'000),
                        measurements[1].result.value_payload_bytes);
    report.expect_equal("sorted pairs address metadata", std::uint64_t(8'000'000),
                        measurements[1].result.address_metadata_bytes);
    report.expect_equal("FormulaOnly payload", std::uint64_t(0),
                        measurements[3].result.value_payload_bytes);
    report.expect_equal("FormulaOnly address metadata", std::uint64_t(0),
                        measurements[3].result.address_metadata_bytes);
    report.expect_true("FormulaOnly metadata stays compact",
                       measurements[3].result.container_bytes < 1'024);
  }
  if (measurements[4].success && measurements[5].success &&
      measurements[6].success && measurements[7].success) {
    report.expect_equal("FormulaWithOverrides 0 percent overrides", std::uint64_t(0),
                        measurements[4].result.override_count);
    report.expect_equal("FormulaWithOverrides 1 percent overrides", std::uint64_t(10'000),
                        measurements[5].result.override_count);
    report.expect_equal("FormulaWithOverrides 10 percent overrides", std::uint64_t(100'000),
                        measurements[6].result.override_count);
    report.expect_equal("FormulaWithOverrides full overrides", kRecordCount,
                        measurements[7].result.override_count);
  }
#else
  report.row("isolated process peak RAM", "unavailable (Linux only)");
#endif

  if (raw_output != nullptr) {
    report.expect_true("table memory raw CSV write", raw_output->good());
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
  if (argc == 3 && std::string_view{argv[1]} == "--isolated-table-child") {
    const auto kind = parse_kind(argv[2]);
    if (!kind.has_value()) return 1;
#if defined(__linux__)
    return run_child(*kind);
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
      std::cerr << "usage: lhme_table_memory_benchmark [--quick] [--iterations N] [--csv FILE]\n";
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
  return run_parent(iterations, raw_output ? &*raw_output : nullptr);
}
