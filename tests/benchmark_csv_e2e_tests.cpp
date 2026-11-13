#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <locale>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

constexpr std::string_view kHeader =
    "benchmark,case_id,metric,iteration,unit,value,validation,notes";

struct Field {
  std::string value;
  bool quoted = false;
};

struct CsvRow {
  std::string benchmark;
  std::string case_id;
  std::string metric;
  std::size_t iteration = 0U;
  std::string unit;
  double value = 0.0;
};

struct Summary {
  double minimum = 0.0;
  double median = 0.0;
  double maximum = 0.0;
  double mean = 0.0;
  double standard_deviation = 0.0;
};

struct GroupExpectation {
  std::string case_id;
  std::string metric;
  std::string unit;
  std::size_t iterations = 0U;
  std::string report_label;
};

enum class ReportStyle { Fair, Standard };

struct RunExpectation {
  std::string name;
  std::string benchmark;
  ReportStyle style = ReportStyle::Standard;
  std::vector<GroupExpectation> groups;
};

[[noreturn]] void fail(std::string_view message) {
  throw std::runtime_error{std::string{message}};
}

[[nodiscard]] std::string read_text(const std::filesystem::path& path) {
  std::ifstream input{path, std::ios::binary};
  if (!input) fail("cannot read " + path.string());
  return {std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
}

[[nodiscard]] std::vector<Field> parse_csv_record(std::string line,
                                                   const std::filesystem::path& path,
                                                   std::size_t line_number) {
  if (!line.empty() && line.back() == '\r') line.pop_back();

  std::vector<Field> fields;
  std::size_t position = 0U;
  while (true) {
    Field field;
    if (position < line.size() && line[position] == '"') {
      field.quoted = true;
      ++position;
      bool closed = false;
      while (position < line.size()) {
        const auto character = line[position++];
        if (character != '"') {
          field.value += character;
          continue;
        }
        if (position < line.size() && line[position] == '"') {
          field.value += '"';
          ++position;
          continue;
        }
        closed = true;
        break;
      }
      if (!closed) {
        fail(path.string() + ":" + std::to_string(line_number) +
             " contains an unterminated quoted CSV field");
      }
    } else {
      const auto start = position;
      while (position < line.size() && line[position] != ',') {
        if (line[position] == '"') {
          fail(path.string() + ":" + std::to_string(line_number) +
               " contains an unescaped quote in an unquoted CSV field");
        }
        ++position;
      }
      field.value = line.substr(start, position - start);
    }
    fields.push_back(std::move(field));
    if (position == line.size()) break;
    if (line[position] != ',') {
      fail(path.string() + ":" + std::to_string(line_number) +
           " has invalid CSV field separation");
    }
    ++position;
    if (position == line.size()) {
      fields.push_back({});
      break;
    }
  }
  return fields;
}

[[nodiscard]] bool has_nine_decimal_places(std::string_view text) {
  if (text.empty()) return false;
  std::size_t position = text.front() == '-' ? 1U : 0U;
  const auto decimal = text.find('.', position);
  if (decimal == std::string_view::npos || decimal == position ||
      text.size() != decimal + 10U) {
    return false;
  }
  for (std::size_t index = position; index < text.size(); ++index) {
    if (index == decimal) continue;
    if (text[index] < '0' || text[index] > '9') return false;
  }
  return true;
}

[[nodiscard]] std::optional<std::size_t> parse_iteration(std::string_view text) {
  if (text.empty()) return std::nullopt;
  for (const auto character : text) {
    if (character < '0' || character > '9') return std::nullopt;
  }
  std::size_t value = 0U;
  const auto [pointer, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || pointer != text.data() + text.size() || value == 0U) {
    return std::nullopt;
  }
  return value;
}

[[nodiscard]] std::optional<double> parse_value(std::string_view text) {
  if (!has_nine_decimal_places(text)) return std::nullopt;
  try {
    const auto value = std::stod(std::string{text});
    return std::isfinite(value) ? std::optional<double>{value} : std::nullopt;
  } catch (...) {
    return std::nullopt;
  }
}

[[nodiscard]] Summary summarize(std::vector<double> values) {
  if (values.empty()) fail("cannot summarize an empty raw-value group");
  std::sort(values.begin(), values.end());
  const auto total = std::accumulate(values.begin(), values.end(), 0.0);
  const auto mean = total / static_cast<double>(values.size());
  double squared_deviation_sum = 0.0;
  for (const auto value : values) {
    const auto delta = value - mean;
    squared_deviation_sum += delta * delta;
  }
  const auto median_index = values.size() / 2U;
  const auto median = values.size() % 2U == 0U
      ? (values[median_index - 1U] + values[median_index]) / 2.0
      : values[median_index];
  const auto divisor = values.size() > 1U ? static_cast<double>(values.size() - 1U) : 1.0;
  return {.minimum = values.front(),
          .median = median,
          .maximum = values.back(),
          .mean = mean,
          .standard_deviation = std::sqrt(squared_deviation_sum / divisor)};
}

[[nodiscard]] std::string format_summary(const Summary& summary, std::size_t count,
                                         std::string_view unit) {
  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << std::fixed << std::setprecision(3) << "median=" << summary.median << ' '
         << unit << "; mean=" << summary.mean << "; min=" << summary.minimum
         << "; max=" << summary.maximum << "; stddev=";
  if (count == 1U) {
    output << "unavailable (n=1)";
  } else {
    output << summary.standard_deviation;
  }
  return output.str();
}

[[nodiscard]] std::string group_key(std::string_view case_id, std::string_view metric) {
  return std::string{case_id} + '\x1f' + std::string{metric};
}

[[nodiscard]] std::vector<GroupExpectation> fair_groups() {
  constexpr std::array strategies{
      "formula-recipe-0-percent-overrides",
      "formula-recipe-with-overrides-0-percent-overrides",
      "dense-vector-0-percent-overrides",
      "sorted-pairs-0-percent-overrides",
      "hash-table-0-percent-overrides",
  };
  constexpr std::array workloads{"sequential", "pseudo-random", "repeat-heavy"};
  std::vector<GroupExpectation> groups;
  groups.reserve(strategies.size() * workloads.size());
  for (const auto strategy : strategies) {
    for (const auto workload : workloads) {
      const auto case_id = std::string{strategy} + "." + workload;
      groups.push_back({.case_id = case_id,
                        .metric = "lookup_elapsed",
                        .unit = "us/workload",
                        .iterations = 11U,
                        .report_label = std::string{strategy} + " " + workload});
    }
  }
  return groups;
}

[[nodiscard]] std::vector<GroupExpectation> table_groups() {
  constexpr std::array cases{
      "dense-vector", "sorted-pairs", "hash-table", "formula-only",
      "formula-with-overrides-0-percent", "formula-with-overrides-1-percent",
      "formula-with-overrides-10-percent", "formula-with-overrides-100-percent",
  };
  std::vector<GroupExpectation> groups;
  groups.reserve(cases.size() * 2U);
  for (const auto case_id : cases) {
    groups.push_back({.case_id = case_id, .metric = "build_elapsed", .unit = "ms/build",
                      .iterations = 1U, .report_label = case_id});
    groups.push_back({.case_id = case_id, .metric = "peak_rss", .unit = "bytes",
                      .iterations = 1U, .report_label = case_id});
  }
  return groups;
}

[[nodiscard]] std::vector<GroupExpectation> kevin_groups() {
  std::vector<GroupExpectation> groups;
  for (const auto workers : {1U, 2U, 4U, 8U}) {
    groups.push_back({.case_id = "modern-domain-filter-workers-" + std::to_string(workers),
                      .metric = "filter_elapsed",
                      .unit = "us/reduction",
                      .iterations = 3U,
                      .report_label = "KEVIN " + std::to_string(workers) + " worker"});
  }
  return groups;
}

[[nodiscard]] std::vector<GroupExpectation> hybrid_groups() {
  constexpr std::array tiers{
      std::pair{"0", "0 percent"}, std::pair{"1000", "0.1 percent"},
      std::pair{"10000", "1 percent"}, std::pair{"100000", "10 percent"},
  };
  std::vector<GroupExpectation> groups;
  groups.reserve(tiers.size() * 2U);
  for (const auto& [count, label] : tiers) {
    const auto case_id = std::string{"formula-overrides-"} + count;
    const auto report_label = std::string{"FormulaWithOverrides "} + label;
    groups.push_back({.case_id = case_id,
                      .metric = "override_insert_elapsed",
                      .unit = "ms/insert-set",
                      .iterations = 1U,
                      .report_label = report_label});
    groups.push_back({.case_id = case_id,
                      .metric = "snapshot_size",
                      .unit = "bytes",
                      .iterations = 1U,
                      .report_label = report_label});
  }
  return groups;
}

[[nodiscard]] std::vector<GroupExpectation> stress_groups() {
  constexpr std::array cases{"million-scan-1", "million-scan-8", "adaptive-hits"};
  std::vector<GroupExpectation> groups;
  groups.reserve(cases.size() * 2U);
  for (const auto case_id : cases) {
    groups.push_back({.case_id = case_id, .metric = "elapsed", .unit = "ms/scenario",
                      .iterations = 1U, .report_label = case_id});
    groups.push_back({.case_id = case_id, .metric = "peak_rss", .unit = "bytes",
                      .iterations = 1U, .report_label = case_id});
  }
  return groups;
}

[[nodiscard]] std::vector<std::string> expected_report_lines(
    const RunExpectation& expectation, const GroupExpectation& group,
    const Summary& summary) {
  if (expectation.style == ReportStyle::Fair) {
    return {
        group.report_label + " lookup minimum microseconds | " + std::to_string(summary.minimum),
        group.report_label + " lookup median microseconds | " + std::to_string(summary.median),
        group.report_label + " lookup mean microseconds | " + std::to_string(summary.mean),
        group.report_label + " lookup maximum microseconds | " + std::to_string(summary.maximum),
        group.report_label + " lookup sample standard deviation microseconds | " +
            std::to_string(summary.standard_deviation),
    };
  }

  const auto formatted = format_summary(summary, group.iterations, group.unit);
  if (expectation.name == "table") {
    return {group.report_label +
            (group.metric == "build_elapsed" ? " build statistics | "
                                               : " isolated process peak RAM statistics | ") +
            formatted};
  }
  if (expectation.name == "kevin") {
    return {group.report_label + " elapsed statistics | " + formatted};
  }
  if (expectation.name == "hybrid") {
    return {group.report_label +
            (group.metric == "override_insert_elapsed" ? " insert statistics | "
                                                         : " snapshot size statistics | ") +
            formatted};
  }
  return {group.report_label +
          (group.metric == "elapsed" ? " elapsed statistics | "
                                      : " isolated process peak RAM statistics | ") +
          formatted};
}

void validate_run(const RunExpectation& expectation, const std::filesystem::path& csv_path,
                  const std::filesystem::path& stdout_path) {
  std::ifstream input{csv_path};
  if (!input) fail(expectation.name + ": CSV was not created at " + csv_path.string());

  std::string header;
  if (!std::getline(input, header)) {
    fail(expectation.name + ": CSV schema does not exactly match the required header");
  }
  if (!header.empty() && header.back() == '\r') header.pop_back();
  if (header != kHeader) {
    fail(expectation.name + ": CSV schema does not exactly match the required header");
  }

  std::map<std::string, GroupExpectation> expected;
  std::size_t expected_rows = 0U;
  for (const auto& group : expectation.groups) {
    const auto key = group_key(group.case_id, group.metric);
    if (!expected.emplace(key, group).second) fail(expectation.name + ": duplicate test expectation");
    expected_rows += group.iterations;
  }

  std::map<std::string, std::map<std::size_t, double>> values;
  std::string line;
  std::size_t line_number = 1U;
  std::size_t row_count = 0U;
  while (std::getline(input, line)) {
    ++line_number;
    const auto fields = parse_csv_record(line, csv_path, line_number);
    if (fields.size() != 8U) {
      fail(expectation.name + ": CSV row " + std::to_string(line_number) +
           " must have exactly eight fields");
    }
    for (const auto index : {0U, 1U, 2U, 4U, 6U, 7U}) {
      if (!fields[index].quoted) {
        fail(expectation.name + ": CSV text field " + std::to_string(index) +
             " must use RFC 4180 quoted-field syntax");
      }
    }
    if (fields[3].quoted || fields[5].quoted) {
      fail(expectation.name + ": CSV numeric fields must not be quoted");
    }
    if (fields[0].value != expectation.benchmark || fields[6].value != "confirmed") {
      fail(expectation.name + ": CSV has an unexpected benchmark identifier or validation state");
    }
    const auto iteration = parse_iteration(fields[3].value);
    const auto value = parse_value(fields[5].value);
    if (!iteration.has_value() || !value.has_value()) {
      fail(expectation.name + ": CSV iteration/value is not a positive integer/fixed-9 finite number");
    }
    const auto key = group_key(fields[1].value, fields[2].value);
    const auto expected_group = expected.find(key);
    if (expected_group == expected.end() || fields[4].value != expected_group->second.unit) {
      fail(expectation.name + ": CSV has an unexpected case, metric, or unit");
    }
    if (!values[key].emplace(*iteration, *value).second) {
      fail(expectation.name + ": CSV contains a duplicate iteration for one case/metric");
    }
    ++row_count;
  }
  if (row_count != expected_rows || values.size() != expected.size()) {
    fail(expectation.name + ": CSV data-row or case/metric-group count differs from quick-mode contract");
  }

  const auto stdout_text = read_text(stdout_path);
  if (stdout_text.find("PASS") == std::string::npos) {
    fail(expectation.name + ": benchmark output did not report PASS");
  }
  for (const auto& [key, group] : expected) {
    const auto found = values.find(key);
    if (found == values.end() || found->second.size() != group.iterations) {
      fail(expectation.name + ": case/metric does not have the expected quick-mode iteration count");
    }
    std::vector<double> samples;
    samples.reserve(group.iterations);
    for (std::size_t iteration = 1U; iteration <= group.iterations; ++iteration) {
      const auto sample = found->second.find(iteration);
      if (sample == found->second.end()) {
        fail(expectation.name + ": case/metric iterations are not the unique sequence 1..N");
      }
      samples.push_back(sample->second);
    }
    const auto summary = summarize(std::move(samples));
    if (!std::isfinite(summary.minimum) || !std::isfinite(summary.mean) ||
        !std::isfinite(summary.maximum) || !std::isfinite(summary.standard_deviation) ||
        summary.minimum > summary.median || summary.median > summary.maximum ||
        summary.mean < summary.minimum || summary.mean > summary.maximum) {
      fail(expectation.name + ": raw values do not form consistent statistic inputs");
    }
    for (const auto& expected_line : expected_report_lines(expectation, group, summary)) {
      if (stdout_text.find(expected_line) == std::string::npos) {
        fail(expectation.name + ": reported statistics do not match the CSV raw values: " +
             expected_line);
      }
    }
  }
}

[[nodiscard]] std::optional<std::filesystem::path> find_argument(
    int argc, char* argv[], std::string_view name) {
  for (int index = 1; index + 1 < argc; ++index) {
    if (std::string_view{argv[index]} == name) return std::filesystem::path{argv[index + 1]};
  }
  return std::nullopt;
}

void validate_named_run(int argc, char* argv[], std::string_view name,
                        const RunExpectation& expectation) {
  const auto csv = find_argument(argc, argv, "--" + std::string{name} + "-csv");
  const auto stdout_path = find_argument(argc, argv, "--" + std::string{name} + "-stdout");
  if (!csv.has_value() || !stdout_path.has_value()) {
    fail("missing CSV/stdout arguments for " + std::string{name});
  }
  validate_run(expectation, *csv, *stdout_path);
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    const auto fair = find_argument(argc, argv, "--fair-csv");
    if (fair.has_value()) {
      validate_named_run(argc, argv, "fair",
                         {.name = "fair",
                          .benchmark = "lhme_fair_storage_benchmark",
                          .style = ReportStyle::Fair,
                          .groups = fair_groups()});
    }
    validate_named_run(argc, argv, "kevin",
                       {.name = "kevin",
                        .benchmark = "lhme_kevin_cpu_benchmark",
                        .groups = kevin_groups()});
    validate_named_run(argc, argv, "hybrid",
                       {.name = "hybrid",
                        .benchmark = "lhme_hybrid_storage_benchmark",
                        .groups = hybrid_groups()});

    const auto table = find_argument(argc, argv, "--table-csv");
    const auto stress = find_argument(argc, argv, "--stress-csv");
    if (table.has_value() || stress.has_value()) {
      validate_named_run(argc, argv, "table",
                         {.name = "table",
                          .benchmark = "lhme_table_memory_benchmark",
                          .groups = table_groups()});
      validate_named_run(argc, argv, "stress",
                         {.name = "stress",
                          .benchmark = "lhme_controlled_stress",
                          .groups = stress_groups()});
    }
  } catch (const std::exception& error) {
    std::cerr << "benchmark CSV end-to-end validation failed: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
