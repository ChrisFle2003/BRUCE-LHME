#include "test.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kExpectedDataRows = 418U;
constexpr std::size_t kExpectedCases = 38U;
constexpr std::size_t kExpectedSamplesPerCase = 11U;

struct CsvField {
  std::string value;
  bool quoted = false;
};

struct Statistics {
  double median = 0.0;
  double mean = 0.0;
  double minimum = 0.0;
  double maximum = 0.0;
  double standard_deviation = 0.0;
};

[[noreturn]] void fail(std::string message) {
  throw std::runtime_error(std::move(message));
}

void require(bool condition, std::string message) {
  if (!condition) fail(std::move(message));
}

[[nodiscard]] std::optional<std::vector<CsvField>> parse_csv_record(
    std::string_view record, std::string& error) {
  if (record.empty()) {
    error = "record is empty";
    return std::nullopt;
  }

  std::vector<CsvField> fields;
  std::size_t position = 0U;
  while (position < record.size()) {
    CsvField field;
    if (record[position] == '"') {
      field.quoted = true;
      ++position;
      bool closed = false;
      while (position < record.size()) {
        const auto character = record[position++];
        if (character != '"') {
          field.value += character;
          continue;
        }
        if (position < record.size() && record[position] == '"') {
          field.value += '"';
          ++position;
          continue;
        }
        closed = true;
        break;
      }
      if (!closed) {
        error = "quoted field is not terminated";
        return std::nullopt;
      }
      if (position < record.size() && record[position] != ',') {
        error = "unexpected character after quoted field";
        return std::nullopt;
      }
    } else {
      while (position < record.size() && record[position] != ',') {
        if (record[position] == '"') {
          error = "unescaped quote in unquoted field";
          return std::nullopt;
        }
        field.value += record[position++];
      }
    }

    fields.push_back(std::move(field));
    if (position == record.size()) break;
    ++position;
    if (position == record.size()) {
      error = "record ends with a delimiter";
      return std::nullopt;
    }
  }
  return fields;
}

[[nodiscard]] bool has_only_digits(std::string_view value) {
  return !value.empty() && std::all_of(value.begin(), value.end(), [](unsigned char character) {
    return character >= '0' && character <= '9';
  });
}

[[nodiscard]] std::optional<std::size_t> parse_iteration(std::string_view value) {
  if (!has_only_digits(value)) return std::nullopt;
  std::size_t parsed = 0U;
  for (const auto character : value) {
    const auto digit = static_cast<std::size_t>(character - '0');
    if (parsed > (std::numeric_limits<std::size_t>::max() - digit) / 10U) {
      return std::nullopt;
    }
    parsed = parsed * 10U + digit;
  }
  return parsed;
}

[[nodiscard]] std::optional<double> parse_fixed_three_decimal(std::string_view value) {
  const auto decimal = value.find('.');
  if (decimal == std::string_view::npos || decimal == 0U ||
      decimal + 4U != value.size()) {
    return std::nullopt;
  }
  const auto whole = value.substr(0U, decimal);
  const auto fraction = value.substr(decimal + 1U);
  if (!has_only_digits(whole) || !has_only_digits(fraction)) return std::nullopt;

  std::uint64_t parsed_whole = 0U;
  for (const auto character : whole) {
    const auto digit = static_cast<std::uint64_t>(character - '0');
    if (parsed_whole > (std::numeric_limits<std::uint64_t>::max() - digit) / 10U) {
      return std::nullopt;
    }
    parsed_whole = parsed_whole * 10U + digit;
  }
  std::uint64_t parsed_fraction = 0U;
  for (const auto character : fraction) {
    parsed_fraction = parsed_fraction * 10U + static_cast<std::uint64_t>(character - '0');
  }
  return static_cast<double>(parsed_whole) + static_cast<double>(parsed_fraction) / 1'000.0;
}

[[nodiscard]] Statistics summarize(std::vector<double> samples) {
  require(!samples.empty(), "statistics requested for an empty sample set");
  std::sort(samples.begin(), samples.end());
  const auto sum = std::accumulate(samples.begin(), samples.end(), 0.0);
  const auto mean = sum / static_cast<double>(samples.size());
  double squared_deviation_sum = 0.0;
  for (const auto sample : samples) {
    const auto delta = sample - mean;
    squared_deviation_sum += delta * delta;
  }
  const auto middle = samples.size() / 2U;
  const auto median = samples.size() % 2U == 0U
      ? (samples[middle - 1U] + samples[middle]) / 2.0
      : samples[middle];
  return {.median = median,
          .mean = mean,
          .minimum = samples.front(),
          .maximum = samples.back(),
          .standard_deviation = std::sqrt(
              squared_deviation_sum / static_cast<double>(samples.size() - 1U))};
}

[[nodiscard]] std::string format_three_decimal(double value) {
  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << std::fixed << std::setprecision(3) << value;
  return output.str();
}

[[nodiscard]] std::string remove_numeric_thousands_separators(std::string text) {
  std::string normalized;
  normalized.reserve(text.size());
  for (std::size_t index = 0U; index < text.size(); ++index) {
    if (text[index] == ',' && index > 0U && index + 1U < text.size() &&
        text[index - 1U] >= '0' && text[index - 1U] <= '9' &&
        text[index + 1U] >= '0' && text[index + 1U] <= '9') {
      continue;
    }
    normalized += text[index];
  }
  return normalized;
}

[[nodiscard]] std::string read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  require(input.good(), "could not open " + path.string());
  std::ostringstream contents;
  contents << input.rdbuf();
  require(input.good() || input.eof(), "could not read " + path.string());
  return contents.str();
}

[[nodiscard]] std::size_t count_occurrences(std::string_view text, std::string_view needle) {
  std::size_t count = 0U;
  std::size_t position = 0U;
  while ((position = text.find(needle, position)) != std::string_view::npos) {
    ++count;
    position += needle.size();
  }
  return count;
}

}  // namespace

TEST(canonical_benchmark_csv_schema_and_report_statistics_are_consistent) {
  const std::filesystem::path raw_path{CANONICAL_BENCHMARK_RAW_PATH};
  const std::filesystem::path report_path{CANONICAL_BENCHMARK_REPORT_PATH};
  std::ifstream input(raw_path);
  require(input.good(), "could not open canonical CSV: " + raw_path.string());

  std::string line;
  require(static_cast<bool>(std::getline(input, line)), "canonical CSV has no header");
  std::string parse_error;
  const auto header = parse_csv_record(line, parse_error);
  require(header.has_value(), "canonical CSV header is invalid: " + parse_error);
  const std::array<std::string_view, 10> expected_header{
      "case_id", "subsystem", "workload", "variant", "input", "iteration", "unit",
      "value", "validation", "notes"};
  require(header->size() == expected_header.size(),
          "canonical CSV header has " + std::to_string(header->size()) + " fields; expected 10");
  for (std::size_t index = 0U; index < expected_header.size(); ++index) {
    require(!(*header)[index].quoted && (*header)[index].value == expected_header[index],
            "canonical CSV header field " + std::to_string(index + 1U) + " is not " +
                std::string{expected_header[index]});
  }

  std::map<std::string, std::map<std::size_t, double>> samples_by_case;
  std::size_t data_rows = 0U;
  std::size_t line_number = 1U;
  while (std::getline(input, line)) {
    ++line_number;
    const auto fields = parse_csv_record(line, parse_error);
    require(fields.has_value(), "canonical CSV line " + std::to_string(line_number) +
                                    " is invalid: " + parse_error);
    require(fields->size() == expected_header.size(),
            "canonical CSV line " + std::to_string(line_number) + " has " +
                std::to_string(fields->size()) + " fields; expected 10");

    for (const auto field_index : {0U, 1U, 2U, 3U, 4U, 6U, 9U}) {
      require((*fields)[field_index].quoted,
              "canonical CSV line " + std::to_string(line_number) + " field " +
                  std::to_string(field_index + 1U) + " must be quoted");
    }
    for (const auto field_index : {5U, 7U, 8U}) {
      require(!(*fields)[field_index].quoted,
              "canonical CSV line " + std::to_string(line_number) + " field " +
                  std::to_string(field_index + 1U) + " must be unquoted");
    }

    const auto& case_id = (*fields)[0U].value;
    require(!case_id.empty(), "canonical CSV line " + std::to_string(line_number) +
                                  " has an empty case_id");
    const auto iteration = parse_iteration((*fields)[5U].value);
    require(iteration.has_value() && *iteration >= 1U && *iteration <= kExpectedSamplesPerCase,
            "canonical CSV line " + std::to_string(line_number) +
                " has an iteration outside 1..11");
    const auto value = parse_fixed_three_decimal((*fields)[7U].value);
    require(value.has_value(), "canonical CSV line " + std::to_string(line_number) +
                                   " has a nonconforming value; expected NNN.NNN");
    require((*fields)[8U].value == "confirmed",
            "canonical CSV line " + std::to_string(line_number) +
                " has validation=" + (*fields)[8U].value + "; expected confirmed");

    const auto [unused, inserted] = samples_by_case[case_id].emplace(*iteration, *value);
    require(inserted, "canonical CSV has duplicate iteration " + std::to_string(*iteration) +
                          " for case " + case_id);
    ++data_rows;
  }
  require(input.eof(), "failed while reading canonical CSV: " + raw_path.string());
  require(data_rows == kExpectedDataRows,
          "canonical CSV has " + std::to_string(data_rows) + " data rows; expected 418");
  require(samples_by_case.size() == kExpectedCases,
          "canonical CSV has " + std::to_string(samples_by_case.size()) +
              " cases; expected 38");

  const auto normalized_report = remove_numeric_thousands_separators(read_file(report_path));
  std::size_t matched_statistics = 0U;
  for (const auto& [case_id, samples] : samples_by_case) {
    require(samples.size() == kExpectedSamplesPerCase,
            "canonical CSV case " + case_id + " has " + std::to_string(samples.size()) +
                " samples; expected 11");
    std::vector<double> ordered_samples;
    ordered_samples.reserve(kExpectedSamplesPerCase);
    for (std::size_t iteration = 1U; iteration <= kExpectedSamplesPerCase; ++iteration) {
      const auto sample = samples.find(iteration);
      require(sample != samples.end(), "canonical CSV case " + case_id +
                                         " is missing iteration " + std::to_string(iteration));
      ordered_samples.push_back(sample->second);
    }

    const auto statistics = summarize(std::move(ordered_samples));
    const auto tuple = "| " + format_three_decimal(statistics.median) + " | " +
        format_three_decimal(statistics.mean) + " | " +
        format_three_decimal(statistics.minimum) + " | " +
        format_three_decimal(statistics.maximum) + " | " +
        format_three_decimal(statistics.standard_deviation) + " |";
    const auto occurrences = count_occurrences(normalized_report, tuple);
    require(occurrences == 1U, "report statistic tuple for case " + case_id +
                                  " occurs " + std::to_string(occurrences) +
                                  " times; expected exactly once: " + tuple);
    ++matched_statistics;
  }
  require(matched_statistics == kExpectedCases,
          "matched " + std::to_string(matched_statistics) + " report statistic tuples; expected 38");
}
