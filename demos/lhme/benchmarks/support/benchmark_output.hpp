#pragma once

#include "benchmark_statistics.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <locale>
#include <optional>
#include <sstream>
#include <span>
#include <string>
#include <string_view>

namespace bruce::lhme::benchmark::output {

struct RawCsvRow {
  std::string_view benchmark;
  std::string_view case_id;
  std::string_view metric;
  std::size_t iteration = 0U;
  std::string_view unit;
  double value = 0.0;
  std::string_view validation;
  std::string_view notes;
};

[[nodiscard]] inline std::string raw_csv_header() {
  return "benchmark,case_id,metric,iteration,unit,value,validation,notes\n";
}

[[nodiscard]] inline std::string quote_csv_field(std::string_view text) {
  std::string quoted{"\""};
  for (const auto character : text) {
    if (character == '\"') quoted += "\"\"";
    else quoted += character;
  }
  quoted += '\"';
  return quoted;
}

[[nodiscard]] inline std::string format_raw_csv_row(const RawCsvRow& row) {
  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << quote_csv_field(row.benchmark) << ',' << quote_csv_field(row.case_id) << ','
         << quote_csv_field(row.metric) << ',' << row.iteration << ','
         << quote_csv_field(row.unit) << ',' << std::fixed << std::setprecision(9)
         << row.value << ',' << quote_csv_field(row.validation) << ','
         << quote_csv_field(row.notes) << '\n';
  return output.str();
}

class RawCsvWriter {
 public:
  explicit RawCsvWriter(const std::filesystem::path& path) {
    std::error_code error;
    if (std::filesystem::exists(path, error) || error) return;
    output_.open(path);
    if (output_.good()) output_ << raw_csv_header();
  }

  [[nodiscard]] bool good() const { return output_.good(); }

  [[nodiscard]] bool write(const RawCsvRow& row) {
    output_ << format_raw_csv_row(row);
    return output_.good();
  }

 private:
  std::ofstream output_;
};

[[nodiscard]] inline std::optional<std::string> format_summary(
    std::span<const double> samples, std::string_view unit) {
  const auto summary = statistics::summarize(samples);
  if (!summary.has_value()) return std::nullopt;

  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << std::fixed << std::setprecision(3) << "median=" << summary->median << ' '
         << unit << "; mean=" << summary->mean << "; min=" << summary->minimum
         << "; max=" << summary->maximum << "; stddev=";
  if (samples.size() == 1U) {
    output << "unavailable (n=1)";
  } else {
    output << summary->standard_deviation;
  }
  return output.str();
}

}  // namespace bruce::lhme::benchmark::output
