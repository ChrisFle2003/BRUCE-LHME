#include "bruce/lhme/kevin_snapshot.hpp"

#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace {
constexpr char kMagicV1[] = "BRUCE-KEVIN-1\n";
constexpr char kMagicV2[] = "BRUCE-KEVIN-2\n";
constexpr std::uintmax_t kMaximumSnapshotBytes = 16U * 1024U * 1024U;

enum class SnapshotFileReadStatus { Confirmed, IoError, TooLarge };

std::uint64_t checksum(std::string_view bytes) { std::uint64_t hash=14695981039346656037ULL; for (const auto byte:bytes) { hash^=static_cast<unsigned char>(byte); hash*=1099511628211ULL; } return hash; }

SnapshotFileReadStatus read_snapshot_file(const std::filesystem::path& path,
                                          std::string& contents) {
  std::error_code error;
  const auto file_size = std::filesystem::file_size(path, error);
  if (error) return SnapshotFileReadStatus::IoError;
  if (file_size > kMaximumSnapshotBytes) return SnapshotFileReadStatus::TooLarge;

  std::ifstream input(path, std::ios::binary);
  if (!input) return SnapshotFileReadStatus::IoError;

  contents.clear();
  contents.reserve(static_cast<std::size_t>(file_size));
  std::array<char, 4096> buffer{};
  while (input.read(buffer.data(), static_cast<std::streamsize>(buffer.size())) ||
         input.gcount() > 0) {
    const auto count = static_cast<std::size_t>(input.gcount());
    if (count > kMaximumSnapshotBytes - contents.size()) {
      return SnapshotFileReadStatus::TooLarge;
    }
    contents.append(buffer.data(), count);
  }
  return input.eof() ? SnapshotFileReadStatus::Confirmed
                     : SnapshotFileReadStatus::IoError;
}

bool parse_checksum_suffix(std::string_view suffix, std::uint64_t& value) {
  if (suffix.size() < 2 || suffix.back() != '\n') return false;
  const auto digits = suffix.substr(0, suffix.size() - 1U);
  if (digits.empty() || (digits.size() > 1U && digits.front() == '0')) return false;
  const auto [end, error] = std::from_chars(digits.data(),
                                             digits.data() + digits.size(), value);
  return error == std::errc{} && end == digits.data() + digits.size();
}

bool write_string(std::ostream& out, const std::string& value) {
  const auto size = value.size();
  out << size << ':' << value << '\n';
  return static_cast<bool>(out);
}

bool read_string(std::istream& in, std::string& value) {
  std::size_t size = 0; char colon = 0;
  if (!(in >> size) || !(in >> colon) || colon != ':' || size > 4096) return false;
  value.resize(size); in.read(value.data(), static_cast<std::streamsize>(size));
  char newline = 0; in.get(newline); return static_cast<bool>(in) && newline == '\n';
}

std::optional<std::filesystem::path> make_temporary_path(
    const std::filesystem::path& path) {
  static std::atomic<std::uint64_t> sequence = 0;
  std::random_device random;
  for (std::size_t attempt = 0; attempt < 16; ++attempt) {
    const auto nonce = std::to_string(random()) + "." +
                       std::to_string(std::chrono::steady_clock::now()
                                          .time_since_epoch()
                                          .count()) +
                       "." + std::to_string(sequence.fetch_add(1));
    const auto temporary = path.parent_path() /
                           (path.filename().string() + ".tmp." + nonce);
    std::error_code error;
    if (!std::filesystem::exists(temporary, error) && !error) return temporary;
  }
  return std::nullopt;
}

class TemporaryFileGuard {
 public:
  explicit TemporaryFileGuard(std::filesystem::path path) : path_(std::move(path)) {}

  ~TemporaryFileGuard() {
    if (!active_) return;
    std::error_code ignored;
    std::filesystem::remove(path_, ignored);
  }

  [[nodiscard]] const std::filesystem::path& path() const { return path_; }
  void release() noexcept { active_ = false; }

 private:
  std::filesystem::path path_;
  bool active_ = true;
};

bool replace_file(const std::filesystem::path& temporary,
                  const std::filesystem::path& destination) {
#ifdef _WIN32
  return MoveFileExW(temporary.c_str(), destination.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
  std::error_code error;
  std::filesystem::rename(temporary, destination, error);
  return !error;
#endif
}
}

namespace bruce::lhme {

SnapshotStatus save_snapshot(const KevinMemoryStore& store, const std::filesystem::path& path) {
  try {
    const auto temporary = make_temporary_path(path);
    if (!temporary.has_value()) return SnapshotStatus::IoError;
    TemporaryFileGuard cleanup(*temporary);

    std::ofstream out(cleanup.path(), std::ios::binary | std::ios::trunc);
    if (!out) return SnapshotStatus::IoError;
    const auto snapshot = store.snapshot();
    out << kMagicV2 << snapshot.grids.size() << ' ' << snapshot.overrides.size() << '\n';
    for (const auto& grid : snapshot.grids) {
      if (!write_string(out, grid.identity.id) || !write_string(out, grid.technique.id) ||
          !write_string(out, grid.canonical_parameters)) return SnapshotStatus::IoError;
      out << grid.identity.version << ' ' << grid.technique.version << ' '
          << static_cast<unsigned>(grid.max_depth) << ' '
          << grid.default_budget.max_bit_length << ' '
          << grid.default_budget.max_operations << ' '
          << grid.default_budget.max_temporary_bytes << ' '
          << grid.default_budget.max_trace_bytes << ' '
          << grid.default_budget.max_candidate_count << '\n';
    }
    for (const auto& item : snapshot.overrides) {
      if (!write_string(out, item.identity.id) ||
          !write_string(out, item.value.convert_to<std::string>())) {
        return SnapshotStatus::IoError;
      }
      out << item.identity.version << ' ' << item.address.depth();
      for (const auto step : item.address.steps()) {
        out << ' ' << unsigned(step.x) << ' ' << unsigned(step.y);
      }
      out << '\n';
    }
    out.close();
    if (!out) return SnapshotStatus::IoError;

    std::ifstream payload_input(cleanup.path(), std::ios::binary);
    const std::string payload{std::istreambuf_iterator<char>(payload_input), {}};
    if (!payload_input.good() && !payload_input.eof()) return SnapshotStatus::IoError;
    std::ofstream checksum_output(cleanup.path(), std::ios::binary | std::ios::app);
    checksum_output << "CHECKSUM " << checksum(payload) << '\n';
    checksum_output.close();
    if (!checksum_output) return SnapshotStatus::IoError;

    if (!replace_file(cleanup.path(), path)) return SnapshotStatus::IoError;
    cleanup.release();
    return SnapshotStatus::Confirmed;
  } catch (...) {
    return SnapshotStatus::IoError;
  }
}

SnapshotStatus load_snapshot(KevinMemoryStore& store, const std::filesystem::path& path) {
  std::string file;
  switch (read_snapshot_file(path, file)) {
    case SnapshotFileReadStatus::Confirmed:
      break;
    case SnapshotFileReadStatus::IoError:
      return SnapshotStatus::IoError;
    case SnapshotFileReadStatus::TooLarge:
      return SnapshotStatus::InvalidSnapshot;
  }
  const auto marker=file.rfind("CHECKSUM ");
  if (marker==std::string::npos || marker==0 || file[marker-1] != '\n') return SnapshotStatus::InvalidSnapshot;
  std::uint64_t stored=0;
  if (!parse_checksum_suffix(std::string_view{file}.substr(marker+9), stored) ||
      checksum(file.substr(0,marker)) != stored) return SnapshotStatus::InvalidSnapshot;
  std::istringstream in(file.substr(0,marker));
  std::string magic(sizeof(kMagicV2) - 1, '\0'); in.read(magic.data(), magic.size());
  const bool is_v1 = magic == kMagicV1;
  const bool is_v2 = magic == kMagicV2;
  if (!is_v1 && !is_v2) return SnapshotStatus::InvalidSnapshot;
  std::size_t grids = 0, overrides = 0;
  if (!(in >> grids >> overrides) || grids > 100000 || overrides > 100000) return SnapshotStatus::InvalidSnapshot;
  in.ignore(1); KevinSnapshot snapshot;
  for (std::size_t i=0;i<grids;++i) {
    GridSpec grid; unsigned depth=0;
    if (!read_string(in,grid.identity.id) || !read_string(in,grid.technique.id) || !read_string(in,grid.canonical_parameters) ||
        !(in>>grid.identity.version>>grid.technique.version>>depth) || depth>10) return SnapshotStatus::InvalidSnapshot;
    if (is_v2 &&
        !(in >> grid.default_budget.max_bit_length >>
          grid.default_budget.max_operations >>
          grid.default_budget.max_temporary_bytes >>
          grid.default_budget.max_trace_bytes >>
          grid.default_budget.max_candidate_count)) {
      return SnapshotStatus::InvalidSnapshot;
    }
    grid.max_depth=static_cast<std::uint8_t>(depth); in.ignore(1); snapshot.grids.push_back(std::move(grid));
  }
  for (std::size_t i=0;i<overrides;++i) {
    GridIdentity identity; std::string number; std::size_t depth=0;
    if (!read_string(in,identity.id) || !read_string(in,number) || !(in>>identity.version>>depth) || depth>10) return SnapshotStatus::InvalidSnapshot;
    std::vector<GridStep> steps; for(std::size_t j=0;j<depth;++j){unsigned x,y; if(!(in>>x>>y)||x>7||y>7)return SnapshotStatus::InvalidSnapshot; steps.push_back({static_cast<std::uint8_t>(x),static_cast<std::uint8_t>(y)});}
    in.ignore(1); if (number.empty() || (number.size()>1 && number[0]=='0') || (number.size()>2 && number[0]=='-' && number[1]=='0')) return SnapshotStatus::InvalidSnapshot;
    BigInt value; try { value=BigInt{number}; } catch (...) { return SnapshotStatus::InvalidSnapshot; }
    const auto address=GridAddress::from_steps(steps); if(!address) return SnapshotStatus::InvalidSnapshot;
    snapshot.overrides.push_back({std::move(identity), *address, std::move(value)});
  }
  if (in.peek()!=std::char_traits<char>::eof()) return SnapshotStatus::InvalidSnapshot;
  return store.replace_snapshot(snapshot);
}

}  // namespace bruce::lhme
