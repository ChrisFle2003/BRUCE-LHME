#include "test.hpp"
#include "bruce/lhme/kevin_snapshot.hpp"
#include "bruce/lhme/power_offset_kernel.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <csignal>
#include <sys/resource.h>
#endif

namespace {

constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
constexpr std::uint64_t kMaximumSnapshotBytes = 16U * 1024U * 1024U;

std::uint64_t checksum(std::string_view bytes) {
  std::uint64_t hash = kFnvOffset;
  for (const auto byte : bytes) {
    hash ^= static_cast<unsigned char>(byte);
    hash *= kFnvPrime;
  }
  return hash;
}

void write_string(std::ostream& output, std::string_view value) {
  output << value.size() << ':' << value << '\n';
}

void write_legacy_v1_snapshot(const std::filesystem::path& path) {
  std::ostringstream payload;
  payload << "BRUCE-KEVIN-1\n1 0\n";
  write_string(payload, "bruce.snapshot.v1");
  write_string(payload, "bruce.power_offset");
  write_string(payload, "base=1:2;scale=3:100");
  payload << "1 1 1\n";

  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output << payload.str() << "CHECKSUM " << checksum(payload.str()) << '\n';
}

void write_file(const std::filesystem::path& path, std::string_view contents) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output << contents;
}

std::string read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input), {}};
}

std::vector<std::filesystem::path> temporary_files_for(
    const std::filesystem::path& path) {
  const auto prefix = path.filename().string() + ".tmp";
  std::vector<std::filesystem::path> files;
  for (const auto& entry : std::filesystem::directory_iterator(path.parent_path())) {
    if (entry.path().filename().string().starts_with(prefix)) files.push_back(entry.path());
  }
  return files;
}

void remove_temporary_files_for(const std::filesystem::path& path) {
  for (const auto& temporary : temporary_files_for(path)) {
    std::filesystem::remove(temporary);
  }
}

#if defined(__unix__) || defined(__APPLE__)
class ScopedFileSizeLimit {
 public:
  explicit ScopedFileSizeLimit(rlim_t bytes) {
    if (getrlimit(RLIMIT_FSIZE, &original_) != 0) return;
    auto limited = original_;
    limited.rlim_cur = bytes;
    if (setrlimit(RLIMIT_FSIZE, &limited) != 0) return;
    previous_handler_ = std::signal(SIGXFSZ, SIG_IGN);
    active_ = previous_handler_ != SIG_ERR;
  }

  ~ScopedFileSizeLimit() {
    if (active_) std::signal(SIGXFSZ, previous_handler_);
    if (original_.rlim_max != 0) setrlimit(RLIMIT_FSIZE, &original_);
  }

  [[nodiscard]] bool active() const { return active_; }

 private:
  struct rlimit original_ {};
  void (*previous_handler_)(int) = SIG_ERR;
  bool active_ = false;
};
#endif

bruce::lhme::GridSpec test_grid(std::string id) {
  return {{.id = std::move(id), .version = 1},
          {.id = "bruce.power_offset", .version = 1},
          bruce::lhme::encode_power_offset_parameters(2, 100), 3};
}

}  // namespace

TEST(kevin_snapshot_roundtrips_sparse_override) {
  using namespace bruce::lhme;
  const auto path = std::filesystem::temp_directory_path() / "bruce-kevin-snapshot-test.bin";
  std::filesystem::remove(path);
  auto engine = Engine::create({});
  CHECK(engine.register_technique(std::make_shared<PowerOffsetKernel>()));
  GridSpec grid{{.id="bruce.snapshot",.version=1}, {.id="bruce.power_offset",.version=1}, encode_power_offset_parameters(2,100), 1};
  const std::vector<GridStep> steps{{.x=1,.y=1}};
  const auto address = *GridAddress::from_steps(steps);
  KevinMemoryStore source{engine};
  CHECK(source.register_grid(grid));
  CHECK(source.set_override(grid.identity,address,BigInt{-42}) == Status::Confirmed);
  CHECK(save_snapshot(source,path) == SnapshotStatus::Confirmed);
  KevinMemoryStore restored{engine};
  CHECK(load_snapshot(restored,path) == SnapshotStatus::Confirmed);
  CHECK(restored.get(grid.identity,address,EvaluationMode::Exact,Budget::unlimited()).result.exact_integer() == -42);
  std::filesystem::remove(path);
}

TEST(kevin_snapshot_roundtrips_every_nondefault_grid_budget_field) {
  using namespace bruce::lhme;
  const auto path = std::filesystem::temp_directory_path() /
                    "bruce-kevin-budget-v2-snapshot-test.bin";
  std::filesystem::remove(path);

  auto engine = Engine::create({});
  CHECK(engine.register_technique(std::make_shared<PowerOffsetKernel>()));
  const Budget expected_budget{.max_bit_length = 111,
                               .max_operations = 222,
                               .max_temporary_bytes = 333,
                               .max_trace_bytes = 444,
                               .max_candidate_count = 555};
  const GridSpec grid{{.id = "bruce.snapshot.budget", .version = 1},
                      {.id = "bruce.power_offset", .version = 1},
                      encode_power_offset_parameters(2, 100), 1, expected_budget};

  KevinMemoryStore source{engine};
  CHECK(source.register_grid(grid));
  CHECK(save_snapshot(source, path) == SnapshotStatus::Confirmed);

  KevinMemoryStore restored{engine};
  CHECK(load_snapshot(restored, path) == SnapshotStatus::Confirmed);
  const auto restored_snapshot = restored.snapshot();
  CHECK(restored_snapshot.grids.size() == 1);
  const auto& actual_budget = restored_snapshot.grids.front().default_budget;
  CHECK(actual_budget.max_bit_length == expected_budget.max_bit_length);
  CHECK(actual_budget.max_operations == expected_budget.max_operations);
  CHECK(actual_budget.max_temporary_bytes == expected_budget.max_temporary_bytes);
  CHECK(actual_budget.max_trace_bytes == expected_budget.max_trace_bytes);
  CHECK(actual_budget.max_candidate_count == expected_budget.max_candidate_count);
  std::filesystem::remove(path);
}

TEST(kevin_snapshot_loads_v1_with_the_historical_unlimited_budget_default) {
  using namespace bruce::lhme;
  const auto path = std::filesystem::temp_directory_path() /
                    "bruce-kevin-v1-compatibility-snapshot-test.bin";
  std::filesystem::remove(path);
  write_legacy_v1_snapshot(path);

  auto engine = Engine::create({});
  CHECK(engine.register_technique(std::make_shared<PowerOffsetKernel>()));
  KevinMemoryStore store{engine};
  CHECK(load_snapshot(store, path) == SnapshotStatus::Confirmed);
  const auto snapshot = store.snapshot();
  CHECK(snapshot.grids.size() == 1);
  CHECK(snapshot.grids.front().default_budget.max_bit_length ==
        Budget::unlimited().max_bit_length);
  CHECK(snapshot.grids.front().default_budget.max_operations ==
        Budget::unlimited().max_operations);
  CHECK(snapshot.grids.front().default_budget.max_temporary_bytes ==
        Budget::unlimited().max_temporary_bytes);
  CHECK(snapshot.grids.front().default_budget.max_trace_bytes ==
        Budget::unlimited().max_trace_bytes);
  CHECK(snapshot.grids.front().default_budget.max_candidate_count ==
        Budget::unlimited().max_candidate_count);
  std::filesystem::remove(path);
}

TEST(kevin_snapshot_rejects_bad_magic_without_mutating_store) {
  using namespace bruce::lhme;
  const auto path = std::filesystem::temp_directory_path() / "bruce-kevin-bad-snapshot.bin";
  { std::ofstream out(path, std::ios::binary); out << "not-a-bruce-snapshot"; }
  auto engine = Engine::create({});
  CHECK(engine.register_technique(std::make_shared<PowerOffsetKernel>()));
  GridSpec grid{{.id="bruce.snapshot.bad",.version=1}, {.id="bruce.power_offset",.version=1}, encode_power_offset_parameters(2,100), 1};
  const auto address = *GridAddress::from_steps(std::vector<GridStep>{{.x=1,.y=1}});
  KevinMemoryStore store{engine}; CHECK(store.register_grid(grid));
  CHECK(store.set_override(grid.identity,address,BigInt{7}) == Status::Confirmed);
  CHECK(load_snapshot(store,path) == SnapshotStatus::InvalidSnapshot);
  CHECK(store.get(grid.identity,address,EvaluationMode::Exact,Budget::unlimited()).result.exact_integer() == 7);
  std::filesystem::remove(path);
}

TEST(kevin_snapshot_rejects_changed_payload_by_checksum) {
  using namespace bruce::lhme;
  const auto path = std::filesystem::temp_directory_path() / "bruce-kevin-checksum-test.bin";
  auto engine = Engine::create({}); CHECK(engine.register_technique(std::make_shared<PowerOffsetKernel>()));
  GridSpec grid{{.id="bruce.checksum",.version=1}, {.id="bruce.power_offset",.version=1}, encode_power_offset_parameters(2,100), 1};
  KevinMemoryStore store{engine}; CHECK(store.register_grid(grid)); CHECK(save_snapshot(store,path)==SnapshotStatus::Confirmed);
  { std::fstream file(path, std::ios::in|std::ios::out|std::ios::binary); file.seekp(0); file.put('X'); }
  CHECK(load_snapshot(store,path)==SnapshotStatus::InvalidSnapshot);
  std::filesystem::remove(path);
}

TEST(kevin_snapshot_rejects_data_after_a_valid_checksum_without_mutating_store) {
  using namespace bruce::lhme;
  const auto path = std::filesystem::temp_directory_path() /
                    "bruce-kevin-trailing-checksum-data-test.bin";
  std::filesystem::remove(path);

  auto engine = Engine::create({});
  CHECK(engine.register_technique(std::make_shared<PowerOffsetKernel>()));
  const GridSpec grid{{.id = "bruce.checksum.trailing", .version = 1},
                      {.id = "bruce.power_offset", .version = 1},
                      encode_power_offset_parameters(2, 100), 1};
  const auto address = *GridAddress::from_steps(
      std::vector<GridStep>{{.x = 1, .y = 1}});
  KevinMemoryStore source{engine};
  CHECK(source.register_grid(grid));
  CHECK(save_snapshot(source, path) == SnapshotStatus::Confirmed);
  {
    std::ofstream output(path, std::ios::binary | std::ios::app);
    output << " unexpected";
  }

  KevinMemoryStore target{engine};
  CHECK(target.register_grid(grid));
  CHECK(target.set_override(grid.identity, address, BigInt{7}) == Status::Confirmed);
  CHECK(load_snapshot(target, path) == SnapshotStatus::InvalidSnapshot);
  CHECK(target.get(grid.identity, address, EvaluationMode::Exact,
                   Budget::unlimited()).result.exact_integer() == 7);
  std::filesystem::remove(path);
}

TEST(kevin_snapshot_rejects_an_oversized_input_without_mutating_store) {
  using namespace bruce::lhme;
  const auto path = std::filesystem::temp_directory_path() /
                    "bruce-kevin-oversized-snapshot-test.bin";
  std::filesystem::remove(path);
  {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.seekp(static_cast<std::streamoff>(kMaximumSnapshotBytes));
    output.put('\n');
  }

  auto engine = Engine::create({});
  CHECK(engine.register_technique(std::make_shared<PowerOffsetKernel>()));
  const GridSpec grid{{.id = "bruce.oversized", .version = 1},
                      {.id = "bruce.power_offset", .version = 1},
                      encode_power_offset_parameters(2, 100), 1};
  const auto address = *GridAddress::from_steps(
      std::vector<GridStep>{{.x = 1, .y = 1}});
  KevinMemoryStore store{engine};
  CHECK(store.register_grid(grid));
  CHECK(store.set_override(grid.identity, address, BigInt{7}) == Status::Confirmed);
  CHECK(load_snapshot(store, path) == SnapshotStatus::InvalidSnapshot);
  CHECK(store.get(grid.identity, address, EvaluationMode::Exact,
                  Budget::unlimited()).result.exact_integer() == 7);
  std::filesystem::remove(path);
}

TEST(kevin_snapshot_does_not_overwrite_a_legacy_temporary_name) {
  using namespace bruce::lhme;
  const auto path = std::filesystem::temp_directory_path() /
                    "bruce-kevin-unique-temporary-test.bin";
  const auto legacy_temporary = std::filesystem::path(path.string() + ".tmp");
  std::filesystem::remove(path);
  remove_temporary_files_for(path);
  write_file(legacy_temporary, "sentinel");

  auto engine = Engine::create({});
  CHECK(engine.register_technique(std::make_shared<PowerOffsetKernel>()));
  const auto grid = test_grid("bruce.kevin.unique");
  const auto address = *GridAddress::from_steps(std::vector<GridStep>{{.x = 1, .y = 1}});
  KevinMemoryStore store{engine};
  CHECK(store.register_grid(grid));
  CHECK(store.set_override(grid.identity, address, BigInt{9}) == Status::Confirmed);

  const auto status = save_snapshot(store, path);
  const auto legacy_unchanged = read_file(legacy_temporary) == "sentinel";
  const auto target_exists = std::filesystem::exists(path);
  remove_temporary_files_for(path);
  std::filesystem::remove(path);

  CHECK(status == SnapshotStatus::Confirmed);
  CHECK(legacy_unchanged);
  CHECK(target_exists);
}

TEST(kevin_snapshot_replaces_an_existing_target) {
  using namespace bruce::lhme;
  const auto path = std::filesystem::temp_directory_path() /
                    "bruce-kevin-existing-target-test.bin";
  std::filesystem::remove(path);
  remove_temporary_files_for(path);
  write_file(path, "previous snapshot");

  auto engine = Engine::create({});
  CHECK(engine.register_technique(std::make_shared<PowerOffsetKernel>()));
  const auto grid = test_grid("bruce.kevin.existing");
  const auto address = *GridAddress::from_steps(std::vector<GridStep>{{.x = 1, .y = 1}});
  KevinMemoryStore source{engine};
  CHECK(source.register_grid(grid));
  CHECK(source.set_override(grid.identity, address, BigInt{9}) == Status::Confirmed);
  const auto save_status = save_snapshot(source, path);

  KevinMemoryStore restored{engine};
  const auto load_status = load_snapshot(restored, path);
  const auto value = restored.get(grid.identity, address, EvaluationMode::Exact,
                                  Budget::unlimited());
  const auto temporaries = temporary_files_for(path);
  remove_temporary_files_for(path);
  std::filesystem::remove(path);

  CHECK(save_status == SnapshotStatus::Confirmed);
  CHECK(load_status == SnapshotStatus::Confirmed);
  CHECK(value.result.exact_integer() == 9);
  CHECK(temporaries.empty());
}

TEST(kevin_snapshot_cleans_up_after_a_rename_failure) {
  using namespace bruce::lhme;
  const auto path = std::filesystem::temp_directory_path() /
                    "bruce-kevin-rename-failure-test";
  std::filesystem::remove_all(path);
  remove_temporary_files_for(path);
  CHECK(std::filesystem::create_directory(path));

  auto engine = Engine::create({});
  CHECK(engine.register_technique(std::make_shared<PowerOffsetKernel>()));
  const auto grid = test_grid("bruce.kevin.rename");
  const auto address = *GridAddress::from_steps(std::vector<GridStep>{{.x = 1, .y = 1}});
  KevinMemoryStore store{engine};
  CHECK(store.register_grid(grid));
  CHECK(store.set_override(grid.identity, address, BigInt{9}) == Status::Confirmed);
  const auto status = save_snapshot(store, path);
  const auto temporaries = temporary_files_for(path);
  remove_temporary_files_for(path);
  std::filesystem::remove(path);

  CHECK(status == SnapshotStatus::IoError);
  CHECK(temporaries.empty());
}

#if defined(__unix__) || defined(__APPLE__)
TEST(kevin_snapshot_cleans_up_after_a_write_failure) {
  using namespace bruce::lhme;
  const auto path = std::filesystem::temp_directory_path() /
                    "bruce-kevin-write-failure-test.bin";
  std::filesystem::remove(path);
  remove_temporary_files_for(path);
  write_file(path, "previous snapshot");

  auto engine = Engine::create({});
  CHECK(engine.register_technique(std::make_shared<PowerOffsetKernel>()));
  const auto grid = test_grid("bruce.kevin.write.failure");
  KevinMemoryStore store{engine};
  CHECK(store.register_grid(grid));
  for (std::int64_t index = 0; index < 200; ++index) {
    const std::vector<GridStep> steps{{.x = static_cast<std::uint8_t>(index & 7),
                                       .y = static_cast<std::uint8_t>((index >> 3) & 7)},
                                      {.x = static_cast<std::uint8_t>((index >> 6) & 7),
                                       .y = static_cast<std::uint8_t>((index >> 9) & 7)}};
    const auto address = *GridAddress::from_steps(steps);
    CHECK(store.set_override(grid.identity, address, BigInt{index}) == Status::Confirmed);
  }

  SnapshotStatus status = SnapshotStatus::Confirmed;
  {
    ScopedFileSizeLimit limit{1024};
    CHECK(limit.active());
    status = save_snapshot(store, path);
  }
  const auto target_unchanged = read_file(path) == "previous snapshot";
  const auto temporaries = temporary_files_for(path);
  remove_temporary_files_for(path);
  std::filesystem::remove(path);

  CHECK(status == SnapshotStatus::IoError);
  CHECK(target_unchanged);
  CHECK(temporaries.empty());
}
#endif
