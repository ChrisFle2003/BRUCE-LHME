#include "test.hpp"

#include "bruce/lhme/formula_override_snapshot.hpp"
#include "bruce/lhme/power_offset_kernel.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <csignal>
#include <sys/resource.h>
#endif

namespace {

constexpr std::uint64_t kMaximumSnapshotBytes = 16U * 1024U * 1024U;

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

}  // namespace

TEST(formula_override_snapshot_round_trips_binding_and_value) {
  using namespace bruce::lhme;
  const auto path = std::filesystem::temp_directory_path() /
                    "bruce-formula-override-snapshot-test.bin";
  std::filesystem::remove(path);

  auto source_engine = Engine::create({});
  const auto binding = source_engine.register_formula(make_power_offset_spec(2, 100, 1));
  FormulaOverrideStore source{source_engine};
  CHECK(source.set_override(binding, {.index = 10}, BigInt{-42}) == Status::Confirmed);
  CHECK(save_formula_override_snapshot(source, path) ==
        FormulaOverrideSnapshotStatus::Confirmed);

  auto restored_engine = Engine::create({});
  const auto restored_binding =
      restored_engine.register_formula(make_power_offset_spec(2, 100, 1));
  FormulaOverrideStore restored{restored_engine};
  CHECK(load_formula_override_snapshot(restored, path) ==
        FormulaOverrideSnapshotStatus::Confirmed);
  const auto value = restored.evaluate({.binding = restored_binding,
                                        .address = {.index = 10},
                                        .mode = EvaluationMode::Exact,
                                        .budget = Budget::unlimited()});
  CHECK(value.source == FormulaValueSource::Override);
  CHECK(value.result.exact_integer() == -42);
  std::filesystem::remove(path);
}

TEST(formula_override_snapshot_rejects_bruce_kevin_legacy_magic) {
  using namespace bruce::lhme;
  const auto path = std::filesystem::temp_directory_path() /
                    "bruce-formula-override-legacy-magic.bin";
  {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << "BRUCE-KEVIN-1\n";
  }
  auto engine = Engine::create({});
  const auto binding = engine.register_formula(make_power_offset_spec(2, 100, 1));
  FormulaOverrideStore store{engine};
  CHECK(store.set_override(binding, {.index = 3}, BigInt{7}) == Status::Confirmed);

  CHECK(load_formula_override_snapshot(store, path) ==
        FormulaOverrideSnapshotStatus::InvalidSnapshot);
  CHECK(store.evaluate({.binding = binding,
                        .address = {.index = 3},
                        .mode = EvaluationMode::Exact,
                        .budget = Budget::unlimited()})
            .result.exact_integer() == 7);
  std::filesystem::remove(path);
}

TEST(formula_override_snapshot_round_trips_a_technique_binding) {
  using namespace bruce::lhme;
  const auto path = std::filesystem::temp_directory_path() /
                    "bruce-formula-override-technique-test.bin";
  std::filesystem::remove(path);

  auto source_engine = Engine::create({});
  CHECK(source_engine.register_technique(std::make_shared<PowerOffsetKernel>()));
  const TechniqueBinding binding{{.id = "bruce.power_offset", .version = 1},
                                 encode_power_offset_parameters(2, 100)};
  FormulaOverrideStore source{source_engine};
  CHECK(source.set_override(binding, {.index = 8}, BigInt{13}) == Status::Confirmed);
  CHECK(save_formula_override_snapshot(source, path) ==
        FormulaOverrideSnapshotStatus::Confirmed);

  auto restored_engine = Engine::create({});
  CHECK(restored_engine.register_technique(std::make_shared<PowerOffsetKernel>()));
  FormulaOverrideStore restored{restored_engine};
  CHECK(load_formula_override_snapshot(restored, path) ==
        FormulaOverrideSnapshotStatus::Confirmed);
  const auto value = restored.evaluate({.binding = binding,
                                        .address = {.index = 8},
                                        .mode = EvaluationMode::Exact,
                                        .budget = Budget::unlimited()});
  CHECK(value.source == FormulaValueSource::Override);
  CHECK(value.result.exact_integer() == 13);
  std::filesystem::remove(path);
}

TEST(formula_override_snapshot_rejects_trailing_data_without_mutating_store) {
  using namespace bruce::lhme;
  const auto path = std::filesystem::temp_directory_path() /
                    "bruce-formula-override-trailing-test.bin";
  std::filesystem::remove(path);

  auto source_engine = Engine::create({});
  const auto binding = source_engine.register_formula(make_power_offset_spec(2, 100, 1));
  FormulaOverrideStore source{source_engine};
  CHECK(source.set_override(binding, {.index = 6}, BigInt{17}) == Status::Confirmed);
  CHECK(save_formula_override_snapshot(source, path) ==
        FormulaOverrideSnapshotStatus::Confirmed);
  {
    std::ofstream output(path, std::ios::binary | std::ios::app);
    output << "unexpected trailing data";
  }

  auto target_engine = Engine::create({});
  const auto target_binding =
      target_engine.register_formula(make_power_offset_spec(2, 100, 1));
  FormulaOverrideStore target{target_engine};
  CHECK(target.set_override(target_binding, {.index = 3}, BigInt{7}) ==
        Status::Confirmed);
  CHECK(load_formula_override_snapshot(target, path) ==
        FormulaOverrideSnapshotStatus::InvalidSnapshot);
  CHECK(target.evaluate({.binding = target_binding,
                         .address = {.index = 3},
                         .mode = EvaluationMode::Exact,
                         .budget = Budget::unlimited()})
            .result.exact_integer() == 7);
  std::filesystem::remove(path);
}

TEST(formula_override_snapshot_rejects_an_oversized_input_without_mutating_store) {
  using namespace bruce::lhme;
  const auto path = std::filesystem::temp_directory_path() /
                    "bruce-formula-override-oversized-test.bin";
  std::filesystem::remove(path);
  {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.seekp(static_cast<std::streamoff>(kMaximumSnapshotBytes));
    output.put('\n');
  }

  auto engine = Engine::create({});
  const auto binding = engine.register_formula(make_power_offset_spec(2, 100, 1));
  FormulaOverrideStore store{engine};
  CHECK(store.set_override(binding, {.index = 3}, BigInt{7}) == Status::Confirmed);
  CHECK(load_formula_override_snapshot(store, path) ==
        FormulaOverrideSnapshotStatus::InvalidSnapshot);
  CHECK(store.evaluate({.binding = binding,
                        .address = {.index = 3},
                        .mode = EvaluationMode::Exact,
                        .budget = Budget::unlimited()})
            .result.exact_integer() == 7);
  std::filesystem::remove(path);
}

TEST(formula_override_snapshot_does_not_overwrite_a_legacy_temporary_name) {
  using namespace bruce::lhme;
  const auto path = std::filesystem::temp_directory_path() /
                    "bruce-formula-override-unique-temporary-test.bin";
  const auto legacy_temporary = std::filesystem::path(path.string() + ".tmp");
  std::filesystem::remove(path);
  remove_temporary_files_for(path);
  write_file(legacy_temporary, "sentinel");

  auto engine = Engine::create({});
  const auto binding = engine.register_formula(make_power_offset_spec(2, 100, 1));
  FormulaOverrideStore store{engine};
  CHECK(store.set_override(binding, {.index = 4}, BigInt{9}) == Status::Confirmed);

  const auto status = save_formula_override_snapshot(store, path);
  const auto legacy_unchanged = read_file(legacy_temporary) == "sentinel";
  const auto target_exists = std::filesystem::exists(path);
  remove_temporary_files_for(path);
  std::filesystem::remove(path);

  CHECK(status == FormulaOverrideSnapshotStatus::Confirmed);
  CHECK(legacy_unchanged);
  CHECK(target_exists);
}

TEST(formula_override_snapshot_replaces_an_existing_target) {
  using namespace bruce::lhme;
  const auto path = std::filesystem::temp_directory_path() /
                    "bruce-formula-override-existing-target-test.bin";
  std::filesystem::remove(path);
  remove_temporary_files_for(path);
  write_file(path, "previous snapshot");

  auto source_engine = Engine::create({});
  const auto source_binding =
      source_engine.register_formula(make_power_offset_spec(2, 100, 1));
  FormulaOverrideStore source{source_engine};
  CHECK(source.set_override(source_binding, {.index = 4}, BigInt{9}) ==
        Status::Confirmed);
  const auto save_status = save_formula_override_snapshot(source, path);

  auto restored_engine = Engine::create({});
  const auto restored_binding =
      restored_engine.register_formula(make_power_offset_spec(2, 100, 1));
  FormulaOverrideStore restored{restored_engine};
  const auto load_status = load_formula_override_snapshot(restored, path);
  const auto value = restored.evaluate({.binding = restored_binding,
                                        .address = {.index = 4},
                                        .mode = EvaluationMode::Exact,
                                        .budget = Budget::unlimited()});
  const auto temporaries = temporary_files_for(path);
  remove_temporary_files_for(path);
  std::filesystem::remove(path);

  CHECK(save_status == FormulaOverrideSnapshotStatus::Confirmed);
  CHECK(load_status == FormulaOverrideSnapshotStatus::Confirmed);
  CHECK(value.source == FormulaValueSource::Override);
  CHECK(value.result.exact_integer() == 9);
  CHECK(temporaries.empty());
}

TEST(formula_override_snapshot_cleans_up_after_a_rename_failure) {
  using namespace bruce::lhme;
  const auto path = std::filesystem::temp_directory_path() /
                    "bruce-formula-override-rename-failure-test";
  std::filesystem::remove_all(path);
  remove_temporary_files_for(path);
  CHECK(std::filesystem::create_directory(path));

  auto engine = Engine::create({});
  const auto binding = engine.register_formula(make_power_offset_spec(2, 100, 1));
  FormulaOverrideStore store{engine};
  CHECK(store.set_override(binding, {.index = 4}, BigInt{9}) == Status::Confirmed);
  const auto status = save_formula_override_snapshot(store, path);
  const auto temporaries = temporary_files_for(path);
  remove_temporary_files_for(path);
  std::filesystem::remove(path);

  CHECK(status == FormulaOverrideSnapshotStatus::IoError);
  CHECK(temporaries.empty());
}

#if defined(__unix__) || defined(__APPLE__)
TEST(formula_override_snapshot_cleans_up_after_a_write_failure) {
  using namespace bruce::lhme;
  const auto path = std::filesystem::temp_directory_path() /
                    "bruce-formula-override-write-failure-test.bin";
  std::filesystem::remove(path);
  remove_temporary_files_for(path);
  write_file(path, "previous snapshot");

  auto engine = Engine::create({});
  const auto binding = engine.register_formula(make_power_offset_spec(2, 100, 1));
  FormulaOverrideStore store{engine};
  for (std::int64_t index = 0; index < 200; ++index) {
    CHECK(store.set_override(binding, {.index = index}, BigInt{index}) ==
          Status::Confirmed);
  }

  FormulaOverrideSnapshotStatus status = FormulaOverrideSnapshotStatus::Confirmed;
  {
    ScopedFileSizeLimit limit{1024};
    CHECK(limit.active());
    status = save_formula_override_snapshot(store, path);
  }
  const auto target_unchanged = read_file(path) == "previous snapshot";
  const auto temporaries = temporary_files_for(path);
  remove_temporary_files_for(path);
  std::filesystem::remove(path);

  CHECK(status == FormulaOverrideSnapshotStatus::IoError);
  CHECK(target_unchanged);
  CHECK(temporaries.empty());
}
#endif
