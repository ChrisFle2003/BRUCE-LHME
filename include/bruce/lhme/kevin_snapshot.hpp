#pragma once

#include <filesystem>

#include "bruce/lhme/kevin_memory_store.hpp"

namespace bruce::lhme {
// New snapshots use the V2 format and preserve every GridSpec budget field.
// The loader also accepts V1 snapshots, whose omitted budgets are interpreted
// as Budget::unlimited() for backward compatibility.
[[nodiscard]] SnapshotStatus save_snapshot(const KevinMemoryStore& store,
                                           const std::filesystem::path& path);
[[nodiscard]] SnapshotStatus load_snapshot(KevinMemoryStore& store,
                                           const std::filesystem::path& path);
}  // namespace bruce::lhme
