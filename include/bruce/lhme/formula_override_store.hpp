#pragma once

#include <atomic>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <variant>
#include <vector>

#include "bruce/lhme/engine.hpp"
#include "bruce/lhme/streaming.hpp"

namespace bruce::lhme {

struct TechniqueBinding {
  TechniqueIdentity identity;
  std::string canonical_parameters;
  Budget default_budget = Budget::unlimited();
};

using FormulaBinding = std::variant<FormulaSpec, TechniqueBinding>;

[[nodiscard]] bool valid_formula_binding(const FormulaBinding& binding);
[[nodiscard]] std::string canonical_formula_binding_identity(
    const FormulaBinding& binding);

struct FormulaOverrideRequest {
  FormulaBinding binding;
  FormulaAddress address;
  EvaluationMode mode = EvaluationMode::Symbolic;
  Budget budget = Budget::unlimited();
  std::optional<BigInt> modulus;
  bool cancellation_requested = false;
  std::shared_ptr<const std::atomic_bool> cancellation_signal;
};

enum class FormulaValueSource { Override, Computed };

struct FormulaValue {
  EvaluationResult result;
  FormulaValueSource source = FormulaValueSource::Computed;
};

// A range request is always evaluated exactly after its optional modular
// prefilter. The binding and budget are shared by every address in the range.
// Cancellation is cooperative: it is observed before dispatch, between
// addresses or tiles, and before publishing retained results. It does not
// preempt a Formula or Technique evaluation that is already running.
struct FormulaStreamingRequest {
  FormulaBinding binding;
  Budget budget = Budget::unlimited();
  bool cancellation_requested = false;
  std::shared_ptr<const std::atomic_bool> cancellation_signal;
};

struct FormulaValueAtAddress {
  FormulaAddress address;
  FormulaValue value;
};

struct FormulaStreamingRangeResult {
  Status status = Status::InternalError;
  GlobalSummary summary;
  std::vector<FormulaValueAtAddress> retained_candidates;
  RetentionMode effective_retention_mode = RetentionMode::SummaryOnly;
  std::uint64_t filtered_out_count = 0;
  std::uint64_t exact_evaluation_count = 0;
};

struct FormulaOverrideRecord {
  FormulaBinding binding;
  FormulaAddress address;
  BigInt value;
};

struct FormulaOverrideSnapshot {
  std::vector<FormulaOverrideRecord> overrides;
};

enum class FormulaOverrideSnapshotStatus {
  Confirmed,
  IoError,
  InvalidSnapshot,
  CapacityExceeded,
};

// These limits apply to the deterministic payload accounting used by this
// store: the canonical override key plus the magnitude bytes of the BigInt
// value (zero occupies one value byte). They do not claim to cap allocator or
// process memory. Defaults preserve the historical unbounded Store behaviour.
struct FormulaOverrideStoreConfig {
  std::size_t max_override_count = std::numeric_limits<std::size_t>::max();
  std::uint64_t max_override_storage_bytes =
      std::numeric_limits<std::uint64_t>::max();
  std::uint64_t max_override_value_bytes =
      std::numeric_limits<std::uint64_t>::max();
};

struct FormulaOverrideStoreStatistics {
  std::size_t override_count = 0;
  std::uint64_t override_storage_bytes = 0;
  std::uint64_t computed_exact_evaluation_count = 0;
  CacheStatistics engine_cache;
};

class FormulaOverrideStore {
 public:
  explicit FormulaOverrideStore(Engine& engine,
                                FormulaOverrideStoreConfig config = {});

  [[nodiscard]] Status set_override(const FormulaBinding& binding,
                                    FormulaAddress address, BigInt value);
  [[nodiscard]] bool remove_override(const FormulaBinding& binding,
                                     FormulaAddress address);
  [[nodiscard]] FormulaValue evaluate(const FormulaOverrideRequest& request);
  // An observed cancellation returns Cancelled and no retained candidates.
  // The observation points are the cooperative stream boundaries documented
  // by FormulaStreamingRequest.
  [[nodiscard]] FormulaStreamingRangeResult evaluate_streaming_range(
      const FormulaStreamingRequest& request, AddressRange range,
      StreamingOptions options);
  [[nodiscard]] FormulaOverrideSnapshot snapshot() const;
  [[nodiscard]] FormulaOverrideSnapshotStatus replace_snapshot(
      const FormulaOverrideSnapshot& snapshot);
  [[nodiscard]] FormulaOverrideStoreStatistics inspect() const;

 private:
  [[nodiscard]] bool has_registered_binding(const FormulaBinding& binding) const;

  Engine& engine_;
  FormulaOverrideStoreConfig config_;
  std::map<std::string, FormulaOverrideRecord> overrides_;
  std::uint64_t override_storage_bytes_ = 0;
  std::atomic<std::uint64_t> computed_exact_evaluation_count_ = 0;
  std::shared_ptr<std::shared_mutex> mutex_;
};

}  // namespace bruce::lhme
