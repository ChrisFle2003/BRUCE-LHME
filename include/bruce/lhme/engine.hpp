#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>

#include "bruce/lhme/cache_equivalence.hpp"
#include "bruce/lhme/cache.hpp"
#include "bruce/lhme/reduction.hpp"
#include "bruce/lhme/technique.hpp"

namespace bruce::lhme {

struct EngineConfig {
  std::size_t cache_max_entries = 0;
  std::size_t cache_max_bytes = 0;
  std::uint32_t worker_count = 1;
  std::uint64_t tile_size = 1;
};

class Engine {
 public:
  [[nodiscard]] static Engine create(EngineConfig config);

  // Non-canonical or invalid specs are returned unchanged but are not
  // registered; evaluating them through this Engine returns InvalidArgument.
  FormulaSpec register_formula(FormulaSpec spec);
  [[nodiscard]] bool register_technique(
      std::shared_ptr<const TechniqueKernel> kernel);
  // Registers an immutable, exact-only cache-equivalence declaration.  An
  // identical registration succeeds; a conflicting registration is rejected.
  [[nodiscard]] bool register_exact_formula_cache_binding(
      ExactFormulaTechniqueCacheBinding binding);
  [[nodiscard]] bool has_technique(const TechniqueIdentity& identity) const;
  [[nodiscard]] EvaluationResult evaluate(const EvaluationRequest& request);
  [[nodiscard]] EvaluationResult evaluate_technique(
      const TechniqueRequest& request);
  [[nodiscard]] RangeResult evaluate_range(const EvaluationRequest& request,
                                           AddressRange range);
  [[nodiscard]] CacheStatistics inspect() const;
  void cancel() noexcept;

 private:
  struct CanonicalRangeIdentity {
    EvaluationMode mode = EvaluationMode::Symbolic;
    std::int64_t start = 0;
    std::int64_t end = -1;

    auto operator==(const CanonicalRangeIdentity&) const -> bool = default;
  };

  struct InflightCancellationGroup {
    std::shared_ptr<std::atomic_bool> signal;
    std::size_t lease_count = 0;
    std::string calculation_key;
    std::string base_key;
    bool range_scope = false;
    std::optional<CanonicalRangeIdentity> range_identity;
    bool modular = false;
    std::optional<BigInt> modulus;
  };

  struct ExactFormulaTechniqueCacheBindingKey {
    std::string technique_id;
    std::uint32_t technique_version = 0;
    std::string accepted_parameters;

    auto operator<=>(const ExactFormulaTechniqueCacheBindingKey&) const = default;
  };

  class EvaluationCancellationLease;

  explicit Engine(EngineConfig config);
  [[nodiscard]] std::shared_ptr<InflightCancellationGroup> begin_evaluation(
      const std::string& group_key, const std::string& base_key,
      const std::string& calculation_key, bool range_scope,
      EvaluationMode mode, bool modular,
      std::optional<BigInt> modulus,
      std::optional<AddressRange> range);
  void end_evaluation(
      const std::shared_ptr<InflightCancellationGroup>& group) noexcept;
  [[nodiscard]] bool is_registered(const FormulaSpec& spec) const;

  EngineConfig config_;
  ResultCache cache_;
  std::map<std::string, FormulaSpec> formulas_;
  std::map<std::string, std::shared_ptr<const TechniqueKernel>> techniques_;
  std::map<ExactFormulaTechniqueCacheBindingKey, FormulaSpec>
      exact_formula_cache_bindings_;
  std::shared_ptr<std::mutex> cancellation_mutex_;
  std::map<std::string, std::shared_ptr<InflightCancellationGroup>>
      cancellation_groups_;
  std::shared_ptr<std::shared_mutex> registry_mutex_;
};

}  // namespace bruce::lhme
