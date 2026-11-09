#include "bruce/lhme/engine.hpp"

#include "bruce/lhme/trace.hpp"

#include <mutex>

#if defined(BRUCE_LHME_TESTING)
#include "engine_formula_cache_hit_finalization_test_barrier.hpp"
#include "engine_formula_cache_hit_post_finalization_test_barrier.hpp"
#include "engine_formula_cache_miss_post_finalization_test_barrier.hpp"
#include "engine_formula_finalization_test_barrier.hpp"
#include "engine_technique_cache_hit_finalization_test_barrier.hpp"
#include "engine_technique_cache_hit_post_finalization_test_barrier.hpp"
#include "engine_technique_cache_miss_post_finalization_test_barrier.hpp"
#include "engine_technique_finalization_test_barrier.hpp"
#include "engine_technique_noncacheable_finalization_test_barrier.hpp"
#include "engine_technique_noncacheable_post_finalization_test_barrier.hpp"
#include "engine_range_finalization_test_barrier.hpp"
#include "engine_range_post_finalization_test_barrier.hpp"
#include "engine_range_test_barrier.hpp"
#endif

namespace bruce::lhme {
namespace {

EvaluationResult cancelled_result(EvaluationMode mode) {
  EvaluationResult result;
  result.status = Status::Cancelled;
  result.mode = mode;
  result.trace = make_trace({{.scope = GateScope::Run,
                              .status = Status::Cancelled,
                              .reason = "engine cancellation requested"}});
  return result;
}

std::string technique_registry_key(const TechniqueIdentity& identity) {
  return std::to_string(identity.id.size()) + ":" + identity.id + ":" +
         std::to_string(identity.version);
}

struct InflightGroupDescriptor {
  std::string key;
  std::string base_key;
  std::string calculation_key;
  bool range_scope = false;
  std::optional<AddressRange> range;
  EvaluationMode mode = EvaluationMode::Symbolic;
  bool modular = false;
  std::optional<BigInt> modulus;
};

std::string inflight_calculation_key(const ComputationIdentity& identity) {
  return "computation:" + std::to_string(identity.value.size()) + ":" +
         identity.value;
}

std::string inflight_evaluation_address_identity(
    FormulaAddress address) {
  return "address:index:" + std::to_string(address.index);
}

std::string inflight_evaluation_address_identity(AddressRange range) {
  return "range:start:" + std::to_string(range.start) +
         ":end:" + std::to_string(range.end);
}

std::string inflight_base_key(const std::string& calculation_key,
                              const std::string& address_identity) {
  return calculation_key + ":" + address_identity;
}

std::string inflight_range_base_key(const std::string& calculation_key,
                                    AddressRange range, EvaluationMode mode) {
  const auto range_key = inflight_base_key(
      calculation_key, inflight_evaluation_address_identity(range));
  return range_key + ":range-mode:" +
         std::to_string(static_cast<int>(mode));
}

std::string inflight_modular_key(const std::string& base_key,
                                 const std::optional<BigInt>& modulus) {
  if (!modulus.has_value()) return base_key + ":modular:none";
  const auto text = modulus->convert_to<std::string>();
  return base_key + ":modular:" + std::to_string(text.size()) + ":" + text;
}

std::string inflight_nonmodular_direct_key(const std::string& base_key) {
  return "direct-nonmodular:" + std::to_string(base_key.size()) + ":" +
         base_key;
}

InflightGroupDescriptor inflight_group_descriptor(
    const ComputationIdentity& identity, const std::string& address_identity,
    bool range_scope, std::optional<AddressRange> range, EvaluationMode mode,
    const std::optional<BigInt>& modulus) {
  InflightGroupDescriptor descriptor;
  descriptor.calculation_key = inflight_calculation_key(identity);
  descriptor.base_key = inflight_base_key(descriptor.calculation_key,
                                          address_identity);
  descriptor.range_scope = range_scope;
  descriptor.range = range;
  descriptor.mode = mode;
  descriptor.modular = mode == EvaluationMode::Modular;
  if (descriptor.modular) {
    descriptor.modulus = modulus;
    descriptor.key = inflight_modular_key(descriptor.base_key,
                                          descriptor.modulus);
  } else {
    descriptor.key = descriptor.base_key;
  }
  return descriptor;
}

InflightGroupDescriptor inflight_group_descriptor(
    const EvaluationRequest& request) {
  return inflight_group_descriptor(canonical_computation_identity(request.spec),
                                   inflight_evaluation_address_identity(
                                       request.address),
                                   false, std::nullopt, request.mode,
                                   request.modulus);
}

InflightGroupDescriptor inflight_group_descriptor(
    const EvaluationRequest& request, AddressRange range) {
  auto descriptor = inflight_group_descriptor(
      canonical_computation_identity(request.spec),
      inflight_evaluation_address_identity(range), true, range, request.mode,
      request.modulus);
  descriptor.base_key =
      inflight_range_base_key(descriptor.calculation_key, range, request.mode);
  descriptor.key = descriptor.modular
                       ? inflight_modular_key(descriptor.base_key,
                                              descriptor.modulus)
                       : descriptor.base_key;
  return descriptor;
}

ComputationIdentity fallback_computation_identity(
    const TechniqueRequest& request, const TechniqueKernel& kernel) {
  const auto canonical_parameters = kernel.canonical_parameters(request);
  const auto& parameters =
      canonical_parameters.has_value() && !canonical_parameters->empty() &&
              canonical_parameters->size() <= 4096
          ? *canonical_parameters
          : request.parameters;
  return {.value = "technique:" + technique_registry_key(request.identity) +
                   ":parameters:" +
                   std::to_string(parameters.size()) + ":" + parameters};
}

InflightGroupDescriptor inflight_group_descriptor(
    const TechniqueRequest& request, const TechniqueKernel& kernel) {
  const auto identity = kernel.canonical_computation_identity(request);
  return inflight_group_descriptor(
      identity.has_value() && valid_computation_identity(*identity)
          ? *identity
          : fallback_computation_identity(request, kernel),
      inflight_evaluation_address_identity(request.address), false,
      std::nullopt, request.mode, request.modulus);
}

EvaluationResult technique_rejection(EvaluationMode mode, const char* reason) {
  EvaluationResult result;
  result.status = Status::InvalidArgument;
  result.mode = mode;
  result.trace = make_trace({{.scope = GateScope::Branch,
                              .status = result.status,
                              .reason = reason}});
  return result;
}

EvaluationResult technique_internal_error(EvaluationMode mode) {
  EvaluationResult result;
  result.status = Status::InternalError;
  result.mode = mode;
  result.trace = make_trace({{.scope = GateScope::Branch,
                              .status = result.status,
                              .reason = "technique kernel threw an exception"}});
  return result;
}

EvaluationResult within_trace_budget(EvaluationResult result,
                                     const Budget& budget) {
  enforce_trace_budget(result, budget.max_trace_bytes);
  return result;
}

bool same_formula_spec(const FormulaSpec& left, const FormulaSpec& right) {
  return left.operation == right.operation && left.base == right.base &&
         left.scale == right.scale &&
         left.nonnegative_index == right.nonnegative_index &&
         left.version == right.version &&
         left.specification_hash == right.specification_hash;
}

void append_cache_event(EvaluationResult& result, const char* reason) {
  std::vector<GateEvent> events;
  if (result.trace != nullptr) {
    events = result.trace->events;
  }
  events.push_back({.scope = GateScope::Cache,
                    .status = result.status,
                    .reason = reason});
  result.trace = make_trace(std::move(events));
}

void replace_with_cross_api_cache_trace(EvaluationResult& result) {
  result.trace = make_trace({{.scope = GateScope::Cache,
                              .status = result.status,
                              .reason = "cross-api exact cache hit"}});
}

void append_engine_cancellation(EvaluationResult& result) {
  std::vector<GateEvent> events;
  if (result.trace != nullptr) {
    events = result.trace->events;
  }
  result.status = Status::Cancelled;
  result.integer.reset();
  result.metric_bit_length.reset();
  result.symbolic_expression.reset();
  events.push_back({.scope = GateScope::Run,
                    .status = result.status,
                    .reason = "engine cancellation requested"});
  result.trace = make_trace(std::move(events));
}

std::uint64_t bit_length(const BigInt& value) {
  if (value == 0) return 0;
  const auto magnitude = value < 0 ? -value : value;
  return static_cast<std::uint64_t>(boost::multiprecision::msb(magnitude)) + 1U;
}

std::uint64_t estimated_operations(std::int64_t index) {
  if (index <= 0) return 0;
  std::uint64_t value = static_cast<std::uint64_t>(index);
  std::uint64_t bits = 0;
  while (value > 0) {
    ++bits;
    value >>= 1U;
  }
  return bits * 2U;
}

std::uint64_t bits_to_bytes(std::uint64_t bits) {
  return bits / 8U + (bits % 8U == 0 ? 0U : 1U);
}

template <typename Request>
bool cache_budget_rejects(const Request& request, std::uint64_t value_bit_length) {
  return value_bit_length > request.budget.max_bit_length ||
         estimated_operations(request.address.index) > request.budget.max_operations ||
         bits_to_bytes(value_bit_length) > request.budget.max_temporary_bytes;
}

template <typename Request>
EvaluationResult value_exceeds_budget(const Request& request,
                                      std::uint64_t value_bit_length) {
  EvaluationResult result;
  result.status = Status::BudgetExceeded;
  result.mode = request.mode;
  result.metric_bit_length = value_bit_length;
  result.trace = make_trace({{.scope = GateScope::Publish,
                              .status = result.status,
                              .reason = "result exceeded caller resource budget"}});
  return result;
}

std::optional<std::uint64_t> result_bit_length(const EvaluationResult& result) {
  if (result.integer.has_value()) return bit_length(*result.integer);
  return result.metric_bit_length;
}

}  // namespace

class Engine::EvaluationCancellationLease {
 public:
  EvaluationCancellationLease(Engine& engine, const std::string& group_key,
                              const std::string& base_key,
                              const std::string& calculation_key,
                              bool range_scope, EvaluationMode mode,
                              bool modular,
                              std::optional<BigInt> modulus,
                              std::optional<AddressRange> range)
      : engine_(engine),
        group_(engine_.begin_evaluation(group_key, base_key, calculation_key,
                                        range_scope, mode, modular,
                                        std::move(modulus), std::move(range))),
        signal_(group_->signal) {}

  ~EvaluationCancellationLease() { engine_.end_evaluation(group_); }

  [[nodiscard]] const std::shared_ptr<const std::atomic_bool>& signal() const {
    return signal_;
  }

 private:
  Engine& engine_;
  std::shared_ptr<InflightCancellationGroup> group_;
  std::shared_ptr<const std::atomic_bool> signal_;
};

Engine::Engine(EngineConfig config)
    : config_(config),
      cache_({.max_entries = config.cache_max_entries,
              .max_bytes = config.cache_max_bytes}),
      cancellation_mutex_(std::make_shared<std::mutex>()),
      registry_mutex_(std::make_shared<std::shared_mutex>()) {}

Engine Engine::create(EngineConfig config) { return Engine(config); }

FormulaSpec Engine::register_formula(FormulaSpec spec) {
  if (!valid_formula_spec(spec)) return spec;
  const std::unique_lock lock(*registry_mutex_);
  formulas_.insert_or_assign(canonical_formula_identity(spec), spec);
  return spec;
}

bool Engine::register_technique(std::shared_ptr<const TechniqueKernel> kernel) {
  try {
    if (kernel == nullptr) return false;
    const auto identity = kernel->identity();
    if (!valid_technique_identity(identity)) return false;
    const auto key = technique_registry_key(identity);
    const std::unique_lock lock(*registry_mutex_);
    if (techniques_.contains(key)) return false;
    techniques_.emplace(key, std::move(kernel));
    return true;
  } catch (...) {
    return false;
  }
}

bool Engine::register_exact_formula_cache_binding(
    ExactFormulaTechniqueCacheBinding binding) {
  try {
    if (!valid_technique_identity(binding.technique) ||
        binding.accepted_parameters.empty() ||
        binding.accepted_parameters.size() > 4096 ||
        !valid_formula_spec(binding.formula)) {
      return false;
    }

    const ExactFormulaTechniqueCacheBindingKey key{
        .technique_id = binding.technique.id,
        .technique_version = binding.technique.version,
        .accepted_parameters = std::move(binding.accepted_parameters)};
    const auto technique_key = technique_registry_key(binding.technique);
    const auto formula_key = canonical_formula_identity(binding.formula);

    const std::unique_lock lock(*registry_mutex_);
    if (!techniques_.contains(technique_key)) return false;
    const auto registered_formula = formulas_.find(formula_key);
    if (registered_formula == formulas_.end() ||
        !same_formula_spec(registered_formula->second, binding.formula)) {
      return false;
    }

    const auto existing = exact_formula_cache_bindings_.find(key);
    if (existing == exact_formula_cache_bindings_.end()) {
      exact_formula_cache_bindings_.emplace(std::move(key),
                                             std::move(binding.formula));
      return true;
    }
    // Idempotent registration is accepted, but a previously accepted binding
    // is immutable and cannot be redirected to another FormulaSpec.
    return same_formula_spec(existing->second, binding.formula);
  } catch (...) {
    return false;
  }
}

bool Engine::has_technique(const TechniqueIdentity& identity) const {
  const std::shared_lock lock(*registry_mutex_);
  return valid_technique_identity(identity) &&
         techniques_.contains(technique_registry_key(identity));
}

bool Engine::is_registered(const FormulaSpec& spec) const {
  if (!valid_formula_spec(spec)) return false;
  const std::shared_lock lock(*registry_mutex_);
  const auto it = formulas_.find(canonical_formula_identity(spec));
  return it != formulas_.end() && it->second.version == spec.version &&
         it->second.operation == spec.operation && it->second.base == spec.base &&
         it->second.scale == spec.scale &&
         it->second.nonnegative_index == spec.nonnegative_index;
}

std::shared_ptr<Engine::InflightCancellationGroup> Engine::begin_evaluation(
    const std::string& group_key, const std::string& base_key,
    const std::string& calculation_key, bool range_scope, EvaluationMode mode,
    bool modular,
    std::optional<BigInt> modulus, std::optional<AddressRange> range) {
  const std::lock_guard lock(*cancellation_mutex_);
  std::optional<CanonicalRangeIdentity> range_identity;
  if (range.has_value()) {
    range_identity = CanonicalRangeIdentity{.mode = mode,
                                             .start = range->start,
                                             .end = range->end};
  }

  std::shared_ptr<InflightCancellationGroup> unique_direct_modular_group;
  bool ambiguous_direct_modular_groups = false;
  std::shared_ptr<InflightCancellationGroup>
      unique_cancelled_direct_modular_group;
  bool ambiguous_cancelled_direct_modular_groups = false;
  if (!range_scope && !modular) {
    for (const auto& [_, candidate] : cancellation_groups_) {
      if (candidate->range_scope || !candidate->modular ||
          candidate->base_key != base_key) {
        continue;
      }
      if (!ambiguous_direct_modular_groups &&
          unique_direct_modular_group == nullptr) {
        unique_direct_modular_group = candidate;
      } else if (!ambiguous_direct_modular_groups &&
                 unique_direct_modular_group != candidate) {
        ambiguous_direct_modular_groups = true;
        unique_direct_modular_group.reset();
      }
      if (!candidate->signal->load(std::memory_order_acquire)) {
        continue;
      }
      if (!ambiguous_cancelled_direct_modular_groups &&
          unique_cancelled_direct_modular_group == nullptr) {
        unique_cancelled_direct_modular_group = candidate;
      } else if (!ambiguous_cancelled_direct_modular_groups &&
                 unique_cancelled_direct_modular_group != candidate) {
        ambiguous_cancelled_direct_modular_groups = true;
        unique_cancelled_direct_modular_group.reset();
      }
    }
  }
  const auto nonmodular_direct_key =
      inflight_nonmodular_direct_key(base_key);

  if (!range_scope && !modular) {
    if (const auto direct = cancellation_groups_.find(nonmodular_direct_key);
        direct != cancellation_groups_.end()) {
      ++direct->second->lease_count;
      return direct->second;
    }
    if (unique_cancelled_direct_modular_group != nullptr) {
      ++unique_cancelled_direct_modular_group->lease_count;
      return unique_cancelled_direct_modular_group;
    }
    if (unique_direct_modular_group != nullptr) {
      ++unique_direct_modular_group->lease_count;
      return unique_direct_modular_group;
    }
  }
  if (!ambiguous_direct_modular_groups) {
    if (const auto direct = cancellation_groups_.find(group_key);
             direct != cancellation_groups_.end()) {
      ++direct->second->lease_count;
      return direct->second;
    }
  }

  const auto unique_compatible_group = [&](bool require_range_scope) {
    std::shared_ptr<InflightCancellationGroup> match;
    for (const auto& [_, candidate] : cancellation_groups_) {
      if (candidate->calculation_key != calculation_key ||
          (require_range_scope && !candidate->range_scope) ||
          (range_scope && candidate->range_scope &&
           candidate->range_identity != range_identity) ||
          (modular && candidate->modular && candidate->modulus != modulus)) {
        continue;
      }
      if (match == nullptr) {
        match = candidate;
      } else if (match != candidate) {
        return std::shared_ptr<InflightCancellationGroup>{};
      }
    }
    return match;
  };

  if (range_scope) {
    if (const auto existing = unique_compatible_group(false);
        existing != nullptr) {
      if (modular && !existing->modular) {
        existing->modular = true;
        existing->modulus = std::move(modulus);
        cancellation_groups_.emplace(base_key, existing);
      }
      cancellation_groups_.emplace(group_key, existing);
      ++existing->lease_count;
      return existing;
    }
  } else if (const auto range_group = unique_compatible_group(true);
             range_group != nullptr) {
    if (modular && !range_group->modular) {
      range_group->modular = true;
      range_group->modulus = std::move(modulus);
    }
    cancellation_groups_.emplace(group_key, range_group);
    cancellation_groups_.emplace(base_key, range_group);
    ++range_group->lease_count;
    return range_group;
  }

  std::shared_ptr<InflightCancellationGroup> group;
  if (modular) {
    const auto base = cancellation_groups_.find(base_key);
    if (base != cancellation_groups_.end() && !base->second->modular) {
      group = base->second;
      group->modular = true;
      group->modulus = std::move(modulus);
      cancellation_groups_.emplace(group_key, group);
    } else {
      group = std::make_shared<InflightCancellationGroup>();
      group->signal = std::make_shared<std::atomic_bool>(false);
      group->calculation_key = calculation_key;
      group->base_key = base_key;
      group->range_scope = range_scope;
      group->range_identity = range_identity;
      group->modular = true;
      group->modulus = std::move(modulus);
      cancellation_groups_.emplace(group_key, group);
      if (base == cancellation_groups_.end()) {
        cancellation_groups_.emplace(base_key, group);
      }
    }
  } else {
    group = std::make_shared<InflightCancellationGroup>();
    group->signal = std::make_shared<std::atomic_bool>(false);
    group->calculation_key = calculation_key;
    group->base_key = base_key;
    group->range_scope = range_scope;
    group->range_identity = range_identity;
    cancellation_groups_.emplace(
        ambiguous_direct_modular_groups ? nonmodular_direct_key : group_key,
        group);
  }
  ++group->lease_count;
  return group;
}

void Engine::end_evaluation(
    const std::shared_ptr<InflightCancellationGroup>& group) noexcept {
  const std::lock_guard lock(*cancellation_mutex_);
  --group->lease_count;
  if (group->lease_count != 0) return;
  for (auto it = cancellation_groups_.begin();
       it != cancellation_groups_.end();) {
    if (it->second == group) {
      it = cancellation_groups_.erase(it);
    } else {
      ++it;
    }
  }
}

EvaluationResult Engine::evaluate(const EvaluationRequest& request) {
  const auto group = inflight_group_descriptor(request);
  EvaluationCancellationLease cancellation(*this, group.key, group.base_key,
                                           group.calculation_key,
                                           group.range_scope, group.mode,
                                           group.modular,
                                           group.modulus, group.range);
  const auto& cancellation_signal = cancellation.signal();
  if (cancellation_signal->load() || request.cancellation_requested) {
    return within_trace_budget(cancelled_result(request.mode), request.budget);
  }
  if (!is_registered(request.spec)) {
    EvaluationResult result;
    result.status = Status::InvalidArgument;
    result.mode = request.mode;
    result.trace = make_trace({{.scope = GateScope::Branch,
                                .status = result.status,
                                .reason = "formula is not registered"}});
    return within_trace_budget(std::move(result), request.budget);
  }

  const bool cacheable = request.mode == EvaluationMode::Exact ||
                         request.mode == EvaluationMode::Metric;
  CacheKey key{};
  if (cacheable) {
    key = request.mode == EvaluationMode::Exact
              ? CacheKey::exact(request.spec, request.address.index)
              : CacheKey::metric(request.spec, request.address.index);
    if (const auto cached = cache_.lookup_with_producer(key); cached.has_value()) {
      auto result = cached->result;
      const bool cross_api_hit =
          request.mode == EvaluationMode::Exact &&
          cached->producer == CacheEntryProducer::TechniqueApi;
      if (cross_api_hit) replace_with_cross_api_cache_trace(result);
      if (request.mode == EvaluationMode::Exact && result.integer.has_value() &&
          cache_budget_rejects(request, bit_length(*result.integer))) {
        auto rejection = value_exceeds_budget(request, bit_length(*result.integer));
        append_cache_event(rejection, "cache rejected by caller budget");
        return within_trace_budget(std::move(rejection), request.budget);
      }
#if defined(BRUCE_LHME_TESTING)
      test_support::wait_for_engine_formula_cache_hit_finalization_test_barrier();
#endif
      // This snapshot linearizes the cached formula result against
      // Engine::cancel(). A cancellation observed here changes only the local
      // copy; the existing cache entry remains available to later evaluations.
      if (cancellation_signal->load(std::memory_order_acquire)) {
        append_engine_cancellation(result);
        return within_trace_budget(std::move(result), request.budget);
      }
#if defined(BRUCE_LHME_TESTING)
      test_support::wait_for_engine_formula_cache_hit_post_finalization_test_barrier();
#endif
      if (!cross_api_hit) append_cache_event(result, "cache hit");
      return within_trace_budget(std::move(result), request.budget);
    }
  }

  auto effective_request = request;
  effective_request.cancellation_signal = cancellation_signal;
  EvaluationResult result = bruce::lhme::evaluate(effective_request);
#if defined(BRUCE_LHME_TESTING)
  test_support::wait_for_engine_formula_finalization_test_barrier();
#endif
  // This snapshot linearizes the formula result against Engine::cancel().
  // A cancellation observed here prevents the core result from being cached;
  // a later cancellation belongs to a result already finalized for this call.
  if (cancellation_signal->load(std::memory_order_acquire)) {
    append_engine_cancellation(result);
    return within_trace_budget(std::move(result), request.budget);
  }
#if defined(BRUCE_LHME_TESTING)
  test_support::wait_for_engine_formula_cache_miss_post_finalization_test_barrier();
#endif
  if (cacheable) {
    append_cache_event(result, "cache miss");
    cache_.admit(key, result, CacheEntryProducer::FormulaApi);
  }
  return within_trace_budget(std::move(result), request.budget);
}

EvaluationResult Engine::evaluate_technique(const TechniqueRequest& request) {
  if (!valid_technique_identity(request.identity)) {
    return within_trace_budget(
        technique_rejection(request.mode, "technique identity is invalid"),
        request.budget);
  }
  if (request.parameters.empty() || request.parameters.size() > 4096) {
    return within_trace_budget(
        technique_rejection(request.mode, "technique parameters are invalid"),
        request.budget);
  }
  std::shared_ptr<const TechniqueKernel> kernel;
  std::optional<FormulaSpec> exact_cache_formula;
  {
    const std::shared_lock lock(*registry_mutex_);
    const auto found = techniques_.find(technique_registry_key(request.identity));
    if (found == techniques_.end()) {
      return within_trace_budget(
          technique_rejection(request.mode, "technique is not registered"),
          request.budget);
    }
    kernel = found->second;
    if (request.mode == EvaluationMode::Exact) {
      const ExactFormulaTechniqueCacheBindingKey binding_key{
          .technique_id = request.identity.id,
          .technique_version = request.identity.version,
          .accepted_parameters = request.parameters};
      if (const auto binding = exact_formula_cache_bindings_.find(binding_key);
          binding != exact_formula_cache_bindings_.end()) {
        exact_cache_formula = binding->second;
      }
    }
  }

  try {
    const auto group = inflight_group_descriptor(request, *kernel);
    EvaluationCancellationLease cancellation(*this, group.key, group.base_key,
                                             group.calculation_key,
                                             group.range_scope, group.mode,
                                             group.modular,
                                             group.modulus, group.range);
    const auto& cancellation_signal = cancellation.signal();
    if (cancellation_signal->load()) {
      return within_trace_budget(cancelled_result(request.mode), request.budget);
    }
    auto effective_request = request;
    effective_request.cancellation_signal = cancellation_signal;
    const bool cacheable = request.mode == EvaluationMode::Exact ||
                           request.mode == EvaluationMode::Metric;
    CacheKey key{};
    if (cacheable) {
      key = exact_cache_formula.has_value()
                ? CacheKey::exact(*exact_cache_formula, request.address.index)
                : CacheKey::technique(request.identity, request.parameters,
                                      request.address.index, request.mode);
      if (const auto cached = cache_.lookup_with_producer(key);
          cached.has_value()) {
        auto result = cached->result;
        const bool cross_api_hit =
            exact_cache_formula.has_value() &&
            cached->producer == CacheEntryProducer::FormulaApi;
        if (cross_api_hit) replace_with_cross_api_cache_trace(result);
        if (request.mode == EvaluationMode::Exact && result.integer.has_value() &&
            cache_budget_rejects(request, bit_length(*result.integer))) {
          auto rejection = value_exceeds_budget(request, bit_length(*result.integer));
          append_cache_event(rejection, "cache rejected by caller budget");
          return within_trace_budget(std::move(rejection), request.budget);
        }
        if (cancellation_signal->load()) {
          append_engine_cancellation(result);
          return within_trace_budget(std::move(result), request.budget);
        }
#if defined(BRUCE_LHME_TESTING)
        test_support::wait_for_engine_technique_cache_hit_finalization_test_barrier();
#endif
        // This snapshot linearizes the cached technique result against
        // Engine::cancel(). A cancellation observed here changes only the
        // local copy; the existing cache entry remains available to later
        // evaluations.
        if (cancellation_signal->load(std::memory_order_acquire)) {
          append_engine_cancellation(result);
          return within_trace_budget(std::move(result), request.budget);
        }
#if defined(BRUCE_LHME_TESTING)
        test_support::wait_for_engine_technique_cache_hit_post_finalization_test_barrier();
#endif
        if (!cross_api_hit) append_cache_event(result, "cache hit");
        return within_trace_budget(std::move(result), request.budget);
      }
    }

    EvaluationResult result = kernel->evaluate(effective_request);
    if (cancellation_signal->load()) {
      append_engine_cancellation(result);
      return within_trace_budget(std::move(result), request.budget);
    }
    if (result.status == Status::Confirmed) {
      if (const auto bits = result_bit_length(result);
          bits.has_value() && cache_budget_rejects(request, *bits)) {
        return within_trace_budget(value_exceeds_budget(request, *bits),
                                   request.budget);
      }
    }
    if (cacheable) {
#if defined(BRUCE_LHME_TESTING)
      test_support::wait_for_engine_technique_finalization_test_barrier();
#endif
      // This snapshot linearizes the cache-miss technique result against
      // Engine::cancel(). A cancellation observed here prevents the kernel
      // result from entering the cache; a later cancellation belongs to a
      // result already finalized for this invocation.
      if (cancellation_signal->load(std::memory_order_acquire)) {
        append_engine_cancellation(result);
        return within_trace_budget(std::move(result), request.budget);
      }
#if defined(BRUCE_LHME_TESTING)
      test_support::wait_for_engine_technique_cache_miss_post_finalization_test_barrier();
#endif
      append_cache_event(result, "cache miss");
      cache_.admit(key, result, CacheEntryProducer::TechniqueApi);
    }
#if defined(BRUCE_LHME_TESTING)
    if (!cacheable) {
      test_support::wait_for_engine_technique_noncacheable_finalization_test_barrier();
    }
#endif
    // This snapshot linearizes a non-cacheable technique result against
    // Engine::cancel(). A later cancellation belongs to a result already
    // finalized for this invocation.
    if (!cacheable && cancellation_signal->load(std::memory_order_acquire)) {
      append_engine_cancellation(result);
    }
#if defined(BRUCE_LHME_TESTING)
    if (!cacheable) {
      test_support::wait_for_engine_technique_noncacheable_post_finalization_test_barrier();
    }
#endif
    return within_trace_budget(std::move(result), request.budget);
  } catch (...) {
    return within_trace_budget(technique_internal_error(request.mode), request.budget);
  }
}

RangeResult Engine::evaluate_range(const EvaluationRequest& request,
                                   AddressRange range) {
  const auto group = inflight_group_descriptor(request, range);
  EvaluationCancellationLease cancellation(*this, group.key, group.base_key,
                                           group.calculation_key,
                                           group.range_scope, group.mode,
                                           group.modular,
                                           group.modulus, group.range);
  const auto& cancellation_signal = cancellation.signal();
  if (cancellation_signal->load()) {
    return {.status = Status::Cancelled};
  }
  if (!is_registered(request.spec)) {
    return {.status = Status::InvalidArgument};
  }
  auto effective_request = request;
  effective_request.cancellation_signal = cancellation_signal;
#if defined(BRUCE_LHME_TESTING)
  test_support::wait_for_range_evaluation_test_barrier();
#endif
  auto result = bruce::lhme::evaluate_range(
      effective_request, range,
      {.worker_count = config_.worker_count, .tile_size = config_.tile_size});
#if defined(BRUCE_LHME_TESTING)
  test_support::wait_for_engine_range_finalization_test_barrier();
#endif
  // This snapshot linearizes the range result against Engine::cancel().
  // A cancellation observed here changes the scheduler result; a later one
  // belongs to a result that is already finalized for this invocation.
  if (cancellation_signal->load(std::memory_order_acquire)) {
    result.status = Status::Cancelled;
    result.candidates.clear();
  }
#if defined(BRUCE_LHME_TESTING)
  test_support::wait_for_engine_range_post_finalization_test_barrier();
#endif
  return result;
}

CacheStatistics Engine::inspect() const { return cache_.statistics(); }

void Engine::cancel() noexcept {
  const std::lock_guard lock(*cancellation_mutex_);
  for (const auto& [_, group] : cancellation_groups_) {
    group->signal->store(true);
    group->signal->notify_all();
  }
}

}  // namespace bruce::lhme
