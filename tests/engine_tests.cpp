#include "test.hpp"

#include "engine_range_test_barrier.hpp"
#include "engine_range_finalization_test_barrier.hpp"
#include "engine_range_post_finalization_test_barrier.hpp"
#include "engine_formula_finalization_test_barrier.hpp"
#include "engine_formula_cache_miss_post_finalization_test_barrier.hpp"
#include "engine_formula_cache_hit_finalization_test_barrier.hpp"
#include "engine_formula_cache_hit_post_finalization_test_barrier.hpp"
#include "engine_technique_finalization_test_barrier.hpp"
#include "engine_technique_noncacheable_finalization_test_barrier.hpp"
#include "engine_technique_noncacheable_post_finalization_test_barrier.hpp"
#include "engine_technique_cache_miss_post_finalization_test_barrier.hpp"
#include "engine_technique_cache_hit_finalization_test_barrier.hpp"
#include "engine_technique_cache_hit_post_finalization_test_barrier.hpp"
#include "scheduler_candidate_test_barrier.hpp"
#include "scheduler_final_tile_aggregation_test_barrier.hpp"
#include "scheduler_published_candidate_test_barrier.hpp"
#include "scheduler_tile_test_barrier.hpp"

#include "bruce/lhme/engine.hpp"
#include "bruce/lhme/fibonacci_kernel.hpp"
#include "bruce/lhme/power_offset_kernel.hpp"
#include "bruce/lhme/trace.hpp"

#include <atomic>
#include <future>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>

namespace {

class SignalProbeKernel final : public bruce::lhme::TechniqueKernel {
 public:
  [[nodiscard]] bruce::lhme::TechniqueIdentity identity() const override {
    return {.id = "test.signal_probe", .version = 1};
  }

  [[nodiscard]] bruce::lhme::EvaluationResult evaluate(
      const bruce::lhme::TechniqueRequest& request) const override {
    received_signal = request.cancellation_signal;
    return {.status = request.cancellation_signal != nullptr &&
                      request.cancellation_signal->load()
                  ? bruce::lhme::Status::Cancelled
                  : bruce::lhme::Status::Confirmed,
            .mode = request.mode};
  }

  mutable std::shared_ptr<const std::atomic_bool> received_signal;
};

class OversizedExactKernel final : public bruce::lhme::TechniqueKernel {
 public:
  [[nodiscard]] bruce::lhme::TechniqueIdentity identity() const override {
    return {.id = "test.oversized", .version = 1};
  }

  [[nodiscard]] bruce::lhme::EvaluationResult evaluate(
      const bruce::lhme::TechniqueRequest& request) const override {
    return {.status = bruce::lhme::Status::Confirmed,
            .mode = request.mode,
            .integer = bruce::lhme::BigInt{1024}};
  }
};

class TraceProducingKernel final : public bruce::lhme::TechniqueKernel {
 public:
  [[nodiscard]] bruce::lhme::TechniqueIdentity identity() const override {
    return {.id = "test.trace_producing", .version = 1};
  }

  [[nodiscard]] bruce::lhme::EvaluationResult evaluate(
      const bruce::lhme::TechniqueRequest& request) const override {
    return {.status = bruce::lhme::Status::Confirmed,
            .mode = request.mode,
            .integer = bruce::lhme::BigInt{7},
            .trace = bruce::lhme::make_trace({{
                .scope = bruce::lhme::GateScope::Publish,
                .status = bruce::lhme::Status::Confirmed,
                .reason = "kernel result",
            }})};
  }
};

class ThrowingIdentityKernel final : public bruce::lhme::TechniqueKernel {
 public:
  [[nodiscard]] bruce::lhme::TechniqueIdentity identity() const override {
    throw std::runtime_error("identity failed");
  }

  [[nodiscard]] bruce::lhme::EvaluationResult evaluate(
      const bruce::lhme::TechniqueRequest&) const override {
    return {};
  }
};

class CancellationTraceKernel final : public bruce::lhme::TechniqueKernel {
 public:
  CancellationTraceKernel()
      : entered_(std::make_shared<std::atomic_bool>(false)) {}

  [[nodiscard]] bruce::lhme::TechniqueIdentity identity() const override {
    return {.id = "test.cancellation_trace", .version = 1};
  }

  [[nodiscard]] bruce::lhme::EvaluationResult evaluate(
      const bruce::lhme::TechniqueRequest& request) const override {
    entered_->store(true);
    while (!request.cancellation_signal->load()) {
      std::this_thread::yield();
    }
    return {.status = bruce::lhme::Status::Cancelled, .mode = request.mode};
  }

  std::shared_ptr<std::atomic_bool> entered_;
};

class GroupCancellationKernel final : public bruce::lhme::TechniqueKernel {
 public:
  explicit GroupCancellationKernel(
      std::shared_ptr<std::atomic_uint> entered_evaluations)
      : entered_evaluations_(std::move(entered_evaluations)) {}

  [[nodiscard]] bruce::lhme::TechniqueIdentity identity() const override {
    return {.id = "test.group_cancellation", .version = 1};
  }

  [[nodiscard]] bruce::lhme::EvaluationResult evaluate(
      const bruce::lhme::TechniqueRequest& request) const override {
    if (request.cancellation_signal == nullptr) {
      return {.status = bruce::lhme::Status::InternalError, .mode = request.mode};
    }

    entered_evaluations_->fetch_add(1, std::memory_order_release);
    entered_evaluations_->notify_all();
    request.cancellation_signal->wait(false, std::memory_order_acquire);
    return {.status = bruce::lhme::Status::Cancelled, .mode = request.mode};
  }

 private:
  std::shared_ptr<std::atomic_uint> entered_evaluations_;
};

struct GroupLifetimeCancellationState {
  std::atomic_uint entered_evaluations{0};
  std::atomic_bool held_evaluation_observed_cancellation{false};
  std::atomic_bool fresh_evaluation_entered{false};
  std::atomic_bool release_held_evaluation{false};
};

class GroupLifetimeCancellationKernel final : public bruce::lhme::TechniqueKernel {
 public:
  explicit GroupLifetimeCancellationKernel(
      std::shared_ptr<GroupLifetimeCancellationState> state)
      : state_(std::move(state)) {}

  [[nodiscard]] bruce::lhme::TechniqueIdentity identity() const override {
    return {.id = "test.group_lifetime_cancellation", .version = 1};
  }

  [[nodiscard]] std::optional<bruce::lhme::ComputationIdentity>
  canonical_computation_identity(
      const bruce::lhme::TechniqueRequest&) const override {
    return bruce::lhme::canonical_computation_identity(
        bruce::lhme::make_power_offset_spec(2, 100, 1));
  }

  [[nodiscard]] bruce::lhme::EvaluationResult evaluate(
      const bruce::lhme::TechniqueRequest& request) const override {
    if (request.cancellation_signal == nullptr) {
      return {.status = bruce::lhme::Status::InternalError, .mode = request.mode};
    }
    if (request.parameters != "return_after_cancel" &&
        request.parameters != "hold_after_cancel" &&
        request.parameters != "return_confirmed" &&
        request.parameters != "hold_without_cancel") {
      return {.status = bruce::lhme::Status::InvalidArgument, .mode = request.mode};
    }

    state_->entered_evaluations.fetch_add(1, std::memory_order_release);
    state_->entered_evaluations.notify_all();
    if (request.parameters == "return_confirmed") {
      return {.status = bruce::lhme::Status::Confirmed, .mode = request.mode,
              .integer = bruce::lhme::BigInt{1}};
    }
    if (request.parameters == "hold_without_cancel") {
      state_->fresh_evaluation_entered.store(true, std::memory_order_release);
      state_->fresh_evaluation_entered.notify_all();
      while (!state_->release_held_evaluation.load(std::memory_order_acquire)) {
        state_->release_held_evaluation.wait(false, std::memory_order_relaxed);
      }
      return {.status = bruce::lhme::Status::Confirmed, .mode = request.mode,
              .integer = bruce::lhme::BigInt{2}};
    }
    request.cancellation_signal->wait(false, std::memory_order_acquire);
    if (request.parameters == "hold_after_cancel") {
      state_->held_evaluation_observed_cancellation.store(
          true, std::memory_order_release);
      state_->held_evaluation_observed_cancellation.notify_all();
      while (!state_->release_held_evaluation.load(std::memory_order_acquire)) {
        state_->release_held_evaluation.wait(false, std::memory_order_relaxed);
      }
    }
    return {.status = bruce::lhme::Status::Cancelled, .mode = request.mode};
  }

 private:
  std::shared_ptr<GroupLifetimeCancellationState> state_;
};

class CanonicalParameterAliasKernel final : public bruce::lhme::TechniqueKernel {
 public:
  [[nodiscard]] bruce::lhme::TechniqueIdentity identity() const override {
    return {.id = "test.canonical_parameter_alias", .version = 1};
  }

  [[nodiscard]] std::optional<std::string> canonical_parameters(
      const bruce::lhme::TechniqueRequest& request) const override {
    if (request.parameters == "left=1;right=2" ||
        request.parameters == "right=2;left=1") {
      return "left=1;right=2";
    }
    if (request.parameters == "left=1;right=3") {
      return "left=1;right=3";
    }
    return std::nullopt;
  }

  [[nodiscard]] bruce::lhme::EvaluationResult evaluate(
      const bruce::lhme::TechniqueRequest& request) const override {
    const auto value = value_for(request.parameters, request.address);
    if (!value.has_value()) {
      return {.status = bruce::lhme::Status::InvalidArgument,
              .mode = request.mode};
    }
    if (request.mode != bruce::lhme::EvaluationMode::Exact &&
        request.mode != bruce::lhme::EvaluationMode::Modular) {
      return {.status = bruce::lhme::Status::UnsupportedMode,
              .mode = request.mode};
    }
    if (request.mode == bruce::lhme::EvaluationMode::Modular) {
      if (!request.modulus.has_value() || *request.modulus <= 0) {
        return {.status = bruce::lhme::Status::InvalidArgument,
                .mode = request.mode};
      }
      return {.status = bruce::lhme::Status::Confirmed,
              .mode = request.mode,
              .integer = *value % *request.modulus};
    }
    return {.status = bruce::lhme::Status::Confirmed,
            .mode = request.mode,
            .integer = *value};
  }

 private:
  [[nodiscard]] static std::optional<bruce::lhme::BigInt> value_for(
      const std::string& parameters, bruce::lhme::FormulaAddress address) {
    if (parameters == "left=1;right=2" ||
        parameters == "right=2;left=1") {
      if (address.index == 10) return bruce::lhme::BigInt{3};
      if (address.index == 12) return bruce::lhme::BigInt{5};
      return std::nullopt;
    }
    if (parameters == "left=1;right=3" && address.index == 10) {
      return bruce::lhme::BigInt{4};
    }
    return std::nullopt;
  }
};

class HeldEvaluationReleaseGuard final {
 public:
  explicit HeldEvaluationReleaseGuard(
      std::shared_ptr<GroupLifetimeCancellationState> state)
      : state_(std::move(state)) {}

  ~HeldEvaluationReleaseGuard() { release(); }

  void release() const noexcept {
    state_->release_held_evaluation.store(true, std::memory_order_release);
    state_->release_held_evaluation.notify_all();
  }

 private:
  std::shared_ptr<GroupLifetimeCancellationState> state_;
};

bruce::lhme::TechniqueRequest power_offset_request(
    bruce::lhme::Budget budget = bruce::lhme::Budget::unlimited()) {
  return {.identity = {.id = "bruce.power_offset", .version = 1},
          .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
          .address = {.index = 10},
          .mode = bruce::lhme::EvaluationMode::Exact,
          .budget = budget};
}

}  // namespace

TEST(cpp_engine_registers_and_evaluates_reference_formula) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));

  const auto result = engine.evaluate({.spec = formula,
                                       .address = {.index = 10},
                                       .mode = bruce::lhme::EvaluationMode::Exact,
                                       .budget = bruce::lhme::Budget::unlimited()});

  CHECK(result.status == bruce::lhme::Status::Confirmed);
  CHECK(result.exact_integer() == 102300);
}

TEST(cpp_engine_reuses_confirmed_exact_results_from_its_bounded_cache) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  (void)engine.evaluate(request);
  const auto second = engine.evaluate(request);

  CHECK(second.exact_integer() == 102300);
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_does_not_confuse_formulae_with_ambiguous_decimal_fields) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto first = engine.register_formula(
      bruce::lhme::make_power_offset_spec(1, 23, 1));
  const auto second = engine.register_formula(
      bruce::lhme::make_power_offset_spec(12, 3, 1));

  CHECK(first.specification_hash != second.specification_hash);
  CHECK(engine.evaluate({.spec = first,
                         .address = {.index = 2},
                         .mode = bruce::lhme::EvaluationMode::Exact,
                         .budget = bruce::lhme::Budget::unlimited()})
            .exact_integer() == 0);
  CHECK(engine.evaluate({.spec = second,
                         .address = {.index = 2},
                         .mode = bruce::lhme::EvaluationMode::Exact,
                         .budget = bruce::lhme::Budget::unlimited()})
            .exact_integer() == 429);
}

TEST(cpp_engine_rejects_a_formula_with_a_forced_hash_collision) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto first = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  auto colliding = bruce::lhme::make_power_offset_spec(3, 100, 1);
  colliding.specification_hash = first.specification_hash;
  (void)engine.register_formula(colliding);

  CHECK(engine.evaluate({.spec = first,
                         .address = {.index = 2},
                         .mode = bruce::lhme::EvaluationMode::Exact,
                         .budget = bruce::lhme::Budget::unlimited()})
            .exact_integer() == 300);
  CHECK(engine.evaluate({.spec = colliding,
                         .address = {.index = 2},
                         .mode = bruce::lhme::EvaluationMode::Exact,
                         .budget = bruce::lhme::Budget::unlimited()})
            .status == bruce::lhme::Status::InvalidArgument);
}

TEST(cpp_engine_rejects_mutated_and_invalid_formula_specs_at_registration) {
  auto engine = bruce::lhme::Engine::create({});
  const auto canonical = bruce::lhme::make_power_offset_spec(2, 100, 1);
  auto mutated_hash = canonical;
  mutated_hash.specification_hash[0] ^= std::byte{0x01};
  auto invalid_base = canonical;
  invalid_base.base = 0;
  auto invalid_scale = canonical;
  invalid_scale.scale = -1;
  auto invalid_index_contract = canonical;
  invalid_index_contract.nonnegative_index = false;
  auto invalid_version = canonical;
  invalid_version.version = 0;
  auto invalid_operation = canonical;
  invalid_operation.operation = static_cast<bruce::lhme::FormulaOperation>(99);

  for (const auto& invalid : {mutated_hash, invalid_base, invalid_scale,
                              invalid_index_contract, invalid_version,
                              invalid_operation}) {
    const auto returned = engine.register_formula(invalid);
    CHECK(returned.specification_hash == invalid.specification_hash);
    CHECK(engine.evaluate({.spec = invalid,
                           .address = {.index = 2},
                           .mode = bruce::lhme::EvaluationMode::Exact,
                           .budget = bruce::lhme::Budget::unlimited()})
              .status == bruce::lhme::Status::InvalidArgument);
  }
}

TEST(cpp_engine_registers_distinct_canonical_formula_specs_without_colliding) {
  auto engine = bruce::lhme::Engine::create({});
  const auto first = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const auto second = engine.register_formula(
      bruce::lhme::make_power_offset_spec(3, 100, 1));

  CHECK(first.specification_hash != second.specification_hash);
  CHECK(engine.evaluate({.spec = first,
                         .address = {.index = 2},
                         .mode = bruce::lhme::EvaluationMode::Exact,
                         .budget = bruce::lhme::Budget::unlimited()})
            .exact_integer() == 300);
  CHECK(engine.evaluate({.spec = second,
                         .address = {.index = 2},
                         .mode = bruce::lhme::EvaluationMode::Exact,
                         .budget = bruce::lhme::Budget::unlimited()})
            .exact_integer() == 800);
}

TEST(cpp_engine_enforces_trace_budget_after_adding_formula_cache_events) {
  auto engine = bruce::lhme::Engine::create({});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const auto expected_trace = bruce::lhme::make_trace({{
      .scope = bruce::lhme::GateScope::Cache,
      .status = bruce::lhme::Status::Confirmed,
      .reason = "cache miss",
  }});
  const auto exact_trace_budget =
      bruce::lhme::trace_budget_bytes(*expected_trace);
  const bruce::lhme::EvaluationRequest exact_budget{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = {.max_trace_bytes = exact_trace_budget}};
  auto one_byte_short = exact_budget;
  one_byte_short.budget.max_trace_bytes = exact_trace_budget - 1;

  const auto accepted = engine.evaluate(exact_budget);
  const auto rejected = engine.evaluate(one_byte_short);

  CHECK(accepted.status == bruce::lhme::Status::Confirmed);
  CHECK(accepted.trace != nullptr);
  CHECK(accepted.trace->events.size() == 1);
  CHECK(rejected.status == bruce::lhme::Status::BudgetExceeded);
  CHECK(!rejected.integer.has_value());
  CHECK(rejected.trace == nullptr);
}

TEST(cpp_engine_rechecks_the_callers_budget_before_returning_a_cache_hit) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest unrestricted{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  (void)engine.evaluate(unrestricted);

  auto restricted = unrestricted;
  restricted.budget.max_bit_length = 0;
  const auto bit_limited_result = engine.evaluate(restricted);
  restricted.budget.max_bit_length = UINT64_MAX;
  restricted.budget.max_operations = 0;
  const auto operation_limited_result = engine.evaluate(restricted);

  CHECK(bit_limited_result.status == bruce::lhme::Status::BudgetExceeded);
  CHECK(operation_limited_result.status == bruce::lhme::Status::BudgetExceeded);
  CHECK(!bit_limited_result.integer.has_value());
  CHECK(!operation_limited_result.integer.has_value());
}

TEST(cpp_engine_registry_rejects_duplicate_and_unknown_techniques) {
  auto engine = bruce::lhme::Engine::create({});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();

  CHECK(engine.register_technique(kernel));
  CHECK(!engine.register_technique(kernel));
  CHECK(engine.evaluate_technique({.identity = {.id = "missing", .version = 1}})
            .status == bruce::lhme::Status::InvalidArgument);
}

TEST(cpp_engine_registry_rejects_a_technique_whose_identity_throws) {
  auto engine = bruce::lhme::Engine::create({});

  CHECK(!engine.register_technique(std::make_shared<ThrowingIdentityKernel>()));
  CHECK(engine.register_technique(std::make_shared<bruce::lhme::PowerOffsetKernel>()));
}

TEST(cpp_engine_technique_cache_rechecks_tighter_exact_budget) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  CHECK(engine.register_technique(
      std::make_shared<bruce::lhme::PowerOffsetKernel>()));

  CHECK(engine.evaluate_technique(power_offset_request()).exact_integer() == 102300);
  CHECK(engine.inspect().resident_entries == 1);

  auto restricted = power_offset_request();
  restricted.budget.max_bit_length = 0;
  const auto result = engine.evaluate_technique(restricted);
  CHECK(result.status == bruce::lhme::Status::BudgetExceeded);
  CHECK(!result.integer.has_value());
}

TEST(cpp_engine_technique_power_offset_uses_the_generic_modulus_contract) {
  auto engine = bruce::lhme::Engine::create({});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));

  const auto result = engine.evaluate_technique(
      {.identity = kernel->identity(),
       .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
       .address = {.index = 10},
       .mode = bruce::lhme::EvaluationMode::Modular,
       .budget = bruce::lhme::Budget::unlimited(),
       .modulus = 17});

  CHECK(result.status == bruce::lhme::Status::Confirmed);
  CHECK(result.exact_integer() == 11);
}

TEST(cpp_engine_caches_exact_fibonacci_but_not_modular_fibonacci) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto kernel = std::make_shared<bruce::lhme::FibonacciKernel>();
  CHECK(engine.register_technique(kernel));

  const bruce::lhme::TechniqueRequest exact{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::kFibonacciFastDoublingParameters,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  CHECK(engine.evaluate_technique(exact).exact_integer() == 55);
  CHECK(engine.evaluate_technique(exact).exact_integer() == 55);
  CHECK(engine.inspect().resident_entries == 1);

  auto metric = exact;
  metric.mode = bruce::lhme::EvaluationMode::Metric;
  const auto first_metric = engine.evaluate_technique(metric);
  const auto second_metric = engine.evaluate_technique(metric);
  CHECK(first_metric.status == bruce::lhme::Status::Confirmed);
  CHECK(second_metric.status == bruce::lhme::Status::Confirmed);
  CHECK(second_metric.metric_bit_length == 7);
  CHECK(!second_metric.integer.has_value());
  CHECK(engine.inspect().resident_entries == 2);

  auto modular = exact;
  modular.mode = bruce::lhme::EvaluationMode::Modular;
  modular.modulus = 17;
  const auto modular_result = engine.evaluate_technique(modular);
  CHECK(modular_result.status == bruce::lhme::Status::Confirmed);
  CHECK(modular_result.exact_integer() == 4);
  CHECK(engine.inspect().resident_entries == 2);
}

TEST(cpp_engine_rejects_a_fresh_technique_result_that_exceeds_its_budget) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  CHECK(engine.register_technique(std::make_shared<OversizedExactKernel>()));

  const auto result = engine.evaluate_technique(
      {.identity = {.id = "test.oversized", .version = 1},
       .parameters = "oversized",
       .address = {.index = 1},
       .mode = bruce::lhme::EvaluationMode::Exact,
       .budget = {.max_bit_length = 1}});

  CHECK(result.status == bruce::lhme::Status::BudgetExceeded);
  CHECK(!result.integer.has_value());
  CHECK(engine.inspect().resident_entries == 0);
}

TEST(cpp_engine_enforces_trace_budget_for_kernel_results_after_cache_events) {
  auto engine = bruce::lhme::Engine::create({});
  const auto kernel = std::make_shared<TraceProducingKernel>();
  CHECK(engine.register_technique(kernel));
  const auto expected_trace = bruce::lhme::make_trace({
      {.scope = bruce::lhme::GateScope::Publish,
       .status = bruce::lhme::Status::Confirmed,
       .reason = "kernel result"},
      {.scope = bruce::lhme::GateScope::Cache,
       .status = bruce::lhme::Status::Confirmed,
       .reason = "cache miss"},
  });
  const auto exact_trace_budget =
      bruce::lhme::trace_budget_bytes(*expected_trace);

  const bruce::lhme::TechniqueRequest exact_budget{
      .identity = kernel->identity(),
      .parameters = "trace",
      .address = {.index = 1},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = {.max_trace_bytes = exact_trace_budget}};
  auto one_byte_short = exact_budget;
  one_byte_short.budget.max_trace_bytes = exact_trace_budget - 1;

  const auto accepted = engine.evaluate_technique(exact_budget);
  const auto rejected = engine.evaluate_technique(one_byte_short);

  CHECK(accepted.status == bruce::lhme::Status::Confirmed);
  CHECK(accepted.trace != nullptr);
  CHECK(accepted.trace->events.size() == 2);
  CHECK(rejected.status == bruce::lhme::Status::BudgetExceeded);
  CHECK(!rejected.integer.has_value());
  CHECK(rejected.trace == nullptr);
}

TEST(cpp_engine_replaces_a_technique_callers_cancellation_signal) {
  auto engine = bruce::lhme::Engine::create({});
  const auto kernel = std::make_shared<SignalProbeKernel>();
  const auto caller_signal = std::make_shared<std::atomic_bool>(false);
  CHECK(engine.register_technique(kernel));

  const auto result = engine.evaluate_technique(
      {.identity = kernel->identity(),
       .parameters = "probe",
       .cancellation_signal = caller_signal});
  CHECK(result.status == bruce::lhme::Status::Confirmed);
  CHECK(kernel->received_signal != caller_signal);

  engine.cancel();
  CHECK(engine.evaluate_technique({.identity = kernel->identity(),
                                  .parameters = "probe"})
            .status == bruce::lhme::Status::Confirmed);
}

TEST(cpp_engine_traces_cancellation_that_arrives_during_technique_evaluation) {
  auto engine = bruce::lhme::Engine::create({});
  const auto kernel = std::make_shared<CancellationTraceKernel>();
  CHECK(engine.register_technique(kernel));

  auto pending = std::async(std::launch::async, [&engine, kernel] {
    return engine.evaluate_technique(
        {.identity = kernel->identity(),
         .parameters = "wait",
         .address = {.index = 1},
         .mode = bruce::lhme::EvaluationMode::Metric,
         .budget = bruce::lhme::Budget::unlimited()});
  });
  while (!kernel->entered_->load()) {
    std::this_thread::yield();
  }
  engine.cancel();

  const auto result = pending.get();
  CHECK(result.status == bruce::lhme::Status::Cancelled);
  CHECK(result.trace != nullptr);
  CHECK(result.trace->contains(bruce::lhme::GateScope::Run));
}

TEST(cpp_engine_reuses_after_cancelling_an_active_technique_evaluation) {
  auto engine = bruce::lhme::Engine::create({});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const auto kernel = std::make_shared<CancellationTraceKernel>();
  CHECK(engine.register_technique(kernel));

  auto pending = std::async(std::launch::async, [&engine, kernel] {
    return engine.evaluate_technique(
        {.identity = kernel->identity(),
         .parameters = "wait",
         .address = {.index = 1},
         .mode = bruce::lhme::EvaluationMode::Metric,
         .budget = bruce::lhme::Budget::unlimited()});
  });
  while (!kernel->entered_->load()) {
    std::this_thread::yield();
  }
  engine.cancel();

  CHECK(pending.get().status == bruce::lhme::Status::Cancelled);
  const auto reused = engine.evaluate(
      {.spec = formula,
       .address = {.index = 10},
       .mode = bruce::lhme::EvaluationMode::Exact,
       .budget = bruce::lhme::Budget::unlimited()});
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.exact_integer() == 102300);
}

TEST(cpp_engine_cancels_all_active_evaluations_and_reuses_after_the_group_finishes) {
  constexpr unsigned int kEvaluationCount = 2;
  auto engine = bruce::lhme::Engine::create({});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const auto entered_evaluations = std::make_shared<std::atomic_uint>(0);
  const auto kernel =
      std::make_shared<GroupCancellationKernel>(entered_evaluations);
  CHECK(engine.register_technique(kernel));

  const auto evaluate_active_kernel = [&engine, kernel] {
    return engine.evaluate_technique(
        {.identity = kernel->identity(),
         .parameters = "wait",
         .address = {.index = 1},
         .mode = bruce::lhme::EvaluationMode::Metric,
         .budget = bruce::lhme::Budget::unlimited()});
  };
  auto first = std::async(std::launch::async, evaluate_active_kernel);
  auto second = std::async(std::launch::async, evaluate_active_kernel);

  while (entered_evaluations->load(std::memory_order_acquire) < kEvaluationCount) {
    const auto observed = entered_evaluations->load(std::memory_order_relaxed);
    entered_evaluations->wait(observed, std::memory_order_relaxed);
  }
  engine.cancel();

  const auto first_result = first.get();
  const auto second_result = second.get();
  CHECK(first_result.status == bruce::lhme::Status::Cancelled);
  CHECK(second_result.status == bruce::lhme::Status::Cancelled);

  const auto reused = engine.evaluate(
      {.spec = formula,
       .address = {.index = 10},
       .mode = bruce::lhme::EvaluationMode::Exact,
       .budget = bruce::lhme::Budget::unlimited()});
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.exact_integer() == 102300);
}

TEST(cpp_engine_keeps_cancellation_for_evaluations_joining_an_active_cancelled_group) {
  constexpr unsigned int kEvaluationCount = 2;
  auto engine = bruce::lhme::Engine::create({});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const auto state = std::make_shared<GroupLifetimeCancellationState>();
  const auto kernel = std::make_shared<GroupLifetimeCancellationKernel>(state);
  CHECK(engine.register_technique(kernel));

  const auto evaluate_kernel = [&engine, kernel](const char* mode) {
    return engine.evaluate_technique(
        {.identity = kernel->identity(),
         .parameters = mode,
         .address = {.index = 10},
         .mode = bruce::lhme::EvaluationMode::Metric,
         .budget = bruce::lhme::Budget::unlimited()});
  };
  auto returning = std::async(std::launch::async, evaluate_kernel,
                              "return_after_cancel");
  auto holding = std::async(std::launch::async, evaluate_kernel,
                            "hold_after_cancel");
  HeldEvaluationReleaseGuard release_held_evaluation(state);

  while (state->entered_evaluations.load(std::memory_order_acquire) <
         kEvaluationCount) {
    const auto observed =
        state->entered_evaluations.load(std::memory_order_relaxed);
    state->entered_evaluations.wait(observed, std::memory_order_relaxed);
  }
  engine.cancel();

  while (!state->held_evaluation_observed_cancellation.load(
      std::memory_order_acquire)) {
    state->held_evaluation_observed_cancellation.wait(
        false, std::memory_order_relaxed);
  }
  const auto returning_result = returning.get();
  CHECK(returning_result.status == bruce::lhme::Status::Cancelled);

  const auto joining_result = engine.evaluate(
      {.spec = formula,
       .address = {.index = 10},
       .mode = bruce::lhme::EvaluationMode::Exact,
       .budget = bruce::lhme::Budget::unlimited()});
  CHECK(joining_result.status == bruce::lhme::Status::Cancelled);

  release_held_evaluation.release();
  const auto holding_result = holding.get();
  CHECK(holding_result.status == bruce::lhme::Status::Cancelled);

  const auto reused = engine.evaluate(
      {.spec = formula,
       .address = {.index = 10},
       .mode = bruce::lhme::EvaluationMode::Exact,
       .budget = bruce::lhme::Budget::unlimited()});
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.exact_integer() == 102300);
}

TEST(cpp_engine_cancels_a_range_joining_an_active_cancelled_group_and_reuses_after_release) {
  constexpr unsigned int kEvaluationCount = 2;
  auto engine = bruce::lhme::Engine::create({});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const auto state = std::make_shared<GroupLifetimeCancellationState>();
  const auto kernel = std::make_shared<GroupLifetimeCancellationKernel>(state);
  CHECK(engine.register_technique(kernel));

  const auto evaluate_kernel = [&engine, kernel](const char* mode) {
    return engine.evaluate_technique(
        {.identity = kernel->identity(),
         .parameters = mode,
         .address = {.index = 1},
         .mode = bruce::lhme::EvaluationMode::Metric,
         .budget = bruce::lhme::Budget::unlimited()});
  };
  auto returning = std::async(std::launch::async, evaluate_kernel,
                              "return_after_cancel");
  auto holding = std::async(std::launch::async, evaluate_kernel,
                            "hold_after_cancel");
  HeldEvaluationReleaseGuard release_held_evaluation(state);

  while (state->entered_evaluations.load(std::memory_order_acquire) <
         kEvaluationCount) {
    const auto observed =
        state->entered_evaluations.load(std::memory_order_relaxed);
    state->entered_evaluations.wait(observed, std::memory_order_relaxed);
  }
  engine.cancel();

  while (!state->held_evaluation_observed_cancellation.load(
      std::memory_order_acquire)) {
    state->held_evaluation_observed_cancellation.wait(
        false, std::memory_order_relaxed);
  }
  const auto returning_result = returning.get();
  CHECK(returning_result.status == bruce::lhme::Status::Cancelled);

  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const auto cancelled_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(cancelled_range.status == bruce::lhme::Status::Cancelled);
  CHECK(cancelled_range.candidates.empty());

  release_held_evaluation.release();
  const auto holding_result = holding.get();
  CHECK(holding_result.status == bruce::lhme::Status::Cancelled);

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.summary.invalid_argument_count == 0);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
}

TEST(cpp_engine_cancels_an_active_range_and_reuses_after_completion) {
  auto engine = bruce::lhme::Engine::create({});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::RangeResult> pending;
  bruce::lhme::test_support::RangeEvaluationBarrier range_barrier;
  pending = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  range_barrier.wait_until_reached();

  engine.cancel();
  range_barrier.release();
  const auto cancelled_range = pending.get();
  CHECK(cancelled_range.status == bruce::lhme::Status::Cancelled);
  CHECK(cancelled_range.candidates.empty());

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.summary.invalid_argument_count == 0);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
}

TEST(cpp_engine_cancels_a_range_after_a_scheduler_tile_starts_and_reuses_after_completion) {
  auto engine = bruce::lhme::Engine::create(
      {.worker_count = 1, .tile_size = 2});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::RangeResult> pending;
  bruce::lhme::test_support::SchedulerTileBarrier tile_barrier;
  pending = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  tile_barrier.wait_until_reached();

  engine.cancel();
  tile_barrier.release();
  const auto cancelled_range = pending.get();
  CHECK(cancelled_range.status == bruce::lhme::Status::Cancelled);
  CHECK(cancelled_range.candidates.empty());

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.summary.invalid_argument_count == 0);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
}

TEST(cpp_engine_cancels_a_range_after_its_first_scheduler_candidate_and_reuses_after_completion) {
  auto engine = bruce::lhme::Engine::create(
      {.worker_count = 1, .tile_size = 2});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::RangeResult> pending;
  bruce::lhme::test_support::SchedulerCandidateBarrier candidate_barrier;
  pending = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  candidate_barrier.wait_until_reached();

  engine.cancel();
  candidate_barrier.release();
  const auto cancelled_range = pending.get();
  CHECK(cancelled_range.status == bruce::lhme::Status::Cancelled);
  CHECK(cancelled_range.candidates.empty());

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.summary.invalid_argument_count == 0);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
}

TEST(cpp_engine_cancels_a_range_after_a_scheduler_candidate_is_published_and_reuses_after_completion) {
  auto engine = bruce::lhme::Engine::create(
      {.worker_count = 2, .tile_size = 1});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::RangeResult> pending;
  bruce::lhme::test_support::SchedulerPublishedCandidateBarrier
      published_candidate_barrier;
  pending = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  published_candidate_barrier.wait_until_reached();

  engine.cancel();
  published_candidate_barrier.release();
  const auto cancelled_range = pending.get();
  CHECK(cancelled_range.status == bruce::lhme::Status::Cancelled);
  CHECK(cancelled_range.candidates.empty());

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.summary.invalid_argument_count == 0);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
}

TEST(cpp_engine_cancels_a_range_after_its_last_scheduler_tile_is_aggregated_and_reuses_after_completion) {
  auto engine = bruce::lhme::Engine::create(
      {.worker_count = 1, .tile_size = 2});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::RangeResult> pending;
  bruce::lhme::test_support::SchedulerFinalTileAggregationBarrier
      final_tile_aggregation_barrier;
  pending = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  final_tile_aggregation_barrier.wait_until_reached();

  engine.cancel();
  final_tile_aggregation_barrier.release();
  const auto cancelled_range = pending.get();
  CHECK(cancelled_range.status == bruce::lhme::Status::Cancelled);
  CHECK(cancelled_range.candidates.empty());

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.summary.invalid_argument_count == 0);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
}

TEST(cpp_engine_cancels_a_range_after_scheduler_completion_and_before_engine_finalization) {
  auto engine = bruce::lhme::Engine::create(
      {.worker_count = 1, .tile_size = 2});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::RangeResult> pending;
  bruce::lhme::test_support::EngineRangeFinalizationBarrier
      finalization_barrier;
  pending = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  finalization_barrier.wait_until_reached();

  engine.cancel();
  finalization_barrier.release();
  const auto cancelled_range = pending.get();
  CHECK(cancelled_range.status == bruce::lhme::Status::Cancelled);
  CHECK(cancelled_range.candidates.empty());

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.summary.invalid_argument_count == 0);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
}

TEST(cpp_engine_keeps_a_range_result_final_after_its_completion_snapshot) {
  auto engine = bruce::lhme::Engine::create(
      {.worker_count = 1, .tile_size = 2});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::RangeResult> pending;
  bruce::lhme::test_support::EngineRangePostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  post_finalization_barrier.release();
  const auto finalized_range = pending.get();
  CHECK(finalized_range.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_range.summary.checked_count == 2);
  CHECK(finalized_range.summary.confirmed_count == 2);
  CHECK(finalized_range.summary.invalid_argument_count == 0);
  CHECK(finalized_range.candidates.size() == 2);
  CHECK(finalized_range.candidates[0].exact_integer() == 0);
  CHECK(finalized_range.candidates[1].exact_integer() == 100);

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.summary.invalid_argument_count == 0);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
}

TEST(cpp_engine_cancels_a_range_joining_a_post_snapshot_cancelled_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.worker_count = 1, .tile_size = 2});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::RangeResult> pending;
  bruce::lhme::test_support::EngineRangePostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_range = engine.evaluate_range(range_request, {0, 1});
  post_finalization_barrier.release();
  const auto finalized_range = pending.get();

  CHECK(joining_range.status == bruce::lhme::Status::Cancelled);
  CHECK(joining_range.candidates.empty());
  CHECK(finalized_range.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_range.summary.checked_count == 2);
  CHECK(finalized_range.summary.confirmed_count == 2);
  CHECK(finalized_range.summary.invalid_argument_count == 0);
  CHECK(finalized_range.candidates.size() == 2);
  CHECK(finalized_range.candidates[0].exact_integer() == 0);
  CHECK(finalized_range.candidates[1].exact_integer() == 100);

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.summary.invalid_argument_count == 0);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
}

TEST(cpp_engine_cancels_a_technique_joining_a_post_snapshot_cancelled_range_group_without_mutating_the_finalized_range) {
  auto engine = bruce::lhme::Engine::create(
      {.worker_count = 1, .tile_size = 2});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  CHECK(engine.register_technique(
      std::make_shared<bruce::lhme::PowerOffsetKernel>()));
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::RangeResult> pending;
  bruce::lhme::test_support::EngineRangePostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_technique = engine.evaluate_technique(power_offset_request());
  post_finalization_barrier.release();
  const auto finalized_range = pending.get();

  CHECK(joining_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(finalized_range.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_range.summary.checked_count == 2);
  CHECK(finalized_range.summary.confirmed_count == 2);
  CHECK(finalized_range.summary.invalid_argument_count == 0);
  CHECK(finalized_range.candidates.size() == 2);
  CHECK(finalized_range.candidates[0].exact_integer() == 0);
  CHECK(finalized_range.candidates[1].exact_integer() == 100);

  const auto reused_technique = engine.evaluate_technique(power_offset_request());
  CHECK(reused_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_technique.exact_integer() == 102300);
}

TEST(cpp_engine_cancels_a_metric_technique_evaluation_joining_a_post_snapshot_cancelled_range_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  CHECK(engine.register_technique(
      std::make_shared<bruce::lhme::PowerOffsetKernel>()));
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  auto technique_request = power_offset_request();
  technique_request.mode = bruce::lhme::EvaluationMode::Metric;

  std::future<bruce::lhme::RangeResult> pending;
  bruce::lhme::test_support::EngineRangePostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_technique = engine.evaluate_technique(technique_request);
  CHECK(joining_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining_technique.integer.has_value());
  CHECK(!joining_technique.metric_bit_length.has_value());
  CHECK(!joining_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized_range = pending.get();
  CHECK(finalized_range.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_range.summary.checked_count == 2);
  CHECK(finalized_range.summary.confirmed_count == 2);
  CHECK(finalized_range.candidates.size() == 2);
  CHECK(finalized_range.candidates[0].exact_integer() == 0);
  CHECK(finalized_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_technique = engine.evaluate_technique(technique_request);
  CHECK(reused_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_technique.metric_bit_length == 17);
  CHECK(!reused_technique.integer.has_value());
  CHECK(!reused_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_cancels_a_modular_technique_evaluation_joining_a_post_snapshot_cancelled_range_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  CHECK(engine.register_technique(
      std::make_shared<bruce::lhme::PowerOffsetKernel>()));
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  auto technique_request = power_offset_request();
  technique_request.mode = bruce::lhme::EvaluationMode::Modular;
  technique_request.modulus = 17;

  std::future<bruce::lhme::RangeResult> pending;
  bruce::lhme::test_support::EngineRangePostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_technique = engine.evaluate_technique(technique_request);
  CHECK(joining_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining_technique.integer.has_value());
  CHECK(!joining_technique.metric_bit_length.has_value());
  CHECK(!joining_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized_range = pending.get();
  CHECK(finalized_range.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_range.summary.checked_count == 2);
  CHECK(finalized_range.summary.confirmed_count == 2);
  CHECK(finalized_range.candidates.size() == 2);
  CHECK(finalized_range.candidates[0].exact_integer() == 0);
  CHECK(finalized_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_technique = engine.evaluate_technique(technique_request);
  CHECK(reused_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_technique.exact_integer() == 11);
  CHECK(engine.inspect().resident_entries == 0);
}

TEST(cpp_engine_cancels_a_symbolic_technique_evaluation_joining_a_post_snapshot_cancelled_range_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  CHECK(engine.register_technique(
      std::make_shared<bruce::lhme::PowerOffsetKernel>()));
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  auto technique_request = power_offset_request();
  technique_request.mode = bruce::lhme::EvaluationMode::Symbolic;

  std::future<bruce::lhme::RangeResult> pending;
  bruce::lhme::test_support::EngineRangePostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_technique = engine.evaluate_technique(technique_request);
  CHECK(joining_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining_technique.integer.has_value());
  CHECK(!joining_technique.metric_bit_length.has_value());
  CHECK(!joining_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized_range = pending.get();
  CHECK(finalized_range.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_range.summary.checked_count == 2);
  CHECK(finalized_range.summary.confirmed_count == 2);
  CHECK(finalized_range.candidates.size() == 2);
  CHECK(finalized_range.candidates[0].exact_integer() == 0);
  CHECK(finalized_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_technique = engine.evaluate_technique(technique_request);
  CHECK(reused_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_technique.symbolic_expression == "100*(2^10-1)");
  CHECK(!reused_technique.integer.has_value());
  CHECK(!reused_technique.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 0);
}

TEST(cpp_engine_cancels_a_metric_formula_evaluation_joining_a_post_snapshot_cancelled_range_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::RangeResult> pending;
  bruce::lhme::test_support::EngineRangePostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_formula = engine.evaluate(formula_request);
  CHECK(joining_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining_formula.integer.has_value());
  CHECK(!joining_formula.metric_bit_length.has_value());
  CHECK(!joining_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized_range = pending.get();
  CHECK(finalized_range.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_range.summary.checked_count == 2);
  CHECK(finalized_range.summary.confirmed_count == 2);
  CHECK(finalized_range.candidates.size() == 2);
  CHECK(finalized_range.candidates[0].exact_integer() == 0);
  CHECK(finalized_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_formula = engine.evaluate(formula_request);
  CHECK(reused_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_formula.metric_bit_length == 17);
  CHECK(!reused_formula.integer.has_value());
  CHECK(!reused_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_cancels_a_modular_formula_evaluation_joining_a_post_snapshot_cancelled_range_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};

  std::future<bruce::lhme::RangeResult> pending;
  bruce::lhme::test_support::EngineRangePostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_formula = engine.evaluate(formula_request);
  CHECK(joining_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining_formula.integer.has_value());
  CHECK(!joining_formula.metric_bit_length.has_value());
  CHECK(!joining_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized_range = pending.get();
  CHECK(finalized_range.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_range.summary.checked_count == 2);
  CHECK(finalized_range.summary.confirmed_count == 2);
  CHECK(finalized_range.candidates.size() == 2);
  CHECK(finalized_range.candidates[0].exact_integer() == 0);
  CHECK(finalized_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_formula = engine.evaluate(formula_request);
  CHECK(reused_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_formula.exact_integer() == 11);
  CHECK(!reused_formula.metric_bit_length.has_value());
  CHECK(!reused_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);
}

TEST(cpp_engine_cancels_a_symbolic_formula_evaluation_joining_a_post_snapshot_cancelled_range_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::RangeResult> pending;
  bruce::lhme::test_support::EngineRangePostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_formula = engine.evaluate(formula_request);
  CHECK(joining_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining_formula.integer.has_value());
  CHECK(!joining_formula.metric_bit_length.has_value());
  CHECK(!joining_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized_range = pending.get();
  CHECK(finalized_range.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_range.summary.checked_count == 2);
  CHECK(finalized_range.summary.confirmed_count == 2);
  CHECK(finalized_range.candidates.size() == 2);
  CHECK(finalized_range.candidates[0].exact_integer() == 0);
  CHECK(finalized_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_formula = engine.evaluate(formula_request);
  CHECK(reused_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_formula.symbolic_expression == "100*(2^10-1)");
  CHECK(!reused_formula.integer.has_value());
  CHECK(!reused_formula.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 0);
}

TEST(cpp_engine_cancels_a_formula_evaluation_joining_a_post_snapshot_cancelled_range_group_without_mutating_the_finalized_range) {
  auto engine = bruce::lhme::Engine::create(
      {.worker_count = 1, .tile_size = 2});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::RangeResult> pending;
  bruce::lhme::test_support::EngineRangePostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_formula = engine.evaluate(formula_request);
  post_finalization_barrier.release();
  const auto finalized_range = pending.get();

  CHECK(joining_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(finalized_range.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_range.summary.checked_count == 2);
  CHECK(finalized_range.summary.confirmed_count == 2);
  CHECK(finalized_range.summary.invalid_argument_count == 0);
  CHECK(finalized_range.candidates.size() == 2);
  CHECK(finalized_range.candidates[0].exact_integer() == 0);
  CHECK(finalized_range.candidates[1].exact_integer() == 100);

  const auto reused_formula = engine.evaluate(formula_request);
  CHECK(reused_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_formula.exact_integer() == 102300);
}

TEST(cpp_engine_cancels_a_formula_after_core_completion_and_reuses_after_completion) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      finalization_barrier;
  pending = std::async(std::launch::async, [&engine, request] {
    return engine.evaluate(request);
  });
  finalization_barrier.wait_until_reached();

  engine.cancel();
  finalization_barrier.release();
  const auto cancelled = pending.get();

  CHECK(cancelled.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled.integer.has_value());
  CHECK(!cancelled.metric_bit_length.has_value());
  CHECK(!cancelled.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused = engine.evaluate(request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.exact_integer() == 102300);
}

TEST(cpp_engine_cancels_a_symbolic_noncacheable_formula_before_its_completion_snapshot_without_publishing_symbolic_output) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      finalization_barrier;
  pending = std::async(std::launch::async, [&engine, request] {
    return engine.evaluate(request);
  });
  finalization_barrier.wait_until_reached();

  engine.cancel();
  finalization_barrier.release();
  const auto cancelled = pending.get();

  CHECK(cancelled.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled.integer.has_value());
  CHECK(!cancelled.metric_bit_length.has_value());
  CHECK(!cancelled.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused = engine.evaluate(request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.symbolic_expression == "100*(2^10-1)");
  CHECK(!reused.integer.has_value());
  CHECK(!reused.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 0);
}

TEST(cpp_engine_cancels_a_modular_noncacheable_formula_before_its_completion_snapshot_and_reuses_after_completion) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      finalization_barrier;
  pending = std::async(std::launch::async, [&engine, request] {
    return engine.evaluate(request);
  });
  finalization_barrier.wait_until_reached();

  engine.cancel();
  finalization_barrier.release();
  const auto cancelled = pending.get();

  CHECK(cancelled.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled.integer.has_value());
  CHECK(!cancelled.metric_bit_length.has_value());
  CHECK(!cancelled.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused = engine.evaluate(request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.exact_integer() == 11);
  CHECK(!reused.metric_bit_length.has_value());
  CHECK(!reused.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);
}

TEST(cpp_engine_cancels_a_metric_formula_cache_miss_before_its_completion_snapshot_and_reuses_after_completion) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      finalization_barrier;
  pending = std::async(std::launch::async, [&engine, request] {
    return engine.evaluate(request);
  });
  finalization_barrier.wait_until_reached();

  engine.cancel();
  finalization_barrier.release();
  const auto cancelled = pending.get();

  CHECK(cancelled.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled.integer.has_value());
  CHECK(!cancelled.metric_bit_length.has_value());
  CHECK(!cancelled.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused = engine.evaluate(request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.metric_bit_length == 17);
  CHECK(!reused.integer.has_value());
  CHECK(!reused.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_keeps_a_formula_cache_miss_final_after_its_completion_snapshot) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, request] {
    return engine.evaluate(request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  post_finalization_barrier.release();
  const auto finalized = pending.get();

  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.exact_integer() == 102300);
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused = engine.evaluate(request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.exact_integer() == 102300);
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_keeps_a_symbolic_noncacheable_formula_result_final_after_its_completion_snapshot) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, request] {
    return engine.evaluate(request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  post_finalization_barrier.release();
  const auto finalized = pending.get();

  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.symbolic_expression == "100*(2^10-1)");
  CHECK(!finalized.integer.has_value());
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused = engine.evaluate(request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.symbolic_expression == "100*(2^10-1)");
  CHECK(!reused.integer.has_value());
  CHECK(!reused.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 0);
}

TEST(cpp_engine_keeps_a_modular_noncacheable_formula_result_final_after_its_completion_snapshot) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, request] {
    return engine.evaluate(request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  post_finalization_barrier.release();
  const auto finalized = pending.get();

  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.exact_integer() == 11);
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused = engine.evaluate(request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.exact_integer() == 11);
  CHECK(!reused.metric_bit_length.has_value());
  CHECK(!reused.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);
}

TEST(cpp_engine_cancels_a_modular_noncacheable_formula_joining_a_post_snapshot_cancelled_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, request] {
    return engine.evaluate(request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining = engine.evaluate(request);
  CHECK(joining.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining.integer.has_value());
  CHECK(!joining.metric_bit_length.has_value());
  CHECK(!joining.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.exact_integer() == 11);
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused = engine.evaluate(request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.exact_integer() == 11);
  CHECK(!reused.metric_bit_length.has_value());
  CHECK(!reused.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);
}

TEST(cpp_engine_cancels_a_metric_technique_evaluation_joining_a_post_snapshot_cancelled_modular_noncacheable_formula_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  CHECK(engine.register_technique(
      std::make_shared<bruce::lhme::PowerOffsetKernel>()));
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  auto technique_request = power_offset_request();
  technique_request.mode = bruce::lhme::EvaluationMode::Metric;

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, formula_request] {
    return engine.evaluate(formula_request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_technique = engine.evaluate_technique(technique_request);
  CHECK(joining_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining_technique.integer.has_value());
  CHECK(!joining_technique.metric_bit_length.has_value());
  CHECK(!joining_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.exact_integer() == 11);
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_technique = engine.evaluate_technique(technique_request);
  CHECK(reused_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_technique.metric_bit_length == 17);
  CHECK(!reused_technique.integer.has_value());
  CHECK(!reused_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_cancels_an_exact_technique_evaluation_joining_a_post_snapshot_cancelled_modular_noncacheable_formula_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  CHECK(engine.register_technique(
      std::make_shared<bruce::lhme::PowerOffsetKernel>()));
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const auto technique_request = power_offset_request();

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, formula_request] {
    return engine.evaluate(formula_request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_technique = engine.evaluate_technique(technique_request);
  CHECK(joining_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining_technique.integer.has_value());
  CHECK(!joining_technique.metric_bit_length.has_value());
  CHECK(!joining_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.exact_integer() == 11);
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_technique = engine.evaluate_technique(technique_request);
  CHECK(reused_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_technique.exact_integer() == 102300);
  CHECK(!reused_technique.metric_bit_length.has_value());
  CHECK(!reused_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_cancels_a_modular_technique_evaluation_joining_a_post_snapshot_cancelled_modular_noncacheable_formula_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  CHECK(engine.register_technique(
      std::make_shared<bruce::lhme::PowerOffsetKernel>()));
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  auto technique_request = power_offset_request();
  technique_request.mode = bruce::lhme::EvaluationMode::Modular;
  technique_request.modulus = 17;

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, formula_request] {
    return engine.evaluate(formula_request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_technique = engine.evaluate_technique(technique_request);
  CHECK(joining_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining_technique.integer.has_value());
  CHECK(!joining_technique.metric_bit_length.has_value());
  CHECK(!joining_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.exact_integer() == 11);
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_technique = engine.evaluate_technique(technique_request);
  CHECK(reused_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_technique.exact_integer() == 11);
  CHECK(!reused_technique.metric_bit_length.has_value());
  CHECK(!reused_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);
}

TEST(cpp_engine_cancels_a_symbolic_technique_evaluation_joining_a_post_snapshot_cancelled_modular_noncacheable_formula_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  CHECK(engine.register_technique(
      std::make_shared<bruce::lhme::PowerOffsetKernel>()));
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  auto technique_request = power_offset_request();
  technique_request.mode = bruce::lhme::EvaluationMode::Symbolic;

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, formula_request] {
    return engine.evaluate(formula_request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_technique = engine.evaluate_technique(technique_request);
  CHECK(joining_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining_technique.integer.has_value());
  CHECK(!joining_technique.metric_bit_length.has_value());
  CHECK(!joining_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.exact_integer() == 11);
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_technique = engine.evaluate_technique(technique_request);
  CHECK(reused_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_technique.symbolic_expression == "100*(2^10-1)");
  CHECK(!reused_technique.integer.has_value());
  CHECK(!reused_technique.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 0);
}

TEST(cpp_engine_cancels_a_range_joining_a_post_snapshot_cancelled_modular_noncacheable_formula_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, formula_request] {
    return engine.evaluate(formula_request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(joining_range.status == bruce::lhme::Status::Cancelled);
  CHECK(joining_range.candidates.empty());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.exact_integer() == 11);
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 0);
}

TEST(cpp_engine_cancels_an_exact_formula_evaluation_joining_a_post_snapshot_cancelled_modular_noncacheable_formula_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest modular_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest exact_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, modular_request] {
    return engine.evaluate(modular_request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_exact = engine.evaluate(exact_request);
  CHECK(joining_exact.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining_exact.integer.has_value());
  CHECK(!joining_exact.metric_bit_length.has_value());
  CHECK(!joining_exact.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.exact_integer() == 11);
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_exact = engine.evaluate(exact_request);
  CHECK(reused_exact.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_exact.exact_integer() == 102300);
  CHECK(!reused_exact.metric_bit_length.has_value());
  CHECK(!reused_exact.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_cancels_a_metric_formula_evaluation_joining_a_post_snapshot_cancelled_modular_noncacheable_formula_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest modular_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest metric_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, modular_request] {
    return engine.evaluate(modular_request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_metric = engine.evaluate(metric_request);
  CHECK(joining_metric.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining_metric.integer.has_value());
  CHECK(!joining_metric.metric_bit_length.has_value());
  CHECK(!joining_metric.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.exact_integer() == 11);
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_metric = engine.evaluate(metric_request);
  CHECK(reused_metric.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_metric.metric_bit_length == 17);
  CHECK(!reused_metric.integer.has_value());
  CHECK(!reused_metric.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_cancels_a_symbolic_formula_evaluation_joining_a_post_snapshot_cancelled_modular_noncacheable_formula_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest modular_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest symbolic_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, modular_request] {
    return engine.evaluate(modular_request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_symbolic = engine.evaluate(symbolic_request);
  CHECK(joining_symbolic.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining_symbolic.integer.has_value());
  CHECK(!joining_symbolic.metric_bit_length.has_value());
  CHECK(!joining_symbolic.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.exact_integer() == 11);
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_symbolic = engine.evaluate(symbolic_request);
  CHECK(reused_symbolic.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_symbolic.symbolic_expression == "100*(2^10-1)");
  CHECK(!reused_symbolic.integer.has_value());
  CHECK(!reused_symbolic.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 0);
}

TEST(cpp_engine_cancels_a_symbolic_noncacheable_formula_joining_a_post_snapshot_cancelled_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, request] {
    return engine.evaluate(request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining = engine.evaluate(request);
  CHECK(joining.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining.integer.has_value());
  CHECK(!joining.metric_bit_length.has_value());
  CHECK(!joining.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.symbolic_expression == "100*(2^10-1)");
  CHECK(!finalized.integer.has_value());
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused = engine.evaluate(request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.symbolic_expression == "100*(2^10-1)");
  CHECK(!reused.integer.has_value());
  CHECK(!reused.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 0);
}

TEST(cpp_engine_cancels_an_exact_technique_evaluation_joining_a_post_snapshot_cancelled_symbolic_noncacheable_formula_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  CHECK(engine.register_technique(
      std::make_shared<bruce::lhme::PowerOffsetKernel>()));
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};
  const auto technique_request = power_offset_request();

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, formula_request] {
    return engine.evaluate(formula_request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_technique = engine.evaluate_technique(technique_request);
  CHECK(joining_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining_technique.integer.has_value());
  CHECK(!joining_technique.metric_bit_length.has_value());
  CHECK(!joining_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.symbolic_expression == "100*(2^10-1)");
  CHECK(!finalized.integer.has_value());
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_technique = engine.evaluate_technique(technique_request);
  CHECK(reused_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_technique.exact_integer() == 102300);
  CHECK(!reused_technique.metric_bit_length.has_value());
  CHECK(!reused_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_cancels_a_metric_technique_evaluation_joining_a_post_snapshot_cancelled_symbolic_noncacheable_formula_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  CHECK(engine.register_technique(
      std::make_shared<bruce::lhme::PowerOffsetKernel>()));
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};
  auto technique_request = power_offset_request();
  technique_request.mode = bruce::lhme::EvaluationMode::Metric;

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, formula_request] {
    return engine.evaluate(formula_request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_technique = engine.evaluate_technique(technique_request);
  CHECK(joining_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining_technique.integer.has_value());
  CHECK(!joining_technique.metric_bit_length.has_value());
  CHECK(!joining_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.symbolic_expression == "100*(2^10-1)");
  CHECK(!finalized.integer.has_value());
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_technique = engine.evaluate_technique(technique_request);
  CHECK(reused_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_technique.metric_bit_length == 17);
  CHECK(!reused_technique.integer.has_value());
  CHECK(!reused_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_cancels_a_modular_technique_evaluation_joining_a_post_snapshot_cancelled_symbolic_noncacheable_formula_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  CHECK(engine.register_technique(
      std::make_shared<bruce::lhme::PowerOffsetKernel>()));
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};
  auto technique_request = power_offset_request();
  technique_request.mode = bruce::lhme::EvaluationMode::Modular;
  technique_request.modulus = 17;

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, formula_request] {
    return engine.evaluate(formula_request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_technique = engine.evaluate_technique(technique_request);
  CHECK(joining_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining_technique.integer.has_value());
  CHECK(!joining_technique.metric_bit_length.has_value());
  CHECK(!joining_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.symbolic_expression == "100*(2^10-1)");
  CHECK(!finalized.integer.has_value());
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_technique = engine.evaluate_technique(technique_request);
  CHECK(reused_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_technique.exact_integer() == 11);
  CHECK(!reused_technique.metric_bit_length.has_value());
  CHECK(!reused_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);
}

TEST(cpp_engine_cancels_a_symbolic_technique_evaluation_joining_a_post_snapshot_cancelled_symbolic_noncacheable_formula_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  CHECK(engine.register_technique(
      std::make_shared<bruce::lhme::PowerOffsetKernel>()));
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};
  auto technique_request = power_offset_request();
  technique_request.mode = bruce::lhme::EvaluationMode::Symbolic;

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, formula_request] {
    return engine.evaluate(formula_request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_technique = engine.evaluate_technique(technique_request);
  CHECK(joining_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining_technique.integer.has_value());
  CHECK(!joining_technique.metric_bit_length.has_value());
  CHECK(!joining_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.symbolic_expression == "100*(2^10-1)");
  CHECK(!finalized.integer.has_value());
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_technique = engine.evaluate_technique(technique_request);
  CHECK(reused_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_technique.symbolic_expression == "100*(2^10-1)");
  CHECK(!reused_technique.integer.has_value());
  CHECK(!reused_technique.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 0);
}

TEST(cpp_engine_cancels_a_range_joining_a_post_snapshot_cancelled_symbolic_noncacheable_formula_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, formula_request] {
    return engine.evaluate(formula_request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(joining_range.status == bruce::lhme::Status::Cancelled);
  CHECK(joining_range.candidates.empty());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.symbolic_expression == "100*(2^10-1)");
  CHECK(!finalized.integer.has_value());
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 0);
}

TEST(cpp_engine_cancels_an_exact_formula_evaluation_joining_a_post_snapshot_cancelled_symbolic_noncacheable_formula_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest symbolic_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest exact_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, symbolic_request] {
    return engine.evaluate(symbolic_request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_exact = engine.evaluate(exact_request);
  CHECK(joining_exact.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining_exact.integer.has_value());
  CHECK(!joining_exact.metric_bit_length.has_value());
  CHECK(!joining_exact.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.symbolic_expression == "100*(2^10-1)");
  CHECK(!finalized.integer.has_value());
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_exact = engine.evaluate(exact_request);
  CHECK(reused_exact.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_exact.exact_integer() == 102300);
  CHECK(!reused_exact.metric_bit_length.has_value());
  CHECK(!reused_exact.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_cancels_a_metric_formula_evaluation_joining_a_post_snapshot_cancelled_symbolic_noncacheable_formula_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest symbolic_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest metric_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, symbolic_request] {
    return engine.evaluate(symbolic_request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_metric = engine.evaluate(metric_request);
  CHECK(joining_metric.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining_metric.integer.has_value());
  CHECK(!joining_metric.metric_bit_length.has_value());
  CHECK(!joining_metric.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.symbolic_expression == "100*(2^10-1)");
  CHECK(!finalized.integer.has_value());
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_metric = engine.evaluate(metric_request);
  CHECK(reused_metric.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_metric.metric_bit_length == 17);
  CHECK(!reused_metric.integer.has_value());
  CHECK(!reused_metric.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_cancels_a_modular_formula_evaluation_joining_a_post_snapshot_cancelled_symbolic_noncacheable_formula_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest symbolic_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest modular_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, symbolic_request] {
    return engine.evaluate(symbolic_request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_modular = engine.evaluate(modular_request);
  CHECK(joining_modular.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining_modular.integer.has_value());
  CHECK(!joining_modular.metric_bit_length.has_value());
  CHECK(!joining_modular.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.symbolic_expression == "100*(2^10-1)");
  CHECK(!finalized.integer.has_value());
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_modular = engine.evaluate(modular_request);
  CHECK(reused_modular.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_modular.exact_integer() == 11);
  CHECK(!reused_modular.metric_bit_length.has_value());
  CHECK(!reused_modular.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);
}

TEST(cpp_engine_cancels_a_formula_cache_miss_joining_a_post_snapshot_cancelled_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, request] {
    return engine.evaluate(request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining = engine.evaluate(request);
  CHECK(joining.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining.integer.has_value());
  CHECK(!joining.metric_bit_length.has_value());
  CHECK(!joining.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.exact_integer() == 102300);
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused = engine.evaluate(request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.exact_integer() == 102300);
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_cancels_a_metric_formula_cache_miss_joining_a_post_snapshot_cancelled_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, request] {
    return engine.evaluate(request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining = engine.evaluate(request);
  CHECK(joining.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining.integer.has_value());
  CHECK(!joining.metric_bit_length.has_value());
  CHECK(!joining.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.metric_bit_length == 17);
  CHECK(!finalized.integer.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused = engine.evaluate(request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.metric_bit_length == 17);
  CHECK(!reused.integer.has_value());
  CHECK(!reused.symbolic_expression.has_value());
  CHECK(reused.trace != nullptr);
  CHECK(reused.trace->events.back().scope == bruce::lhme::GateScope::Cache);
  CHECK(reused.trace->events.back().reason == "cache hit");
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_cancels_a_metric_technique_evaluation_joining_a_post_snapshot_cancelled_metric_formula_cache_miss_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  CHECK(engine.register_technique(
      std::make_shared<bruce::lhme::PowerOffsetKernel>()));
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};
  auto technique_request = power_offset_request();
  technique_request.mode = bruce::lhme::EvaluationMode::Metric;

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, formula_request] {
    return engine.evaluate(formula_request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_technique = engine.evaluate_technique(technique_request);
  CHECK(joining_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining_technique.integer.has_value());
  CHECK(!joining_technique.metric_bit_length.has_value());
  CHECK(!joining_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.metric_bit_length == 17);
  CHECK(!finalized.integer.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_technique = engine.evaluate_technique(technique_request);
  CHECK(reused_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_technique.metric_bit_length == 17);
  CHECK(!reused_technique.integer.has_value());
  CHECK(!reused_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 2);
}

TEST(cpp_engine_cancels_a_range_joining_a_post_snapshot_cancelled_metric_formula_cache_miss_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, formula_request] {
    return engine.evaluate(formula_request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(joining_range.status == bruce::lhme::Status::Cancelled);
  CHECK(joining_range.candidates.empty());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.metric_bit_length == 17);
  CHECK(!finalized.integer.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_cancels_a_formula_cache_hit_after_cache_completion_and_reuses_after_completion) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  const auto warmed = engine.evaluate(request);
  CHECK(warmed.status == bruce::lhme::Status::Confirmed);
  CHECK(warmed.exact_integer() == 102300);
  CHECK(engine.inspect().resident_entries == 1);

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineFormulaCacheHitFinalizationBarrier
      finalization_barrier;
  pending = std::async(std::launch::async, [&engine, request] {
    return engine.evaluate(request);
  });
  finalization_barrier.wait_until_reached();

  engine.cancel();
  finalization_barrier.release();
  const auto cancelled = pending.get();

  CHECK(cancelled.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled.integer.has_value());
  CHECK(!cancelled.metric_bit_length.has_value());
  CHECK(!cancelled.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused = engine.evaluate(request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.exact_integer() == 102300);
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_cancels_a_metric_formula_cache_hit_before_its_completion_snapshot_and_reuses_after_completion) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};

  const auto warmed = engine.evaluate(request);
  CHECK(warmed.status == bruce::lhme::Status::Confirmed);
  CHECK(warmed.metric_bit_length == 17);
  CHECK(!warmed.integer.has_value());
  CHECK(!warmed.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineFormulaCacheHitFinalizationBarrier
      finalization_barrier;
  pending = std::async(std::launch::async, [&engine, request] {
    return engine.evaluate(request);
  });
  finalization_barrier.wait_until_reached();

  engine.cancel();
  finalization_barrier.release();
  const auto cancelled = pending.get();

  CHECK(cancelled.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled.integer.has_value());
  CHECK(!cancelled.metric_bit_length.has_value());
  CHECK(!cancelled.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused = engine.evaluate(request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.metric_bit_length == 17);
  CHECK(!reused.integer.has_value());
  CHECK(!reused.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_keeps_a_formula_cache_hit_final_after_its_completion_snapshot) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  const auto warmed = engine.evaluate(request);
  CHECK(warmed.status == bruce::lhme::Status::Confirmed);
  CHECK(warmed.exact_integer() == 102300);
  CHECK(engine.inspect().resident_entries == 1);

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineFormulaCacheHitPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, request] {
    return engine.evaluate(request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  post_finalization_barrier.release();
  const auto finalized = pending.get();

  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.exact_integer() == 102300);
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused = engine.evaluate(request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.exact_integer() == 102300);
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_cancels_a_formula_cache_hit_joining_a_post_snapshot_cancelled_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  const auto warmed = engine.evaluate(request);
  CHECK(warmed.status == bruce::lhme::Status::Confirmed);
  CHECK(warmed.exact_integer() == 102300);
  CHECK(engine.inspect().resident_entries == 1);

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineFormulaCacheHitPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, request] {
    return engine.evaluate(request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining = engine.evaluate(request);
  CHECK(joining.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining.integer.has_value());
  CHECK(!joining.metric_bit_length.has_value());
  CHECK(!joining.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.exact_integer() == 102300);
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused = engine.evaluate(request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.exact_integer() == 102300);
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_cancels_a_metric_technique_evaluation_joining_a_post_snapshot_cancelled_metric_formula_cache_hit_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  CHECK(engine.register_technique(
      std::make_shared<bruce::lhme::PowerOffsetKernel>()));
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};
  auto technique_request = power_offset_request();
  technique_request.mode = bruce::lhme::EvaluationMode::Metric;

  const auto warmed = engine.evaluate(formula_request);
  CHECK(warmed.status == bruce::lhme::Status::Confirmed);
  CHECK(warmed.metric_bit_length == 17);
  CHECK(!warmed.integer.has_value());
  CHECK(!warmed.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineFormulaCacheHitPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, formula_request] {
    return engine.evaluate(formula_request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_technique = engine.evaluate_technique(technique_request);
  CHECK(joining_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining_technique.integer.has_value());
  CHECK(!joining_technique.metric_bit_length.has_value());
  CHECK(!joining_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.metric_bit_length == 17);
  CHECK(!finalized.integer.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_technique = engine.evaluate_technique(technique_request);
  CHECK(reused_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_technique.metric_bit_length == 17);
  CHECK(!reused_technique.integer.has_value());
  CHECK(!reused_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 2);
}

TEST(cpp_engine_cancels_a_range_joining_a_post_snapshot_cancelled_metric_formula_cache_hit_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  const auto warmed = engine.evaluate(formula_request);
  CHECK(warmed.status == bruce::lhme::Status::Confirmed);
  CHECK(warmed.metric_bit_length == 17);
  CHECK(!warmed.integer.has_value());
  CHECK(!warmed.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineFormulaCacheHitPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, formula_request] {
    return engine.evaluate(formula_request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(joining_range.status == bruce::lhme::Status::Cancelled);
  CHECK(joining_range.candidates.empty());
  CHECK(engine.inspect().resident_entries == 1);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.metric_bit_length == 17);
  CHECK(!finalized.integer.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_cancels_a_metric_formula_cache_hit_joining_a_post_snapshot_cancelled_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::EvaluationRequest request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};

  const auto warmed = engine.evaluate(request);
  CHECK(warmed.status == bruce::lhme::Status::Confirmed);
  CHECK(warmed.metric_bit_length == 17);
  CHECK(!warmed.integer.has_value());
  CHECK(!warmed.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineFormulaCacheHitPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, request] {
    return engine.evaluate(request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining = engine.evaluate(request);
  CHECK(joining.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining.integer.has_value());
  CHECK(!joining.metric_bit_length.has_value());
  CHECK(!joining.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.metric_bit_length == 17);
  CHECK(!finalized.integer.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused = engine.evaluate(request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.metric_bit_length == 17);
  CHECK(!reused.integer.has_value());
  CHECK(!reused.symbolic_expression.has_value());
  CHECK(reused.trace != nullptr);
  CHECK(reused.trace->events.back().scope == bruce::lhme::GateScope::Cache);
  CHECK(reused.trace->events.back().reason == "cache hit");
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_cancels_a_technique_after_kernel_completion_and_reuses_after_completion) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  CHECK(engine.register_technique(
      std::make_shared<bruce::lhme::PowerOffsetKernel>()));
  const auto request = power_offset_request();

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueFinalizationBarrier
      finalization_barrier;
  pending = std::async(std::launch::async, [&engine, request] {
    return engine.evaluate_technique(request);
  });
  finalization_barrier.wait_until_reached();

  engine.cancel();
  finalization_barrier.release();
  const auto cancelled = pending.get();

  CHECK(cancelled.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled.integer.has_value());
  CHECK(!cancelled.metric_bit_length.has_value());
  CHECK(!cancelled.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused = engine.evaluate_technique(request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.exact_integer() == 102300);
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_cancels_a_metric_technique_cache_miss_before_its_completion_snapshot_and_reuses_after_completion) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  CHECK(engine.register_technique(
      std::make_shared<bruce::lhme::PowerOffsetKernel>()));
  auto request = power_offset_request();
  request.mode = bruce::lhme::EvaluationMode::Metric;

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueFinalizationBarrier
      finalization_barrier;
  pending = std::async(std::launch::async, [&engine, request] {
    return engine.evaluate_technique(request);
  });
  finalization_barrier.wait_until_reached();

  engine.cancel();
  finalization_barrier.release();
  const auto cancelled = pending.get();

  CHECK(cancelled.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled.integer.has_value());
  CHECK(!cancelled.metric_bit_length.has_value());
  CHECK(!cancelled.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused = engine.evaluate_technique(request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.metric_bit_length == 17);
  CHECK(!reused.integer.has_value());
  CHECK(!reused.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_cancels_a_modular_noncacheable_technique_before_its_completion_snapshot_and_reuses_after_completion) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const bruce::lhme::TechniqueRequest request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueNoncacheableFinalizationBarrier
      finalization_barrier;
  pending = std::async(std::launch::async, [&engine, request] {
    return engine.evaluate_technique(request);
  });
  finalization_barrier.wait_until_reached();

  engine.cancel();
  finalization_barrier.release();
  const auto cancelled = pending.get();

  CHECK(cancelled.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled.integer.has_value());
  CHECK(!cancelled.metric_bit_length.has_value());
  CHECK(!cancelled.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused = engine.evaluate_technique(request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.exact_integer() == 11);
  CHECK(!reused.metric_bit_length.has_value());
  CHECK(!reused.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);
}

TEST(cpp_engine_cancels_a_symbolic_noncacheable_technique_before_its_completion_snapshot_without_publishing_symbolic_output) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const bruce::lhme::TechniqueRequest request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueNoncacheableFinalizationBarrier
      finalization_barrier;
  pending = std::async(std::launch::async, [&engine, request] {
    return engine.evaluate_technique(request);
  });
  finalization_barrier.wait_until_reached();

  engine.cancel();
  finalization_barrier.release();
  const auto cancelled = pending.get();

  CHECK(cancelled.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled.integer.has_value());
  CHECK(!cancelled.metric_bit_length.has_value());
  CHECK(!cancelled.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused = engine.evaluate_technique(request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.symbolic_expression == "100*(2^10-1)");
  CHECK(!reused.integer.has_value());
  CHECK(!reused.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 0);
}

TEST(cpp_engine_keeps_a_symbolic_noncacheable_technique_result_final_after_its_completion_snapshot) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const bruce::lhme::TechniqueRequest request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, request] {
    return engine.evaluate_technique(request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  post_finalization_barrier.release();
  const auto finalized = pending.get();

  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.symbolic_expression == "100*(2^10-1)");
  CHECK(!finalized.integer.has_value());
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused = engine.evaluate_technique(request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.symbolic_expression == "100*(2^10-1)");
  CHECK(!reused.integer.has_value());
  CHECK(!reused.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 0);
}

TEST(cpp_engine_keeps_a_modular_noncacheable_technique_result_final_after_its_completion_snapshot) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const bruce::lhme::TechniqueRequest request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, request] {
    return engine.evaluate_technique(request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  post_finalization_barrier.release();
  const auto finalized = pending.get();

  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.exact_integer() == 11);
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused = engine.evaluate_technique(request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.exact_integer() == 11);
  CHECK(!reused.metric_bit_length.has_value());
  CHECK(!reused.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);
}

TEST(cpp_engine_cancels_a_modular_noncacheable_technique_joining_a_post_snapshot_cancelled_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const bruce::lhme::TechniqueRequest request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, request] {
    return engine.evaluate_technique(request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining = engine.evaluate_technique(request);
  CHECK(joining.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining.integer.has_value());
  CHECK(!joining.metric_bit_length.has_value());
  CHECK(!joining.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.exact_integer() == 11);
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused = engine.evaluate_technique(request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.exact_integer() == 11);
  CHECK(!reused.metric_bit_length.has_value());
  CHECK(!reused.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);
}

TEST(cpp_engine_cancels_a_range_joining_a_post_snapshot_cancelled_modular_noncacheable_technique_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::TechniqueRequest technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, technique_request] {
    return engine.evaluate_technique(technique_request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(joining_range.status == bruce::lhme::Status::Cancelled);
  CHECK(joining_range.candidates.empty());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.exact_integer() == 11);
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.summary.invalid_argument_count == 0);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 0);
}

TEST(cpp_engine_keeps_a_post_snapshot_cancelled_modular_noncacheable_technique_group_cancelled_across_multiple_joiner_lease_endings) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::TechniqueRequest technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, technique_request] {
    return engine.evaluate_technique(technique_request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_formula = engine.evaluate(formula_request);
  CHECK(joining_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining_formula.integer.has_value());
  CHECK(!joining_formula.metric_bit_length.has_value());
  CHECK(!joining_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto joining_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(joining_range.status == bruce::lhme::Status::Cancelled);
  CHECK(joining_range.candidates.empty());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.exact_integer() == 11);
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_formula = engine.evaluate(formula_request);
  CHECK(reused_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_formula.exact_integer() == 102300);
  CHECK(!reused_formula.metric_bit_length.has_value());
  CHECK(!reused_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.summary.invalid_argument_count == 0);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_keeps_a_post_snapshot_cancelled_modular_noncacheable_technique_group_cancelled_after_repeated_cancels_and_joiner_lease_endings) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::TechniqueRequest technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, technique_request] {
    return engine.evaluate_technique(technique_request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_formula = engine.evaluate(formula_request);
  CHECK(joining_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining_formula.integer.has_value());
  CHECK(!joining_formula.metric_bit_length.has_value());
  CHECK(!joining_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  const auto joining_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(joining_range.status == bruce::lhme::Status::Cancelled);
  CHECK(joining_range.candidates.empty());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.exact_integer() == 11);
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_formula = engine.evaluate(formula_request);
  CHECK(reused_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_formula.exact_integer() == 102300);
  CHECK(!reused_formula.metric_bit_length.has_value());
  CHECK(!reused_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.summary.invalid_argument_count == 0);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_cancels_concurrently_released_formula_and_range_joiners_of_a_post_snapshot_cancelled_modular_noncacheable_technique_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::TechniqueRequest technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, technique_request] {
    return engine.evaluate_technique(technique_request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  std::atomic_bool joiners_start{false};
  auto formula_joiner = std::async(
      std::launch::async, [&engine, formula_request, &joiners_start] {
        joiners_start.wait(false, std::memory_order_acquire);
        return engine.evaluate(formula_request);
      });
  auto range_joiner = std::async(
      std::launch::async, [&engine, range_request, &joiners_start] {
        joiners_start.wait(false, std::memory_order_acquire);
        return engine.evaluate_range(range_request, {0, 1});
      });
  joiners_start.store(true, std::memory_order_release);
  joiners_start.notify_all();

  const auto joining_formula = formula_joiner.get();
  const auto joining_range = range_joiner.get();
  CHECK(joining_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining_formula.integer.has_value());
  CHECK(!joining_formula.metric_bit_length.has_value());
  CHECK(!joining_formula.symbolic_expression.has_value());
  CHECK(joining_range.status == bruce::lhme::Status::Cancelled);
  CHECK(joining_range.candidates.empty());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.exact_integer() == 11);
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_formula = engine.evaluate(formula_request);
  CHECK(reused_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_formula.exact_integer() == 102300);
  CHECK(!reused_formula.metric_bit_length.has_value());
  CHECK(!reused_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.summary.invalid_argument_count == 0);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_cancels_concurrently_released_formula_range_and_technique_joiners_of_a_post_snapshot_cancelled_modular_noncacheable_technique_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::TechniqueRequest modular_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::TechniqueRequest metric_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(
      std::launch::async, [&engine, modular_technique_request] {
        return engine.evaluate_technique(modular_technique_request);
      });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  std::atomic_bool joiners_start{false};
  auto formula_joiner = std::async(
      std::launch::async, [&engine, formula_request, &joiners_start] {
        joiners_start.wait(false, std::memory_order_acquire);
        return engine.evaluate(formula_request);
      });
  auto range_joiner = std::async(
      std::launch::async, [&engine, range_request, &joiners_start] {
        joiners_start.wait(false, std::memory_order_acquire);
        return engine.evaluate_range(range_request, {0, 1});
      });
  auto technique_joiner = std::async(
      std::launch::async,
      [&engine, metric_technique_request, &joiners_start] {
        joiners_start.wait(false, std::memory_order_acquire);
        return engine.evaluate_technique(metric_technique_request);
      });
  joiners_start.store(true, std::memory_order_release);
  joiners_start.notify_all();

  const auto joining_formula = formula_joiner.get();
  const auto joining_range = range_joiner.get();
  const auto joining_technique = technique_joiner.get();
  CHECK(joining_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining_formula.integer.has_value());
  CHECK(!joining_formula.metric_bit_length.has_value());
  CHECK(!joining_formula.symbolic_expression.has_value());
  CHECK(joining_range.status == bruce::lhme::Status::Cancelled);
  CHECK(joining_range.candidates.empty());
  CHECK(joining_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining_technique.integer.has_value());
  CHECK(!joining_technique.metric_bit_length.has_value());
  CHECK(!joining_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.exact_integer() == 11);
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_formula = engine.evaluate(formula_request);
  CHECK(reused_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_formula.exact_integer() == 102300);
  CHECK(!reused_formula.metric_bit_length.has_value());
  CHECK(!reused_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.summary.invalid_argument_count == 0);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_technique = engine.evaluate_technique(metric_technique_request);
  CHECK(reused_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_technique.metric_bit_length == 17);
  CHECK(!reused_technique.integer.has_value());
  CHECK(!reused_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 2);
}

TEST(cpp_engine_cancels_concurrently_released_formula_modular_formula_range_and_technique_joiners_of_a_post_snapshot_cancelled_modular_noncacheable_technique_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::TechniqueRequest modular_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::TechniqueRequest metric_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest exact_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest modular_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(
      std::launch::async, [&engine, modular_technique_request] {
        return engine.evaluate_technique(modular_technique_request);
      });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  std::atomic_bool joiners_start{false};
  auto exact_formula_joiner = std::async(
      std::launch::async, [&engine, exact_formula_request, &joiners_start] {
        joiners_start.wait(false, std::memory_order_acquire);
        return engine.evaluate(exact_formula_request);
      });
  auto modular_formula_joiner = std::async(
      std::launch::async,
      [&engine, modular_formula_request, &joiners_start] {
        joiners_start.wait(false, std::memory_order_acquire);
        return engine.evaluate(modular_formula_request);
      });
  auto range_joiner = std::async(
      std::launch::async, [&engine, range_request, &joiners_start] {
        joiners_start.wait(false, std::memory_order_acquire);
        return engine.evaluate_range(range_request, {0, 1});
      });
  auto technique_joiner = std::async(
      std::launch::async,
      [&engine, metric_technique_request, &joiners_start] {
        joiners_start.wait(false, std::memory_order_acquire);
        return engine.evaluate_technique(metric_technique_request);
      });
  joiners_start.store(true, std::memory_order_release);
  joiners_start.notify_all();

  const auto joining_exact_formula = exact_formula_joiner.get();
  const auto joining_modular_formula = modular_formula_joiner.get();
  const auto joining_range = range_joiner.get();
  const auto joining_technique = technique_joiner.get();
  CHECK(joining_exact_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining_exact_formula.integer.has_value());
  CHECK(!joining_exact_formula.metric_bit_length.has_value());
  CHECK(!joining_exact_formula.symbolic_expression.has_value());
  CHECK(joining_modular_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining_modular_formula.integer.has_value());
  CHECK(!joining_modular_formula.metric_bit_length.has_value());
  CHECK(!joining_modular_formula.symbolic_expression.has_value());
  CHECK(joining_range.status == bruce::lhme::Status::Cancelled);
  CHECK(joining_range.candidates.empty());
  CHECK(joining_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining_technique.integer.has_value());
  CHECK(!joining_technique.metric_bit_length.has_value());
  CHECK(!joining_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.exact_integer() == 11);
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_exact_formula = engine.evaluate(exact_formula_request);
  CHECK(reused_exact_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_exact_formula.exact_integer() == 102300);
  CHECK(!reused_exact_formula.metric_bit_length.has_value());
  CHECK(!reused_exact_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_modular_formula = engine.evaluate(modular_formula_request);
  CHECK(reused_modular_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_modular_formula.exact_integer() == 11);
  CHECK(!reused_modular_formula.metric_bit_length.has_value());
  CHECK(!reused_modular_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.summary.invalid_argument_count == 0);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_technique = engine.evaluate_technique(metric_technique_request);
  CHECK(reused_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_technique.metric_bit_length == 17);
  CHECK(!reused_technique.integer.has_value());
  CHECK(!reused_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 2);
}

TEST(cpp_engine_cancels_an_exact_formula_evaluation_joining_a_post_snapshot_cancelled_modular_noncacheable_technique_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::TechniqueRequest technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, technique_request] {
    return engine.evaluate_technique(technique_request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_formula = engine.evaluate(formula_request);
  CHECK(joining_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining_formula.integer.has_value());
  CHECK(!joining_formula.metric_bit_length.has_value());
  CHECK(!joining_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.exact_integer() == 11);
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_formula = engine.evaluate(formula_request);
  CHECK(reused_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_formula.exact_integer() == 102300);
  CHECK(!reused_formula.metric_bit_length.has_value());
  CHECK(!reused_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_cancels_a_metric_formula_evaluation_joining_a_post_snapshot_cancelled_modular_noncacheable_technique_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::TechniqueRequest technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, technique_request] {
    return engine.evaluate_technique(technique_request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_formula = engine.evaluate(formula_request);
  CHECK(joining_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining_formula.integer.has_value());
  CHECK(!joining_formula.metric_bit_length.has_value());
  CHECK(!joining_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.exact_integer() == 11);
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_formula = engine.evaluate(formula_request);
  CHECK(reused_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_formula.metric_bit_length == 17);
  CHECK(!reused_formula.integer.has_value());
  CHECK(!reused_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_cancels_a_modular_formula_evaluation_joining_a_post_snapshot_cancelled_modular_noncacheable_technique_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::TechniqueRequest technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, technique_request] {
    return engine.evaluate_technique(technique_request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_formula = engine.evaluate(formula_request);
  CHECK(joining_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining_formula.integer.has_value());
  CHECK(!joining_formula.metric_bit_length.has_value());
  CHECK(!joining_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.exact_integer() == 11);
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_formula = engine.evaluate(formula_request);
  CHECK(reused_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_formula.exact_integer() == 11);
  CHECK(!reused_formula.metric_bit_length.has_value());
  CHECK(!reused_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);
}

TEST(cpp_engine_cancels_a_symbolic_formula_evaluation_joining_a_post_snapshot_cancelled_modular_noncacheable_technique_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::TechniqueRequest technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, technique_request] {
    return engine.evaluate_technique(technique_request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_formula = engine.evaluate(formula_request);
  CHECK(joining_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining_formula.integer.has_value());
  CHECK(!joining_formula.metric_bit_length.has_value());
  CHECK(!joining_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.exact_integer() == 11);
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_formula = engine.evaluate(formula_request);
  CHECK(reused_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_formula.symbolic_expression == "100*(2^10-1)");
  CHECK(!reused_formula.integer.has_value());
  CHECK(!reused_formula.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 0);
}

TEST(cpp_engine_cancels_an_exact_technique_evaluation_joining_a_post_snapshot_cancelled_modular_noncacheable_technique_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const bruce::lhme::TechniqueRequest modular_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::TechniqueRequest exact_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, modular_request] {
    return engine.evaluate_technique(modular_request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining = engine.evaluate_technique(exact_request);
  CHECK(joining.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining.integer.has_value());
  CHECK(!joining.metric_bit_length.has_value());
  CHECK(!joining.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.exact_integer() == 11);
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused = engine.evaluate_technique(exact_request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.exact_integer() == 102300);
  CHECK(!reused.metric_bit_length.has_value());
  CHECK(!reused.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_cancels_a_metric_technique_evaluation_joining_a_post_snapshot_cancelled_modular_noncacheable_technique_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const bruce::lhme::TechniqueRequest modular_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::TechniqueRequest metric_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, modular_request] {
    return engine.evaluate_technique(modular_request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining = engine.evaluate_technique(metric_request);
  CHECK(joining.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining.integer.has_value());
  CHECK(!joining.metric_bit_length.has_value());
  CHECK(!joining.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.exact_integer() == 11);
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused = engine.evaluate_technique(metric_request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.metric_bit_length == 17);
  CHECK(!reused.integer.has_value());
  CHECK(!reused.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_cancels_a_symbolic_technique_evaluation_joining_a_post_snapshot_cancelled_modular_noncacheable_technique_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const bruce::lhme::TechniqueRequest modular_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::TechniqueRequest symbolic_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, modular_request] {
    return engine.evaluate_technique(modular_request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining = engine.evaluate_technique(symbolic_request);
  CHECK(joining.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining.integer.has_value());
  CHECK(!joining.metric_bit_length.has_value());
  CHECK(!joining.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.exact_integer() == 11);
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused = engine.evaluate_technique(symbolic_request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.symbolic_expression == "100*(2^10-1)");
  CHECK(!reused.integer.has_value());
  CHECK(!reused.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 0);
}

TEST(cpp_engine_cancels_a_symbolic_noncacheable_technique_joining_a_post_snapshot_cancelled_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const bruce::lhme::TechniqueRequest request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, request] {
    return engine.evaluate_technique(request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining = engine.evaluate_technique(request);
  CHECK(joining.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining.integer.has_value());
  CHECK(!joining.metric_bit_length.has_value());
  CHECK(!joining.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.symbolic_expression == "100*(2^10-1)");
  CHECK(!finalized.integer.has_value());
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused = engine.evaluate_technique(request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.symbolic_expression == "100*(2^10-1)");
  CHECK(!reused.integer.has_value());
  CHECK(!reused.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 0);
}

TEST(cpp_engine_cancels_a_range_joining_a_post_snapshot_cancelled_symbolic_noncacheable_technique_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::TechniqueRequest technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, technique_request] {
    return engine.evaluate_technique(technique_request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(joining_range.status == bruce::lhme::Status::Cancelled);
  CHECK(joining_range.candidates.empty());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.symbolic_expression == "100*(2^10-1)");
  CHECK(!finalized.integer.has_value());
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.summary.invalid_argument_count == 0);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 0);
}

TEST(cpp_engine_cancels_a_formula_evaluation_joining_a_post_snapshot_cancelled_symbolic_noncacheable_technique_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::TechniqueRequest technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, technique_request] {
    return engine.evaluate_technique(technique_request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_formula = engine.evaluate(formula_request);
  CHECK(joining_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining_formula.integer.has_value());
  CHECK(!joining_formula.metric_bit_length.has_value());
  CHECK(!joining_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.symbolic_expression == "100*(2^10-1)");
  CHECK(!finalized.integer.has_value());
  CHECK(!finalized.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_formula = engine.evaluate(formula_request);
  CHECK(reused_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_formula.exact_integer() == 102300);
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_keeps_a_noncacheable_technique_result_final_after_its_completion_snapshot) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const bruce::lhme::TechniqueRequest request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, request] {
    return engine.evaluate_technique(request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  post_finalization_barrier.release();
  const auto finalized = pending.get();

  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.exact_integer() == 11);
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused = engine.evaluate_technique(request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.exact_integer() == 11);
  CHECK(engine.inspect().resident_entries == 0);
}

TEST(cpp_engine_cancels_a_noncacheable_technique_joining_a_post_snapshot_cancelled_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const bruce::lhme::TechniqueRequest request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, request] {
    return engine.evaluate_technique(request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining = engine.evaluate_technique(request);
  CHECK(joining.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining.integer.has_value());
  CHECK(!joining.metric_bit_length.has_value());
  CHECK(!joining.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.exact_integer() == 11);
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused = engine.evaluate_technique(request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.exact_integer() == 11);
  CHECK(engine.inspect().resident_entries == 0);
}

TEST(cpp_engine_keeps_a_technique_cache_miss_final_after_its_completion_snapshot) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  CHECK(engine.register_technique(
      std::make_shared<bruce::lhme::PowerOffsetKernel>()));
  const auto request = power_offset_request();

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueCacheMissPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, request] {
    return engine.evaluate_technique(request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  post_finalization_barrier.release();
  const auto finalized = pending.get();

  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.exact_integer() == 102300);
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused = engine.evaluate_technique(request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.exact_integer() == 102300);
  CHECK(reused.trace != nullptr);
  CHECK(reused.trace->events.back().scope == bruce::lhme::GateScope::Cache);
  CHECK(reused.trace->events.back().reason == "cache hit");
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_keeps_a_metric_technique_cache_miss_final_after_its_completion_snapshot) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  CHECK(engine.register_technique(
      std::make_shared<bruce::lhme::PowerOffsetKernel>()));
  auto request = power_offset_request();
  request.mode = bruce::lhme::EvaluationMode::Metric;

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueCacheMissPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, request] {
    return engine.evaluate_technique(request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  post_finalization_barrier.release();
  const auto finalized = pending.get();

  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.metric_bit_length == 17);
  CHECK(!finalized.integer.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused = engine.evaluate_technique(request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.metric_bit_length == 17);
  CHECK(!reused.integer.has_value());
  CHECK(!reused.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_cancels_a_metric_technique_cache_miss_joining_a_post_snapshot_cancelled_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  CHECK(engine.register_technique(
      std::make_shared<bruce::lhme::PowerOffsetKernel>()));
  auto request = power_offset_request();
  request.mode = bruce::lhme::EvaluationMode::Metric;

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueCacheMissPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, request] {
    return engine.evaluate_technique(request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining = engine.evaluate_technique(request);
  CHECK(joining.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining.integer.has_value());
  CHECK(!joining.metric_bit_length.has_value());
  CHECK(!joining.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.metric_bit_length == 17);
  CHECK(!finalized.integer.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused = engine.evaluate_technique(request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.metric_bit_length == 17);
  CHECK(!reused.integer.has_value());
  CHECK(!reused.symbolic_expression.has_value());
  CHECK(reused.trace != nullptr);
  CHECK(reused.trace->events.back().scope == bruce::lhme::GateScope::Cache);
  CHECK(reused.trace->events.back().reason == "cache hit");
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_cancels_a_metric_formula_evaluation_joining_a_post_snapshot_cancelled_metric_technique_cache_miss_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  CHECK(engine.register_technique(
      std::make_shared<bruce::lhme::PowerOffsetKernel>()));
  auto technique_request = power_offset_request();
  technique_request.mode = bruce::lhme::EvaluationMode::Metric;
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueCacheMissPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, technique_request] {
    return engine.evaluate_technique(technique_request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_formula = engine.evaluate(formula_request);
  CHECK(joining_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining_formula.integer.has_value());
  CHECK(!joining_formula.metric_bit_length.has_value());
  CHECK(!joining_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.metric_bit_length == 17);
  CHECK(!finalized.integer.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_formula = engine.evaluate(formula_request);
  CHECK(reused_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_formula.metric_bit_length == 17);
  CHECK(!reused_formula.integer.has_value());
  CHECK(!reused_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 2);
}

TEST(cpp_engine_cancels_a_range_joining_a_post_snapshot_cancelled_metric_technique_cache_miss_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  auto technique_request = power_offset_request();
  technique_request.mode = bruce::lhme::EvaluationMode::Metric;
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueCacheMissPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, technique_request] {
    return engine.evaluate_technique(technique_request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(joining_range.status == bruce::lhme::Status::Cancelled);
  CHECK(joining_range.candidates.empty());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.metric_bit_length == 17);
  CHECK(!finalized.integer.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_cancels_a_technique_cache_miss_joining_a_post_snapshot_cancelled_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  CHECK(engine.register_technique(
      std::make_shared<bruce::lhme::PowerOffsetKernel>()));
  const auto request = power_offset_request();

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueCacheMissPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, request] {
    return engine.evaluate_technique(request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining = engine.evaluate_technique(request);
  CHECK(joining.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining.integer.has_value());
  CHECK(!joining.metric_bit_length.has_value());
  CHECK(!joining.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.exact_integer() == 102300);
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused = engine.evaluate_technique(request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.exact_integer() == 102300);
  CHECK(reused.trace != nullptr);
  CHECK(reused.trace->events.back().scope == bruce::lhme::GateScope::Cache);
  CHECK(reused.trace->events.back().reason == "cache hit");
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_cancels_a_technique_cache_hit_after_cache_completion_and_reuses_after_completion) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  CHECK(engine.register_technique(
      std::make_shared<bruce::lhme::PowerOffsetKernel>()));
  const auto request = power_offset_request();

  const auto warmed = engine.evaluate_technique(request);
  CHECK(warmed.status == bruce::lhme::Status::Confirmed);
  CHECK(warmed.exact_integer() == 102300);
  CHECK(engine.inspect().resident_entries == 1);

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueCacheHitFinalizationBarrier
      finalization_barrier;
  pending = std::async(std::launch::async, [&engine, request] {
    return engine.evaluate_technique(request);
  });
  finalization_barrier.wait_until_reached();

  engine.cancel();
  finalization_barrier.release();
  const auto cancelled = pending.get();

  CHECK(cancelled.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled.integer.has_value());
  CHECK(!cancelled.metric_bit_length.has_value());
  CHECK(!cancelled.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused = engine.evaluate_technique(request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.exact_integer() == 102300);
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_cancels_a_metric_technique_cache_hit_before_its_completion_snapshot_and_reuses_after_completion) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  CHECK(engine.register_technique(
      std::make_shared<bruce::lhme::PowerOffsetKernel>()));
  auto request = power_offset_request();
  request.mode = bruce::lhme::EvaluationMode::Metric;

  const auto warmed = engine.evaluate_technique(request);
  CHECK(warmed.status == bruce::lhme::Status::Confirmed);
  CHECK(warmed.metric_bit_length == 17);
  CHECK(!warmed.integer.has_value());
  CHECK(!warmed.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueCacheHitFinalizationBarrier
      finalization_barrier;
  pending = std::async(std::launch::async, [&engine, request] {
    return engine.evaluate_technique(request);
  });
  finalization_barrier.wait_until_reached();

  engine.cancel();
  finalization_barrier.release();
  const auto cancelled = pending.get();

  CHECK(cancelled.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled.integer.has_value());
  CHECK(!cancelled.metric_bit_length.has_value());
  CHECK(!cancelled.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused = engine.evaluate_technique(request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.metric_bit_length == 17);
  CHECK(!reused.integer.has_value());
  CHECK(!reused.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_keeps_a_technique_cache_hit_final_after_its_completion_snapshot) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  CHECK(engine.register_technique(
      std::make_shared<bruce::lhme::PowerOffsetKernel>()));
  const auto request = power_offset_request();

  const auto warmed = engine.evaluate_technique(request);
  CHECK(warmed.status == bruce::lhme::Status::Confirmed);
  CHECK(warmed.exact_integer() == 102300);
  CHECK(engine.inspect().resident_entries == 1);

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueCacheHitPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, request] {
    return engine.evaluate_technique(request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  post_finalization_barrier.release();
  const auto finalized = pending.get();

  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.exact_integer() == 102300);
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused = engine.evaluate_technique(request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.exact_integer() == 102300);
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_keeps_a_metric_technique_cache_hit_final_after_its_completion_snapshot) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  CHECK(engine.register_technique(
      std::make_shared<bruce::lhme::PowerOffsetKernel>()));
  auto request = power_offset_request();
  request.mode = bruce::lhme::EvaluationMode::Metric;

  const auto warmed = engine.evaluate_technique(request);
  CHECK(warmed.status == bruce::lhme::Status::Confirmed);
  CHECK(warmed.metric_bit_length == 17);
  CHECK(!warmed.integer.has_value());
  CHECK(!warmed.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueCacheHitPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, request] {
    return engine.evaluate_technique(request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  post_finalization_barrier.release();
  const auto finalized = pending.get();

  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.metric_bit_length == 17);
  CHECK(!finalized.integer.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused = engine.evaluate_technique(request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.metric_bit_length == 17);
  CHECK(!reused.integer.has_value());
  CHECK(!reused.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_cancels_a_metric_technique_cache_hit_joining_a_post_snapshot_cancelled_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  CHECK(engine.register_technique(
      std::make_shared<bruce::lhme::PowerOffsetKernel>()));
  auto request = power_offset_request();
  request.mode = bruce::lhme::EvaluationMode::Metric;

  const auto warmed = engine.evaluate_technique(request);
  CHECK(warmed.status == bruce::lhme::Status::Confirmed);
  CHECK(warmed.metric_bit_length == 17);
  CHECK(!warmed.integer.has_value());
  CHECK(!warmed.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueCacheHitPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, request] {
    return engine.evaluate_technique(request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining = engine.evaluate_technique(request);
  CHECK(joining.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining.integer.has_value());
  CHECK(!joining.metric_bit_length.has_value());
  CHECK(!joining.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.metric_bit_length == 17);
  CHECK(!finalized.integer.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused = engine.evaluate_technique(request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.metric_bit_length == 17);
  CHECK(!reused.integer.has_value());
  CHECK(!reused.symbolic_expression.has_value());
  CHECK(reused.trace != nullptr);
  CHECK(reused.trace->events.back().scope == bruce::lhme::GateScope::Cache);
  CHECK(reused.trace->events.back().reason == "cache hit");
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_cancels_a_range_joining_a_post_snapshot_cancelled_metric_technique_cache_hit_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  auto technique_request = power_offset_request();
  technique_request.mode = bruce::lhme::EvaluationMode::Metric;
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  const auto warmed = engine.evaluate_technique(technique_request);
  CHECK(warmed.status == bruce::lhme::Status::Confirmed);
  CHECK(warmed.metric_bit_length == 17);
  CHECK(!warmed.integer.has_value());
  CHECK(!warmed.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueCacheHitPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, technique_request] {
    return engine.evaluate_technique(technique_request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(joining_range.status == bruce::lhme::Status::Cancelled);
  CHECK(joining_range.candidates.empty());
  CHECK(engine.inspect().resident_entries == 1);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.metric_bit_length == 17);
  CHECK(!finalized.integer.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_cancels_a_formula_evaluation_joining_a_post_snapshot_cancelled_metric_technique_cache_hit_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  auto technique_request = power_offset_request();
  technique_request.mode = bruce::lhme::EvaluationMode::Metric;
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  const auto warmed = engine.evaluate_technique(technique_request);
  CHECK(warmed.status == bruce::lhme::Status::Confirmed);
  CHECK(warmed.metric_bit_length == 17);
  CHECK(!warmed.integer.has_value());
  CHECK(!warmed.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueCacheHitPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, technique_request] {
    return engine.evaluate_technique(technique_request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining_formula = engine.evaluate(formula_request);
  CHECK(joining_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining_formula.integer.has_value());
  CHECK(!joining_formula.metric_bit_length.has_value());
  CHECK(!joining_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.metric_bit_length == 17);
  CHECK(!finalized.integer.has_value());
  CHECK(!finalized.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_formula = engine.evaluate(formula_request);
  CHECK(reused_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_formula.exact_integer() == 102300);
  CHECK(engine.inspect().resident_entries == 2);
}

TEST(cpp_engine_cancels_a_technique_cache_hit_joining_a_post_snapshot_cancelled_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  CHECK(engine.register_technique(
      std::make_shared<bruce::lhme::PowerOffsetKernel>()));
  const auto request = power_offset_request();

  const auto warmed = engine.evaluate_technique(request);
  CHECK(warmed.status == bruce::lhme::Status::Confirmed);
  CHECK(warmed.exact_integer() == 102300);
  CHECK(engine.inspect().resident_entries == 1);

  std::future<bruce::lhme::EvaluationResult> pending;
  bruce::lhme::test_support::EngineTechniqueCacheHitPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&engine, request] {
    return engine.evaluate_technique(request);
  });
  post_finalization_barrier.wait_until_reached();

  engine.cancel();
  const auto joining = engine.evaluate_technique(request);
  CHECK(joining.status == bruce::lhme::Status::Cancelled);
  CHECK(!joining.integer.has_value());
  CHECK(!joining.metric_bit_length.has_value());
  CHECK(!joining.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  post_finalization_barrier.release();
  const auto finalized = pending.get();
  CHECK(finalized.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized.exact_integer() == 102300);
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused = engine.evaluate_technique(request);
  CHECK(reused.status == bruce::lhme::Status::Confirmed);
  CHECK(reused.exact_integer() == 102300);
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_releases_cancellation_state_after_an_invalid_evaluation) {
  auto engine = bruce::lhme::Engine::create({});
  const auto registered = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const auto unregistered = bruce::lhme::make_power_offset_spec(3, 100, 1);

  CHECK(engine.evaluate(
            {.spec = unregistered,
             .address = {.index = 10},
             .mode = bruce::lhme::EvaluationMode::Exact,
             .budget = bruce::lhme::Budget::unlimited()})
            .status == bruce::lhme::Status::InvalidArgument);
  engine.cancel();
  CHECK(engine.evaluate(
            {.spec = registered,
             .address = {.index = 10},
             .mode = bruce::lhme::EvaluationMode::Exact,
             .budget = bruce::lhme::Budget::unlimited()})
            .status == bruce::lhme::Status::Confirmed);
}

TEST(cpp_engine_cancels_an_inflight_exact_formula_joining_a_post_snapshot_finalized_modular_noncacheable_technique_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::TechniqueRequest technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_technique;
  std::future<bruce::lhme::EvaluationResult> pending_formula;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      technique_post_finalization_barrier;
  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  pending_technique = std::async(std::launch::async, [&engine, technique_request] {
    return engine.evaluate_technique(technique_request);
  });
  technique_post_finalization_barrier.wait_until_reached();

  pending_formula = std::async(std::launch::async, [&engine, formula_request] {
    return engine.evaluate(formula_request);
  });
  formula_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_formula = pending_formula.get();

  CHECK(cancelled_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_formula.integer.has_value());
  CHECK(!cancelled_formula.metric_bit_length.has_value());
  CHECK(!cancelled_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  technique_post_finalization_barrier.release();
  const auto finalized_technique = pending_technique.get();

  CHECK(finalized_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_technique.exact_integer() == 11);
  CHECK(!finalized_technique.metric_bit_length.has_value());
  CHECK(!finalized_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_formula = engine.evaluate(formula_request);
  CHECK(reused_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_formula.exact_integer() == 102300);
  CHECK(!reused_formula.metric_bit_length.has_value());
  CHECK(!reused_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_cancels_inflight_formula_and_range_joiners_of_a_post_snapshot_finalized_modular_noncacheable_technique_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::TechniqueRequest technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_technique;
  std::future<bruce::lhme::EvaluationResult> pending_formula;
  std::future<bruce::lhme::RangeResult> pending_range;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      technique_post_finalization_barrier;
  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  bruce::lhme::test_support::EngineRangeFinalizationBarrier
      range_finalization_barrier;
  pending_technique = std::async(std::launch::async, [&engine, technique_request] {
    return engine.evaluate_technique(technique_request);
  });
  technique_post_finalization_barrier.wait_until_reached();

  pending_formula = std::async(std::launch::async, [&engine, formula_request] {
    return engine.evaluate(formula_request);
  });
  formula_finalization_barrier.wait_until_reached();

  pending_range = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  range_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  range_finalization_barrier.release();
  const auto cancelled_formula = pending_formula.get();
  const auto cancelled_range = pending_range.get();

  CHECK(cancelled_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_formula.integer.has_value());
  CHECK(!cancelled_formula.metric_bit_length.has_value());
  CHECK(!cancelled_formula.symbolic_expression.has_value());
  CHECK(cancelled_range.status == bruce::lhme::Status::Cancelled);
  CHECK(cancelled_range.candidates.empty());
  CHECK(engine.inspect().resident_entries == 0);

  technique_post_finalization_barrier.release();
  const auto finalized_technique = pending_technique.get();

  CHECK(finalized_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_technique.exact_integer() == 11);
  CHECK(!finalized_technique.metric_bit_length.has_value());
  CHECK(!finalized_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_formula = engine.evaluate(formula_request);
  CHECK(reused_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_formula.exact_integer() == 102300);
  CHECK(!reused_formula.metric_bit_length.has_value());
  CHECK(!reused_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.summary.invalid_argument_count == 0);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_cancels_inflight_formula_range_and_metric_technique_joiners_of_a_post_snapshot_finalized_modular_noncacheable_technique_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::TechniqueRequest modular_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest metric_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_formula;
  std::future<bruce::lhme::RangeResult> pending_range;
  std::future<bruce::lhme::EvaluationResult> pending_metric_technique;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier;
  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  bruce::lhme::test_support::EngineRangeFinalizationBarrier
      range_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueFinalizationBarrier
      technique_finalization_barrier;
  pending_origin = std::async(std::launch::async,
                              [&engine, modular_technique_request] {
    return engine.evaluate_technique(modular_technique_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  pending_formula = std::async(std::launch::async, [&engine, formula_request] {
    return engine.evaluate(formula_request);
  });
  formula_finalization_barrier.wait_until_reached();

  pending_range = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  range_finalization_barrier.wait_until_reached();

  pending_metric_technique = std::async(
      std::launch::async, [&engine, metric_technique_request] {
        return engine.evaluate_technique(metric_technique_request);
      });
  technique_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  range_finalization_barrier.release();
  technique_finalization_barrier.release();
  const auto cancelled_formula = pending_formula.get();
  const auto cancelled_range = pending_range.get();
  const auto cancelled_metric_technique = pending_metric_technique.get();

  CHECK(cancelled_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_formula.integer.has_value());
  CHECK(!cancelled_formula.metric_bit_length.has_value());
  CHECK(!cancelled_formula.symbolic_expression.has_value());
  CHECK(cancelled_range.status == bruce::lhme::Status::Cancelled);
  CHECK(cancelled_range.candidates.empty());
  CHECK(cancelled_metric_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_metric_technique.integer.has_value());
  CHECK(!cancelled_metric_technique.metric_bit_length.has_value());
  CHECK(!cancelled_metric_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();

  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_formula = engine.evaluate(formula_request);
  CHECK(reused_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_formula.exact_integer() == 102300);
  CHECK(!reused_formula.metric_bit_length.has_value());
  CHECK(!reused_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.summary.invalid_argument_count == 0);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_metric_technique = engine.evaluate_technique(
      metric_technique_request);
  CHECK(reused_metric_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_metric_technique.metric_bit_length == 17);
  CHECK(!reused_metric_technique.integer.has_value());
  CHECK(!reused_metric_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 2);
}

TEST(cpp_engine_cancels_inflight_formula_range_metric_and_symbolic_technique_joiners_of_a_post_snapshot_finalized_modular_noncacheable_technique_group_without_mutating_the_finalized_result) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::TechniqueRequest modular_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest metric_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest symbolic_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_formula;
  std::future<bruce::lhme::RangeResult> pending_range;
  std::future<bruce::lhme::EvaluationResult> pending_metric_technique;
  std::future<bruce::lhme::EvaluationResult> pending_symbolic_technique;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async,
                              [&engine, modular_technique_request] {
    return engine.evaluate_technique(modular_technique_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  bruce::lhme::test_support::EngineRangeFinalizationBarrier
      range_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueFinalizationBarrier
      technique_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueNoncacheableFinalizationBarrier
      noncacheable_technique_finalization_barrier;

  pending_formula = std::async(std::launch::async, [&engine, formula_request] {
    return engine.evaluate(formula_request);
  });
  formula_finalization_barrier.wait_until_reached();

  pending_range = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  range_finalization_barrier.wait_until_reached();

  pending_metric_technique = std::async(
      std::launch::async, [&engine, metric_technique_request] {
        return engine.evaluate_technique(metric_technique_request);
      });
  technique_finalization_barrier.wait_until_reached();

  pending_symbolic_technique = std::async(
      std::launch::async, [&engine, symbolic_technique_request] {
        return engine.evaluate_technique(symbolic_technique_request);
      });
  noncacheable_technique_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  range_finalization_barrier.release();
  technique_finalization_barrier.release();
  noncacheable_technique_finalization_barrier.release();
  const auto cancelled_formula = pending_formula.get();
  const auto cancelled_range = pending_range.get();
  const auto cancelled_metric_technique = pending_metric_technique.get();
  const auto cancelled_symbolic_technique = pending_symbolic_technique.get();

  CHECK(cancelled_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_formula.integer.has_value());
  CHECK(!cancelled_formula.metric_bit_length.has_value());
  CHECK(!cancelled_formula.symbolic_expression.has_value());
  CHECK(cancelled_range.status == bruce::lhme::Status::Cancelled);
  CHECK(cancelled_range.candidates.empty());
  CHECK(cancelled_metric_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_metric_technique.integer.has_value());
  CHECK(!cancelled_metric_technique.metric_bit_length.has_value());
  CHECK(!cancelled_metric_technique.symbolic_expression.has_value());
  CHECK(cancelled_symbolic_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_symbolic_technique.integer.has_value());
  CHECK(!cancelled_symbolic_technique.metric_bit_length.has_value());
  CHECK(!cancelled_symbolic_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();

  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_formula = engine.evaluate(formula_request);
  CHECK(reused_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_formula.exact_integer() == 102300);
  CHECK(!reused_formula.metric_bit_length.has_value());
  CHECK(!reused_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.summary.invalid_argument_count == 0);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_metric_technique = engine.evaluate_technique(
      metric_technique_request);
  CHECK(reused_metric_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_metric_technique.metric_bit_length == 17);
  CHECK(!reused_metric_technique.integer.has_value());
  CHECK(!reused_metric_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 2);

  const auto reused_symbolic_technique = engine.evaluate_technique(
      symbolic_technique_request);
  CHECK(reused_symbolic_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_symbolic_technique.symbolic_expression == "100*(2^10-1)");
  CHECK(!reused_symbolic_technique.integer.has_value());
  CHECK(!reused_symbolic_technique.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 2);
}

TEST(cpp_engine_keeps_a_post_snapshot_cancelled_modular_noncacheable_technique_group_cancelled_for_a_late_formula_joiner_after_all_inflight_joiner_lease_endings_and_repeated_cancels) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::TechniqueRequest modular_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest metric_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest symbolic_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_formula;
  std::future<bruce::lhme::RangeResult> pending_range;
  std::future<bruce::lhme::EvaluationResult> pending_metric_technique;
  std::future<bruce::lhme::EvaluationResult> pending_symbolic_technique;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async,
                              [&engine, modular_technique_request] {
    return engine.evaluate_technique(modular_technique_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  bruce::lhme::test_support::EngineRangeFinalizationBarrier
      range_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueFinalizationBarrier
      technique_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueNoncacheableFinalizationBarrier
      noncacheable_technique_finalization_barrier;

  pending_formula = std::async(std::launch::async, [&engine, formula_request] {
    return engine.evaluate(formula_request);
  });
  formula_finalization_barrier.wait_until_reached();
  pending_range = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  range_finalization_barrier.wait_until_reached();
  pending_metric_technique = std::async(
      std::launch::async, [&engine, metric_technique_request] {
        return engine.evaluate_technique(metric_technique_request);
      });
  technique_finalization_barrier.wait_until_reached();
  pending_symbolic_technique = std::async(
      std::launch::async, [&engine, symbolic_technique_request] {
        return engine.evaluate_technique(symbolic_technique_request);
      });
  noncacheable_technique_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_formula = pending_formula.get();
  CHECK(cancelled_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_formula.integer.has_value());
  CHECK(!cancelled_formula.metric_bit_length.has_value());
  CHECK(!cancelled_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  range_finalization_barrier.release();
  const auto cancelled_range = pending_range.get();
  CHECK(cancelled_range.status == bruce::lhme::Status::Cancelled);
  CHECK(cancelled_range.candidates.empty());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  technique_finalization_barrier.release();
  noncacheable_technique_finalization_barrier.release();
  const auto cancelled_metric_technique = pending_metric_technique.get();
  const auto cancelled_symbolic_technique = pending_symbolic_technique.get();
  CHECK(cancelled_metric_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_metric_technique.integer.has_value());
  CHECK(!cancelled_metric_technique.metric_bit_length.has_value());
  CHECK(!cancelled_metric_technique.symbolic_expression.has_value());
  CHECK(cancelled_symbolic_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_symbolic_technique.integer.has_value());
  CHECK(!cancelled_symbolic_technique.metric_bit_length.has_value());
  CHECK(!cancelled_symbolic_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  const auto late_cancelled_formula = engine.evaluate(formula_request);
  CHECK(late_cancelled_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!late_cancelled_formula.integer.has_value());
  CHECK(!late_cancelled_formula.metric_bit_length.has_value());
  CHECK(!late_cancelled_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();
  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_formula = engine.evaluate(formula_request);
  CHECK(reused_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_formula.exact_integer() == 102300);
  CHECK(!reused_formula.metric_bit_length.has_value());
  CHECK(!reused_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);
  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.summary.invalid_argument_count == 0);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);
  const auto reused_metric_technique = engine.evaluate_technique(
      metric_technique_request);
  CHECK(reused_metric_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_metric_technique.metric_bit_length == 17);
  CHECK(!reused_metric_technique.integer.has_value());
  CHECK(!reused_metric_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 2);
  const auto reused_symbolic_technique = engine.evaluate_technique(
      symbolic_technique_request);
  CHECK(reused_symbolic_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_symbolic_technique.symbolic_expression == "100*(2^10-1)");
  CHECK(!reused_symbolic_technique.integer.has_value());
  CHECK(!reused_symbolic_technique.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 2);
}

TEST(cpp_engine_keeps_a_post_snapshot_cancelled_modular_noncacheable_technique_group_cancelled_for_a_late_range_joiner_after_all_inflight_joiner_lease_endings_and_repeated_cancels) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::TechniqueRequest modular_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest metric_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest symbolic_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_formula;
  std::future<bruce::lhme::RangeResult> pending_range;
  std::future<bruce::lhme::EvaluationResult> pending_metric_technique;
  std::future<bruce::lhme::EvaluationResult> pending_symbolic_technique;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async,
                              [&engine, modular_technique_request] {
    return engine.evaluate_technique(modular_technique_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  bruce::lhme::test_support::EngineRangeFinalizationBarrier
      range_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueFinalizationBarrier
      technique_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueNoncacheableFinalizationBarrier
      noncacheable_technique_finalization_barrier;

  pending_formula = std::async(std::launch::async, [&engine, formula_request] {
    return engine.evaluate(formula_request);
  });
  formula_finalization_barrier.wait_until_reached();
  pending_range = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  range_finalization_barrier.wait_until_reached();
  pending_metric_technique = std::async(
      std::launch::async, [&engine, metric_technique_request] {
        return engine.evaluate_technique(metric_technique_request);
      });
  technique_finalization_barrier.wait_until_reached();
  pending_symbolic_technique = std::async(
      std::launch::async, [&engine, symbolic_technique_request] {
        return engine.evaluate_technique(symbolic_technique_request);
      });
  noncacheable_technique_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_formula = pending_formula.get();
  CHECK(cancelled_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_formula.integer.has_value());
  CHECK(!cancelled_formula.metric_bit_length.has_value());
  CHECK(!cancelled_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  range_finalization_barrier.release();
  const auto cancelled_range = pending_range.get();
  CHECK(cancelled_range.status == bruce::lhme::Status::Cancelled);
  CHECK(cancelled_range.candidates.empty());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  technique_finalization_barrier.release();
  noncacheable_technique_finalization_barrier.release();
  const auto cancelled_metric_technique = pending_metric_technique.get();
  const auto cancelled_symbolic_technique = pending_symbolic_technique.get();
  CHECK(cancelled_metric_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_metric_technique.integer.has_value());
  CHECK(!cancelled_metric_technique.metric_bit_length.has_value());
  CHECK(!cancelled_metric_technique.symbolic_expression.has_value());
  CHECK(cancelled_symbolic_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_symbolic_technique.integer.has_value());
  CHECK(!cancelled_symbolic_technique.metric_bit_length.has_value());
  CHECK(!cancelled_symbolic_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  const auto late_cancelled_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(late_cancelled_range.status == bruce::lhme::Status::Cancelled);
  CHECK(late_cancelled_range.candidates.empty());
  CHECK(engine.inspect().resident_entries == 0);

  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();
  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_formula = engine.evaluate(formula_request);
  CHECK(reused_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_formula.exact_integer() == 102300);
  CHECK(!reused_formula.metric_bit_length.has_value());
  CHECK(!reused_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);
  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.summary.invalid_argument_count == 0);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);
  const auto reused_metric_technique = engine.evaluate_technique(
      metric_technique_request);
  CHECK(reused_metric_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_metric_technique.metric_bit_length == 17);
  CHECK(!reused_metric_technique.integer.has_value());
  CHECK(!reused_metric_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 2);
  const auto reused_symbolic_technique = engine.evaluate_technique(
      symbolic_technique_request);
  CHECK(reused_symbolic_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_symbolic_technique.symbolic_expression == "100*(2^10-1)");
  CHECK(!reused_symbolic_technique.integer.has_value());
  CHECK(!reused_symbolic_technique.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 2);
}

TEST(cpp_engine_keeps_a_post_snapshot_cancelled_modular_noncacheable_technique_group_cancelled_for_remaining_inflight_cross_api_joiners_after_a_repeated_cancel) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::TechniqueRequest modular_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest metric_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest symbolic_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_formula;
  std::future<bruce::lhme::RangeResult> pending_range;
  std::future<bruce::lhme::EvaluationResult> pending_metric_technique;
  std::future<bruce::lhme::EvaluationResult> pending_symbolic_technique;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async,
                              [&engine, modular_technique_request] {
    return engine.evaluate_technique(modular_technique_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  bruce::lhme::test_support::EngineRangeFinalizationBarrier
      range_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueFinalizationBarrier
      technique_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueNoncacheableFinalizationBarrier
      noncacheable_technique_finalization_barrier;

  pending_formula = std::async(std::launch::async, [&engine, formula_request] {
    return engine.evaluate(formula_request);
  });
  formula_finalization_barrier.wait_until_reached();

  pending_range = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  range_finalization_barrier.wait_until_reached();

  pending_metric_technique = std::async(
      std::launch::async, [&engine, metric_technique_request] {
        return engine.evaluate_technique(metric_technique_request);
      });
  technique_finalization_barrier.wait_until_reached();

  pending_symbolic_technique = std::async(
      std::launch::async, [&engine, symbolic_technique_request] {
        return engine.evaluate_technique(symbolic_technique_request);
      });
  noncacheable_technique_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_formula = pending_formula.get();

  CHECK(cancelled_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_formula.integer.has_value());
  CHECK(!cancelled_formula.metric_bit_length.has_value());
  CHECK(!cancelled_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  range_finalization_barrier.release();
  technique_finalization_barrier.release();
  noncacheable_technique_finalization_barrier.release();
  const auto cancelled_range = pending_range.get();
  const auto cancelled_metric_technique = pending_metric_technique.get();
  const auto cancelled_symbolic_technique = pending_symbolic_technique.get();

  CHECK(cancelled_range.status == bruce::lhme::Status::Cancelled);
  CHECK(cancelled_range.candidates.empty());
  CHECK(cancelled_metric_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_metric_technique.integer.has_value());
  CHECK(!cancelled_metric_technique.metric_bit_length.has_value());
  CHECK(!cancelled_metric_technique.symbolic_expression.has_value());
  CHECK(cancelled_symbolic_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_symbolic_technique.integer.has_value());
  CHECK(!cancelled_symbolic_technique.metric_bit_length.has_value());
  CHECK(!cancelled_symbolic_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();

  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_formula = engine.evaluate(formula_request);
  CHECK(reused_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_formula.exact_integer() == 102300);
  CHECK(!reused_formula.metric_bit_length.has_value());
  CHECK(!reused_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.summary.invalid_argument_count == 0);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_metric_technique = engine.evaluate_technique(
      metric_technique_request);
  CHECK(reused_metric_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_metric_technique.metric_bit_length == 17);
  CHECK(!reused_metric_technique.integer.has_value());
  CHECK(!reused_metric_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 2);

  const auto reused_symbolic_technique = engine.evaluate_technique(
      symbolic_technique_request);
  CHECK(reused_symbolic_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_symbolic_technique.symbolic_expression == "100*(2^10-1)");
  CHECK(!reused_symbolic_technique.integer.has_value());
  CHECK(!reused_symbolic_technique.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 2);
}

TEST(cpp_engine_keeps_a_post_snapshot_cancelled_modular_noncacheable_technique_group_cancelled_across_staged_inflight_cross_api_joiner_lease_endings_and_repeated_cancels) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::TechniqueRequest modular_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest metric_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest symbolic_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_formula;
  std::future<bruce::lhme::RangeResult> pending_range;
  std::future<bruce::lhme::EvaluationResult> pending_metric_technique;
  std::future<bruce::lhme::EvaluationResult> pending_symbolic_technique;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async,
                              [&engine, modular_technique_request] {
    return engine.evaluate_technique(modular_technique_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  bruce::lhme::test_support::EngineRangeFinalizationBarrier
      range_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueFinalizationBarrier
      technique_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueNoncacheableFinalizationBarrier
      noncacheable_technique_finalization_barrier;

  pending_formula = std::async(std::launch::async, [&engine, formula_request] {
    return engine.evaluate(formula_request);
  });
  formula_finalization_barrier.wait_until_reached();

  pending_range = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  range_finalization_barrier.wait_until_reached();

  pending_metric_technique = std::async(
      std::launch::async, [&engine, metric_technique_request] {
        return engine.evaluate_technique(metric_technique_request);
      });
  technique_finalization_barrier.wait_until_reached();

  pending_symbolic_technique = std::async(
      std::launch::async, [&engine, symbolic_technique_request] {
        return engine.evaluate_technique(symbolic_technique_request);
      });
  noncacheable_technique_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_formula = pending_formula.get();

  CHECK(cancelled_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_formula.integer.has_value());
  CHECK(!cancelled_formula.metric_bit_length.has_value());
  CHECK(!cancelled_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  range_finalization_barrier.release();
  const auto cancelled_range = pending_range.get();

  CHECK(cancelled_range.status == bruce::lhme::Status::Cancelled);
  CHECK(cancelled_range.candidates.empty());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  technique_finalization_barrier.release();
  noncacheable_technique_finalization_barrier.release();
  const auto cancelled_metric_technique = pending_metric_technique.get();
  const auto cancelled_symbolic_technique = pending_symbolic_technique.get();

  CHECK(cancelled_metric_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_metric_technique.integer.has_value());
  CHECK(!cancelled_metric_technique.metric_bit_length.has_value());
  CHECK(!cancelled_metric_technique.symbolic_expression.has_value());
  CHECK(cancelled_symbolic_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_symbolic_technique.integer.has_value());
  CHECK(!cancelled_symbolic_technique.metric_bit_length.has_value());
  CHECK(!cancelled_symbolic_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();

  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_formula = engine.evaluate(formula_request);
  CHECK(reused_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_formula.exact_integer() == 102300);
  CHECK(!reused_formula.metric_bit_length.has_value());
  CHECK(!reused_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.summary.invalid_argument_count == 0);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_metric_technique = engine.evaluate_technique(
      metric_technique_request);
  CHECK(reused_metric_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_metric_technique.metric_bit_length == 17);
  CHECK(!reused_metric_technique.integer.has_value());
  CHECK(!reused_metric_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 2);

  const auto reused_symbolic_technique = engine.evaluate_technique(
      symbolic_technique_request);
  CHECK(reused_symbolic_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_symbolic_technique.symbolic_expression == "100*(2^10-1)");
  CHECK(!reused_symbolic_technique.integer.has_value());
  CHECK(!reused_symbolic_technique.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 2);
}

TEST(cpp_engine_keeps_a_post_snapshot_cancelled_modular_noncacheable_technique_group_cancelled_for_a_late_exact_technique_joiner_after_all_inflight_joiner_lease_endings_and_repeated_cancels) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::TechniqueRequest modular_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest exact_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest metric_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest symbolic_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_formula;
  std::future<bruce::lhme::RangeResult> pending_range;
  std::future<bruce::lhme::EvaluationResult> pending_metric_technique;
  std::future<bruce::lhme::EvaluationResult> pending_symbolic_technique;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async,
                              [&engine, modular_technique_request] {
    return engine.evaluate_technique(modular_technique_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  bruce::lhme::test_support::EngineRangeFinalizationBarrier
      range_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueFinalizationBarrier
      technique_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueNoncacheableFinalizationBarrier
      noncacheable_technique_finalization_barrier;

  pending_formula = std::async(std::launch::async, [&engine, formula_request] {
    return engine.evaluate(formula_request);
  });
  formula_finalization_barrier.wait_until_reached();

  pending_range = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  range_finalization_barrier.wait_until_reached();

  pending_metric_technique = std::async(
      std::launch::async, [&engine, metric_technique_request] {
        return engine.evaluate_technique(metric_technique_request);
      });
  technique_finalization_barrier.wait_until_reached();

  pending_symbolic_technique = std::async(
      std::launch::async, [&engine, symbolic_technique_request] {
        return engine.evaluate_technique(symbolic_technique_request);
      });
  noncacheable_technique_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_formula = pending_formula.get();

  CHECK(cancelled_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_formula.integer.has_value());
  CHECK(!cancelled_formula.metric_bit_length.has_value());
  CHECK(!cancelled_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  range_finalization_barrier.release();
  const auto cancelled_range = pending_range.get();

  CHECK(cancelled_range.status == bruce::lhme::Status::Cancelled);
  CHECK(cancelled_range.candidates.empty());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  technique_finalization_barrier.release();
  noncacheable_technique_finalization_barrier.release();
  const auto cancelled_metric_technique = pending_metric_technique.get();
  const auto cancelled_symbolic_technique = pending_symbolic_technique.get();

  CHECK(cancelled_metric_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_metric_technique.integer.has_value());
  CHECK(!cancelled_metric_technique.metric_bit_length.has_value());
  CHECK(!cancelled_metric_technique.symbolic_expression.has_value());
  CHECK(cancelled_symbolic_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_symbolic_technique.integer.has_value());
  CHECK(!cancelled_symbolic_technique.metric_bit_length.has_value());
  CHECK(!cancelled_symbolic_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  const auto cancelled_late_exact_technique = engine.evaluate_technique(
      exact_technique_request);

  CHECK(cancelled_late_exact_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_late_exact_technique.integer.has_value());
  CHECK(!cancelled_late_exact_technique.metric_bit_length.has_value());
  CHECK(!cancelled_late_exact_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();

  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_formula = engine.evaluate(formula_request);
  CHECK(reused_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_formula.exact_integer() == 102300);
  CHECK(!reused_formula.metric_bit_length.has_value());
  CHECK(!reused_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.summary.invalid_argument_count == 0);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_exact_technique = engine.evaluate_technique(
      exact_technique_request);
  CHECK(reused_exact_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_exact_technique.exact_integer() == 102300);
  CHECK(!reused_exact_technique.metric_bit_length.has_value());
  CHECK(!reused_exact_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 2);

  const auto reused_metric_technique = engine.evaluate_technique(
      metric_technique_request);
  CHECK(reused_metric_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_metric_technique.metric_bit_length == 17);
  CHECK(!reused_metric_technique.integer.has_value());
  CHECK(!reused_metric_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 3);

  const auto reused_symbolic_technique = engine.evaluate_technique(
      symbolic_technique_request);
  CHECK(reused_symbolic_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_symbolic_technique.symbolic_expression == "100*(2^10-1)");
  CHECK(!reused_symbolic_technique.integer.has_value());
  CHECK(!reused_symbolic_technique.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 3);
}

TEST(cpp_engine_prefers_a_unique_cancelled_modular_group_over_a_new_uncancelled_group_with_a_different_modulus) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto state = std::make_shared<GroupLifetimeCancellationState>();
  const auto kernel = std::make_shared<GroupLifetimeCancellationKernel>(state);
  CHECK(engine.register_technique(kernel));

  const bruce::lhme::TechniqueRequest cancelled_origin{
      .identity = kernel->identity(),
      .parameters = "return_confirmed",
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::TechniqueRequest fresh_other_modulus{
      .identity = kernel->identity(),
      .parameters = "hold_without_cancel",
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 19};
  const bruce::lhme::TechniqueRequest late_exact{
      .identity = kernel->identity(),
      .parameters = "return_confirmed",
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> origin;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  origin = std::async(std::launch::async, [&engine, cancelled_origin] {
    return engine.evaluate_technique(cancelled_origin);
  });
  origin_post_finalization_barrier.wait_until_reached();

  engine.cancel();
  auto fresh = std::async(std::launch::async, [&engine, fresh_other_modulus] {
    return engine.evaluate_technique(fresh_other_modulus);
  });
  HeldEvaluationReleaseGuard fresh_release_guard{state};
  while (!state->fresh_evaluation_entered.load(std::memory_order_acquire)) {
    state->fresh_evaluation_entered.wait(false, std::memory_order_relaxed);
  }

  const auto late = engine.evaluate_technique(late_exact);
  CHECK(late.status == bruce::lhme::Status::Cancelled);
  CHECK(!late.integer.has_value());
  CHECK(!late.metric_bit_length.has_value());
  CHECK(!late.symbolic_expression.has_value());

  fresh_release_guard.release();
  const auto finalized_fresh = fresh.get();
  CHECK(finalized_fresh.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_fresh.exact_integer() == 2);

  origin_post_finalization_barrier.release();
  const auto finalized_origin = origin.get();
  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 1);
}

TEST(cpp_engine_forms_a_fresh_late_exact_formula_group_when_two_post_snapshot_cancelled_modular_groups_of_the_same_computation_and_address_have_different_moduli) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));

  const bruce::lhme::TechniqueRequest modulus_17_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest modulus_19_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 19};
  const bruce::lhme::EvaluationRequest exact_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest modulus_19_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 19};

  std::future<bruce::lhme::EvaluationResult> pending_modulus_17_technique;
  std::future<bruce::lhme::EvaluationResult> pending_modulus_19_formula;
  std::future<bruce::lhme::EvaluationResult> pending_modulus_19_technique;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      modulus_17_technique_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_modulus_17_technique = std::async(
      std::launch::async, [&engine, modulus_17_technique_request] {
        return engine.evaluate_technique(modulus_17_technique_request);
      });
  modulus_17_technique_post_finalization_barrier.wait_until_reached();

  engine.cancel();

  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      modulus_19_formula_post_finalization_barrier;
  pending_modulus_19_formula = std::async(
      std::launch::async, [&engine, modulus_19_formula_request] {
        return engine.evaluate(modulus_19_formula_request);
      });
  modulus_19_formula_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 0);

  bruce::lhme::test_support::EngineTechniqueNoncacheableFinalizationBarrier
      modulus_19_technique_finalization_barrier;
  pending_modulus_19_technique = std::async(
      std::launch::async, [&engine, modulus_19_technique_request] {
        return engine.evaluate_technique(modulus_19_technique_request);
      });
  modulus_19_technique_finalization_barrier.wait_until_reached();

  engine.cancel();

  modulus_19_formula_post_finalization_barrier.release();
  const auto confirmed_late_exact_formula = engine.evaluate(exact_formula_request);
  CHECK(confirmed_late_exact_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(confirmed_late_exact_formula.exact_integer() == 102300);
  CHECK(!confirmed_late_exact_formula.metric_bit_length.has_value());
  CHECK(!confirmed_late_exact_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto finalized_modulus_19_formula = pending_modulus_19_formula.get();
  CHECK(finalized_modulus_19_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_modulus_19_formula.exact_integer() == 4);
  CHECK(!finalized_modulus_19_formula.metric_bit_length.has_value());
  CHECK(!finalized_modulus_19_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  modulus_19_technique_finalization_barrier.release();
  const auto cancelled_modulus_19_technique =
      pending_modulus_19_technique.get();
  CHECK(cancelled_modulus_19_technique.status ==
        bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_modulus_19_technique.integer.has_value());
  CHECK(!cancelled_modulus_19_technique.metric_bit_length.has_value());
  CHECK(!cancelled_modulus_19_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  modulus_17_technique_post_finalization_barrier.release();
  const auto finalized_modulus_17_technique =
      pending_modulus_17_technique.get();
  CHECK(finalized_modulus_17_technique.status ==
        bruce::lhme::Status::Confirmed);
  CHECK(finalized_modulus_17_technique.exact_integer() == 11);
  CHECK(!finalized_modulus_17_technique.metric_bit_length.has_value());
  CHECK(!finalized_modulus_17_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_exact_formula = engine.evaluate(exact_formula_request);
  CHECK(reused_exact_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_exact_formula.exact_integer() == 102300);
  CHECK(!reused_exact_formula.metric_bit_length.has_value());
  CHECK(!reused_exact_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_forms_a_fresh_late_exact_technique_group_when_two_post_snapshot_cancelled_modular_groups_of_the_same_computation_and_address_have_different_moduli) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));

  const bruce::lhme::TechniqueRequest modulus_17_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest modulus_19_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 19};
  const bruce::lhme::TechniqueRequest exact_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_modulus_17_technique;
  std::future<bruce::lhme::EvaluationResult> pending_modulus_19_formula;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      modulus_17_technique_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_modulus_17_technique = std::async(
      std::launch::async, [&engine, modulus_17_technique_request] {
        return engine.evaluate_technique(modulus_17_technique_request);
      });
  modulus_17_technique_post_finalization_barrier.wait_until_reached();

  engine.cancel();

  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      modulus_19_formula_post_finalization_barrier;
  pending_modulus_19_formula = std::async(
      std::launch::async, [&engine, modulus_19_formula_request] {
        return engine.evaluate(modulus_19_formula_request);
      });
  modulus_19_formula_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();

  const auto confirmed_late_exact_technique = engine.evaluate_technique(
      exact_technique_request);
  CHECK(confirmed_late_exact_technique.status ==
        bruce::lhme::Status::Confirmed);
  CHECK(confirmed_late_exact_technique.exact_integer() == 102300);
  CHECK(!confirmed_late_exact_technique.metric_bit_length.has_value());
  CHECK(!confirmed_late_exact_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  modulus_19_formula_post_finalization_barrier.release();
  const auto finalized_modulus_19_formula = pending_modulus_19_formula.get();
  CHECK(finalized_modulus_19_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_modulus_19_formula.exact_integer() == 4);
  CHECK(!finalized_modulus_19_formula.metric_bit_length.has_value());
  CHECK(!finalized_modulus_19_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  modulus_17_technique_post_finalization_barrier.release();
  const auto finalized_modulus_17_technique =
      pending_modulus_17_technique.get();
  CHECK(finalized_modulus_17_technique.status ==
        bruce::lhme::Status::Confirmed);
  CHECK(finalized_modulus_17_technique.exact_integer() == 11);
  CHECK(!finalized_modulus_17_technique.metric_bit_length.has_value());
  CHECK(!finalized_modulus_17_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_exact_technique = engine.evaluate_technique(
      exact_technique_request);
  CHECK(reused_exact_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_exact_technique.exact_integer() == 102300);
  CHECK(!reused_exact_technique.metric_bit_length.has_value());
  CHECK(!reused_exact_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_keeps_a_unique_post_snapshot_cancelled_secondary_modular_group_cancelled_for_a_late_exact_technique_joiner_after_the_primary_modular_base_alias_group_lease_ends) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));

  const bruce::lhme::TechniqueRequest modulus_17_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest modulus_19_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 19};
  const bruce::lhme::TechniqueRequest exact_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_modulus_17_technique;
  std::future<bruce::lhme::EvaluationResult> pending_modulus_19_formula;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      modulus_17_technique_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_modulus_17_technique = std::async(
      std::launch::async, [&engine, modulus_17_technique_request] {
        return engine.evaluate_technique(modulus_17_technique_request);
      });
  modulus_17_technique_post_finalization_barrier.wait_until_reached();

  engine.cancel();

  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      modulus_19_formula_post_finalization_barrier;
  pending_modulus_19_formula = std::async(
      std::launch::async, [&engine, modulus_19_formula_request] {
        return engine.evaluate(modulus_19_formula_request);
      });
  modulus_19_formula_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();

  modulus_17_technique_post_finalization_barrier.release();
  const auto finalized_modulus_17_technique =
      pending_modulus_17_technique.get();
  CHECK(finalized_modulus_17_technique.status ==
        bruce::lhme::Status::Confirmed);
  CHECK(finalized_modulus_17_technique.exact_integer() == 11);
  CHECK(!finalized_modulus_17_technique.metric_bit_length.has_value());
  CHECK(!finalized_modulus_17_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto cancelled_late_exact_technique = engine.evaluate_technique(
      exact_technique_request);
  CHECK(cancelled_late_exact_technique.status ==
        bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_late_exact_technique.integer.has_value());
  CHECK(!cancelled_late_exact_technique.metric_bit_length.has_value());
  CHECK(!cancelled_late_exact_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  modulus_19_formula_post_finalization_barrier.release();
  const auto finalized_modulus_19_formula = pending_modulus_19_formula.get();
  CHECK(finalized_modulus_19_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_modulus_19_formula.exact_integer() == 4);
  CHECK(!finalized_modulus_19_formula.metric_bit_length.has_value());
  CHECK(!finalized_modulus_19_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_exact_technique = engine.evaluate_technique(
      exact_technique_request);
  CHECK(reused_exact_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_exact_technique.exact_integer() == 102300);
  CHECK(!reused_exact_technique.metric_bit_length.has_value());
  CHECK(!reused_exact_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_keeps_a_unique_post_snapshot_cancelled_secondary_modular_group_cancelled_for_a_late_exact_formula_joiner_after_the_primary_modular_base_alias_group_lease_ends) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));

  const bruce::lhme::TechniqueRequest modulus_17_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest modulus_19_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 19};
  const bruce::lhme::TechniqueRequest modulus_19_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 19};
  const bruce::lhme::EvaluationRequest exact_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_modulus_17_technique;
  std::future<bruce::lhme::EvaluationResult> pending_modulus_19_formula;
  std::future<bruce::lhme::EvaluationResult> pending_modulus_19_technique;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      modulus_17_technique_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_modulus_17_technique = std::async(
      std::launch::async, [&engine, modulus_17_technique_request] {
        return engine.evaluate_technique(modulus_17_technique_request);
      });
  modulus_17_technique_post_finalization_barrier.wait_until_reached();

  engine.cancel();

  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      modulus_19_formula_post_finalization_barrier;
  pending_modulus_19_formula = std::async(
      std::launch::async, [&engine, modulus_19_formula_request] {
        return engine.evaluate(modulus_19_formula_request);
      });
  modulus_19_formula_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 0);

  bruce::lhme::test_support::EngineTechniqueNoncacheableFinalizationBarrier
      modulus_19_technique_finalization_barrier;
  pending_modulus_19_technique = std::async(
      std::launch::async, [&engine, modulus_19_technique_request] {
        return engine.evaluate_technique(modulus_19_technique_request);
      });
  modulus_19_technique_finalization_barrier.wait_until_reached();

  engine.cancel();

  modulus_19_formula_post_finalization_barrier.release();
  const auto finalized_modulus_19_formula = pending_modulus_19_formula.get();
  CHECK(finalized_modulus_19_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_modulus_19_formula.exact_integer() == 4);
  CHECK(!finalized_modulus_19_formula.metric_bit_length.has_value());
  CHECK(!finalized_modulus_19_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  modulus_17_technique_post_finalization_barrier.release();
  const auto finalized_modulus_17_technique =
      pending_modulus_17_technique.get();
  CHECK(finalized_modulus_17_technique.status ==
        bruce::lhme::Status::Confirmed);
  CHECK(finalized_modulus_17_technique.exact_integer() == 11);
  CHECK(!finalized_modulus_17_technique.metric_bit_length.has_value());
  CHECK(!finalized_modulus_17_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto cancelled_late_exact_formula = engine.evaluate(exact_formula_request);
  CHECK(cancelled_late_exact_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_late_exact_formula.integer.has_value());
  CHECK(!cancelled_late_exact_formula.metric_bit_length.has_value());
  CHECK(!cancelled_late_exact_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  modulus_19_technique_finalization_barrier.release();
  const auto cancelled_modulus_19_technique =
      pending_modulus_19_technique.get();
  CHECK(cancelled_modulus_19_technique.status ==
        bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_modulus_19_technique.integer.has_value());
  CHECK(!cancelled_modulus_19_technique.metric_bit_length.has_value());
  CHECK(!cancelled_modulus_19_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_exact_formula = engine.evaluate(exact_formula_request);
  CHECK(reused_exact_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_exact_formula.exact_integer() == 102300);
  CHECK(!reused_exact_formula.metric_bit_length.has_value());
  CHECK(!reused_exact_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_prefers_a_held_post_ambiguity_nonmodular_direct_group_over_a_now_unique_cancelled_secondary_modular_group_for_a_late_exact_technique_joiner) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));

  const bruce::lhme::TechniqueRequest modulus_17_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest modulus_19_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 19};
  const bruce::lhme::TechniqueRequest modulus_19_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 19};
  const bruce::lhme::EvaluationRequest exact_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest exact_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_modulus_17_technique;
  std::future<bruce::lhme::EvaluationResult> pending_modulus_19_formula;
  std::future<bruce::lhme::EvaluationResult> pending_modulus_19_technique;
  std::future<bruce::lhme::EvaluationResult>
      pending_post_ambiguity_exact_formula;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      modulus_17_technique_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_modulus_17_technique = std::async(
      std::launch::async, [&engine, modulus_17_technique_request] {
        return engine.evaluate_technique(modulus_17_technique_request);
      });
  modulus_17_technique_post_finalization_barrier.wait_until_reached();

  engine.cancel();

  std::optional<
      bruce::lhme::test_support::EngineTechniqueNoncacheableFinalizationBarrier>
      modulus_19_technique_finalization_barrier;
  {
    bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
        modulus_19_formula_post_finalization_barrier;
    pending_modulus_19_formula = std::async(
        std::launch::async, [&engine, modulus_19_formula_request] {
          return engine.evaluate(modulus_19_formula_request);
        });
    modulus_19_formula_post_finalization_barrier.wait_until_reached();
    CHECK(engine.inspect().resident_entries == 0);

    modulus_19_technique_finalization_barrier.emplace();
    pending_modulus_19_technique = std::async(
        std::launch::async, [&engine, modulus_19_technique_request] {
          return engine.evaluate_technique(modulus_19_technique_request);
        });
    modulus_19_technique_finalization_barrier->wait_until_reached();

    engine.cancel();

    modulus_19_formula_post_finalization_barrier.release();
    const auto finalized_modulus_19_formula = pending_modulus_19_formula.get();
    CHECK(finalized_modulus_19_formula.status ==
          bruce::lhme::Status::Confirmed);
    CHECK(finalized_modulus_19_formula.exact_integer() == 4);
    CHECK(!finalized_modulus_19_formula.metric_bit_length.has_value());
    CHECK(!finalized_modulus_19_formula.symbolic_expression.has_value());
    CHECK(engine.inspect().resident_entries == 0);
  }

  {
    bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
        post_ambiguity_exact_formula_post_finalization_barrier;
    pending_post_ambiguity_exact_formula = std::async(
        std::launch::async, [&engine, exact_formula_request] {
          return engine.evaluate(exact_formula_request);
        });
    post_ambiguity_exact_formula_post_finalization_barrier.wait_until_reached();
    CHECK(engine.inspect().resident_entries == 0);

    modulus_17_technique_post_finalization_barrier.release();
    const auto finalized_modulus_17_technique =
        pending_modulus_17_technique.get();
    CHECK(finalized_modulus_17_technique.status ==
          bruce::lhme::Status::Confirmed);
    CHECK(finalized_modulus_17_technique.exact_integer() == 11);
    CHECK(!finalized_modulus_17_technique.metric_bit_length.has_value());
    CHECK(!finalized_modulus_17_technique.symbolic_expression.has_value());
    CHECK(engine.inspect().resident_entries == 0);

    const auto confirmed_late_exact_technique = engine.evaluate_technique(
        exact_technique_request);
    CHECK(confirmed_late_exact_technique.status ==
          bruce::lhme::Status::Confirmed);
    CHECK(confirmed_late_exact_technique.exact_integer() == 102300);
    CHECK(!confirmed_late_exact_technique.metric_bit_length.has_value());
    CHECK(!confirmed_late_exact_technique.symbolic_expression.has_value());
    CHECK(engine.inspect().resident_entries == 1);

    post_ambiguity_exact_formula_post_finalization_barrier.release();
    const auto finalized_post_ambiguity_exact_formula =
        pending_post_ambiguity_exact_formula.get();
    CHECK(finalized_post_ambiguity_exact_formula.status ==
          bruce::lhme::Status::Confirmed);
    CHECK(finalized_post_ambiguity_exact_formula.exact_integer() == 102300);
    CHECK(!finalized_post_ambiguity_exact_formula.metric_bit_length.has_value());
    CHECK(!finalized_post_ambiguity_exact_formula.symbolic_expression.has_value());
    CHECK(engine.inspect().resident_entries == 2);
  }

  modulus_19_technique_finalization_barrier->release();
  const auto cancelled_modulus_19_technique =
      pending_modulus_19_technique.get();
  CHECK(cancelled_modulus_19_technique.status ==
        bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_modulus_19_technique.integer.has_value());
  CHECK(!cancelled_modulus_19_technique.metric_bit_length.has_value());
  CHECK(!cancelled_modulus_19_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 2);

  const auto reused_exact_formula = engine.evaluate(exact_formula_request);
  CHECK(reused_exact_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_exact_formula.exact_integer() == 102300);
  CHECK(!reused_exact_formula.metric_bit_length.has_value());
  CHECK(!reused_exact_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 2);
}

TEST(cpp_engine_falls_back_to_a_unique_post_snapshot_cancelled_exact_range_group_for_a_late_exact_formula_when_multiple_uncancelled_direct_modular_groups_of_the_same_computation_and_address_exist) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));

  const bruce::lhme::EvaluationRequest exact_range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest first_range_symbolic_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 12},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest modulus_17_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest modulus_19_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 19};
  const bruce::lhme::EvaluationRequest late_exact_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::RangeResult> pending_first_range;
  std::future<bruce::lhme::EvaluationResult>
      pending_first_range_symbolic_technique;
  std::future<bruce::lhme::RangeResult> pending_second_range;
  std::future<bruce::lhme::EvaluationResult> pending_modulus_17_technique;
  std::future<bruce::lhme::EvaluationResult> pending_modulus_19_formula;
  std::optional<
      bruce::lhme::test_support::
          EngineTechniqueNoncacheablePostFinalizationBarrier>
      first_range_symbolic_technique_post_finalization_barrier;

  {
    bruce::lhme::test_support::EngineRangePostFinalizationBarrier
        first_range_post_finalization_barrier;
    pending_first_range = std::async(
        std::launch::async, [&engine, exact_range_request] {
          return engine.evaluate_range(exact_range_request, {0, 1});
        });
    first_range_post_finalization_barrier.wait_until_reached();

    first_range_symbolic_technique_post_finalization_barrier.emplace(
        bruce::lhme::test_support::
            EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                FirstArrivalOnly);
    pending_first_range_symbolic_technique = std::async(
        std::launch::async,
        [&engine, first_range_symbolic_technique_request] {
          return engine.evaluate_technique(
              first_range_symbolic_technique_request);
        });
    first_range_symbolic_technique_post_finalization_barrier
        ->wait_until_reached();

    first_range_post_finalization_barrier.release();
    const auto finalized_first_range = pending_first_range.get();
    CHECK(finalized_first_range.status == bruce::lhme::Status::Confirmed);
    CHECK(finalized_first_range.summary.checked_count == 2);
    CHECK(finalized_first_range.summary.confirmed_count == 2);
    CHECK(finalized_first_range.summary.invalid_argument_count == 0);
    CHECK(finalized_first_range.candidates.size() == 2);
    CHECK(finalized_first_range.candidates[0].exact_integer() == 0);
    CHECK(finalized_first_range.candidates[1].exact_integer() == 100);
    CHECK(engine.inspect().resident_entries == 0);
  }

  {
    bruce::lhme::test_support::EngineRangePostFinalizationBarrier
        second_range_post_finalization_barrier;
    pending_second_range = std::async(
        std::launch::async, [&engine, exact_range_request] {
          return engine.evaluate_range(exact_range_request, {1, 2});
        });
    second_range_post_finalization_barrier.wait_until_reached();
    CHECK(engine.inspect().resident_entries == 0);

    engine.cancel();

    bruce::lhme::test_support::EngineTechniqueNoncacheableFinalizationBarrier
        modulus_17_technique_finalization_barrier;
    pending_modulus_17_technique = std::async(
        std::launch::async, [&engine, modulus_17_technique_request] {
          return engine.evaluate_technique(modulus_17_technique_request);
        });
    modulus_17_technique_finalization_barrier.wait_until_reached();

    bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
        modulus_19_formula_post_finalization_barrier;
    pending_modulus_19_formula = std::async(
        std::launch::async, [&engine, modulus_19_formula_request] {
          return engine.evaluate(modulus_19_formula_request);
        });
    modulus_19_formula_post_finalization_barrier.wait_until_reached();
    CHECK(engine.inspect().resident_entries == 0);

    second_range_post_finalization_barrier.release();
    const auto finalized_second_range = pending_second_range.get();
    CHECK(finalized_second_range.status == bruce::lhme::Status::Confirmed);
    CHECK(finalized_second_range.summary.checked_count == 2);
    CHECK(finalized_second_range.summary.confirmed_count == 2);
    CHECK(finalized_second_range.summary.invalid_argument_count == 0);
    CHECK(finalized_second_range.candidates.size() == 2);
    CHECK(finalized_second_range.candidates[0].exact_integer() == 100);
    CHECK(finalized_second_range.candidates[1].exact_integer() == 300);
    CHECK(engine.inspect().resident_entries == 0);

    const auto cancelled_late_exact_formula = engine.evaluate(
        late_exact_formula_request);
    CHECK(cancelled_late_exact_formula.status ==
          bruce::lhme::Status::Cancelled);
    CHECK(!cancelled_late_exact_formula.integer.has_value());
    CHECK(!cancelled_late_exact_formula.metric_bit_length.has_value());
    CHECK(!cancelled_late_exact_formula.symbolic_expression.has_value());
    CHECK(engine.inspect().resident_entries == 0);

    modulus_19_formula_post_finalization_barrier.release();
    const auto finalized_modulus_19_formula = pending_modulus_19_formula.get();
    CHECK(finalized_modulus_19_formula.status ==
          bruce::lhme::Status::Confirmed);
    CHECK(finalized_modulus_19_formula.exact_integer() == 4);
    CHECK(!finalized_modulus_19_formula.metric_bit_length.has_value());
    CHECK(!finalized_modulus_19_formula.symbolic_expression.has_value());
    CHECK(engine.inspect().resident_entries == 0);

    modulus_17_technique_finalization_barrier.release();
    const auto finalized_modulus_17_technique =
        pending_modulus_17_technique.get();
    CHECK(finalized_modulus_17_technique.status ==
          bruce::lhme::Status::Confirmed);
    CHECK(finalized_modulus_17_technique.exact_integer() == 11);
    CHECK(!finalized_modulus_17_technique.metric_bit_length.has_value());
    CHECK(!finalized_modulus_17_technique.symbolic_expression.has_value());
    CHECK(engine.inspect().resident_entries == 0);
  }

  first_range_symbolic_technique_post_finalization_barrier->release();
  const auto finalized_first_range_symbolic_technique =
      pending_first_range_symbolic_technique.get();
  CHECK(finalized_first_range_symbolic_technique.status ==
        bruce::lhme::Status::Confirmed);
  CHECK(!finalized_first_range_symbolic_technique.integer.has_value());
  CHECK(!finalized_first_range_symbolic_technique.metric_bit_length
             .has_value());
  CHECK(finalized_first_range_symbolic_technique.symbolic_expression ==
        "100*(2^12-1)");
  CHECK(engine.inspect().resident_entries == 0);

  const auto fresh_exact_formula = engine.evaluate(late_exact_formula_request);
  CHECK(fresh_exact_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(fresh_exact_formula.exact_integer() == 102300);
  CHECK(!fresh_exact_formula.metric_bit_length.has_value());
  CHECK(!fresh_exact_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_falls_back_to_a_unique_post_snapshot_cancelled_exact_range_group_for_a_late_exact_technique_when_multiple_uncancelled_direct_modular_groups_of_the_same_computation_and_address_exist) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));

  const bruce::lhme::EvaluationRequest exact_range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest first_range_symbolic_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 12},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest modulus_17_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest modulus_19_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 19};
  const bruce::lhme::TechniqueRequest exact_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::RangeResult> pending_first_range;
  std::future<bruce::lhme::EvaluationResult>
      pending_first_range_symbolic_technique;
  std::future<bruce::lhme::RangeResult> pending_second_range;
  std::future<bruce::lhme::EvaluationResult> pending_modulus_17_technique;
  std::future<bruce::lhme::EvaluationResult> pending_modulus_19_formula;
  std::optional<
      bruce::lhme::test_support::
          EngineTechniqueNoncacheablePostFinalizationBarrier>
      first_range_symbolic_technique_post_finalization_barrier;

  {
    bruce::lhme::test_support::EngineRangePostFinalizationBarrier
        first_range_post_finalization_barrier;
    pending_first_range = std::async(
        std::launch::async, [&engine, exact_range_request] {
          return engine.evaluate_range(exact_range_request, {0, 1});
        });
    first_range_post_finalization_barrier.wait_until_reached();

    first_range_symbolic_technique_post_finalization_barrier.emplace(
        bruce::lhme::test_support::
            EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                FirstArrivalOnly);
    pending_first_range_symbolic_technique = std::async(
        std::launch::async,
        [&engine, first_range_symbolic_technique_request] {
          return engine.evaluate_technique(
              first_range_symbolic_technique_request);
        });
    first_range_symbolic_technique_post_finalization_barrier
        ->wait_until_reached();

    first_range_post_finalization_barrier.release();
    const auto finalized_first_range = pending_first_range.get();
    CHECK(finalized_first_range.status == bruce::lhme::Status::Confirmed);
    CHECK(finalized_first_range.summary.checked_count == 2);
    CHECK(finalized_first_range.summary.confirmed_count == 2);
    CHECK(finalized_first_range.summary.invalid_argument_count == 0);
    CHECK(finalized_first_range.candidates.size() == 2);
    CHECK(finalized_first_range.candidates[0].exact_integer() == 0);
    CHECK(finalized_first_range.candidates[1].exact_integer() == 100);
    CHECK(engine.inspect().resident_entries == 0);
  }

  {
    bruce::lhme::test_support::EngineRangePostFinalizationBarrier
        second_range_post_finalization_barrier;
    pending_second_range = std::async(
        std::launch::async, [&engine, exact_range_request] {
          return engine.evaluate_range(exact_range_request, {1, 2});
        });
    second_range_post_finalization_barrier.wait_until_reached();
    CHECK(engine.inspect().resident_entries == 0);

    engine.cancel();

    bruce::lhme::test_support::EngineTechniqueNoncacheableFinalizationBarrier
        modulus_17_technique_finalization_barrier;
    pending_modulus_17_technique = std::async(
        std::launch::async, [&engine, modulus_17_technique_request] {
          return engine.evaluate_technique(modulus_17_technique_request);
        });
    modulus_17_technique_finalization_barrier.wait_until_reached();

    bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
        modulus_19_formula_post_finalization_barrier;
    pending_modulus_19_formula = std::async(
        std::launch::async, [&engine, modulus_19_formula_request] {
          return engine.evaluate(modulus_19_formula_request);
        });
    modulus_19_formula_post_finalization_barrier.wait_until_reached();
    CHECK(engine.inspect().resident_entries == 0);

    second_range_post_finalization_barrier.release();
    const auto finalized_second_range = pending_second_range.get();
    CHECK(finalized_second_range.status == bruce::lhme::Status::Confirmed);
    CHECK(finalized_second_range.summary.checked_count == 2);
    CHECK(finalized_second_range.summary.confirmed_count == 2);
    CHECK(finalized_second_range.summary.invalid_argument_count == 0);
    CHECK(finalized_second_range.candidates.size() == 2);
    CHECK(finalized_second_range.candidates[0].exact_integer() == 100);
    CHECK(finalized_second_range.candidates[1].exact_integer() == 300);
    CHECK(engine.inspect().resident_entries == 0);

    const auto cancelled_late_exact_technique = engine.evaluate_technique(
        exact_technique_request);
    CHECK(cancelled_late_exact_technique.status ==
          bruce::lhme::Status::Cancelled);
    CHECK(!cancelled_late_exact_technique.integer.has_value());
    CHECK(!cancelled_late_exact_technique.metric_bit_length.has_value());
    CHECK(!cancelled_late_exact_technique.symbolic_expression.has_value());
    CHECK(engine.inspect().resident_entries == 0);

    modulus_19_formula_post_finalization_barrier.release();
    const auto finalized_modulus_19_formula = pending_modulus_19_formula.get();
    CHECK(finalized_modulus_19_formula.status ==
          bruce::lhme::Status::Confirmed);
    CHECK(finalized_modulus_19_formula.exact_integer() == 4);
    CHECK(!finalized_modulus_19_formula.metric_bit_length.has_value());
    CHECK(!finalized_modulus_19_formula.symbolic_expression.has_value());
    CHECK(engine.inspect().resident_entries == 0);

    modulus_17_technique_finalization_barrier.release();
    const auto finalized_modulus_17_technique =
        pending_modulus_17_technique.get();
    CHECK(finalized_modulus_17_technique.status ==
          bruce::lhme::Status::Confirmed);
    CHECK(finalized_modulus_17_technique.exact_integer() == 11);
    CHECK(!finalized_modulus_17_technique.metric_bit_length.has_value());
    CHECK(!finalized_modulus_17_technique.symbolic_expression.has_value());
    CHECK(engine.inspect().resident_entries == 0);
  }

  first_range_symbolic_technique_post_finalization_barrier->release();
  const auto finalized_first_range_symbolic_technique =
      pending_first_range_symbolic_technique.get();
  CHECK(finalized_first_range_symbolic_technique.status ==
        bruce::lhme::Status::Confirmed);
  CHECK(!finalized_first_range_symbolic_technique.integer.has_value());
  CHECK(!finalized_first_range_symbolic_technique.metric_bit_length
             .has_value());
  CHECK(finalized_first_range_symbolic_technique.symbolic_expression ==
        "100*(2^12-1)");
  CHECK(engine.inspect().resident_entries == 0);

  const auto fresh_exact_technique = engine.evaluate_technique(
      exact_technique_request);
  CHECK(fresh_exact_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(fresh_exact_technique.exact_integer() == 102300);
  CHECK(!fresh_exact_technique.metric_bit_length.has_value());
  CHECK(!fresh_exact_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_prefers_a_held_post_ambiguity_nonmodular_direct_group_over_a_unique_post_snapshot_cancelled_exact_range_group_for_a_late_exact_technique_joiner) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));

  const bruce::lhme::EvaluationRequest exact_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest exact_range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest first_range_symbolic_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 12},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest modulus_17_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::TechniqueRequest modulus_19_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 19};
  const bruce::lhme::TechniqueRequest exact_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  const auto warmed_exact_formula = engine.evaluate(exact_formula_request);
  CHECK(warmed_exact_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(warmed_exact_formula.exact_integer() == 102300);
  CHECK(!warmed_exact_formula.metric_bit_length.has_value());
  CHECK(!warmed_exact_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  std::future<bruce::lhme::RangeResult> pending_first_range;
  std::future<bruce::lhme::EvaluationResult>
      pending_first_range_symbolic_technique;
  std::future<bruce::lhme::RangeResult> pending_second_range;
  std::future<bruce::lhme::EvaluationResult> pending_modulus_17_formula;
  std::future<bruce::lhme::EvaluationResult> pending_modulus_19_technique;
  std::future<bruce::lhme::EvaluationResult>
      pending_post_ambiguity_exact_formula;
  std::optional<
      bruce::lhme::test_support::
          EngineTechniqueNoncacheablePostFinalizationBarrier>
      first_range_symbolic_technique_post_finalization_barrier;

  {
    bruce::lhme::test_support::EngineRangePostFinalizationBarrier
        first_range_post_finalization_barrier;
    pending_first_range = std::async(
        std::launch::async, [&engine, exact_range_request] {
          return engine.evaluate_range(exact_range_request, {0, 1});
        });
    first_range_post_finalization_barrier.wait_until_reached();

    first_range_symbolic_technique_post_finalization_barrier.emplace(
        bruce::lhme::test_support::
            EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                FirstArrivalOnly);
    pending_first_range_symbolic_technique = std::async(
        std::launch::async,
        [&engine, first_range_symbolic_technique_request] {
          return engine.evaluate_technique(
              first_range_symbolic_technique_request);
        });
    first_range_symbolic_technique_post_finalization_barrier
        ->wait_until_reached();

    first_range_post_finalization_barrier.release();
    const auto finalized_first_range = pending_first_range.get();
    CHECK(finalized_first_range.status == bruce::lhme::Status::Confirmed);
    CHECK(finalized_first_range.summary.checked_count == 2);
    CHECK(finalized_first_range.summary.confirmed_count == 2);
    CHECK(finalized_first_range.summary.invalid_argument_count == 0);
    CHECK(finalized_first_range.candidates.size() == 2);
    CHECK(finalized_first_range.candidates[0].exact_integer() == 0);
    CHECK(finalized_first_range.candidates[1].exact_integer() == 100);
    CHECK(engine.inspect().resident_entries == 1);
  }

  {
    bruce::lhme::test_support::EngineRangePostFinalizationBarrier
        second_range_post_finalization_barrier;
    pending_second_range = std::async(
        std::launch::async, [&engine, exact_range_request] {
          return engine.evaluate_range(exact_range_request, {1, 2});
        });
    second_range_post_finalization_barrier.wait_until_reached();
    CHECK(engine.inspect().resident_entries == 1);

    engine.cancel();

    bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
        modulus_17_formula_post_finalization_barrier;
    pending_modulus_17_formula = std::async(
        std::launch::async, [&engine, modulus_17_formula_request] {
          return engine.evaluate(modulus_17_formula_request);
        });
    modulus_17_formula_post_finalization_barrier.wait_until_reached();
    CHECK(engine.inspect().resident_entries == 1);

    bruce::lhme::test_support::EngineTechniqueNoncacheableFinalizationBarrier
        modulus_19_technique_finalization_barrier;
    pending_modulus_19_technique = std::async(
        std::launch::async, [&engine, modulus_19_technique_request] {
          return engine.evaluate_technique(modulus_19_technique_request);
        });
    modulus_19_technique_finalization_barrier.wait_until_reached();

    bruce::lhme::test_support::EngineFormulaCacheHitPostFinalizationBarrier
        post_ambiguity_exact_formula_post_finalization_barrier;
    pending_post_ambiguity_exact_formula = std::async(
        std::launch::async, [&engine, exact_formula_request] {
          return engine.evaluate(exact_formula_request);
        });
    post_ambiguity_exact_formula_post_finalization_barrier.wait_until_reached();
    CHECK(engine.inspect().resident_entries == 1);

    second_range_post_finalization_barrier.release();
    const auto finalized_second_range = pending_second_range.get();
    CHECK(finalized_second_range.status == bruce::lhme::Status::Confirmed);
    CHECK(finalized_second_range.summary.checked_count == 2);
    CHECK(finalized_second_range.summary.confirmed_count == 2);
    CHECK(finalized_second_range.summary.invalid_argument_count == 0);
    CHECK(finalized_second_range.candidates.size() == 2);
    CHECK(finalized_second_range.candidates[0].exact_integer() == 100);
    CHECK(finalized_second_range.candidates[1].exact_integer() == 300);
    CHECK(engine.inspect().resident_entries == 1);

    modulus_17_formula_post_finalization_barrier.release();
    const auto finalized_modulus_17_formula = pending_modulus_17_formula.get();
    CHECK(finalized_modulus_17_formula.status ==
          bruce::lhme::Status::Confirmed);
    CHECK(finalized_modulus_17_formula.exact_integer() == 11);
    CHECK(!finalized_modulus_17_formula.metric_bit_length.has_value());
    CHECK(!finalized_modulus_17_formula.symbolic_expression.has_value());
    CHECK(engine.inspect().resident_entries == 1);

    modulus_19_technique_finalization_barrier.release();
    const auto finalized_modulus_19_technique =
        pending_modulus_19_technique.get();
    CHECK(finalized_modulus_19_technique.status ==
          bruce::lhme::Status::Confirmed);
    CHECK(finalized_modulus_19_technique.exact_integer() == 4);
    CHECK(!finalized_modulus_19_technique.metric_bit_length.has_value());
    CHECK(!finalized_modulus_19_technique.symbolic_expression.has_value());
    CHECK(engine.inspect().resident_entries == 1);

    const auto confirmed_late_exact_technique = engine.evaluate_technique(
        exact_technique_request);
    CHECK(confirmed_late_exact_technique.status ==
          bruce::lhme::Status::Confirmed);
    CHECK(confirmed_late_exact_technique.exact_integer() == 102300);
    CHECK(!confirmed_late_exact_technique.metric_bit_length.has_value());
    CHECK(!confirmed_late_exact_technique.symbolic_expression.has_value());
    CHECK(engine.inspect().resident_entries == 2);

    post_ambiguity_exact_formula_post_finalization_barrier.release();
    const auto finalized_post_ambiguity_exact_formula =
        pending_post_ambiguity_exact_formula.get();
    CHECK(finalized_post_ambiguity_exact_formula.status ==
          bruce::lhme::Status::Confirmed);
    CHECK(finalized_post_ambiguity_exact_formula.exact_integer() == 102300);
    CHECK(!finalized_post_ambiguity_exact_formula.metric_bit_length.has_value());
    CHECK(!finalized_post_ambiguity_exact_formula.symbolic_expression.has_value());
    CHECK(engine.inspect().resident_entries == 2);
  }

  first_range_symbolic_technique_post_finalization_barrier->release();
  const auto finalized_first_range_symbolic_technique =
      pending_first_range_symbolic_technique.get();
  CHECK(finalized_first_range_symbolic_technique.status ==
        bruce::lhme::Status::Confirmed);
  CHECK(!finalized_first_range_symbolic_technique.integer.has_value());
  CHECK(!finalized_first_range_symbolic_technique.metric_bit_length
             .has_value());
  CHECK(finalized_first_range_symbolic_technique.symbolic_expression ==
        "100*(2^12-1)");
  CHECK(engine.inspect().resident_entries == 2);

  const auto reused_exact_formula = engine.evaluate(exact_formula_request);
  CHECK(reused_exact_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_exact_formula.exact_integer() == 102300);
  CHECK(!reused_exact_formula.metric_bit_length.has_value());
  CHECK(!reused_exact_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 2);
}

TEST(cpp_engine_uses_kernel_canonical_parameters_to_keep_an_equivalent_late_modular_technique_request_in_a_cancelled_group) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<CanonicalParameterAliasKernel>();
  CHECK(engine.register_technique(kernel));

  const bruce::lhme::TechniqueRequest origin_modular_request{
      .identity = kernel->identity(),
      .parameters = "left=1;right=2",
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::TechniqueRequest exact_joiner_request{
      .identity = kernel->identity(),
      .parameters = "left=1;right=2",
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest equivalent_alias_modular_request{
      .identity = kernel->identity(),
      .parameters = "right=2;left=1",
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::TechniqueRequest different_parameters_modular_request{
      .identity = kernel->identity(),
      .parameters = "left=1;right=3",
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_exact_joiner;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async, [&engine, origin_modular_request] {
    return engine.evaluate_technique(origin_modular_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineTechniqueFinalizationBarrier
      exact_finalization_barrier;
  pending_exact_joiner = std::async(std::launch::async,
                                    [&engine, exact_joiner_request] {
    return engine.evaluate_technique(exact_joiner_request);
  });
  exact_finalization_barrier.wait_until_reached();

  engine.cancel();
  exact_finalization_barrier.release();
  const auto cancelled_exact_joiner = pending_exact_joiner.get();

  CHECK(cancelled_exact_joiner.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_exact_joiner.integer.has_value());
  CHECK(!cancelled_exact_joiner.metric_bit_length.has_value());
  CHECK(!cancelled_exact_joiner.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  auto late_equivalent_alias = std::async(
      std::launch::async, [&engine, equivalent_alias_modular_request] {
        return engine.evaluate_technique(equivalent_alias_modular_request);
      });
  auto late_different_parameters = std::async(
      std::launch::async, [&engine, different_parameters_modular_request] {
        return engine.evaluate_technique(different_parameters_modular_request);
      });
  const auto cancelled_equivalent_alias = late_equivalent_alias.get();
  const auto confirmed_different_parameters = late_different_parameters.get();

  CHECK(cancelled_equivalent_alias.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_equivalent_alias.integer.has_value());
  CHECK(!cancelled_equivalent_alias.metric_bit_length.has_value());
  CHECK(!cancelled_equivalent_alias.symbolic_expression.has_value());
  CHECK(confirmed_different_parameters.status == bruce::lhme::Status::Confirmed);
  CHECK(confirmed_different_parameters.exact_integer() == 4);
  CHECK(!confirmed_different_parameters.metric_bit_length.has_value());
  CHECK(!confirmed_different_parameters.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();

  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 3);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto fresh_equivalent_alias = engine.evaluate_technique(
      equivalent_alias_modular_request);
  CHECK(fresh_equivalent_alias.status == bruce::lhme::Status::Confirmed);
  CHECK(fresh_equivalent_alias.exact_integer() == 3);
  CHECK(!fresh_equivalent_alias.metric_bit_length.has_value());
  CHECK(!fresh_equivalent_alias.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);
}

TEST(cpp_engine_does_not_join_a_post_snapshot_cancelled_modular_technique_group_for_a_late_canonical_parameter_alias_at_a_different_address) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<CanonicalParameterAliasKernel>();
  CHECK(engine.register_technique(kernel));

  const bruce::lhme::TechniqueRequest origin_modular_request{
      .identity = kernel->identity(),
      .parameters = "left=1;right=2",
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::TechniqueRequest exact_joiner_request{
      .identity = kernel->identity(),
      .parameters = "left=1;right=2",
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest same_address_alias_modular_request{
      .identity = kernel->identity(),
      .parameters = "right=2;left=1",
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::TechniqueRequest different_address_alias_modular_request{
      .identity = kernel->identity(),
      .parameters = "right=2;left=1",
      .address = {.index = 12},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_exact_joiner;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async, [&engine, origin_modular_request] {
    return engine.evaluate_technique(origin_modular_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineTechniqueFinalizationBarrier
      exact_finalization_barrier;
  pending_exact_joiner = std::async(std::launch::async,
                                    [&engine, exact_joiner_request] {
    return engine.evaluate_technique(exact_joiner_request);
  });
  exact_finalization_barrier.wait_until_reached();

  engine.cancel();
  exact_finalization_barrier.release();
  const auto cancelled_exact_joiner = pending_exact_joiner.get();

  CHECK(cancelled_exact_joiner.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_exact_joiner.integer.has_value());
  CHECK(!cancelled_exact_joiner.metric_bit_length.has_value());
  CHECK(!cancelled_exact_joiner.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  auto late_same_address_alias = std::async(
      std::launch::async, [&engine, same_address_alias_modular_request] {
        return engine.evaluate_technique(same_address_alias_modular_request);
      });
  auto late_different_address_alias = std::async(
      std::launch::async, [&engine, different_address_alias_modular_request] {
        return engine.evaluate_technique(different_address_alias_modular_request);
      });
  const auto cancelled_same_address_alias = late_same_address_alias.get();
  const auto confirmed_different_address_alias =
      late_different_address_alias.get();

  CHECK(cancelled_same_address_alias.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_same_address_alias.integer.has_value());
  CHECK(!cancelled_same_address_alias.metric_bit_length.has_value());
  CHECK(!cancelled_same_address_alias.symbolic_expression.has_value());
  CHECK(confirmed_different_address_alias.status ==
        bruce::lhme::Status::Confirmed);
  CHECK(confirmed_different_address_alias.exact_integer() == 5);
  CHECK(!confirmed_different_address_alias.metric_bit_length.has_value());
  CHECK(!confirmed_different_address_alias.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();

  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 3);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto fresh_same_address_alias = engine.evaluate_technique(
      same_address_alias_modular_request);
  CHECK(fresh_same_address_alias.status == bruce::lhme::Status::Confirmed);
  CHECK(fresh_same_address_alias.exact_integer() == 3);
  CHECK(!fresh_same_address_alias.metric_bit_length.has_value());
  CHECK(!fresh_same_address_alias.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);
}

TEST(cpp_engine_does_not_join_a_post_snapshot_cancelled_modular_technique_group_for_a_late_exact_formula_evaluation_at_a_different_address) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));

  const bruce::lhme::TechniqueRequest origin_modular_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest original_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest same_address_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest different_address_formula_request{
      .spec = formula,
      .address = {.index = 12},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_original_formula;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async, [&engine, origin_modular_request] {
    return engine.evaluate_technique(origin_modular_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  pending_original_formula = std::async(
      std::launch::async, [&engine, original_formula_request] {
        return engine.evaluate(original_formula_request);
      });
  formula_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_original_formula = pending_original_formula.get();

  CHECK(cancelled_original_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_original_formula.integer.has_value());
  CHECK(!cancelled_original_formula.metric_bit_length.has_value());
  CHECK(!cancelled_original_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  auto late_same_address_formula = std::async(
      std::launch::async, [&engine, same_address_formula_request] {
        return engine.evaluate(same_address_formula_request);
      });
  auto late_different_address_formula = std::async(
      std::launch::async, [&engine, different_address_formula_request] {
        return engine.evaluate(different_address_formula_request);
      });
  const auto cancelled_same_address_formula = late_same_address_formula.get();
  const auto confirmed_different_address_formula =
      late_different_address_formula.get();

  CHECK(cancelled_same_address_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_same_address_formula.integer.has_value());
  CHECK(!cancelled_same_address_formula.metric_bit_length.has_value());
  CHECK(!cancelled_same_address_formula.symbolic_expression.has_value());
  CHECK(confirmed_different_address_formula.status ==
        bruce::lhme::Status::Confirmed);
  CHECK(confirmed_different_address_formula.exact_integer() == 409500);
  CHECK(!confirmed_different_address_formula.metric_bit_length.has_value());
  CHECK(!confirmed_different_address_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();

  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto fresh_same_address_formula = engine.evaluate(
      same_address_formula_request);
  CHECK(fresh_same_address_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(fresh_same_address_formula.exact_integer() == 102300);
  CHECK(!fresh_same_address_formula.metric_bit_length.has_value());
  CHECK(!fresh_same_address_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 2);
}

TEST(cpp_engine_does_not_join_a_post_snapshot_cancelled_modular_technique_group_for_a_late_metric_formula_evaluation_at_a_different_address) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));

  const bruce::lhme::TechniqueRequest origin_modular_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest original_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest same_address_metric_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest different_address_metric_request{
      .spec = formula,
      .address = {.index = 12},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_original_formula;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async, [&engine, origin_modular_request] {
    return engine.evaluate_technique(origin_modular_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  pending_original_formula = std::async(
      std::launch::async, [&engine, original_formula_request] {
        return engine.evaluate(original_formula_request);
      });
  formula_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_original_formula = pending_original_formula.get();

  CHECK(cancelled_original_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_original_formula.integer.has_value());
  CHECK(!cancelled_original_formula.metric_bit_length.has_value());
  CHECK(!cancelled_original_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  auto late_same_address_metric = std::async(
      std::launch::async, [&engine, same_address_metric_request] {
        return engine.evaluate(same_address_metric_request);
      });
  auto late_different_address_metric = std::async(
      std::launch::async, [&engine, different_address_metric_request] {
        return engine.evaluate(different_address_metric_request);
      });
  const auto cancelled_same_address_metric = late_same_address_metric.get();
  const auto confirmed_different_address_metric =
      late_different_address_metric.get();

  CHECK(cancelled_same_address_metric.status ==
        bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_same_address_metric.integer.has_value());
  CHECK(!cancelled_same_address_metric.metric_bit_length.has_value());
  CHECK(!cancelled_same_address_metric.symbolic_expression.has_value());
  CHECK(confirmed_different_address_metric.status ==
        bruce::lhme::Status::Confirmed);
  CHECK(!confirmed_different_address_metric.integer.has_value());
  CHECK(confirmed_different_address_metric.metric_bit_length == 19);
  CHECK(!confirmed_different_address_metric.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();

  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto fresh_same_address_metric = engine.evaluate(
      same_address_metric_request);
  CHECK(fresh_same_address_metric.status == bruce::lhme::Status::Confirmed);
  CHECK(!fresh_same_address_metric.integer.has_value());
  CHECK(fresh_same_address_metric.metric_bit_length == 17);
  CHECK(!fresh_same_address_metric.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 2);
}

TEST(cpp_engine_does_not_join_a_post_snapshot_cancelled_modular_technique_group_for_a_late_symbolic_formula_evaluation_at_a_different_address) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));

  const bruce::lhme::TechniqueRequest origin_modular_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest original_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest same_address_symbolic_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest different_address_symbolic_request{
      .spec = formula,
      .address = {.index = 12},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_original_formula;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async, [&engine, origin_modular_request] {
    return engine.evaluate_technique(origin_modular_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  pending_original_formula = std::async(
      std::launch::async, [&engine, original_formula_request] {
        return engine.evaluate(original_formula_request);
      });
  formula_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_original_formula = pending_original_formula.get();

  CHECK(cancelled_original_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_original_formula.integer.has_value());
  CHECK(!cancelled_original_formula.metric_bit_length.has_value());
  CHECK(!cancelled_original_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  auto late_same_address_symbolic = std::async(
      std::launch::async, [&engine, same_address_symbolic_request] {
        return engine.evaluate(same_address_symbolic_request);
      });
  auto late_different_address_symbolic = std::async(
      std::launch::async, [&engine, different_address_symbolic_request] {
        return engine.evaluate(different_address_symbolic_request);
      });
  const auto cancelled_same_address_symbolic =
      late_same_address_symbolic.get();
  const auto confirmed_different_address_symbolic =
      late_different_address_symbolic.get();

  CHECK(cancelled_same_address_symbolic.status ==
        bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_same_address_symbolic.integer.has_value());
  CHECK(!cancelled_same_address_symbolic.metric_bit_length.has_value());
  CHECK(!cancelled_same_address_symbolic.symbolic_expression.has_value());
  CHECK(confirmed_different_address_symbolic.status ==
        bruce::lhme::Status::Confirmed);
  CHECK(!confirmed_different_address_symbolic.integer.has_value());
  CHECK(!confirmed_different_address_symbolic.metric_bit_length.has_value());
  CHECK(confirmed_different_address_symbolic.symbolic_expression ==
        "100*(2^12-1)");
  CHECK(engine.inspect().resident_entries == 0);

  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();

  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto fresh_same_address_symbolic = engine.evaluate(
      same_address_symbolic_request);
  CHECK(fresh_same_address_symbolic.status ==
        bruce::lhme::Status::Confirmed);
  CHECK(!fresh_same_address_symbolic.integer.has_value());
  CHECK(!fresh_same_address_symbolic.metric_bit_length.has_value());
  CHECK(fresh_same_address_symbolic.symbolic_expression == "100*(2^10-1)");
  CHECK(engine.inspect().resident_entries == 0);
}

TEST(cpp_engine_does_not_join_a_post_snapshot_cancelled_modular_technique_group_for_a_late_modular_formula_evaluation_at_a_different_address) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));

  const bruce::lhme::TechniqueRequest origin_modular_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest original_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest same_address_modular_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest different_address_modular_request{
      .spec = formula,
      .address = {.index = 12},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_original_formula;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async, [&engine, origin_modular_request] {
    return engine.evaluate_technique(origin_modular_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  pending_original_formula = std::async(
      std::launch::async, [&engine, original_formula_request] {
        return engine.evaluate(original_formula_request);
      });
  formula_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_original_formula = pending_original_formula.get();

  CHECK(cancelled_original_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_original_formula.integer.has_value());
  CHECK(!cancelled_original_formula.metric_bit_length.has_value());
  CHECK(!cancelled_original_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  auto late_same_address_modular = std::async(
      std::launch::async, [&engine, same_address_modular_request] {
        return engine.evaluate(same_address_modular_request);
      });
  auto late_different_address_modular = std::async(
      std::launch::async, [&engine, different_address_modular_request] {
        return engine.evaluate(different_address_modular_request);
      });
  const auto cancelled_same_address_modular = late_same_address_modular.get();
  const auto confirmed_different_address_modular =
      late_different_address_modular.get();

  CHECK(cancelled_same_address_modular.status ==
        bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_same_address_modular.integer.has_value());
  CHECK(!cancelled_same_address_modular.metric_bit_length.has_value());
  CHECK(!cancelled_same_address_modular.symbolic_expression.has_value());
  CHECK(confirmed_different_address_modular.status ==
        bruce::lhme::Status::Confirmed);
  CHECK(confirmed_different_address_modular.exact_integer() == 4);
  CHECK(!confirmed_different_address_modular.metric_bit_length.has_value());
  CHECK(!confirmed_different_address_modular.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();

  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto fresh_same_address_modular = engine.evaluate(
      same_address_modular_request);
  CHECK(fresh_same_address_modular.status ==
        bruce::lhme::Status::Confirmed);
  CHECK(fresh_same_address_modular.exact_integer() == 11);
  CHECK(!fresh_same_address_modular.metric_bit_length.has_value());
  CHECK(!fresh_same_address_modular.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);
}

TEST(cpp_engine_does_not_join_a_late_exact_range_evaluation_to_a_leased_cancelled_cross_api_group_when_multiple_groups_of_the_same_computation_exist) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));

  const bruce::lhme::TechniqueRequest origin_modular_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest original_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest held_index_12_formula_request{
      .spec = formula,
      .address = {.index = 12},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_original_formula;
  std::future<bruce::lhme::EvaluationResult> pending_index_12_formula;
  std::future<bruce::lhme::RangeResult> pending_late_range;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async, [&engine, origin_modular_request] {
    return engine.evaluate_technique(origin_modular_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  pending_original_formula = std::async(
      std::launch::async, [&engine, original_formula_request] {
        return engine.evaluate(original_formula_request);
      });
  formula_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_original_formula = pending_original_formula.get();

  CHECK(cancelled_original_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_original_formula.integer.has_value());
  CHECK(!cancelled_original_formula.metric_bit_length.has_value());
  CHECK(!cancelled_original_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      index_12_formula_post_finalization_barrier;
  pending_index_12_formula = std::async(
      std::launch::async, [&engine, held_index_12_formula_request] {
        return engine.evaluate(held_index_12_formula_request);
      });
  index_12_formula_post_finalization_barrier.wait_until_reached();

  CHECK(engine.inspect().resident_entries == 0);

  pending_late_range = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  const auto confirmed_late_range = pending_late_range.get();

  CHECK(confirmed_late_range.status == bruce::lhme::Status::Confirmed);
  CHECK(confirmed_late_range.summary.checked_count == 2);
  CHECK(confirmed_late_range.summary.confirmed_count == 2);
  CHECK(confirmed_late_range.summary.invalid_argument_count == 0);
  CHECK(confirmed_late_range.candidates.size() == 2);
  CHECK(confirmed_late_range.candidates[0].exact_integer() == 0);
  CHECK(confirmed_late_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 0);

  index_12_formula_post_finalization_barrier.release();
  const auto finalized_index_12_formula = pending_index_12_formula.get();

  CHECK(finalized_index_12_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_index_12_formula.exact_integer() == 409500);
  CHECK(!finalized_index_12_formula.metric_bit_length.has_value());
  CHECK(!finalized_index_12_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();

  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto fresh_index_10_formula = engine.evaluate(original_formula_request);
  CHECK(fresh_index_10_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(fresh_index_10_formula.exact_integer() == 102300);
  CHECK(!fresh_index_10_formula.metric_bit_length.has_value());
  CHECK(!fresh_index_10_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 2);
}

TEST(cpp_engine_forms_a_fresh_late_exact_range_group_after_a_third_cancel_instead_of_joining_any_of_multiple_leased_cross_api_groups_of_the_same_computation) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));

  const bruce::lhme::TechniqueRequest origin_modular_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest original_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest held_index_12_formula_request{
      .spec = formula,
      .address = {.index = 12},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_original_formula;
  std::future<bruce::lhme::EvaluationResult> pending_index_12_formula;
  std::future<bruce::lhme::RangeResult> pending_late_range;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async, [&engine, origin_modular_request] {
    return engine.evaluate_technique(origin_modular_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  pending_original_formula = std::async(
      std::launch::async, [&engine, original_formula_request] {
        return engine.evaluate(original_formula_request);
      });
  formula_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_original_formula = pending_original_formula.get();

  CHECK(cancelled_original_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_original_formula.integer.has_value());
  CHECK(!cancelled_original_formula.metric_bit_length.has_value());
  CHECK(!cancelled_original_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      index_12_formula_post_finalization_barrier;
  pending_index_12_formula = std::async(
      std::launch::async, [&engine, held_index_12_formula_request] {
        return engine.evaluate(held_index_12_formula_request);
      });
  index_12_formula_post_finalization_barrier.wait_until_reached();

  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  pending_late_range = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  const auto confirmed_late_range = pending_late_range.get();

  CHECK(confirmed_late_range.status == bruce::lhme::Status::Confirmed);
  CHECK(confirmed_late_range.summary.checked_count == 2);
  CHECK(confirmed_late_range.summary.confirmed_count == 2);
  CHECK(confirmed_late_range.summary.invalid_argument_count == 0);
  CHECK(confirmed_late_range.candidates.size() == 2);
  CHECK(confirmed_late_range.candidates[0].exact_integer() == 0);
  CHECK(confirmed_late_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 0);

  index_12_formula_post_finalization_barrier.release();
  const auto finalized_index_12_formula = pending_index_12_formula.get();

  CHECK(finalized_index_12_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_index_12_formula.exact_integer() == 409500);
  CHECK(!finalized_index_12_formula.metric_bit_length.has_value());
  CHECK(!finalized_index_12_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();

  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto fresh_index_10_formula = engine.evaluate(original_formula_request);
  CHECK(fresh_index_10_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(fresh_index_10_formula.exact_integer() == 102300);
  CHECK(!fresh_index_10_formula.metric_bit_length.has_value());
  CHECK(!fresh_index_10_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 2);
}

TEST(cpp_engine_keeps_a_fresh_post_snapshot_cancelled_late_exact_range_group_cancelled_for_a_late_exact_formula_joiner_after_all_prior_cross_api_group_lease_endings_and_a_fourth_cancel) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));

  const bruce::lhme::TechniqueRequest origin_modular_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest index_10_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest index_12_formula_request{
      .spec = formula,
      .address = {.index = 12},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_original_formula;
  std::future<bruce::lhme::EvaluationResult> pending_index_12_formula;
  std::future<bruce::lhme::RangeResult> pending_late_range;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async, [&engine, origin_modular_request] {
    return engine.evaluate_technique(origin_modular_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  pending_original_formula = std::async(
      std::launch::async, [&engine, index_10_formula_request] {
        return engine.evaluate(index_10_formula_request);
      });
  formula_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_original_formula = pending_original_formula.get();

  CHECK(cancelled_original_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_original_formula.integer.has_value());
  CHECK(!cancelled_original_formula.metric_bit_length.has_value());
  CHECK(!cancelled_original_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      index_12_formula_post_finalization_barrier;
  pending_index_12_formula = std::async(
      std::launch::async, [&engine, index_12_formula_request] {
        return engine.evaluate(index_12_formula_request);
      });
  index_12_formula_post_finalization_barrier.wait_until_reached();

  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  bruce::lhme::test_support::EngineRangePostFinalizationBarrier
      late_range_post_finalization_barrier;
  pending_late_range = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  late_range_post_finalization_barrier.wait_until_reached();

  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  index_12_formula_post_finalization_barrier.release();
  const auto finalized_index_12_formula = pending_index_12_formula.get();

  CHECK(finalized_index_12_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_index_12_formula.exact_integer() == 409500);
  CHECK(!finalized_index_12_formula.metric_bit_length.has_value());
  CHECK(!finalized_index_12_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();

  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto cancelled_late_formula = engine.evaluate(index_10_formula_request);
  CHECK(cancelled_late_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_late_formula.integer.has_value());
  CHECK(!cancelled_late_formula.metric_bit_length.has_value());
  CHECK(!cancelled_late_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  late_range_post_finalization_barrier.release();
  const auto finalized_late_range = pending_late_range.get();

  CHECK(finalized_late_range.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_late_range.summary.checked_count == 2);
  CHECK(finalized_late_range.summary.confirmed_count == 2);
  CHECK(finalized_late_range.summary.invalid_argument_count == 0);
  CHECK(finalized_late_range.candidates.size() == 2);
  CHECK(finalized_late_range.candidates[0].exact_integer() == 0);
  CHECK(finalized_late_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);

  const auto fresh_index_10_formula = engine.evaluate(index_10_formula_request);
  CHECK(fresh_index_10_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(fresh_index_10_formula.exact_integer() == 102300);
  CHECK(!fresh_index_10_formula.metric_bit_length.has_value());
  CHECK(!fresh_index_10_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 2);
}

TEST(cpp_engine_keeps_a_fresh_post_snapshot_cancelled_late_exact_range_group_cancelled_for_a_late_exact_technique_joiner_after_all_prior_cross_api_group_lease_endings_and_a_fourth_cancel) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));

  const bruce::lhme::TechniqueRequest origin_modular_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest index_10_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest index_12_formula_request{
      .spec = formula,
      .address = {.index = 12},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest index_10_exact_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_original_formula;
  std::future<bruce::lhme::EvaluationResult> pending_index_12_formula;
  std::future<bruce::lhme::RangeResult> pending_late_range;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async, [&engine, origin_modular_request] {
    return engine.evaluate_technique(origin_modular_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  pending_original_formula = std::async(
      std::launch::async, [&engine, index_10_formula_request] {
        return engine.evaluate(index_10_formula_request);
      });
  formula_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_original_formula = pending_original_formula.get();

  CHECK(cancelled_original_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_original_formula.integer.has_value());
  CHECK(!cancelled_original_formula.metric_bit_length.has_value());
  CHECK(!cancelled_original_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      index_12_formula_post_finalization_barrier;
  pending_index_12_formula = std::async(
      std::launch::async, [&engine, index_12_formula_request] {
        return engine.evaluate(index_12_formula_request);
      });
  index_12_formula_post_finalization_barrier.wait_until_reached();

  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  bruce::lhme::test_support::EngineRangePostFinalizationBarrier
      late_range_post_finalization_barrier;
  pending_late_range = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  late_range_post_finalization_barrier.wait_until_reached();

  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  index_12_formula_post_finalization_barrier.release();
  const auto finalized_index_12_formula = pending_index_12_formula.get();

  CHECK(finalized_index_12_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_index_12_formula.exact_integer() == 409500);
  CHECK(!finalized_index_12_formula.metric_bit_length.has_value());
  CHECK(!finalized_index_12_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();

  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto cancelled_late_technique =
      engine.evaluate_technique(index_10_exact_technique_request);
  CHECK(cancelled_late_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_late_technique.integer.has_value());
  CHECK(!cancelled_late_technique.metric_bit_length.has_value());
  CHECK(!cancelled_late_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  late_range_post_finalization_barrier.release();
  const auto finalized_late_range = pending_late_range.get();

  CHECK(finalized_late_range.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_late_range.summary.checked_count == 2);
  CHECK(finalized_late_range.summary.confirmed_count == 2);
  CHECK(finalized_late_range.summary.invalid_argument_count == 0);
  CHECK(finalized_late_range.candidates.size() == 2);
  CHECK(finalized_late_range.candidates[0].exact_integer() == 0);
  CHECK(finalized_late_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);

  const auto fresh_index_10_technique =
      engine.evaluate_technique(index_10_exact_technique_request);
  CHECK(fresh_index_10_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(fresh_index_10_technique.exact_integer() == 102300);
  CHECK(!fresh_index_10_technique.metric_bit_length.has_value());
  CHECK(!fresh_index_10_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 2);
}

TEST(cpp_engine_keeps_a_fresh_post_snapshot_cancelled_late_exact_range_group_cancelled_for_a_late_metric_technique_joiner_after_all_prior_cross_api_group_lease_endings_and_a_fourth_cancel) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));

  const bruce::lhme::TechniqueRequest origin_modular_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest index_10_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest index_12_formula_request{
      .spec = formula,
      .address = {.index = 12},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest index_10_metric_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_original_formula;
  std::future<bruce::lhme::EvaluationResult> pending_index_12_formula;
  std::future<bruce::lhme::RangeResult> pending_late_range;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async, [&engine, origin_modular_request] {
    return engine.evaluate_technique(origin_modular_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  pending_original_formula = std::async(
      std::launch::async, [&engine, index_10_formula_request] {
        return engine.evaluate(index_10_formula_request);
      });
  formula_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_original_formula = pending_original_formula.get();

  CHECK(cancelled_original_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_original_formula.integer.has_value());
  CHECK(!cancelled_original_formula.metric_bit_length.has_value());
  CHECK(!cancelled_original_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      index_12_formula_post_finalization_barrier;
  pending_index_12_formula = std::async(
      std::launch::async, [&engine, index_12_formula_request] {
        return engine.evaluate(index_12_formula_request);
      });
  index_12_formula_post_finalization_barrier.wait_until_reached();

  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  bruce::lhme::test_support::EngineRangePostFinalizationBarrier
      late_range_post_finalization_barrier;
  pending_late_range = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  late_range_post_finalization_barrier.wait_until_reached();

  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  index_12_formula_post_finalization_barrier.release();
  const auto finalized_index_12_formula = pending_index_12_formula.get();

  CHECK(finalized_index_12_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_index_12_formula.exact_integer() == 409500);
  CHECK(!finalized_index_12_formula.metric_bit_length.has_value());
  CHECK(!finalized_index_12_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();

  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto cancelled_late_technique =
      engine.evaluate_technique(index_10_metric_technique_request);
  CHECK(cancelled_late_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_late_technique.integer.has_value());
  CHECK(!cancelled_late_technique.metric_bit_length.has_value());
  CHECK(!cancelled_late_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  late_range_post_finalization_barrier.release();
  const auto finalized_late_range = pending_late_range.get();

  CHECK(finalized_late_range.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_late_range.summary.checked_count == 2);
  CHECK(finalized_late_range.summary.confirmed_count == 2);
  CHECK(finalized_late_range.summary.invalid_argument_count == 0);
  CHECK(finalized_late_range.candidates.size() == 2);
  CHECK(finalized_late_range.candidates[0].exact_integer() == 0);
  CHECK(finalized_late_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);

  const auto fresh_index_10_technique =
      engine.evaluate_technique(index_10_metric_technique_request);
  CHECK(fresh_index_10_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(!fresh_index_10_technique.integer.has_value());
  CHECK(fresh_index_10_technique.metric_bit_length == 17);
  CHECK(!fresh_index_10_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 2);
}

TEST(cpp_engine_keeps_a_fresh_post_snapshot_cancelled_late_exact_range_group_cancelled_for_a_late_symbolic_technique_joiner_after_all_prior_cross_api_group_lease_endings_and_a_fourth_cancel) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));

  const bruce::lhme::TechniqueRequest origin_modular_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest index_10_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest index_12_formula_request{
      .spec = formula,
      .address = {.index = 12},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest index_10_symbolic_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_original_formula;
  std::future<bruce::lhme::EvaluationResult> pending_index_12_formula;
  std::future<bruce::lhme::RangeResult> pending_late_range;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async, [&engine, origin_modular_request] {
    return engine.evaluate_technique(origin_modular_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  pending_original_formula = std::async(
      std::launch::async, [&engine, index_10_formula_request] {
        return engine.evaluate(index_10_formula_request);
      });
  formula_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_original_formula = pending_original_formula.get();
  CHECK(cancelled_original_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_original_formula.integer.has_value());
  CHECK(!cancelled_original_formula.metric_bit_length.has_value());
  CHECK(!cancelled_original_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      index_12_formula_post_finalization_barrier;
  pending_index_12_formula = std::async(
      std::launch::async, [&engine, index_12_formula_request] {
        return engine.evaluate(index_12_formula_request);
      });
  index_12_formula_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  bruce::lhme::test_support::EngineRangePostFinalizationBarrier
      late_range_post_finalization_barrier;
  pending_late_range = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  late_range_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  index_12_formula_post_finalization_barrier.release();
  const auto finalized_index_12_formula = pending_index_12_formula.get();
  CHECK(finalized_index_12_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_index_12_formula.exact_integer() == 409500);
  CHECK(!finalized_index_12_formula.metric_bit_length.has_value());
  CHECK(!finalized_index_12_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();
  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto cancelled_late_technique =
      engine.evaluate_technique(index_10_symbolic_technique_request);
  CHECK(cancelled_late_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_late_technique.integer.has_value());
  CHECK(!cancelled_late_technique.metric_bit_length.has_value());
  CHECK(!cancelled_late_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  late_range_post_finalization_barrier.release();
  const auto finalized_late_range = pending_late_range.get();
  CHECK(finalized_late_range.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_late_range.summary.checked_count == 2);
  CHECK(finalized_late_range.summary.confirmed_count == 2);
  CHECK(finalized_late_range.summary.invalid_argument_count == 0);
  CHECK(finalized_late_range.candidates.size() == 2);
  CHECK(finalized_late_range.candidates[0].exact_integer() == 0);
  CHECK(finalized_late_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);

  const auto fresh_index_10_technique =
      engine.evaluate_technique(index_10_symbolic_technique_request);
  CHECK(fresh_index_10_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(!fresh_index_10_technique.integer.has_value());
  CHECK(!fresh_index_10_technique.metric_bit_length.has_value());
  CHECK(fresh_index_10_technique.symbolic_expression == "100*(2^10-1)");
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_keeps_a_fresh_post_snapshot_cancelled_late_exact_range_group_cancelled_for_a_late_modular_technique_joiner_after_all_prior_cross_api_group_lease_endings_and_a_fourth_cancel) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));

  const bruce::lhme::TechniqueRequest origin_modular_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest index_10_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest index_12_formula_request{
      .spec = formula,
      .address = {.index = 12},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_original_formula;
  std::future<bruce::lhme::EvaluationResult> pending_index_12_formula;
  std::future<bruce::lhme::RangeResult> pending_late_range;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async, [&engine, origin_modular_request] {
    return engine.evaluate_technique(origin_modular_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  pending_original_formula = std::async(
      std::launch::async, [&engine, index_10_formula_request] {
        return engine.evaluate(index_10_formula_request);
      });
  formula_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_original_formula = pending_original_formula.get();
  CHECK(cancelled_original_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_original_formula.integer.has_value());
  CHECK(!cancelled_original_formula.metric_bit_length.has_value());
  CHECK(!cancelled_original_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      index_12_formula_post_finalization_barrier;
  pending_index_12_formula = std::async(
      std::launch::async, [&engine, index_12_formula_request] {
        return engine.evaluate(index_12_formula_request);
      });
  index_12_formula_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  bruce::lhme::test_support::EngineRangePostFinalizationBarrier
      late_range_post_finalization_barrier;
  pending_late_range = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  late_range_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  index_12_formula_post_finalization_barrier.release();
  const auto finalized_index_12_formula = pending_index_12_formula.get();
  CHECK(finalized_index_12_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_index_12_formula.exact_integer() == 409500);
  CHECK(!finalized_index_12_formula.metric_bit_length.has_value());
  CHECK(!finalized_index_12_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();
  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto cancelled_late_technique =
      engine.evaluate_technique(origin_modular_request);
  CHECK(cancelled_late_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_late_technique.integer.has_value());
  CHECK(!cancelled_late_technique.metric_bit_length.has_value());
  CHECK(!cancelled_late_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  late_range_post_finalization_barrier.release();
  const auto finalized_late_range = pending_late_range.get();
  CHECK(finalized_late_range.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_late_range.summary.checked_count == 2);
  CHECK(finalized_late_range.summary.confirmed_count == 2);
  CHECK(finalized_late_range.summary.invalid_argument_count == 0);
  CHECK(finalized_late_range.candidates.size() == 2);
  CHECK(finalized_late_range.candidates[0].exact_integer() == 0);
  CHECK(finalized_late_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);

  const auto fresh_index_10_technique =
      engine.evaluate_technique(origin_modular_request);
  CHECK(fresh_index_10_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(fresh_index_10_technique.exact_integer() == 11);
  CHECK(!fresh_index_10_technique.metric_bit_length.has_value());
  CHECK(!fresh_index_10_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_keeps_a_fresh_post_snapshot_cancelled_late_exact_range_group_cancelled_for_a_late_metric_formula_joiner_after_all_prior_cross_api_group_lease_endings_and_a_fourth_cancel) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));

  const bruce::lhme::TechniqueRequest origin_modular_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest index_10_exact_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest index_10_metric_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest index_12_formula_request{
      .spec = formula,
      .address = {.index = 12},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_original_formula;
  std::future<bruce::lhme::EvaluationResult> pending_index_12_formula;
  std::future<bruce::lhme::RangeResult> pending_late_range;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async, [&engine, origin_modular_request] {
    return engine.evaluate_technique(origin_modular_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  pending_original_formula = std::async(
      std::launch::async, [&engine, index_10_exact_formula_request] {
        return engine.evaluate(index_10_exact_formula_request);
      });
  formula_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_original_formula = pending_original_formula.get();
  CHECK(cancelled_original_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_original_formula.integer.has_value());
  CHECK(!cancelled_original_formula.metric_bit_length.has_value());
  CHECK(!cancelled_original_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      index_12_formula_post_finalization_barrier;
  pending_index_12_formula = std::async(
      std::launch::async, [&engine, index_12_formula_request] {
        return engine.evaluate(index_12_formula_request);
      });
  index_12_formula_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  bruce::lhme::test_support::EngineRangePostFinalizationBarrier
      late_range_post_finalization_barrier;
  pending_late_range = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  late_range_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  index_12_formula_post_finalization_barrier.release();
  const auto finalized_index_12_formula = pending_index_12_formula.get();
  CHECK(finalized_index_12_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_index_12_formula.exact_integer() == 409500);
  CHECK(!finalized_index_12_formula.metric_bit_length.has_value());
  CHECK(!finalized_index_12_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();
  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto cancelled_late_formula = engine.evaluate(index_10_metric_formula_request);
  CHECK(cancelled_late_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_late_formula.integer.has_value());
  CHECK(!cancelled_late_formula.metric_bit_length.has_value());
  CHECK(!cancelled_late_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  late_range_post_finalization_barrier.release();
  const auto finalized_late_range = pending_late_range.get();
  CHECK(finalized_late_range.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_late_range.summary.checked_count == 2);
  CHECK(finalized_late_range.summary.confirmed_count == 2);
  CHECK(finalized_late_range.summary.invalid_argument_count == 0);
  CHECK(finalized_late_range.candidates.size() == 2);
  CHECK(finalized_late_range.candidates[0].exact_integer() == 0);
  CHECK(finalized_late_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);

  const auto fresh_index_10_formula = engine.evaluate(index_10_metric_formula_request);
  CHECK(fresh_index_10_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(!fresh_index_10_formula.integer.has_value());
  CHECK(fresh_index_10_formula.metric_bit_length == 17);
  CHECK(!fresh_index_10_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 2);
}

TEST(cpp_engine_keeps_a_fresh_post_snapshot_cancelled_late_exact_range_group_cancelled_for_a_late_symbolic_formula_joiner_after_all_prior_cross_api_group_lease_endings_and_a_fourth_cancel) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));

  const bruce::lhme::TechniqueRequest origin_modular_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest index_10_exact_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest index_10_symbolic_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest index_12_formula_request{
      .spec = formula,
      .address = {.index = 12},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_original_formula;
  std::future<bruce::lhme::EvaluationResult> pending_index_12_formula;
  std::future<bruce::lhme::RangeResult> pending_late_range;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async, [&engine, origin_modular_request] {
    return engine.evaluate_technique(origin_modular_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  pending_original_formula = std::async(
      std::launch::async, [&engine, index_10_exact_formula_request] {
        return engine.evaluate(index_10_exact_formula_request);
      });
  formula_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_original_formula = pending_original_formula.get();
  CHECK(cancelled_original_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_original_formula.integer.has_value());
  CHECK(!cancelled_original_formula.metric_bit_length.has_value());
  CHECK(!cancelled_original_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      index_12_formula_post_finalization_barrier;
  pending_index_12_formula = std::async(
      std::launch::async, [&engine, index_12_formula_request] {
        return engine.evaluate(index_12_formula_request);
      });
  index_12_formula_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  bruce::lhme::test_support::EngineRangePostFinalizationBarrier
      late_range_post_finalization_barrier;
  pending_late_range = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  late_range_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  index_12_formula_post_finalization_barrier.release();
  const auto finalized_index_12_formula = pending_index_12_formula.get();
  CHECK(finalized_index_12_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_index_12_formula.exact_integer() == 409500);
  CHECK(!finalized_index_12_formula.metric_bit_length.has_value());
  CHECK(!finalized_index_12_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();
  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto cancelled_late_formula =
      engine.evaluate(index_10_symbolic_formula_request);
  CHECK(cancelled_late_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_late_formula.integer.has_value());
  CHECK(!cancelled_late_formula.metric_bit_length.has_value());
  CHECK(!cancelled_late_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  late_range_post_finalization_barrier.release();
  const auto finalized_late_range = pending_late_range.get();
  CHECK(finalized_late_range.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_late_range.summary.checked_count == 2);
  CHECK(finalized_late_range.summary.confirmed_count == 2);
  CHECK(finalized_late_range.summary.invalid_argument_count == 0);
  CHECK(finalized_late_range.candidates.size() == 2);
  CHECK(finalized_late_range.candidates[0].exact_integer() == 0);
  CHECK(finalized_late_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);

  const auto fresh_index_10_formula =
      engine.evaluate(index_10_symbolic_formula_request);
  CHECK(fresh_index_10_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(!fresh_index_10_formula.integer.has_value());
  CHECK(!fresh_index_10_formula.metric_bit_length.has_value());
  CHECK(fresh_index_10_formula.symbolic_expression == "100*(2^10-1)");
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_keeps_a_fresh_post_snapshot_cancelled_late_exact_range_group_cancelled_for_a_late_modular_formula_joiner_after_all_prior_cross_api_group_lease_endings_and_a_fourth_cancel) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));

  const bruce::lhme::TechniqueRequest origin_modular_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest index_10_exact_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest index_10_modular_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest index_12_formula_request{
      .spec = formula,
      .address = {.index = 12},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_original_formula;
  std::future<bruce::lhme::EvaluationResult> pending_index_12_formula;
  std::future<bruce::lhme::RangeResult> pending_late_range;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async, [&engine, origin_modular_request] {
    return engine.evaluate_technique(origin_modular_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  pending_original_formula = std::async(
      std::launch::async, [&engine, index_10_exact_formula_request] {
        return engine.evaluate(index_10_exact_formula_request);
      });
  formula_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_original_formula = pending_original_formula.get();
  CHECK(cancelled_original_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_original_formula.integer.has_value());
  CHECK(!cancelled_original_formula.metric_bit_length.has_value());
  CHECK(!cancelled_original_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      index_12_formula_post_finalization_barrier;
  pending_index_12_formula = std::async(
      std::launch::async, [&engine, index_12_formula_request] {
        return engine.evaluate(index_12_formula_request);
      });
  index_12_formula_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  bruce::lhme::test_support::EngineRangePostFinalizationBarrier
      late_range_post_finalization_barrier;
  pending_late_range = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  late_range_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  index_12_formula_post_finalization_barrier.release();
  const auto finalized_index_12_formula = pending_index_12_formula.get();
  CHECK(finalized_index_12_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_index_12_formula.exact_integer() == 409500);
  CHECK(!finalized_index_12_formula.metric_bit_length.has_value());
  CHECK(!finalized_index_12_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();
  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto cancelled_late_formula =
      engine.evaluate(index_10_modular_formula_request);
  CHECK(cancelled_late_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_late_formula.integer.has_value());
  CHECK(!cancelled_late_formula.metric_bit_length.has_value());
  CHECK(!cancelled_late_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  late_range_post_finalization_barrier.release();
  const auto finalized_late_range = pending_late_range.get();
  CHECK(finalized_late_range.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_late_range.summary.checked_count == 2);
  CHECK(finalized_late_range.summary.confirmed_count == 2);
  CHECK(finalized_late_range.summary.invalid_argument_count == 0);
  CHECK(finalized_late_range.candidates.size() == 2);
  CHECK(finalized_late_range.candidates[0].exact_integer() == 0);
  CHECK(finalized_late_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);

  const auto fresh_index_10_formula =
      engine.evaluate(index_10_modular_formula_request);
  CHECK(fresh_index_10_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(fresh_index_10_formula.exact_integer() == 11);
  CHECK(!fresh_index_10_formula.metric_bit_length.has_value());
  CHECK(!fresh_index_10_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_keeps_a_fresh_post_snapshot_cancelled_late_exact_range_group_cancelled_for_a_late_exact_range_joiner_after_all_prior_cross_api_group_lease_endings_and_a_fourth_cancel) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));

  const bruce::lhme::TechniqueRequest origin_modular_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest index_10_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest index_12_formula_request{
      .spec = formula,
      .address = {.index = 12},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_original_formula;
  std::future<bruce::lhme::EvaluationResult> pending_index_12_formula;
  std::future<bruce::lhme::RangeResult> pending_late_range;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async, [&engine, origin_modular_request] {
    return engine.evaluate_technique(origin_modular_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  pending_original_formula = std::async(
      std::launch::async, [&engine, index_10_formula_request] {
        return engine.evaluate(index_10_formula_request);
      });
  formula_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_original_formula = pending_original_formula.get();
  CHECK(cancelled_original_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_original_formula.integer.has_value());
  CHECK(!cancelled_original_formula.metric_bit_length.has_value());
  CHECK(!cancelled_original_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      index_12_formula_post_finalization_barrier;
  pending_index_12_formula = std::async(
      std::launch::async, [&engine, index_12_formula_request] {
        return engine.evaluate(index_12_formula_request);
      });
  index_12_formula_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  bruce::lhme::test_support::EngineRangePostFinalizationBarrier
      late_range_post_finalization_barrier;
  pending_late_range = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  late_range_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  index_12_formula_post_finalization_barrier.release();
  const auto finalized_index_12_formula = pending_index_12_formula.get();
  CHECK(finalized_index_12_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_index_12_formula.exact_integer() == 409500);
  CHECK(!finalized_index_12_formula.metric_bit_length.has_value());
  CHECK(!finalized_index_12_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();
  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto cancelled_late_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(cancelled_late_range.status == bruce::lhme::Status::Cancelled);
  CHECK(cancelled_late_range.candidates.empty());
  CHECK(engine.inspect().resident_entries == 1);

  late_range_post_finalization_barrier.release();
  const auto finalized_late_range = pending_late_range.get();
  CHECK(finalized_late_range.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_late_range.summary.checked_count == 2);
  CHECK(finalized_late_range.summary.confirmed_count == 2);
  CHECK(finalized_late_range.summary.invalid_argument_count == 0);
  CHECK(finalized_late_range.candidates.size() == 2);
  CHECK(finalized_late_range.candidates[0].exact_integer() == 0);
  CHECK(finalized_late_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);

  const auto fresh_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(fresh_range.status == bruce::lhme::Status::Confirmed);
  CHECK(fresh_range.summary.checked_count == 2);
  CHECK(fresh_range.summary.confirmed_count == 2);
  CHECK(fresh_range.summary.invalid_argument_count == 0);
  CHECK(fresh_range.candidates.size() == 2);
  CHECK(fresh_range.candidates[0].exact_integer() == 0);
  CHECK(fresh_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_does_not_join_a_fresh_post_snapshot_cancelled_late_exact_range_group_for_a_late_overlapping_exact_range_evaluation_after_all_prior_cross_api_group_lease_endings_and_a_fourth_cancel) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));

  const bruce::lhme::TechniqueRequest origin_modular_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest index_10_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest index_12_formula_request{
      .spec = formula,
      .address = {.index = 12},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_original_formula;
  std::future<bruce::lhme::EvaluationResult> pending_index_12_formula;
  std::future<bruce::lhme::EvaluationResult>
      pending_range_group_formula_cache_hit;
  std::future<bruce::lhme::RangeResult> pending_late_range;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async, [&engine, origin_modular_request] {
    return engine.evaluate_technique(origin_modular_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  pending_original_formula = std::async(
      std::launch::async, [&engine, index_10_formula_request] {
        return engine.evaluate(index_10_formula_request);
      });
  formula_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_original_formula = pending_original_formula.get();
  CHECK(cancelled_original_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_original_formula.integer.has_value());
  CHECK(!cancelled_original_formula.metric_bit_length.has_value());
  CHECK(!cancelled_original_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      index_12_formula_post_finalization_barrier;
  pending_index_12_formula = std::async(
      std::launch::async, [&engine, index_12_formula_request] {
        return engine.evaluate(index_12_formula_request);
      });
  index_12_formula_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  bruce::lhme::test_support::EngineRangePostFinalizationBarrier
      late_range_post_finalization_barrier;
  pending_late_range = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  late_range_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 0);

  index_12_formula_post_finalization_barrier.release();
  const auto finalized_index_12_formula = pending_index_12_formula.get();
  CHECK(finalized_index_12_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_index_12_formula.exact_integer() == 409500);
  CHECK(!finalized_index_12_formula.metric_bit_length.has_value());
  CHECK(!finalized_index_12_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  bruce::lhme::test_support::EngineFormulaCacheHitPostFinalizationBarrier
      range_group_formula_cache_hit_post_finalization_barrier;
  pending_range_group_formula_cache_hit = std::async(
      std::launch::async, [&engine, index_12_formula_request] {
        return engine.evaluate(index_12_formula_request);
      });
  range_group_formula_cache_hit_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 1);

  engine.cancel();
  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();
  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  late_range_post_finalization_barrier.release();
  const auto finalized_late_range = pending_late_range.get();
  CHECK(finalized_late_range.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_late_range.summary.checked_count == 2);
  CHECK(finalized_late_range.summary.confirmed_count == 2);
  CHECK(finalized_late_range.summary.invalid_argument_count == 0);
  CHECK(finalized_late_range.candidates.size() == 2);
  CHECK(finalized_late_range.candidates[0].exact_integer() == 0);
  CHECK(finalized_late_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);

  const auto confirmed_overlapping_range =
      engine.evaluate_range(range_request, {1, 2});
  CHECK(confirmed_overlapping_range.status == bruce::lhme::Status::Confirmed);
  CHECK(confirmed_overlapping_range.summary.checked_count == 2);
  CHECK(confirmed_overlapping_range.summary.confirmed_count == 2);
  CHECK(confirmed_overlapping_range.summary.invalid_argument_count == 0);
  CHECK(confirmed_overlapping_range.candidates.size() == 2);
  CHECK(confirmed_overlapping_range.candidates[0].exact_integer() == 100);
  CHECK(confirmed_overlapping_range.candidates[1].exact_integer() == 300);
  CHECK(engine.inspect().resident_entries == 1);

  range_group_formula_cache_hit_post_finalization_barrier.release();
  const auto finalized_range_group_formula_cache_hit =
      pending_range_group_formula_cache_hit.get();
  CHECK(finalized_range_group_formula_cache_hit.status ==
        bruce::lhme::Status::Confirmed);
  CHECK(finalized_range_group_formula_cache_hit.exact_integer() == 409500);
  CHECK(!finalized_range_group_formula_cache_hit.metric_bit_length.has_value());
  CHECK(!finalized_range_group_formula_cache_hit.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto fresh_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(fresh_range.status == bruce::lhme::Status::Confirmed);
  CHECK(fresh_range.summary.checked_count == 2);
  CHECK(fresh_range.summary.confirmed_count == 2);
  CHECK(fresh_range.summary.invalid_argument_count == 0);
  CHECK(fresh_range.candidates.size() == 2);
  CHECK(fresh_range.candidates[0].exact_integer() == 0);
  CHECK(fresh_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_forms_a_fresh_late_exact_formula_group_when_two_distinct_post_snapshot_cancelled_exact_range_groups_of_the_same_computation_are_simultaneously_leased) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));

  const bruce::lhme::TechniqueRequest origin_modular_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest index_10_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest index_12_formula_request{
      .spec = formula,
      .address = {.index = 12},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest exact_range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_original_formula;
  std::future<bruce::lhme::EvaluationResult> pending_index_12_formula;
  std::future<bruce::lhme::EvaluationResult>
      pending_range_group_formula_cache_hit;
  std::future<bruce::lhme::RangeResult> pending_first_range;
  std::future<bruce::lhme::RangeResult> pending_second_range;
  std::optional<
      bruce::lhme::test_support::EngineFormulaCacheHitPostFinalizationBarrier>
      range_group_formula_cache_hit_post_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async, [&engine, origin_modular_request] {
    return engine.evaluate_technique(origin_modular_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  pending_original_formula = std::async(
      std::launch::async, [&engine, index_10_formula_request] {
        return engine.evaluate(index_10_formula_request);
      });
  formula_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_original_formula = pending_original_formula.get();
  CHECK(cancelled_original_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_original_formula.integer.has_value());
  CHECK(!cancelled_original_formula.metric_bit_length.has_value());
  CHECK(!cancelled_original_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      index_12_formula_post_finalization_barrier;
  pending_index_12_formula = std::async(
      std::launch::async, [&engine, index_12_formula_request] {
        return engine.evaluate(index_12_formula_request);
      });
  index_12_formula_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  {
    bruce::lhme::test_support::EngineRangePostFinalizationBarrier
        first_range_post_finalization_barrier;
    pending_first_range =
        std::async(std::launch::async, [&engine, exact_range_request] {
          return engine.evaluate_range(exact_range_request, {0, 1});
        });
    first_range_post_finalization_barrier.wait_until_reached();
    CHECK(engine.inspect().resident_entries == 0);

    index_12_formula_post_finalization_barrier.release();
    const auto finalized_index_12_formula = pending_index_12_formula.get();
    CHECK(finalized_index_12_formula.status ==
          bruce::lhme::Status::Confirmed);
    CHECK(finalized_index_12_formula.exact_integer() == 409500);
    CHECK(!finalized_index_12_formula.metric_bit_length.has_value());
    CHECK(!finalized_index_12_formula.symbolic_expression.has_value());
    CHECK(engine.inspect().resident_entries == 1);

    range_group_formula_cache_hit_post_finalization_barrier.emplace();
    pending_range_group_formula_cache_hit = std::async(
        std::launch::async, [&engine, index_12_formula_request] {
          return engine.evaluate(index_12_formula_request);
        });
    range_group_formula_cache_hit_post_finalization_barrier
        ->wait_until_reached();
    CHECK(engine.inspect().resident_entries == 1);

    engine.cancel();
    origin_post_finalization_barrier.release();
    const auto finalized_origin = pending_origin.get();
    CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
    CHECK(finalized_origin.exact_integer() == 11);
    CHECK(!finalized_origin.metric_bit_length.has_value());
    CHECK(!finalized_origin.symbolic_expression.has_value());
    CHECK(engine.inspect().resident_entries == 1);

    first_range_post_finalization_barrier.release();
    const auto finalized_first_range = pending_first_range.get();
    CHECK(finalized_first_range.status == bruce::lhme::Status::Confirmed);
    CHECK(finalized_first_range.summary.checked_count == 2);
    CHECK(finalized_first_range.summary.confirmed_count == 2);
    CHECK(finalized_first_range.summary.invalid_argument_count == 0);
    CHECK(finalized_first_range.candidates.size() == 2);
    CHECK(finalized_first_range.candidates[0].exact_integer() == 0);
    CHECK(finalized_first_range.candidates[1].exact_integer() == 100);
    CHECK(engine.inspect().resident_entries == 1);
  }

  {
    bruce::lhme::test_support::EngineRangePostFinalizationBarrier
        second_range_post_finalization_barrier;
    pending_second_range =
        std::async(std::launch::async, [&engine, exact_range_request] {
          return engine.evaluate_range(exact_range_request, {1, 2});
        });
    second_range_post_finalization_barrier.wait_until_reached();
    CHECK(engine.inspect().resident_entries == 1);

    engine.cancel();
    const auto confirmed_late_formula = engine.evaluate(index_10_formula_request);
    CHECK(confirmed_late_formula.status == bruce::lhme::Status::Confirmed);
    CHECK(confirmed_late_formula.exact_integer() == 102300);
    CHECK(!confirmed_late_formula.metric_bit_length.has_value());
    CHECK(!confirmed_late_formula.symbolic_expression.has_value());
    CHECK(engine.inspect().resident_entries == 2);

    range_group_formula_cache_hit_post_finalization_barrier->release();
    const auto finalized_range_group_formula_cache_hit =
        pending_range_group_formula_cache_hit.get();
    CHECK(finalized_range_group_formula_cache_hit.status ==
          bruce::lhme::Status::Confirmed);
    CHECK(finalized_range_group_formula_cache_hit.exact_integer() == 409500);
    CHECK(!finalized_range_group_formula_cache_hit.metric_bit_length
               .has_value());
    CHECK(!finalized_range_group_formula_cache_hit.symbolic_expression
               .has_value());
    CHECK(engine.inspect().resident_entries == 2);

    second_range_post_finalization_barrier.release();
    const auto finalized_second_range = pending_second_range.get();
    CHECK(finalized_second_range.status == bruce::lhme::Status::Confirmed);
    CHECK(finalized_second_range.summary.checked_count == 2);
    CHECK(finalized_second_range.summary.confirmed_count == 2);
    CHECK(finalized_second_range.summary.invalid_argument_count == 0);
    CHECK(finalized_second_range.candidates.size() == 2);
    CHECK(finalized_second_range.candidates[0].exact_integer() == 100);
    CHECK(finalized_second_range.candidates[1].exact_integer() == 300);
    CHECK(engine.inspect().resident_entries == 2);
  }

  const auto fresh_first_range =
      engine.evaluate_range(exact_range_request, {0, 1});
  CHECK(fresh_first_range.status == bruce::lhme::Status::Confirmed);
  CHECK(fresh_first_range.summary.checked_count == 2);
  CHECK(fresh_first_range.summary.confirmed_count == 2);
  CHECK(fresh_first_range.summary.invalid_argument_count == 0);
  CHECK(fresh_first_range.candidates.size() == 2);
  CHECK(fresh_first_range.candidates[0].exact_integer() == 0);
  CHECK(fresh_first_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 2);
}

TEST(cpp_engine_forms_a_fresh_late_exact_technique_group_when_two_distinct_post_snapshot_cancelled_exact_range_groups_of_the_same_computation_are_simultaneously_leased) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));

  const bruce::lhme::TechniqueRequest origin_modular_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest index_10_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest index_12_formula_request{
      .spec = formula,
      .address = {.index = 12},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest exact_range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest index_10_exact_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_original_formula;
  std::future<bruce::lhme::EvaluationResult> pending_index_12_formula;
  std::future<bruce::lhme::EvaluationResult>
      pending_range_group_formula_cache_hit;
  std::future<bruce::lhme::RangeResult> pending_first_range;
  std::future<bruce::lhme::RangeResult> pending_second_range;
  std::optional<
      bruce::lhme::test_support::EngineFormulaCacheHitPostFinalizationBarrier>
      range_group_formula_cache_hit_post_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async, [&engine, origin_modular_request] {
    return engine.evaluate_technique(origin_modular_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  pending_original_formula = std::async(
      std::launch::async, [&engine, index_10_formula_request] {
        return engine.evaluate(index_10_formula_request);
      });
  formula_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_original_formula = pending_original_formula.get();
  CHECK(cancelled_original_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_original_formula.integer.has_value());
  CHECK(!cancelled_original_formula.metric_bit_length.has_value());
  CHECK(!cancelled_original_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      index_12_formula_post_finalization_barrier;
  pending_index_12_formula = std::async(
      std::launch::async, [&engine, index_12_formula_request] {
        return engine.evaluate(index_12_formula_request);
      });
  index_12_formula_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  {
    bruce::lhme::test_support::EngineRangePostFinalizationBarrier
        first_range_post_finalization_barrier;
    pending_first_range =
        std::async(std::launch::async, [&engine, exact_range_request] {
          return engine.evaluate_range(exact_range_request, {0, 1});
        });
    first_range_post_finalization_barrier.wait_until_reached();
    CHECK(engine.inspect().resident_entries == 0);

    index_12_formula_post_finalization_barrier.release();
    const auto finalized_index_12_formula = pending_index_12_formula.get();
    CHECK(finalized_index_12_formula.status ==
          bruce::lhme::Status::Confirmed);
    CHECK(finalized_index_12_formula.exact_integer() == 409500);
    CHECK(!finalized_index_12_formula.metric_bit_length.has_value());
    CHECK(!finalized_index_12_formula.symbolic_expression.has_value());
    CHECK(engine.inspect().resident_entries == 1);

    range_group_formula_cache_hit_post_finalization_barrier.emplace();
    pending_range_group_formula_cache_hit = std::async(
        std::launch::async, [&engine, index_12_formula_request] {
          return engine.evaluate(index_12_formula_request);
        });
    range_group_formula_cache_hit_post_finalization_barrier
        ->wait_until_reached();
    CHECK(engine.inspect().resident_entries == 1);

    engine.cancel();
    origin_post_finalization_barrier.release();
    const auto finalized_origin = pending_origin.get();
    CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
    CHECK(finalized_origin.exact_integer() == 11);
    CHECK(!finalized_origin.metric_bit_length.has_value());
    CHECK(!finalized_origin.symbolic_expression.has_value());
    CHECK(engine.inspect().resident_entries == 1);

    first_range_post_finalization_barrier.release();
    const auto finalized_first_range = pending_first_range.get();
    CHECK(finalized_first_range.status == bruce::lhme::Status::Confirmed);
    CHECK(finalized_first_range.summary.checked_count == 2);
    CHECK(finalized_first_range.summary.confirmed_count == 2);
    CHECK(finalized_first_range.summary.invalid_argument_count == 0);
    CHECK(finalized_first_range.candidates.size() == 2);
    CHECK(finalized_first_range.candidates[0].exact_integer() == 0);
    CHECK(finalized_first_range.candidates[1].exact_integer() == 100);
    CHECK(engine.inspect().resident_entries == 1);
  }

  {
    bruce::lhme::test_support::EngineRangePostFinalizationBarrier
        second_range_post_finalization_barrier;
    pending_second_range =
        std::async(std::launch::async, [&engine, exact_range_request] {
          return engine.evaluate_range(exact_range_request, {1, 2});
        });
    second_range_post_finalization_barrier.wait_until_reached();
    CHECK(engine.inspect().resident_entries == 1);

    engine.cancel();
    const auto confirmed_late_technique =
        engine.evaluate_technique(index_10_exact_technique_request);
    CHECK(confirmed_late_technique.status == bruce::lhme::Status::Confirmed);
    CHECK(confirmed_late_technique.exact_integer() == 102300);
    CHECK(!confirmed_late_technique.metric_bit_length.has_value());
    CHECK(!confirmed_late_technique.symbolic_expression.has_value());
    CHECK(engine.inspect().resident_entries == 2);

    range_group_formula_cache_hit_post_finalization_barrier->release();
    const auto finalized_range_group_formula_cache_hit =
        pending_range_group_formula_cache_hit.get();
    CHECK(finalized_range_group_formula_cache_hit.status ==
          bruce::lhme::Status::Confirmed);
    CHECK(finalized_range_group_formula_cache_hit.exact_integer() == 409500);
    CHECK(!finalized_range_group_formula_cache_hit.metric_bit_length
               .has_value());
    CHECK(!finalized_range_group_formula_cache_hit.symbolic_expression
               .has_value());
    CHECK(engine.inspect().resident_entries == 2);

    second_range_post_finalization_barrier.release();
    const auto finalized_second_range = pending_second_range.get();
    CHECK(finalized_second_range.status == bruce::lhme::Status::Confirmed);
    CHECK(finalized_second_range.summary.checked_count == 2);
    CHECK(finalized_second_range.summary.confirmed_count == 2);
    CHECK(finalized_second_range.summary.invalid_argument_count == 0);
    CHECK(finalized_second_range.candidates.size() == 2);
    CHECK(finalized_second_range.candidates[0].exact_integer() == 100);
    CHECK(finalized_second_range.candidates[1].exact_integer() == 300);
    CHECK(engine.inspect().resident_entries == 2);
  }

  const auto fresh_first_range =
      engine.evaluate_range(exact_range_request, {0, 1});
  CHECK(fresh_first_range.status == bruce::lhme::Status::Confirmed);
  CHECK(fresh_first_range.summary.checked_count == 2);
  CHECK(fresh_first_range.summary.confirmed_count == 2);
  CHECK(fresh_first_range.summary.invalid_argument_count == 0);
  CHECK(fresh_first_range.candidates.size() == 2);
  CHECK(fresh_first_range.candidates[0].exact_integer() == 0);
  CHECK(fresh_first_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 2);
}

TEST(cpp_engine_forms_a_fresh_late_metric_formula_group_when_exact_and_metric_post_snapshot_cancelled_range_groups_of_the_same_computation_and_canonical_range_are_simultaneously_leased) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));

  const bruce::lhme::TechniqueRequest origin_modular_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest index_10_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest index_10_metric_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest index_12_formula_request{
      .spec = formula,
      .address = {.index = 12},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest exact_range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest metric_range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_original_formula;
  std::future<bruce::lhme::EvaluationResult> pending_index_12_formula;
  std::future<bruce::lhme::EvaluationResult>
      pending_exact_range_group_formula_cache_hit;
  std::future<bruce::lhme::RangeResult> pending_exact_range;
  std::future<bruce::lhme::RangeResult> pending_metric_range;
  std::optional<
      bruce::lhme::test_support::EngineFormulaCacheHitPostFinalizationBarrier>
      exact_range_group_formula_cache_hit_post_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async, [&engine, origin_modular_request] {
    return engine.evaluate_technique(origin_modular_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  pending_original_formula = std::async(
      std::launch::async, [&engine, index_10_formula_request] {
        return engine.evaluate(index_10_formula_request);
      });
  formula_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_original_formula = pending_original_formula.get();
  CHECK(cancelled_original_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_original_formula.integer.has_value());
  CHECK(!cancelled_original_formula.metric_bit_length.has_value());
  CHECK(!cancelled_original_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      index_12_formula_post_finalization_barrier;
  pending_index_12_formula = std::async(
      std::launch::async, [&engine, index_12_formula_request] {
        return engine.evaluate(index_12_formula_request);
      });
  index_12_formula_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  {
    bruce::lhme::test_support::EngineRangePostFinalizationBarrier
        exact_range_post_finalization_barrier;
    pending_exact_range =
        std::async(std::launch::async, [&engine, exact_range_request] {
          return engine.evaluate_range(exact_range_request, {0, 1});
        });
    exact_range_post_finalization_barrier.wait_until_reached();
    CHECK(engine.inspect().resident_entries == 0);

    index_12_formula_post_finalization_barrier.release();
    const auto finalized_index_12_formula = pending_index_12_formula.get();
    CHECK(finalized_index_12_formula.status ==
          bruce::lhme::Status::Confirmed);
    CHECK(finalized_index_12_formula.exact_integer() == 409500);
    CHECK(!finalized_index_12_formula.metric_bit_length.has_value());
    CHECK(!finalized_index_12_formula.symbolic_expression.has_value());
    CHECK(engine.inspect().resident_entries == 1);

    exact_range_group_formula_cache_hit_post_finalization_barrier.emplace();
    pending_exact_range_group_formula_cache_hit = std::async(
        std::launch::async, [&engine, index_12_formula_request] {
          return engine.evaluate(index_12_formula_request);
        });
    exact_range_group_formula_cache_hit_post_finalization_barrier
        ->wait_until_reached();
    CHECK(engine.inspect().resident_entries == 1);

    engine.cancel();
    origin_post_finalization_barrier.release();
    const auto finalized_origin = pending_origin.get();
    CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
    CHECK(finalized_origin.exact_integer() == 11);
    CHECK(!finalized_origin.metric_bit_length.has_value());
    CHECK(!finalized_origin.symbolic_expression.has_value());
    CHECK(engine.inspect().resident_entries == 1);

    exact_range_post_finalization_barrier.release();
    const auto finalized_exact_range = pending_exact_range.get();
    CHECK(finalized_exact_range.status == bruce::lhme::Status::Confirmed);
    CHECK(finalized_exact_range.summary.checked_count == 2);
    CHECK(finalized_exact_range.summary.confirmed_count == 2);
    CHECK(finalized_exact_range.summary.invalid_argument_count == 0);
    CHECK(finalized_exact_range.candidates.size() == 2);
    CHECK(finalized_exact_range.candidates[0].exact_integer() == 0);
    CHECK(finalized_exact_range.candidates[1].exact_integer() == 100);
    CHECK(engine.inspect().resident_entries == 1);
  }

  {
    bruce::lhme::test_support::EngineRangePostFinalizationBarrier
        metric_range_post_finalization_barrier;
    pending_metric_range =
        std::async(std::launch::async, [&engine, metric_range_request] {
          return engine.evaluate_range(metric_range_request, {0, 1});
        });
    metric_range_post_finalization_barrier.wait_until_reached();
    CHECK(engine.inspect().resident_entries == 1);

    engine.cancel();
    const auto confirmed_late_metric_formula =
        engine.evaluate(index_10_metric_formula_request);
    CHECK(confirmed_late_metric_formula.status ==
          bruce::lhme::Status::Confirmed);
    CHECK(!confirmed_late_metric_formula.integer.has_value());
    CHECK(confirmed_late_metric_formula.metric_bit_length == 17);
    CHECK(!confirmed_late_metric_formula.symbolic_expression.has_value());
    CHECK(engine.inspect().resident_entries == 2);

    exact_range_group_formula_cache_hit_post_finalization_barrier->release();
    const auto finalized_exact_range_group_formula_cache_hit =
        pending_exact_range_group_formula_cache_hit.get();
    CHECK(finalized_exact_range_group_formula_cache_hit.status ==
          bruce::lhme::Status::Confirmed);
    CHECK(finalized_exact_range_group_formula_cache_hit.exact_integer() ==
          409500);
    CHECK(!finalized_exact_range_group_formula_cache_hit.metric_bit_length
               .has_value());
    CHECK(!finalized_exact_range_group_formula_cache_hit.symbolic_expression
               .has_value());
    CHECK(engine.inspect().resident_entries == 2);

    metric_range_post_finalization_barrier.release();
    const auto finalized_metric_range = pending_metric_range.get();
    CHECK(finalized_metric_range.status == bruce::lhme::Status::Confirmed);
    CHECK(finalized_metric_range.summary.checked_count == 2);
    CHECK(finalized_metric_range.summary.confirmed_count == 2);
    CHECK(finalized_metric_range.summary.invalid_argument_count == 0);
    CHECK(finalized_metric_range.candidates.size() == 2);
    CHECK(!finalized_metric_range.candidates[0].integer.has_value());
    CHECK(finalized_metric_range.candidates[0].metric_bit_length == 0);
    CHECK(!finalized_metric_range.candidates[0].symbolic_expression
               .has_value());
    CHECK(!finalized_metric_range.candidates[1].integer.has_value());
    CHECK(finalized_metric_range.candidates[1].metric_bit_length == 7);
    CHECK(!finalized_metric_range.candidates[1].symbolic_expression
               .has_value());
    CHECK(engine.inspect().resident_entries == 2);
  }

  const auto fresh_exact_range =
      engine.evaluate_range(exact_range_request, {0, 1});
  CHECK(fresh_exact_range.status == bruce::lhme::Status::Confirmed);
  CHECK(fresh_exact_range.summary.checked_count == 2);
  CHECK(fresh_exact_range.summary.confirmed_count == 2);
  CHECK(fresh_exact_range.summary.invalid_argument_count == 0);
  CHECK(fresh_exact_range.candidates.size() == 2);
  CHECK(fresh_exact_range.candidates[0].exact_integer() == 0);
  CHECK(fresh_exact_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 2);
}

TEST(cpp_engine_forms_a_fresh_late_metric_technique_group_when_exact_and_metric_post_snapshot_cancelled_range_groups_of_the_same_computation_and_canonical_range_are_simultaneously_leased) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));

  const bruce::lhme::TechniqueRequest origin_modular_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest index_10_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest index_12_formula_request{
      .spec = formula,
      .address = {.index = 12},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest exact_range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest metric_range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest index_10_metric_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_original_formula;
  std::future<bruce::lhme::EvaluationResult> pending_index_12_formula;
  std::future<bruce::lhme::EvaluationResult>
      pending_exact_range_group_formula_cache_hit;
  std::future<bruce::lhme::RangeResult> pending_exact_range;
  std::future<bruce::lhme::RangeResult> pending_metric_range;
  std::optional<
      bruce::lhme::test_support::EngineFormulaCacheHitPostFinalizationBarrier>
      exact_range_group_formula_cache_hit_post_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async, [&engine, origin_modular_request] {
    return engine.evaluate_technique(origin_modular_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  pending_original_formula = std::async(
      std::launch::async, [&engine, index_10_formula_request] {
        return engine.evaluate(index_10_formula_request);
      });
  formula_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_original_formula = pending_original_formula.get();
  CHECK(cancelled_original_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_original_formula.integer.has_value());
  CHECK(!cancelled_original_formula.metric_bit_length.has_value());
  CHECK(!cancelled_original_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      index_12_formula_post_finalization_barrier;
  pending_index_12_formula = std::async(
      std::launch::async, [&engine, index_12_formula_request] {
        return engine.evaluate(index_12_formula_request);
      });
  index_12_formula_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  {
    bruce::lhme::test_support::EngineRangePostFinalizationBarrier
        exact_range_post_finalization_barrier;
    pending_exact_range =
        std::async(std::launch::async, [&engine, exact_range_request] {
          return engine.evaluate_range(exact_range_request, {0, 1});
        });
    exact_range_post_finalization_barrier.wait_until_reached();
    CHECK(engine.inspect().resident_entries == 0);

    index_12_formula_post_finalization_barrier.release();
    const auto finalized_index_12_formula = pending_index_12_formula.get();
    CHECK(finalized_index_12_formula.status ==
          bruce::lhme::Status::Confirmed);
    CHECK(finalized_index_12_formula.exact_integer() == 409500);
    CHECK(!finalized_index_12_formula.metric_bit_length.has_value());
    CHECK(!finalized_index_12_formula.symbolic_expression.has_value());
    CHECK(engine.inspect().resident_entries == 1);

    exact_range_group_formula_cache_hit_post_finalization_barrier.emplace();
    pending_exact_range_group_formula_cache_hit = std::async(
        std::launch::async, [&engine, index_12_formula_request] {
          return engine.evaluate(index_12_formula_request);
        });
    exact_range_group_formula_cache_hit_post_finalization_barrier
        ->wait_until_reached();
    CHECK(engine.inspect().resident_entries == 1);

    engine.cancel();
    origin_post_finalization_barrier.release();
    const auto finalized_origin = pending_origin.get();
    CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
    CHECK(finalized_origin.exact_integer() == 11);
    CHECK(!finalized_origin.metric_bit_length.has_value());
    CHECK(!finalized_origin.symbolic_expression.has_value());
    CHECK(engine.inspect().resident_entries == 1);

    exact_range_post_finalization_barrier.release();
    const auto finalized_exact_range = pending_exact_range.get();
    CHECK(finalized_exact_range.status == bruce::lhme::Status::Confirmed);
    CHECK(finalized_exact_range.summary.checked_count == 2);
    CHECK(finalized_exact_range.summary.confirmed_count == 2);
    CHECK(finalized_exact_range.summary.invalid_argument_count == 0);
    CHECK(finalized_exact_range.candidates.size() == 2);
    CHECK(finalized_exact_range.candidates[0].exact_integer() == 0);
    CHECK(finalized_exact_range.candidates[1].exact_integer() == 100);
    CHECK(engine.inspect().resident_entries == 1);
  }

  {
    bruce::lhme::test_support::EngineRangePostFinalizationBarrier
        metric_range_post_finalization_barrier;
    pending_metric_range =
        std::async(std::launch::async, [&engine, metric_range_request] {
          return engine.evaluate_range(metric_range_request, {0, 1});
        });
    metric_range_post_finalization_barrier.wait_until_reached();
    CHECK(engine.inspect().resident_entries == 1);

    engine.cancel();
    const auto confirmed_late_metric_technique =
        engine.evaluate_technique(index_10_metric_technique_request);
    CHECK(confirmed_late_metric_technique.status ==
          bruce::lhme::Status::Confirmed);
    CHECK(!confirmed_late_metric_technique.integer.has_value());
    CHECK(confirmed_late_metric_technique.metric_bit_length == 17);
    CHECK(!confirmed_late_metric_technique.symbolic_expression.has_value());
    CHECK(engine.inspect().resident_entries == 2);

    exact_range_group_formula_cache_hit_post_finalization_barrier->release();
    const auto finalized_exact_range_group_formula_cache_hit =
        pending_exact_range_group_formula_cache_hit.get();
    CHECK(finalized_exact_range_group_formula_cache_hit.status ==
          bruce::lhme::Status::Confirmed);
    CHECK(finalized_exact_range_group_formula_cache_hit.exact_integer() ==
          409500);
    CHECK(!finalized_exact_range_group_formula_cache_hit.metric_bit_length
               .has_value());
    CHECK(!finalized_exact_range_group_formula_cache_hit.symbolic_expression
               .has_value());
    CHECK(engine.inspect().resident_entries == 2);

    metric_range_post_finalization_barrier.release();
    const auto finalized_metric_range = pending_metric_range.get();
    CHECK(finalized_metric_range.status == bruce::lhme::Status::Confirmed);
    CHECK(finalized_metric_range.summary.checked_count == 2);
    CHECK(finalized_metric_range.summary.confirmed_count == 2);
    CHECK(finalized_metric_range.summary.invalid_argument_count == 0);
    CHECK(finalized_metric_range.candidates.size() == 2);
    CHECK(!finalized_metric_range.candidates[0].integer.has_value());
    CHECK(finalized_metric_range.candidates[0].metric_bit_length == 0);
    CHECK(!finalized_metric_range.candidates[0].symbolic_expression
               .has_value());
    CHECK(!finalized_metric_range.candidates[1].integer.has_value());
    CHECK(finalized_metric_range.candidates[1].metric_bit_length == 7);
    CHECK(!finalized_metric_range.candidates[1].symbolic_expression
               .has_value());
    CHECK(engine.inspect().resident_entries == 2);
  }

  const auto fresh_exact_range =
      engine.evaluate_range(exact_range_request, {0, 1});
  CHECK(fresh_exact_range.status == bruce::lhme::Status::Confirmed);
  CHECK(fresh_exact_range.summary.checked_count == 2);
  CHECK(fresh_exact_range.summary.confirmed_count == 2);
  CHECK(fresh_exact_range.summary.invalid_argument_count == 0);
  CHECK(fresh_exact_range.candidates.size() == 2);
  CHECK(fresh_exact_range.candidates[0].exact_integer() == 0);
  CHECK(fresh_exact_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 2);
}

TEST(cpp_engine_does_not_join_a_post_snapshot_cancelled_exact_range_group_for_a_late_metric_range_evaluation_with_the_same_canonical_range_after_a_fourth_cancel) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));

  const bruce::lhme::TechniqueRequest origin_modular_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest index_10_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest index_12_formula_request{
      .spec = formula,
      .address = {.index = 12},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest exact_range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest metric_range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_original_formula;
  std::future<bruce::lhme::EvaluationResult> pending_index_12_formula;
  std::future<bruce::lhme::EvaluationResult>
      pending_range_group_formula_cache_hit;
  std::future<bruce::lhme::RangeResult> pending_exact_range;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async, [&engine, origin_modular_request] {
    return engine.evaluate_technique(origin_modular_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  pending_original_formula = std::async(
      std::launch::async, [&engine, index_10_formula_request] {
        return engine.evaluate(index_10_formula_request);
      });
  formula_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_original_formula = pending_original_formula.get();
  CHECK(cancelled_original_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_original_formula.integer.has_value());
  CHECK(!cancelled_original_formula.metric_bit_length.has_value());
  CHECK(!cancelled_original_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      index_12_formula_post_finalization_barrier;
  pending_index_12_formula = std::async(
      std::launch::async, [&engine, index_12_formula_request] {
        return engine.evaluate(index_12_formula_request);
      });
  index_12_formula_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  bruce::lhme::test_support::EngineRangePostFinalizationBarrier
      exact_range_post_finalization_barrier;
  pending_exact_range =
      std::async(std::launch::async, [&engine, exact_range_request] {
        return engine.evaluate_range(exact_range_request, {0, 1});
      });
  exact_range_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 0);

  index_12_formula_post_finalization_barrier.release();
  const auto finalized_index_12_formula = pending_index_12_formula.get();
  CHECK(finalized_index_12_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_index_12_formula.exact_integer() == 409500);
  CHECK(!finalized_index_12_formula.metric_bit_length.has_value());
  CHECK(!finalized_index_12_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  bruce::lhme::test_support::EngineFormulaCacheHitPostFinalizationBarrier
      range_group_formula_cache_hit_post_finalization_barrier;
  pending_range_group_formula_cache_hit = std::async(
      std::launch::async, [&engine, index_12_formula_request] {
        return engine.evaluate(index_12_formula_request);
      });
  range_group_formula_cache_hit_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 1);

  engine.cancel();
  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();
  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  exact_range_post_finalization_barrier.release();
  const auto finalized_exact_range = pending_exact_range.get();
  CHECK(finalized_exact_range.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_exact_range.summary.checked_count == 2);
  CHECK(finalized_exact_range.summary.confirmed_count == 2);
  CHECK(finalized_exact_range.summary.invalid_argument_count == 0);
  CHECK(finalized_exact_range.candidates.size() == 2);
  CHECK(finalized_exact_range.candidates[0].exact_integer() == 0);
  CHECK(finalized_exact_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);

  const auto confirmed_metric_range =
      engine.evaluate_range(metric_range_request, {0, 1});
  CHECK(confirmed_metric_range.status == bruce::lhme::Status::Confirmed);
  CHECK(confirmed_metric_range.summary.checked_count == 2);
  CHECK(confirmed_metric_range.summary.confirmed_count == 2);
  CHECK(confirmed_metric_range.summary.invalid_argument_count == 0);
  CHECK(confirmed_metric_range.candidates.size() == 2);
  CHECK(!confirmed_metric_range.candidates[0].integer.has_value());
  CHECK(confirmed_metric_range.candidates[0].metric_bit_length == 0);
  CHECK(!confirmed_metric_range.candidates[0].symbolic_expression.has_value());
  CHECK(!confirmed_metric_range.candidates[1].integer.has_value());
  CHECK(confirmed_metric_range.candidates[1].metric_bit_length == 7);
  CHECK(!confirmed_metric_range.candidates[1].symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  range_group_formula_cache_hit_post_finalization_barrier.release();
  const auto finalized_range_group_formula_cache_hit =
      pending_range_group_formula_cache_hit.get();
  CHECK(finalized_range_group_formula_cache_hit.status ==
        bruce::lhme::Status::Confirmed);
  CHECK(finalized_range_group_formula_cache_hit.exact_integer() == 409500);
  CHECK(!finalized_range_group_formula_cache_hit.metric_bit_length.has_value());
  CHECK(!finalized_range_group_formula_cache_hit.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto fresh_exact_range = engine.evaluate_range(exact_range_request, {0, 1});
  CHECK(fresh_exact_range.status == bruce::lhme::Status::Confirmed);
  CHECK(fresh_exact_range.summary.checked_count == 2);
  CHECK(fresh_exact_range.summary.confirmed_count == 2);
  CHECK(fresh_exact_range.summary.invalid_argument_count == 0);
  CHECK(fresh_exact_range.candidates.size() == 2);
  CHECK(fresh_exact_range.candidates[0].exact_integer() == 0);
  CHECK(fresh_exact_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_does_not_join_a_post_snapshot_cancelled_metric_range_group_for_a_late_exact_range_evaluation_with_the_same_canonical_range_after_a_fourth_cancel) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));

  const bruce::lhme::TechniqueRequest origin_modular_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest index_10_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest index_12_metric_formula_request{
      .spec = formula,
      .address = {.index = 12},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest exact_range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest metric_range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_original_formula;
  std::future<bruce::lhme::EvaluationResult> pending_index_12_metric_formula;
  std::future<bruce::lhme::EvaluationResult>
      pending_range_group_metric_formula_cache_hit;
  std::future<bruce::lhme::RangeResult> pending_metric_range;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async, [&engine, origin_modular_request] {
    return engine.evaluate_technique(origin_modular_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  pending_original_formula = std::async(
      std::launch::async, [&engine, index_10_formula_request] {
        return engine.evaluate(index_10_formula_request);
      });
  formula_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_original_formula = pending_original_formula.get();
  CHECK(cancelled_original_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_original_formula.integer.has_value());
  CHECK(!cancelled_original_formula.metric_bit_length.has_value());
  CHECK(!cancelled_original_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      index_12_metric_formula_post_finalization_barrier;
  pending_index_12_metric_formula = std::async(
      std::launch::async, [&engine, index_12_metric_formula_request] {
        return engine.evaluate(index_12_metric_formula_request);
      });
  index_12_metric_formula_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  bruce::lhme::test_support::EngineRangePostFinalizationBarrier
      metric_range_post_finalization_barrier;
  pending_metric_range =
      std::async(std::launch::async, [&engine, metric_range_request] {
        return engine.evaluate_range(metric_range_request, {0, 1});
      });
  metric_range_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 0);

  index_12_metric_formula_post_finalization_barrier.release();
  const auto finalized_index_12_metric_formula =
      pending_index_12_metric_formula.get();
  CHECK(finalized_index_12_metric_formula.status ==
        bruce::lhme::Status::Confirmed);
  CHECK(!finalized_index_12_metric_formula.integer.has_value());
  CHECK(finalized_index_12_metric_formula.metric_bit_length == 19);
  CHECK(!finalized_index_12_metric_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  bruce::lhme::test_support::EngineFormulaCacheHitPostFinalizationBarrier
      range_group_metric_formula_cache_hit_post_finalization_barrier;
  pending_range_group_metric_formula_cache_hit = std::async(
      std::launch::async, [&engine, index_12_metric_formula_request] {
        return engine.evaluate(index_12_metric_formula_request);
      });
  range_group_metric_formula_cache_hit_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 1);

  engine.cancel();
  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();
  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  metric_range_post_finalization_barrier.release();
  const auto finalized_metric_range = pending_metric_range.get();
  CHECK(finalized_metric_range.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_metric_range.summary.checked_count == 2);
  CHECK(finalized_metric_range.summary.confirmed_count == 2);
  CHECK(finalized_metric_range.summary.invalid_argument_count == 0);
  CHECK(finalized_metric_range.candidates.size() == 2);
  CHECK(!finalized_metric_range.candidates[0].integer.has_value());
  CHECK(finalized_metric_range.candidates[0].metric_bit_length == 0);
  CHECK(!finalized_metric_range.candidates[0].symbolic_expression.has_value());
  CHECK(!finalized_metric_range.candidates[1].integer.has_value());
  CHECK(finalized_metric_range.candidates[1].metric_bit_length == 7);
  CHECK(!finalized_metric_range.candidates[1].symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto confirmed_exact_range =
      engine.evaluate_range(exact_range_request, {0, 1});
  CHECK(confirmed_exact_range.status == bruce::lhme::Status::Confirmed);
  CHECK(confirmed_exact_range.summary.checked_count == 2);
  CHECK(confirmed_exact_range.summary.confirmed_count == 2);
  CHECK(confirmed_exact_range.summary.invalid_argument_count == 0);
  CHECK(confirmed_exact_range.candidates.size() == 2);
  CHECK(confirmed_exact_range.candidates[0].exact_integer() == 0);
  CHECK(!confirmed_exact_range.candidates[0].metric_bit_length.has_value());
  CHECK(!confirmed_exact_range.candidates[0].symbolic_expression.has_value());
  CHECK(confirmed_exact_range.candidates[1].exact_integer() == 100);
  CHECK(!confirmed_exact_range.candidates[1].metric_bit_length.has_value());
  CHECK(!confirmed_exact_range.candidates[1].symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  range_group_metric_formula_cache_hit_post_finalization_barrier.release();
  const auto finalized_range_group_metric_formula_cache_hit =
      pending_range_group_metric_formula_cache_hit.get();
  CHECK(finalized_range_group_metric_formula_cache_hit.status ==
        bruce::lhme::Status::Confirmed);
  CHECK(!finalized_range_group_metric_formula_cache_hit.integer.has_value());
  CHECK(finalized_range_group_metric_formula_cache_hit.metric_bit_length == 19);
  CHECK(!finalized_range_group_metric_formula_cache_hit.symbolic_expression
             .has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto fresh_metric_range =
      engine.evaluate_range(metric_range_request, {0, 1});
  CHECK(fresh_metric_range.status == bruce::lhme::Status::Confirmed);
  CHECK(fresh_metric_range.summary.checked_count == 2);
  CHECK(fresh_metric_range.summary.confirmed_count == 2);
  CHECK(fresh_metric_range.summary.invalid_argument_count == 0);
  CHECK(fresh_metric_range.candidates.size() == 2);
  CHECK(!fresh_metric_range.candidates[0].integer.has_value());
  CHECK(fresh_metric_range.candidates[0].metric_bit_length == 0);
  CHECK(!fresh_metric_range.candidates[0].symbolic_expression.has_value());
  CHECK(!fresh_metric_range.candidates[1].integer.has_value());
  CHECK(fresh_metric_range.candidates[1].metric_bit_length == 7);
  CHECK(!fresh_metric_range.candidates[1].symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_keeps_a_post_snapshot_cancelled_metric_range_group_cancelled_for_a_late_identical_metric_range_joiner_after_a_fourth_cancel) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));

  const bruce::lhme::TechniqueRequest origin_modular_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest index_10_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest index_12_metric_formula_request{
      .spec = formula,
      .address = {.index = 12},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest metric_range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_original_formula;
  std::future<bruce::lhme::EvaluationResult> pending_index_12_metric_formula;
  std::future<bruce::lhme::EvaluationResult>
      pending_range_group_metric_formula_cache_hit;
  std::future<bruce::lhme::RangeResult> pending_metric_range;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async, [&engine, origin_modular_request] {
    return engine.evaluate_technique(origin_modular_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  pending_original_formula = std::async(
      std::launch::async, [&engine, index_10_formula_request] {
        return engine.evaluate(index_10_formula_request);
      });
  formula_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_original_formula = pending_original_formula.get();
  CHECK(cancelled_original_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_original_formula.integer.has_value());
  CHECK(!cancelled_original_formula.metric_bit_length.has_value());
  CHECK(!cancelled_original_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      index_12_metric_formula_post_finalization_barrier;
  pending_index_12_metric_formula = std::async(
      std::launch::async, [&engine, index_12_metric_formula_request] {
        return engine.evaluate(index_12_metric_formula_request);
      });
  index_12_metric_formula_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  bruce::lhme::test_support::EngineRangePostFinalizationBarrier
      metric_range_post_finalization_barrier;
  pending_metric_range =
      std::async(std::launch::async, [&engine, metric_range_request] {
        return engine.evaluate_range(metric_range_request, {0, 1});
      });
  metric_range_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 0);

  index_12_metric_formula_post_finalization_barrier.release();
  const auto finalized_index_12_metric_formula =
      pending_index_12_metric_formula.get();
  CHECK(finalized_index_12_metric_formula.status ==
        bruce::lhme::Status::Confirmed);
  CHECK(!finalized_index_12_metric_formula.integer.has_value());
  CHECK(finalized_index_12_metric_formula.metric_bit_length == 19);
  CHECK(!finalized_index_12_metric_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  bruce::lhme::test_support::EngineFormulaCacheHitPostFinalizationBarrier
      range_group_metric_formula_cache_hit_post_finalization_barrier;
  pending_range_group_metric_formula_cache_hit = std::async(
      std::launch::async, [&engine, index_12_metric_formula_request] {
        return engine.evaluate(index_12_metric_formula_request);
      });
  range_group_metric_formula_cache_hit_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 1);

  engine.cancel();
  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();
  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  metric_range_post_finalization_barrier.release();
  const auto finalized_metric_range = pending_metric_range.get();
  CHECK(finalized_metric_range.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_metric_range.summary.checked_count == 2);
  CHECK(finalized_metric_range.summary.confirmed_count == 2);
  CHECK(finalized_metric_range.summary.invalid_argument_count == 0);
  CHECK(finalized_metric_range.candidates.size() == 2);
  CHECK(!finalized_metric_range.candidates[0].integer.has_value());
  CHECK(finalized_metric_range.candidates[0].metric_bit_length == 0);
  CHECK(!finalized_metric_range.candidates[0].symbolic_expression.has_value());
  CHECK(!finalized_metric_range.candidates[1].integer.has_value());
  CHECK(finalized_metric_range.candidates[1].metric_bit_length == 7);
  CHECK(!finalized_metric_range.candidates[1].symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto cancelled_late_metric_range =
      engine.evaluate_range(metric_range_request, {0, 1});
  CHECK(cancelled_late_metric_range.status == bruce::lhme::Status::Cancelled);
  CHECK(cancelled_late_metric_range.candidates.empty());
  CHECK(engine.inspect().resident_entries == 1);

  range_group_metric_formula_cache_hit_post_finalization_barrier.release();
  const auto finalized_range_group_metric_formula_cache_hit =
      pending_range_group_metric_formula_cache_hit.get();
  CHECK(finalized_range_group_metric_formula_cache_hit.status ==
        bruce::lhme::Status::Confirmed);
  CHECK(!finalized_range_group_metric_formula_cache_hit.integer.has_value());
  CHECK(finalized_range_group_metric_formula_cache_hit.metric_bit_length == 19);
  CHECK(!finalized_range_group_metric_formula_cache_hit.symbolic_expression
             .has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto fresh_metric_range =
      engine.evaluate_range(metric_range_request, {0, 1});
  CHECK(fresh_metric_range.status == bruce::lhme::Status::Confirmed);
  CHECK(fresh_metric_range.summary.checked_count == 2);
  CHECK(fresh_metric_range.summary.confirmed_count == 2);
  CHECK(fresh_metric_range.summary.invalid_argument_count == 0);
  CHECK(fresh_metric_range.candidates.size() == 2);
  CHECK(!fresh_metric_range.candidates[0].integer.has_value());
  CHECK(fresh_metric_range.candidates[0].metric_bit_length == 0);
  CHECK(!fresh_metric_range.candidates[0].symbolic_expression.has_value());
  CHECK(!fresh_metric_range.candidates[1].integer.has_value());
  CHECK(fresh_metric_range.candidates[1].metric_bit_length == 7);
  CHECK(!fresh_metric_range.candidates[1].symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_does_not_join_a_post_snapshot_cancelled_metric_range_group_for_a_late_overlapping_metric_range_evaluation_after_a_fourth_cancel) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));

  const bruce::lhme::TechniqueRequest origin_modular_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest index_10_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest index_12_metric_formula_request{
      .spec = formula,
      .address = {.index = 12},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest metric_range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_original_formula;
  std::future<bruce::lhme::EvaluationResult> pending_index_12_metric_formula;
  std::future<bruce::lhme::EvaluationResult>
      pending_range_group_metric_formula_cache_hit;
  std::future<bruce::lhme::RangeResult> pending_metric_range;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async, [&engine, origin_modular_request] {
    return engine.evaluate_technique(origin_modular_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  pending_original_formula = std::async(
      std::launch::async, [&engine, index_10_formula_request] {
        return engine.evaluate(index_10_formula_request);
      });
  formula_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_original_formula = pending_original_formula.get();
  CHECK(cancelled_original_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_original_formula.integer.has_value());
  CHECK(!cancelled_original_formula.metric_bit_length.has_value());
  CHECK(!cancelled_original_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      index_12_metric_formula_post_finalization_barrier;
  pending_index_12_metric_formula = std::async(
      std::launch::async, [&engine, index_12_metric_formula_request] {
        return engine.evaluate(index_12_metric_formula_request);
      });
  index_12_metric_formula_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  bruce::lhme::test_support::EngineRangePostFinalizationBarrier
      metric_range_post_finalization_barrier;
  pending_metric_range =
      std::async(std::launch::async, [&engine, metric_range_request] {
        return engine.evaluate_range(metric_range_request, {0, 1});
      });
  metric_range_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 0);

  index_12_metric_formula_post_finalization_barrier.release();
  const auto finalized_index_12_metric_formula =
      pending_index_12_metric_formula.get();
  CHECK(finalized_index_12_metric_formula.status ==
        bruce::lhme::Status::Confirmed);
  CHECK(!finalized_index_12_metric_formula.integer.has_value());
  CHECK(finalized_index_12_metric_formula.metric_bit_length == 19);
  CHECK(!finalized_index_12_metric_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  bruce::lhme::test_support::EngineFormulaCacheHitPostFinalizationBarrier
      range_group_metric_formula_cache_hit_post_finalization_barrier;
  pending_range_group_metric_formula_cache_hit = std::async(
      std::launch::async, [&engine, index_12_metric_formula_request] {
        return engine.evaluate(index_12_metric_formula_request);
      });
  range_group_metric_formula_cache_hit_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 1);

  engine.cancel();
  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();
  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  metric_range_post_finalization_barrier.release();
  const auto finalized_metric_range = pending_metric_range.get();
  CHECK(finalized_metric_range.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_metric_range.summary.checked_count == 2);
  CHECK(finalized_metric_range.summary.confirmed_count == 2);
  CHECK(finalized_metric_range.summary.invalid_argument_count == 0);
  CHECK(finalized_metric_range.candidates.size() == 2);
  CHECK(!finalized_metric_range.candidates[0].integer.has_value());
  CHECK(finalized_metric_range.candidates[0].metric_bit_length == 0);
  CHECK(!finalized_metric_range.candidates[0].symbolic_expression.has_value());
  CHECK(!finalized_metric_range.candidates[1].integer.has_value());
  CHECK(finalized_metric_range.candidates[1].metric_bit_length == 7);
  CHECK(!finalized_metric_range.candidates[1].symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto confirmed_overlapping_metric_range =
      engine.evaluate_range(metric_range_request, {1, 2});
  CHECK(confirmed_overlapping_metric_range.status ==
        bruce::lhme::Status::Confirmed);
  CHECK(confirmed_overlapping_metric_range.summary.checked_count == 2);
  CHECK(confirmed_overlapping_metric_range.summary.confirmed_count == 2);
  CHECK(confirmed_overlapping_metric_range.summary.invalid_argument_count == 0);
  CHECK(confirmed_overlapping_metric_range.candidates.size() == 2);
  CHECK(!confirmed_overlapping_metric_range.candidates[0].integer.has_value());
  CHECK(confirmed_overlapping_metric_range.candidates[0].metric_bit_length == 7);
  CHECK(!confirmed_overlapping_metric_range.candidates[0]
             .symbolic_expression.has_value());
  CHECK(!confirmed_overlapping_metric_range.candidates[1].integer.has_value());
  CHECK(confirmed_overlapping_metric_range.candidates[1].metric_bit_length == 9);
  CHECK(!confirmed_overlapping_metric_range.candidates[1]
             .symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  range_group_metric_formula_cache_hit_post_finalization_barrier.release();
  const auto finalized_range_group_metric_formula_cache_hit =
      pending_range_group_metric_formula_cache_hit.get();
  CHECK(finalized_range_group_metric_formula_cache_hit.status ==
        bruce::lhme::Status::Confirmed);
  CHECK(!finalized_range_group_metric_formula_cache_hit.integer.has_value());
  CHECK(finalized_range_group_metric_formula_cache_hit.metric_bit_length == 19);
  CHECK(!finalized_range_group_metric_formula_cache_hit.symbolic_expression
             .has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto fresh_metric_range =
      engine.evaluate_range(metric_range_request, {0, 1});
  CHECK(fresh_metric_range.status == bruce::lhme::Status::Confirmed);
  CHECK(fresh_metric_range.summary.checked_count == 2);
  CHECK(fresh_metric_range.summary.confirmed_count == 2);
  CHECK(fresh_metric_range.summary.invalid_argument_count == 0);
  CHECK(fresh_metric_range.candidates.size() == 2);
  CHECK(!fresh_metric_range.candidates[0].integer.has_value());
  CHECK(fresh_metric_range.candidates[0].metric_bit_length == 0);
  CHECK(!fresh_metric_range.candidates[0].symbolic_expression.has_value());
  CHECK(!fresh_metric_range.candidates[1].integer.has_value());
  CHECK(fresh_metric_range.candidates[1].metric_bit_length == 7);
  CHECK(!fresh_metric_range.candidates[1].symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_keeps_a_post_snapshot_cancelled_metric_range_group_cancelled_for_a_concurrent_late_identical_metric_range_joiner_while_an_overlapping_metric_range_forms_a_confirmed_group_after_a_fourth_cancel) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));

  const bruce::lhme::TechniqueRequest origin_modular_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest index_10_formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest index_12_metric_formula_request{
      .spec = formula,
      .address = {.index = 12},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest metric_range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_original_formula;
  std::future<bruce::lhme::EvaluationResult> pending_index_12_metric_formula;
  std::future<bruce::lhme::EvaluationResult>
      pending_range_group_metric_formula_cache_hit;
  std::future<bruce::lhme::RangeResult> pending_metric_range;
  std::future<bruce::lhme::RangeResult> pending_late_identical_metric_range;
  std::future<bruce::lhme::RangeResult> pending_late_overlapping_metric_range;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async, [&engine, origin_modular_request] {
    return engine.evaluate_technique(origin_modular_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  pending_original_formula = std::async(
      std::launch::async, [&engine, index_10_formula_request] {
        return engine.evaluate(index_10_formula_request);
      });
  formula_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_original_formula = pending_original_formula.get();
  CHECK(cancelled_original_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_original_formula.integer.has_value());
  CHECK(!cancelled_original_formula.metric_bit_length.has_value());
  CHECK(!cancelled_original_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      index_12_metric_formula_post_finalization_barrier;
  pending_index_12_metric_formula = std::async(
      std::launch::async, [&engine, index_12_metric_formula_request] {
        return engine.evaluate(index_12_metric_formula_request);
      });
  index_12_metric_formula_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  bruce::lhme::test_support::EngineRangePostFinalizationBarrier
      metric_range_post_finalization_barrier;
  pending_metric_range =
      std::async(std::launch::async, [&engine, metric_range_request] {
        return engine.evaluate_range(metric_range_request, {0, 1});
      });
  metric_range_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 0);

  index_12_metric_formula_post_finalization_barrier.release();
  const auto finalized_index_12_metric_formula =
      pending_index_12_metric_formula.get();
  CHECK(finalized_index_12_metric_formula.status ==
        bruce::lhme::Status::Confirmed);
  CHECK(!finalized_index_12_metric_formula.integer.has_value());
  CHECK(finalized_index_12_metric_formula.metric_bit_length == 19);
  CHECK(!finalized_index_12_metric_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  bruce::lhme::test_support::EngineFormulaCacheHitPostFinalizationBarrier
      range_group_metric_formula_cache_hit_post_finalization_barrier;
  pending_range_group_metric_formula_cache_hit = std::async(
      std::launch::async, [&engine, index_12_metric_formula_request] {
        return engine.evaluate(index_12_metric_formula_request);
      });
  range_group_metric_formula_cache_hit_post_finalization_barrier.wait_until_reached();
  CHECK(engine.inspect().resident_entries == 1);

  engine.cancel();
  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();
  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  metric_range_post_finalization_barrier.release();
  const auto finalized_metric_range = pending_metric_range.get();
  CHECK(finalized_metric_range.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_metric_range.summary.checked_count == 2);
  CHECK(finalized_metric_range.summary.confirmed_count == 2);
  CHECK(finalized_metric_range.summary.invalid_argument_count == 0);
  CHECK(finalized_metric_range.candidates.size() == 2);
  CHECK(!finalized_metric_range.candidates[0].integer.has_value());
  CHECK(finalized_metric_range.candidates[0].metric_bit_length == 0);
  CHECK(!finalized_metric_range.candidates[0].symbolic_expression.has_value());
  CHECK(!finalized_metric_range.candidates[1].integer.has_value());
  CHECK(finalized_metric_range.candidates[1].metric_bit_length == 7);
  CHECK(!finalized_metric_range.candidates[1].symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  pending_late_identical_metric_range =
      std::async(std::launch::async, [&engine, metric_range_request] {
        return engine.evaluate_range(metric_range_request, {0, 1});
      });
  pending_late_overlapping_metric_range =
      std::async(std::launch::async, [&engine, metric_range_request] {
        return engine.evaluate_range(metric_range_request, {1, 2});
      });

  const auto cancelled_late_identical_metric_range =
      pending_late_identical_metric_range.get();
  CHECK(cancelled_late_identical_metric_range.status ==
        bruce::lhme::Status::Cancelled);
  CHECK(cancelled_late_identical_metric_range.candidates.empty());
  CHECK(engine.inspect().resident_entries == 1);

  const auto confirmed_late_overlapping_metric_range =
      pending_late_overlapping_metric_range.get();
  CHECK(confirmed_late_overlapping_metric_range.status ==
        bruce::lhme::Status::Confirmed);
  CHECK(confirmed_late_overlapping_metric_range.summary.checked_count == 2);
  CHECK(confirmed_late_overlapping_metric_range.summary.confirmed_count == 2);
  CHECK(confirmed_late_overlapping_metric_range.summary.invalid_argument_count ==
        0);
  CHECK(confirmed_late_overlapping_metric_range.candidates.size() == 2);
  CHECK(!confirmed_late_overlapping_metric_range.candidates[0].integer
             .has_value());
  CHECK(confirmed_late_overlapping_metric_range.candidates[0]
            .metric_bit_length == 7);
  CHECK(!confirmed_late_overlapping_metric_range.candidates[0]
             .symbolic_expression.has_value());
  CHECK(!confirmed_late_overlapping_metric_range.candidates[1].integer
             .has_value());
  CHECK(confirmed_late_overlapping_metric_range.candidates[1]
            .metric_bit_length == 9);
  CHECK(!confirmed_late_overlapping_metric_range.candidates[1]
             .symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  range_group_metric_formula_cache_hit_post_finalization_barrier.release();
  const auto finalized_range_group_metric_formula_cache_hit =
      pending_range_group_metric_formula_cache_hit.get();
  CHECK(finalized_range_group_metric_formula_cache_hit.status ==
        bruce::lhme::Status::Confirmed);
  CHECK(!finalized_range_group_metric_formula_cache_hit.integer.has_value());
  CHECK(finalized_range_group_metric_formula_cache_hit.metric_bit_length == 19);
  CHECK(!finalized_range_group_metric_formula_cache_hit.symbolic_expression
             .has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto fresh_metric_range =
      engine.evaluate_range(metric_range_request, {0, 1});
  CHECK(fresh_metric_range.status == bruce::lhme::Status::Confirmed);
  CHECK(fresh_metric_range.summary.checked_count == 2);
  CHECK(fresh_metric_range.summary.confirmed_count == 2);
  CHECK(fresh_metric_range.summary.invalid_argument_count == 0);
  CHECK(fresh_metric_range.candidates.size() == 2);
  CHECK(!fresh_metric_range.candidates[0].integer.has_value());
  CHECK(fresh_metric_range.candidates[0].metric_bit_length == 0);
  CHECK(!fresh_metric_range.candidates[0].symbolic_expression.has_value());
  CHECK(!fresh_metric_range.candidates[1].integer.has_value());
  CHECK(fresh_metric_range.candidates[1].metric_bit_length == 7);
  CHECK(!fresh_metric_range.candidates[1].symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(cpp_engine_does_not_join_a_post_snapshot_cancelled_power_offset_modular_technique_group_for_a_late_fibonacci_modular_technique_without_an_explicit_computation_identity) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto power_offset_kernel =
      std::make_shared<bruce::lhme::PowerOffsetKernel>();
  const auto fibonacci_kernel = std::make_shared<bruce::lhme::FibonacciKernel>();
  CHECK(engine.register_technique(power_offset_kernel));
  CHECK(engine.register_technique(fibonacci_kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::TechniqueRequest modular_technique_request{
      .identity = power_offset_kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::TechniqueRequest fibonacci_modular_technique_request{
      .identity = fibonacci_kernel->identity(),
      .parameters = bruce::lhme::kFibonacciFastDoublingParameters,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest exact_technique_request{
      .identity = power_offset_kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest metric_technique_request{
      .identity = power_offset_kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest symbolic_technique_request{
      .identity = power_offset_kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_formula;
  std::future<bruce::lhme::RangeResult> pending_range;
  std::future<bruce::lhme::EvaluationResult> pending_metric_technique;
  std::future<bruce::lhme::EvaluationResult> pending_symbolic_technique;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async,
                              [&engine, modular_technique_request] {
    return engine.evaluate_technique(modular_technique_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  bruce::lhme::test_support::EngineRangeFinalizationBarrier
      range_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueFinalizationBarrier
      technique_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueNoncacheableFinalizationBarrier
      noncacheable_technique_finalization_barrier;

  pending_formula = std::async(std::launch::async, [&engine, formula_request] {
    return engine.evaluate(formula_request);
  });
  formula_finalization_barrier.wait_until_reached();

  pending_range = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  range_finalization_barrier.wait_until_reached();

  pending_metric_technique = std::async(
      std::launch::async, [&engine, metric_technique_request] {
        return engine.evaluate_technique(metric_technique_request);
      });
  technique_finalization_barrier.wait_until_reached();

  pending_symbolic_technique = std::async(
      std::launch::async, [&engine, symbolic_technique_request] {
        return engine.evaluate_technique(symbolic_technique_request);
      });
  noncacheable_technique_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_formula = pending_formula.get();

  CHECK(cancelled_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_formula.integer.has_value());
  CHECK(!cancelled_formula.metric_bit_length.has_value());
  CHECK(!cancelled_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  range_finalization_barrier.release();
  const auto cancelled_range = pending_range.get();

  CHECK(cancelled_range.status == bruce::lhme::Status::Cancelled);
  CHECK(cancelled_range.candidates.empty());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  technique_finalization_barrier.release();
  noncacheable_technique_finalization_barrier.release();
  const auto cancelled_metric_technique = pending_metric_technique.get();
  const auto cancelled_symbolic_technique = pending_symbolic_technique.get();

  CHECK(cancelled_metric_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_metric_technique.integer.has_value());
  CHECK(!cancelled_metric_technique.metric_bit_length.has_value());
  CHECK(!cancelled_metric_technique.symbolic_expression.has_value());
  CHECK(cancelled_symbolic_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_symbolic_technique.integer.has_value());
  CHECK(!cancelled_symbolic_technique.metric_bit_length.has_value());
  CHECK(!cancelled_symbolic_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  auto late_power_offset_technique = std::async(
      std::launch::async, [&engine, modular_technique_request] {
        return engine.evaluate_technique(modular_technique_request);
      });
  auto late_fibonacci_technique = std::async(
      std::launch::async, [&engine, fibonacci_modular_technique_request] {
        return engine.evaluate_technique(fibonacci_modular_technique_request);
      });
  const auto cancelled_late_power_offset_technique =
      late_power_offset_technique.get();
  const auto confirmed_late_fibonacci_technique =
      late_fibonacci_technique.get();

  CHECK(cancelled_late_power_offset_technique.status ==
        bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_late_power_offset_technique.integer.has_value());
  CHECK(!cancelled_late_power_offset_technique.metric_bit_length.has_value());
  CHECK(!cancelled_late_power_offset_technique.symbolic_expression.has_value());
  CHECK(confirmed_late_fibonacci_technique.status ==
        bruce::lhme::Status::Confirmed);
  CHECK(confirmed_late_fibonacci_technique.exact_integer() == 4);
  CHECK(!confirmed_late_fibonacci_technique.metric_bit_length.has_value());
  CHECK(!confirmed_late_fibonacci_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();

  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_formula = engine.evaluate(formula_request);
  CHECK(reused_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_formula.exact_integer() == 102300);
  CHECK(!reused_formula.metric_bit_length.has_value());
  CHECK(!reused_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.summary.invalid_argument_count == 0);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_exact_technique = engine.evaluate_technique(
      exact_technique_request);
  CHECK(reused_exact_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_exact_technique.exact_integer() == 102300);
  CHECK(!reused_exact_technique.metric_bit_length.has_value());
  CHECK(!reused_exact_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 2);

  const auto reused_metric_technique = engine.evaluate_technique(
      metric_technique_request);
  CHECK(reused_metric_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_metric_technique.metric_bit_length == 17);
  CHECK(!reused_metric_technique.integer.has_value());
  CHECK(!reused_metric_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 3);

  const auto reused_symbolic_technique = engine.evaluate_technique(
      symbolic_technique_request);
  CHECK(reused_symbolic_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_symbolic_technique.symbolic_expression == "100*(2^10-1)");
  CHECK(!reused_symbolic_technique.integer.has_value());
  CHECK(!reused_symbolic_technique.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 3);

  const auto reused_fibonacci_technique = engine.evaluate_technique(
      fibonacci_modular_technique_request);
  CHECK(reused_fibonacci_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_fibonacci_technique.exact_integer() == 4);
  CHECK(!reused_fibonacci_technique.metric_bit_length.has_value());
  CHECK(!reused_fibonacci_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 3);
}

TEST(cpp_engine_keeps_a_post_snapshot_cancelled_fibonacci_modular_technique_group_cancelled_for_a_late_same_fibonacci_modular_technique_joiner_after_all_inflight_joiner_lease_endings_and_repeated_cancels) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::FibonacciKernel>();
  CHECK(engine.register_technique(kernel));
  const bruce::lhme::TechniqueRequest modular_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::kFibonacciFastDoublingParameters,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::TechniqueRequest exact_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::kFibonacciFastDoublingParameters,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest metric_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::kFibonacciFastDoublingParameters,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest symbolic_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::kFibonacciFastDoublingParameters,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_exact_technique;
  std::future<bruce::lhme::EvaluationResult> pending_symbolic_technique;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async,
                              [&engine, modular_technique_request] {
    return engine.evaluate_technique(modular_technique_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineTechniqueFinalizationBarrier
      technique_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueNoncacheableFinalizationBarrier
      noncacheable_technique_finalization_barrier;
  pending_exact_technique = std::async(
      std::launch::async, [&engine, exact_technique_request] {
        return engine.evaluate_technique(exact_technique_request);
      });
  technique_finalization_barrier.wait_until_reached();

  pending_symbolic_technique = std::async(
      std::launch::async, [&engine, symbolic_technique_request] {
        return engine.evaluate_technique(symbolic_technique_request);
      });
  noncacheable_technique_finalization_barrier.wait_until_reached();

  engine.cancel();
  technique_finalization_barrier.release();
  const auto cancelled_exact_technique = pending_exact_technique.get();

  CHECK(cancelled_exact_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_exact_technique.integer.has_value());
  CHECK(!cancelled_exact_technique.metric_bit_length.has_value());
  CHECK(!cancelled_exact_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  noncacheable_technique_finalization_barrier.release();
  const auto cancelled_symbolic_technique = pending_symbolic_technique.get();

  CHECK(cancelled_symbolic_technique.status ==
        bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_symbolic_technique.integer.has_value());
  CHECK(!cancelled_symbolic_technique.metric_bit_length.has_value());
  CHECK(!cancelled_symbolic_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  auto late_modular_technique = std::async(
      std::launch::async, [&engine, modular_technique_request] {
        return engine.evaluate_technique(modular_technique_request);
      });
  const auto cancelled_late_modular_technique = late_modular_technique.get();

  CHECK(cancelled_late_modular_technique.status ==
        bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_late_modular_technique.integer.has_value());
  CHECK(!cancelled_late_modular_technique.metric_bit_length.has_value());
  CHECK(!cancelled_late_modular_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();

  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 4);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_exact_technique = engine.evaluate_technique(
      exact_technique_request);
  CHECK(reused_exact_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_exact_technique.exact_integer() == 55);
  CHECK(!reused_exact_technique.metric_bit_length.has_value());
  CHECK(!reused_exact_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_metric_technique = engine.evaluate_technique(
      metric_technique_request);
  CHECK(reused_metric_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_metric_technique.metric_bit_length == 7);
  CHECK(!reused_metric_technique.integer.has_value());
  CHECK(!reused_metric_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 2);

  const auto reused_symbolic_technique = engine.evaluate_technique(
      symbolic_technique_request);
  CHECK(reused_symbolic_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_symbolic_technique.symbolic_expression == "F(10)");
  CHECK(!reused_symbolic_technique.integer.has_value());
  CHECK(!reused_symbolic_technique.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 2);

  const auto reused_modular_technique = engine.evaluate_technique(
      modular_technique_request);
  CHECK(reused_modular_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_modular_technique.exact_integer() == 4);
  CHECK(!reused_modular_technique.metric_bit_length.has_value());
  CHECK(!reused_modular_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 2);
}

TEST(cpp_engine_keeps_a_post_snapshot_cancelled_modular_noncacheable_technique_group_cancelled_for_a_second_concurrent_late_cross_api_joiner_wave_after_all_prior_late_joiner_lease_endings_and_repeated_cancels) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::TechniqueRequest modular_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest exact_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest metric_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest symbolic_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_formula;
  std::future<bruce::lhme::RangeResult> pending_range;
  std::future<bruce::lhme::EvaluationResult> pending_metric_technique;
  std::future<bruce::lhme::EvaluationResult> pending_symbolic_technique;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async,
                              [&engine, modular_technique_request] {
    return engine.evaluate_technique(modular_technique_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  bruce::lhme::test_support::EngineRangeFinalizationBarrier
      range_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueFinalizationBarrier
      technique_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueNoncacheableFinalizationBarrier
      noncacheable_technique_finalization_barrier;

  pending_formula = std::async(std::launch::async, [&engine, formula_request] {
    return engine.evaluate(formula_request);
  });
  formula_finalization_barrier.wait_until_reached();

  pending_range = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  range_finalization_barrier.wait_until_reached();

  pending_metric_technique = std::async(
      std::launch::async, [&engine, metric_technique_request] {
        return engine.evaluate_technique(metric_technique_request);
      });
  technique_finalization_barrier.wait_until_reached();

  pending_symbolic_technique = std::async(
      std::launch::async, [&engine, symbolic_technique_request] {
        return engine.evaluate_technique(symbolic_technique_request);
      });
  noncacheable_technique_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_formula = pending_formula.get();

  CHECK(cancelled_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_formula.integer.has_value());
  CHECK(!cancelled_formula.metric_bit_length.has_value());
  CHECK(!cancelled_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  range_finalization_barrier.release();
  const auto cancelled_range = pending_range.get();

  CHECK(cancelled_range.status == bruce::lhme::Status::Cancelled);
  CHECK(cancelled_range.candidates.empty());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  technique_finalization_barrier.release();
  noncacheable_technique_finalization_barrier.release();
  const auto cancelled_metric_technique = pending_metric_technique.get();
  const auto cancelled_symbolic_technique = pending_symbolic_technique.get();

  CHECK(cancelled_metric_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_metric_technique.integer.has_value());
  CHECK(!cancelled_metric_technique.metric_bit_length.has_value());
  CHECK(!cancelled_metric_technique.symbolic_expression.has_value());
  CHECK(cancelled_symbolic_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_symbolic_technique.integer.has_value());
  CHECK(!cancelled_symbolic_technique.metric_bit_length.has_value());
  CHECK(!cancelled_symbolic_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  auto pending_first_late_formula = std::async(
      std::launch::async, [&engine, formula_request] {
        return engine.evaluate(formula_request);
      });
  auto pending_first_late_range = std::async(
      std::launch::async, [&engine, range_request] {
        return engine.evaluate_range(range_request, {0, 1});
      });
  auto pending_first_late_exact_technique = std::async(
      std::launch::async, [&engine, exact_technique_request] {
        return engine.evaluate_technique(exact_technique_request);
      });
  auto pending_first_late_metric_technique = std::async(
      std::launch::async, [&engine, metric_technique_request] {
        return engine.evaluate_technique(metric_technique_request);
      });
  auto pending_first_late_symbolic_technique = std::async(
      std::launch::async, [&engine, symbolic_technique_request] {
        return engine.evaluate_technique(symbolic_technique_request);
      });
  const auto cancelled_first_late_formula = pending_first_late_formula.get();
  const auto cancelled_first_late_range = pending_first_late_range.get();
  const auto cancelled_first_late_exact_technique =
      pending_first_late_exact_technique.get();
  const auto cancelled_first_late_metric_technique =
      pending_first_late_metric_technique.get();
  const auto cancelled_first_late_symbolic_technique =
      pending_first_late_symbolic_technique.get();

  CHECK(cancelled_first_late_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_first_late_formula.integer.has_value());
  CHECK(!cancelled_first_late_formula.metric_bit_length.has_value());
  CHECK(!cancelled_first_late_formula.symbolic_expression.has_value());
  CHECK(cancelled_first_late_range.status == bruce::lhme::Status::Cancelled);
  CHECK(cancelled_first_late_range.candidates.empty());
  CHECK(cancelled_first_late_exact_technique.status ==
        bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_first_late_exact_technique.integer.has_value());
  CHECK(!cancelled_first_late_exact_technique.metric_bit_length.has_value());
  CHECK(!cancelled_first_late_exact_technique.symbolic_expression.has_value());
  CHECK(cancelled_first_late_metric_technique.status ==
        bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_first_late_metric_technique.integer.has_value());
  CHECK(!cancelled_first_late_metric_technique.metric_bit_length.has_value());
  CHECK(!cancelled_first_late_metric_technique.symbolic_expression.has_value());
  CHECK(cancelled_first_late_symbolic_technique.status ==
        bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_first_late_symbolic_technique.integer.has_value());
  CHECK(!cancelled_first_late_symbolic_technique.metric_bit_length.has_value());
  CHECK(!cancelled_first_late_symbolic_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  auto pending_second_late_formula = std::async(
      std::launch::async, [&engine, formula_request] {
        return engine.evaluate(formula_request);
      });
  auto pending_second_late_range = std::async(
      std::launch::async, [&engine, range_request] {
        return engine.evaluate_range(range_request, {0, 1});
      });
  auto pending_second_late_exact_technique = std::async(
      std::launch::async, [&engine, exact_technique_request] {
        return engine.evaluate_technique(exact_technique_request);
      });
  auto pending_second_late_metric_technique = std::async(
      std::launch::async, [&engine, metric_technique_request] {
        return engine.evaluate_technique(metric_technique_request);
      });
  auto pending_second_late_symbolic_technique = std::async(
      std::launch::async, [&engine, symbolic_technique_request] {
        return engine.evaluate_technique(symbolic_technique_request);
      });
  const auto cancelled_second_late_formula = pending_second_late_formula.get();
  const auto cancelled_second_late_range = pending_second_late_range.get();
  const auto cancelled_second_late_exact_technique =
      pending_second_late_exact_technique.get();
  const auto cancelled_second_late_metric_technique =
      pending_second_late_metric_technique.get();
  const auto cancelled_second_late_symbolic_technique =
      pending_second_late_symbolic_technique.get();

  CHECK(cancelled_second_late_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_second_late_formula.integer.has_value());
  CHECK(!cancelled_second_late_formula.metric_bit_length.has_value());
  CHECK(!cancelled_second_late_formula.symbolic_expression.has_value());
  CHECK(cancelled_second_late_range.status == bruce::lhme::Status::Cancelled);
  CHECK(cancelled_second_late_range.candidates.empty());
  CHECK(cancelled_second_late_exact_technique.status ==
        bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_second_late_exact_technique.integer.has_value());
  CHECK(!cancelled_second_late_exact_technique.metric_bit_length.has_value());
  CHECK(!cancelled_second_late_exact_technique.symbolic_expression.has_value());
  CHECK(cancelled_second_late_metric_technique.status ==
        bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_second_late_metric_technique.integer.has_value());
  CHECK(!cancelled_second_late_metric_technique.metric_bit_length.has_value());
  CHECK(!cancelled_second_late_metric_technique.symbolic_expression.has_value());
  CHECK(cancelled_second_late_symbolic_technique.status ==
        bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_second_late_symbolic_technique.integer.has_value());
  CHECK(!cancelled_second_late_symbolic_technique.metric_bit_length.has_value());
  CHECK(!cancelled_second_late_symbolic_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();

  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_formula = engine.evaluate(formula_request);
  CHECK(reused_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_formula.exact_integer() == 102300);
  CHECK(!reused_formula.metric_bit_length.has_value());
  CHECK(!reused_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.summary.invalid_argument_count == 0);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_exact_technique = engine.evaluate_technique(
      exact_technique_request);
  CHECK(reused_exact_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_exact_technique.exact_integer() == 102300);
  CHECK(!reused_exact_technique.metric_bit_length.has_value());
  CHECK(!reused_exact_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 2);

  const auto reused_metric_technique = engine.evaluate_technique(
      metric_technique_request);
  CHECK(reused_metric_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_metric_technique.metric_bit_length == 17);
  CHECK(!reused_metric_technique.integer.has_value());
  CHECK(!reused_metric_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 3);

  const auto reused_symbolic_technique = engine.evaluate_technique(
      symbolic_technique_request);
  CHECK(reused_symbolic_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_symbolic_technique.symbolic_expression == "100*(2^10-1)");
  CHECK(!reused_symbolic_technique.integer.has_value());
  CHECK(!reused_symbolic_technique.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 3);
}

TEST(cpp_engine_keeps_a_post_snapshot_cancelled_modular_noncacheable_technique_group_cancelled_for_a_third_concurrent_late_cross_api_joiner_wave_after_all_prior_late_joiner_lease_endings_and_repeated_cancels) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::TechniqueRequest modular_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest exact_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest metric_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest symbolic_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_formula;
  std::future<bruce::lhme::RangeResult> pending_range;
  std::future<bruce::lhme::EvaluationResult> pending_metric_technique;
  std::future<bruce::lhme::EvaluationResult> pending_symbolic_technique;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async,
                              [&engine, modular_technique_request] {
    return engine.evaluate_technique(modular_technique_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  bruce::lhme::test_support::EngineRangeFinalizationBarrier
      range_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueFinalizationBarrier
      technique_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueNoncacheableFinalizationBarrier
      noncacheable_technique_finalization_barrier;

  pending_formula = std::async(std::launch::async, [&engine, formula_request] {
    return engine.evaluate(formula_request);
  });
  formula_finalization_barrier.wait_until_reached();

  pending_range = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  range_finalization_barrier.wait_until_reached();

  pending_metric_technique = std::async(
      std::launch::async, [&engine, metric_technique_request] {
        return engine.evaluate_technique(metric_technique_request);
      });
  technique_finalization_barrier.wait_until_reached();

  pending_symbolic_technique = std::async(
      std::launch::async, [&engine, symbolic_technique_request] {
        return engine.evaluate_technique(symbolic_technique_request);
      });
  noncacheable_technique_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_formula = pending_formula.get();

  CHECK(cancelled_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_formula.integer.has_value());
  CHECK(!cancelled_formula.metric_bit_length.has_value());
  CHECK(!cancelled_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  range_finalization_barrier.release();
  const auto cancelled_range = pending_range.get();

  CHECK(cancelled_range.status == bruce::lhme::Status::Cancelled);
  CHECK(cancelled_range.candidates.empty());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  technique_finalization_barrier.release();
  noncacheable_technique_finalization_barrier.release();
  const auto cancelled_metric_technique = pending_metric_technique.get();
  const auto cancelled_symbolic_technique = pending_symbolic_technique.get();

  CHECK(cancelled_metric_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_metric_technique.integer.has_value());
  CHECK(!cancelled_metric_technique.metric_bit_length.has_value());
  CHECK(!cancelled_metric_technique.symbolic_expression.has_value());
  CHECK(cancelled_symbolic_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_symbolic_technique.integer.has_value());
  CHECK(!cancelled_symbolic_technique.metric_bit_length.has_value());
  CHECK(!cancelled_symbolic_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  struct LateWave {
    std::future<bruce::lhme::EvaluationResult> formula;
    std::future<bruce::lhme::RangeResult> range;
    std::future<bruce::lhme::EvaluationResult> exact_technique;
    std::future<bruce::lhme::EvaluationResult> metric_technique;
    std::future<bruce::lhme::EvaluationResult> symbolic_technique;
  };
  const auto start_late_wave =
      [&engine, formula_request, range_request, exact_technique_request,
       metric_technique_request, symbolic_technique_request] {
        return LateWave{
            .formula = std::async(std::launch::async,
                                  [&engine, formula_request] {
                                    return engine.evaluate(formula_request);
                                  }),
            .range = std::async(std::launch::async, [&engine, range_request] {
              return engine.evaluate_range(range_request, {0, 1});
            }),
            .exact_technique = std::async(
                std::launch::async, [&engine, exact_technique_request] {
                  return engine.evaluate_technique(exact_technique_request);
                }),
            .metric_technique = std::async(
                std::launch::async, [&engine, metric_technique_request] {
                  return engine.evaluate_technique(metric_technique_request);
                }),
            .symbolic_technique = std::async(
                std::launch::async, [&engine, symbolic_technique_request] {
                  return engine.evaluate_technique(symbolic_technique_request);
                })};
      };
  const auto check_cancelled_late_wave = [&engine](LateWave& late_wave) {
    const auto cancelled_late_formula = late_wave.formula.get();
    const auto cancelled_late_range = late_wave.range.get();
    const auto cancelled_late_exact_technique = late_wave.exact_technique.get();
    const auto cancelled_late_metric_technique =
        late_wave.metric_technique.get();
    const auto cancelled_late_symbolic_technique =
        late_wave.symbolic_technique.get();

    CHECK(cancelled_late_formula.status == bruce::lhme::Status::Cancelled);
    CHECK(!cancelled_late_formula.integer.has_value());
    CHECK(!cancelled_late_formula.metric_bit_length.has_value());
    CHECK(!cancelled_late_formula.symbolic_expression.has_value());
    CHECK(cancelled_late_range.status == bruce::lhme::Status::Cancelled);
    CHECK(cancelled_late_range.candidates.empty());
    CHECK(cancelled_late_exact_technique.status == bruce::lhme::Status::Cancelled);
    CHECK(!cancelled_late_exact_technique.integer.has_value());
    CHECK(!cancelled_late_exact_technique.metric_bit_length.has_value());
    CHECK(!cancelled_late_exact_technique.symbolic_expression.has_value());
    CHECK(cancelled_late_metric_technique.status ==
          bruce::lhme::Status::Cancelled);
    CHECK(!cancelled_late_metric_technique.integer.has_value());
    CHECK(!cancelled_late_metric_technique.metric_bit_length.has_value());
    CHECK(!cancelled_late_metric_technique.symbolic_expression.has_value());
    CHECK(cancelled_late_symbolic_technique.status ==
          bruce::lhme::Status::Cancelled);
    CHECK(!cancelled_late_symbolic_technique.integer.has_value());
    CHECK(!cancelled_late_symbolic_technique.metric_bit_length.has_value());
    CHECK(!cancelled_late_symbolic_technique.symbolic_expression.has_value());
    CHECK(engine.inspect().resident_entries == 0);
  };

  engine.cancel();
  auto first_late_wave = start_late_wave();
  check_cancelled_late_wave(first_late_wave);

  engine.cancel();
  auto second_late_wave = start_late_wave();
  check_cancelled_late_wave(second_late_wave);

  engine.cancel();
  auto third_late_wave = start_late_wave();
  check_cancelled_late_wave(third_late_wave);

  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();

  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_formula = engine.evaluate(formula_request);
  CHECK(reused_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_formula.exact_integer() == 102300);
  CHECK(!reused_formula.metric_bit_length.has_value());
  CHECK(!reused_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.summary.invalid_argument_count == 0);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_exact_technique = engine.evaluate_technique(
      exact_technique_request);
  CHECK(reused_exact_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_exact_technique.exact_integer() == 102300);
  CHECK(!reused_exact_technique.metric_bit_length.has_value());
  CHECK(!reused_exact_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 2);

  const auto reused_metric_technique = engine.evaluate_technique(
      metric_technique_request);
  CHECK(reused_metric_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_metric_technique.metric_bit_length == 17);
  CHECK(!reused_metric_technique.integer.has_value());
  CHECK(!reused_metric_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 3);

  const auto reused_symbolic_technique = engine.evaluate_technique(
      symbolic_technique_request);
  CHECK(reused_symbolic_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_symbolic_technique.symbolic_expression == "100*(2^10-1)");
  CHECK(!reused_symbolic_technique.integer.has_value());
  CHECK(!reused_symbolic_technique.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 3);
}

TEST(cpp_engine_keeps_a_post_snapshot_cancelled_modular_noncacheable_technique_group_cancelled_for_a_late_modular_technique_joiner_after_all_prior_late_cross_api_joiner_lease_endings_and_repeated_cancels) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::TechniqueRequest modular_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest exact_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest metric_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest symbolic_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_formula;
  std::future<bruce::lhme::RangeResult> pending_range;
  std::future<bruce::lhme::EvaluationResult> pending_metric_technique;
  std::future<bruce::lhme::EvaluationResult> pending_symbolic_technique;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async,
                              [&engine, modular_technique_request] {
    return engine.evaluate_technique(modular_technique_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  bruce::lhme::test_support::EngineRangeFinalizationBarrier
      range_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueFinalizationBarrier
      technique_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueNoncacheableFinalizationBarrier
      noncacheable_technique_finalization_barrier;

  pending_formula = std::async(std::launch::async, [&engine, formula_request] {
    return engine.evaluate(formula_request);
  });
  formula_finalization_barrier.wait_until_reached();

  pending_range = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  range_finalization_barrier.wait_until_reached();

  pending_metric_technique = std::async(
      std::launch::async, [&engine, metric_technique_request] {
        return engine.evaluate_technique(metric_technique_request);
      });
  technique_finalization_barrier.wait_until_reached();

  pending_symbolic_technique = std::async(
      std::launch::async, [&engine, symbolic_technique_request] {
        return engine.evaluate_technique(symbolic_technique_request);
      });
  noncacheable_technique_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_formula = pending_formula.get();

  CHECK(cancelled_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_formula.integer.has_value());
  CHECK(!cancelled_formula.metric_bit_length.has_value());
  CHECK(!cancelled_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  range_finalization_barrier.release();
  const auto cancelled_range = pending_range.get();

  CHECK(cancelled_range.status == bruce::lhme::Status::Cancelled);
  CHECK(cancelled_range.candidates.empty());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  technique_finalization_barrier.release();
  noncacheable_technique_finalization_barrier.release();
  const auto cancelled_metric_technique = pending_metric_technique.get();
  const auto cancelled_symbolic_technique = pending_symbolic_technique.get();

  CHECK(cancelled_metric_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_metric_technique.integer.has_value());
  CHECK(!cancelled_metric_technique.metric_bit_length.has_value());
  CHECK(!cancelled_metric_technique.symbolic_expression.has_value());
  CHECK(cancelled_symbolic_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_symbolic_technique.integer.has_value());
  CHECK(!cancelled_symbolic_technique.metric_bit_length.has_value());
  CHECK(!cancelled_symbolic_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  struct LateWave {
    std::future<bruce::lhme::EvaluationResult> formula;
    std::future<bruce::lhme::RangeResult> range;
    std::future<bruce::lhme::EvaluationResult> exact_technique;
    std::future<bruce::lhme::EvaluationResult> metric_technique;
    std::future<bruce::lhme::EvaluationResult> symbolic_technique;
  };
  const auto start_late_wave =
      [&engine, formula_request, range_request, exact_technique_request,
       metric_technique_request, symbolic_technique_request] {
        return LateWave{
            .formula = std::async(std::launch::async,
                                  [&engine, formula_request] {
                                    return engine.evaluate(formula_request);
                                  }),
            .range = std::async(std::launch::async, [&engine, range_request] {
              return engine.evaluate_range(range_request, {0, 1});
            }),
            .exact_technique = std::async(
                std::launch::async, [&engine, exact_technique_request] {
                  return engine.evaluate_technique(exact_technique_request);
                }),
            .metric_technique = std::async(
                std::launch::async, [&engine, metric_technique_request] {
                  return engine.evaluate_technique(metric_technique_request);
                }),
            .symbolic_technique = std::async(
                std::launch::async, [&engine, symbolic_technique_request] {
                  return engine.evaluate_technique(symbolic_technique_request);
                })};
      };
  const auto check_cancelled_late_wave = [&engine](LateWave& late_wave) {
    const auto cancelled_late_formula = late_wave.formula.get();
    const auto cancelled_late_range = late_wave.range.get();
    const auto cancelled_late_exact_technique = late_wave.exact_technique.get();
    const auto cancelled_late_metric_technique =
        late_wave.metric_technique.get();
    const auto cancelled_late_symbolic_technique =
        late_wave.symbolic_technique.get();

    CHECK(cancelled_late_formula.status == bruce::lhme::Status::Cancelled);
    CHECK(!cancelled_late_formula.integer.has_value());
    CHECK(!cancelled_late_formula.metric_bit_length.has_value());
    CHECK(!cancelled_late_formula.symbolic_expression.has_value());
    CHECK(cancelled_late_range.status == bruce::lhme::Status::Cancelled);
    CHECK(cancelled_late_range.candidates.empty());
    CHECK(cancelled_late_exact_technique.status == bruce::lhme::Status::Cancelled);
    CHECK(!cancelled_late_exact_technique.integer.has_value());
    CHECK(!cancelled_late_exact_technique.metric_bit_length.has_value());
    CHECK(!cancelled_late_exact_technique.symbolic_expression.has_value());
    CHECK(cancelled_late_metric_technique.status ==
          bruce::lhme::Status::Cancelled);
    CHECK(!cancelled_late_metric_technique.integer.has_value());
    CHECK(!cancelled_late_metric_technique.metric_bit_length.has_value());
    CHECK(!cancelled_late_metric_technique.symbolic_expression.has_value());
    CHECK(cancelled_late_symbolic_technique.status ==
          bruce::lhme::Status::Cancelled);
    CHECK(!cancelled_late_symbolic_technique.integer.has_value());
    CHECK(!cancelled_late_symbolic_technique.metric_bit_length.has_value());
    CHECK(!cancelled_late_symbolic_technique.symbolic_expression.has_value());
    CHECK(engine.inspect().resident_entries == 0);
  };

  engine.cancel();
  auto first_late_wave = start_late_wave();
  check_cancelled_late_wave(first_late_wave);

  engine.cancel();
  auto second_late_wave = start_late_wave();
  check_cancelled_late_wave(second_late_wave);

  engine.cancel();
  auto third_late_wave = start_late_wave();
  check_cancelled_late_wave(third_late_wave);

  engine.cancel();
  const auto cancelled_late_modular_technique = engine.evaluate_technique(
      modular_technique_request);

  CHECK(cancelled_late_modular_technique.status ==
        bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_late_modular_technique.integer.has_value());
  CHECK(!cancelled_late_modular_technique.metric_bit_length.has_value());
  CHECK(!cancelled_late_modular_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();

  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_modular_technique = engine.evaluate_technique(
      modular_technique_request);
  CHECK(reused_modular_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_modular_technique.exact_integer() == 11);
  CHECK(!reused_modular_technique.metric_bit_length.has_value());
  CHECK(!reused_modular_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_formula = engine.evaluate(formula_request);
  CHECK(reused_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_formula.exact_integer() == 102300);
  CHECK(!reused_formula.metric_bit_length.has_value());
  CHECK(!reused_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.summary.invalid_argument_count == 0);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_exact_technique = engine.evaluate_technique(
      exact_technique_request);
  CHECK(reused_exact_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_exact_technique.exact_integer() == 102300);
  CHECK(!reused_exact_technique.metric_bit_length.has_value());
  CHECK(!reused_exact_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 2);

  const auto reused_metric_technique = engine.evaluate_technique(
      metric_technique_request);
  CHECK(reused_metric_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_metric_technique.metric_bit_length == 17);
  CHECK(!reused_metric_technique.integer.has_value());
  CHECK(!reused_metric_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 3);

  const auto reused_symbolic_technique = engine.evaluate_technique(
      symbolic_technique_request);
  CHECK(reused_symbolic_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_symbolic_technique.symbolic_expression == "100*(2^10-1)");
  CHECK(!reused_symbolic_technique.integer.has_value());
  CHECK(!reused_symbolic_technique.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 3);
}

TEST(cpp_engine_keeps_a_post_snapshot_cancelled_modular_noncacheable_technique_group_cancelled_for_a_concurrent_late_cross_api_joiner_wave_including_modular_technique_after_all_prior_late_joiner_lease_endings_and_repeated_cancels) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::TechniqueRequest modular_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest exact_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest metric_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest symbolic_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_formula;
  std::future<bruce::lhme::RangeResult> pending_range;
  std::future<bruce::lhme::EvaluationResult> pending_metric_technique;
  std::future<bruce::lhme::EvaluationResult> pending_symbolic_technique;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async,
                              [&engine, modular_technique_request] {
    return engine.evaluate_technique(modular_technique_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  bruce::lhme::test_support::EngineRangeFinalizationBarrier
      range_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueFinalizationBarrier
      technique_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueNoncacheableFinalizationBarrier
      noncacheable_technique_finalization_barrier;

  pending_formula = std::async(std::launch::async, [&engine, formula_request] {
    return engine.evaluate(formula_request);
  });
  formula_finalization_barrier.wait_until_reached();

  pending_range = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  range_finalization_barrier.wait_until_reached();

  pending_metric_technique = std::async(
      std::launch::async, [&engine, metric_technique_request] {
        return engine.evaluate_technique(metric_technique_request);
      });
  technique_finalization_barrier.wait_until_reached();

  pending_symbolic_technique = std::async(
      std::launch::async, [&engine, symbolic_technique_request] {
        return engine.evaluate_technique(symbolic_technique_request);
      });
  noncacheable_technique_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_formula = pending_formula.get();

  CHECK(cancelled_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_formula.integer.has_value());
  CHECK(!cancelled_formula.metric_bit_length.has_value());
  CHECK(!cancelled_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  range_finalization_barrier.release();
  const auto cancelled_range = pending_range.get();

  CHECK(cancelled_range.status == bruce::lhme::Status::Cancelled);
  CHECK(cancelled_range.candidates.empty());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  technique_finalization_barrier.release();
  noncacheable_technique_finalization_barrier.release();
  const auto cancelled_metric_technique = pending_metric_technique.get();
  const auto cancelled_symbolic_technique = pending_symbolic_technique.get();

  CHECK(cancelled_metric_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_metric_technique.integer.has_value());
  CHECK(!cancelled_metric_technique.metric_bit_length.has_value());
  CHECK(!cancelled_metric_technique.symbolic_expression.has_value());
  CHECK(cancelled_symbolic_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_symbolic_technique.integer.has_value());
  CHECK(!cancelled_symbolic_technique.metric_bit_length.has_value());
  CHECK(!cancelled_symbolic_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  struct LateWave {
    std::future<bruce::lhme::EvaluationResult> formula;
    std::future<bruce::lhme::RangeResult> range;
    std::future<bruce::lhme::EvaluationResult> exact_technique;
    std::future<bruce::lhme::EvaluationResult> metric_technique;
    std::future<bruce::lhme::EvaluationResult> symbolic_technique;
  };
  const auto start_late_wave =
      [&engine, formula_request, range_request, exact_technique_request,
       metric_technique_request, symbolic_technique_request] {
        return LateWave{
            .formula = std::async(std::launch::async,
                                  [&engine, formula_request] {
                                    return engine.evaluate(formula_request);
                                  }),
            .range = std::async(std::launch::async, [&engine, range_request] {
              return engine.evaluate_range(range_request, {0, 1});
            }),
            .exact_technique = std::async(
                std::launch::async, [&engine, exact_technique_request] {
                  return engine.evaluate_technique(exact_technique_request);
                }),
            .metric_technique = std::async(
                std::launch::async, [&engine, metric_technique_request] {
                  return engine.evaluate_technique(metric_technique_request);
                }),
            .symbolic_technique = std::async(
                std::launch::async, [&engine, symbolic_technique_request] {
                  return engine.evaluate_technique(symbolic_technique_request);
                })};
      };
  const auto check_cancelled_late_wave = [&engine](LateWave& late_wave) {
    const auto cancelled_late_formula = late_wave.formula.get();
    const auto cancelled_late_range = late_wave.range.get();
    const auto cancelled_late_exact_technique = late_wave.exact_technique.get();
    const auto cancelled_late_metric_technique =
        late_wave.metric_technique.get();
    const auto cancelled_late_symbolic_technique =
        late_wave.symbolic_technique.get();

    CHECK(cancelled_late_formula.status == bruce::lhme::Status::Cancelled);
    CHECK(!cancelled_late_formula.integer.has_value());
    CHECK(!cancelled_late_formula.metric_bit_length.has_value());
    CHECK(!cancelled_late_formula.symbolic_expression.has_value());
    CHECK(cancelled_late_range.status == bruce::lhme::Status::Cancelled);
    CHECK(cancelled_late_range.candidates.empty());
    CHECK(cancelled_late_exact_technique.status == bruce::lhme::Status::Cancelled);
    CHECK(!cancelled_late_exact_technique.integer.has_value());
    CHECK(!cancelled_late_exact_technique.metric_bit_length.has_value());
    CHECK(!cancelled_late_exact_technique.symbolic_expression.has_value());
    CHECK(cancelled_late_metric_technique.status ==
          bruce::lhme::Status::Cancelled);
    CHECK(!cancelled_late_metric_technique.integer.has_value());
    CHECK(!cancelled_late_metric_technique.metric_bit_length.has_value());
    CHECK(!cancelled_late_metric_technique.symbolic_expression.has_value());
    CHECK(cancelled_late_symbolic_technique.status ==
          bruce::lhme::Status::Cancelled);
    CHECK(!cancelled_late_symbolic_technique.integer.has_value());
    CHECK(!cancelled_late_symbolic_technique.metric_bit_length.has_value());
    CHECK(!cancelled_late_symbolic_technique.symbolic_expression.has_value());
    CHECK(engine.inspect().resident_entries == 0);
  };

  engine.cancel();
  auto first_late_wave = start_late_wave();
  check_cancelled_late_wave(first_late_wave);

  engine.cancel();
  auto second_late_wave = start_late_wave();
  check_cancelled_late_wave(second_late_wave);

  engine.cancel();
  auto third_late_wave = start_late_wave();
  check_cancelled_late_wave(third_late_wave);

  struct FinalLateWave {
    std::future<bruce::lhme::EvaluationResult> formula;
    std::future<bruce::lhme::RangeResult> range;
    std::future<bruce::lhme::EvaluationResult> exact_technique;
    std::future<bruce::lhme::EvaluationResult> metric_technique;
    std::future<bruce::lhme::EvaluationResult> symbolic_technique;
    std::future<bruce::lhme::EvaluationResult> modular_technique;
  };
  const auto start_final_late_wave =
      [&engine, formula_request, range_request, exact_technique_request,
       metric_technique_request, symbolic_technique_request,
       modular_technique_request] {
        return FinalLateWave{
            .formula = std::async(std::launch::async,
                                  [&engine, formula_request] {
                                    return engine.evaluate(formula_request);
                                  }),
            .range = std::async(std::launch::async, [&engine, range_request] {
              return engine.evaluate_range(range_request, {0, 1});
            }),
            .exact_technique = std::async(
                std::launch::async, [&engine, exact_technique_request] {
                  return engine.evaluate_technique(exact_technique_request);
                }),
            .metric_technique = std::async(
                std::launch::async, [&engine, metric_technique_request] {
                  return engine.evaluate_technique(metric_technique_request);
                }),
            .symbolic_technique = std::async(
                std::launch::async, [&engine, symbolic_technique_request] {
                  return engine.evaluate_technique(symbolic_technique_request);
                }),
            .modular_technique = std::async(
                std::launch::async, [&engine, modular_technique_request] {
                  return engine.evaluate_technique(modular_technique_request);
                })};
      };
  const auto check_cancelled_final_late_wave =
      [&engine](FinalLateWave& late_wave) {
        const auto cancelled_late_formula = late_wave.formula.get();
        const auto cancelled_late_range = late_wave.range.get();
        const auto cancelled_late_exact_technique =
            late_wave.exact_technique.get();
        const auto cancelled_late_metric_technique =
            late_wave.metric_technique.get();
        const auto cancelled_late_symbolic_technique =
            late_wave.symbolic_technique.get();
        const auto cancelled_late_modular_technique =
            late_wave.modular_technique.get();

        CHECK(cancelled_late_formula.status == bruce::lhme::Status::Cancelled);
        CHECK(!cancelled_late_formula.integer.has_value());
        CHECK(!cancelled_late_formula.metric_bit_length.has_value());
        CHECK(!cancelled_late_formula.symbolic_expression.has_value());
        CHECK(cancelled_late_range.status == bruce::lhme::Status::Cancelled);
        CHECK(cancelled_late_range.candidates.empty());
        CHECK(cancelled_late_exact_technique.status ==
              bruce::lhme::Status::Cancelled);
        CHECK(!cancelled_late_exact_technique.integer.has_value());
        CHECK(!cancelled_late_exact_technique.metric_bit_length.has_value());
        CHECK(!cancelled_late_exact_technique.symbolic_expression.has_value());
        CHECK(cancelled_late_metric_technique.status ==
              bruce::lhme::Status::Cancelled);
        CHECK(!cancelled_late_metric_technique.integer.has_value());
        CHECK(!cancelled_late_metric_technique.metric_bit_length.has_value());
        CHECK(!cancelled_late_metric_technique.symbolic_expression.has_value());
        CHECK(cancelled_late_symbolic_technique.status ==
              bruce::lhme::Status::Cancelled);
        CHECK(!cancelled_late_symbolic_technique.integer.has_value());
        CHECK(!cancelled_late_symbolic_technique.metric_bit_length.has_value());
        CHECK(!cancelled_late_symbolic_technique.symbolic_expression.has_value());
        CHECK(cancelled_late_modular_technique.status ==
              bruce::lhme::Status::Cancelled);
        CHECK(!cancelled_late_modular_technique.integer.has_value());
        CHECK(!cancelled_late_modular_technique.metric_bit_length.has_value());
        CHECK(!cancelled_late_modular_technique.symbolic_expression.has_value());
        CHECK(engine.inspect().resident_entries == 0);
      };

  engine.cancel();
  auto final_late_wave = start_final_late_wave();
  check_cancelled_final_late_wave(final_late_wave);

  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();

  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_modular_technique = engine.evaluate_technique(
      modular_technique_request);
  CHECK(reused_modular_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_modular_technique.exact_integer() == 11);
  CHECK(!reused_modular_technique.metric_bit_length.has_value());
  CHECK(!reused_modular_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_formula = engine.evaluate(formula_request);
  CHECK(reused_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_formula.exact_integer() == 102300);
  CHECK(!reused_formula.metric_bit_length.has_value());
  CHECK(!reused_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.summary.invalid_argument_count == 0);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_exact_technique = engine.evaluate_technique(
      exact_technique_request);
  CHECK(reused_exact_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_exact_technique.exact_integer() == 102300);
  CHECK(!reused_exact_technique.metric_bit_length.has_value());
  CHECK(!reused_exact_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 2);

  const auto reused_metric_technique = engine.evaluate_technique(
      metric_technique_request);
  CHECK(reused_metric_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_metric_technique.metric_bit_length == 17);
  CHECK(!reused_metric_technique.integer.has_value());
  CHECK(!reused_metric_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 3);

  const auto reused_symbolic_technique = engine.evaluate_technique(
      symbolic_technique_request);
  CHECK(reused_symbolic_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_symbolic_technique.symbolic_expression == "100*(2^10-1)");
  CHECK(!reused_symbolic_technique.integer.has_value());
  CHECK(!reused_symbolic_technique.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 3);
}

TEST(cpp_engine_keeps_a_post_snapshot_cancelled_modular_noncacheable_technique_group_cancelled_for_a_concurrent_late_cross_api_joiner_wave_while_a_different_modular_modulus_forms_a_confirmed_group_after_all_prior_late_joiner_lease_endings_and_repeated_cancels) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::TechniqueRequest modular_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::TechniqueRequest other_modular_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 19};
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest exact_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest metric_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest symbolic_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_formula;
  std::future<bruce::lhme::RangeResult> pending_range;
  std::future<bruce::lhme::EvaluationResult> pending_metric_technique;
  std::future<bruce::lhme::EvaluationResult> pending_symbolic_technique;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async,
                              [&engine, modular_technique_request] {
    return engine.evaluate_technique(modular_technique_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  bruce::lhme::test_support::EngineRangeFinalizationBarrier
      range_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueFinalizationBarrier
      technique_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueNoncacheableFinalizationBarrier
      noncacheable_technique_finalization_barrier;

  pending_formula = std::async(std::launch::async, [&engine, formula_request] {
    return engine.evaluate(formula_request);
  });
  formula_finalization_barrier.wait_until_reached();

  pending_range = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  range_finalization_barrier.wait_until_reached();

  pending_metric_technique = std::async(
      std::launch::async, [&engine, metric_technique_request] {
        return engine.evaluate_technique(metric_technique_request);
      });
  technique_finalization_barrier.wait_until_reached();

  pending_symbolic_technique = std::async(
      std::launch::async, [&engine, symbolic_technique_request] {
        return engine.evaluate_technique(symbolic_technique_request);
      });
  noncacheable_technique_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_formula = pending_formula.get();

  CHECK(cancelled_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_formula.integer.has_value());
  CHECK(!cancelled_formula.metric_bit_length.has_value());
  CHECK(!cancelled_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  range_finalization_barrier.release();
  const auto cancelled_range = pending_range.get();

  CHECK(cancelled_range.status == bruce::lhme::Status::Cancelled);
  CHECK(cancelled_range.candidates.empty());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  technique_finalization_barrier.release();
  noncacheable_technique_finalization_barrier.release();
  const auto cancelled_metric_technique = pending_metric_technique.get();
  const auto cancelled_symbolic_technique = pending_symbolic_technique.get();

  CHECK(cancelled_metric_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_metric_technique.integer.has_value());
  CHECK(!cancelled_metric_technique.metric_bit_length.has_value());
  CHECK(!cancelled_metric_technique.symbolic_expression.has_value());
  CHECK(cancelled_symbolic_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_symbolic_technique.integer.has_value());
  CHECK(!cancelled_symbolic_technique.metric_bit_length.has_value());
  CHECK(!cancelled_symbolic_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  struct LateWave {
    std::future<bruce::lhme::EvaluationResult> formula;
    std::future<bruce::lhme::RangeResult> range;
    std::future<bruce::lhme::EvaluationResult> exact_technique;
    std::future<bruce::lhme::EvaluationResult> metric_technique;
    std::future<bruce::lhme::EvaluationResult> symbolic_technique;
  };
  const auto start_late_wave =
      [&engine, formula_request, range_request, exact_technique_request,
       metric_technique_request, symbolic_technique_request] {
        return LateWave{
            .formula = std::async(std::launch::async,
                                  [&engine, formula_request] {
                                    return engine.evaluate(formula_request);
                                  }),
            .range = std::async(std::launch::async, [&engine, range_request] {
              return engine.evaluate_range(range_request, {0, 1});
            }),
            .exact_technique = std::async(
                std::launch::async, [&engine, exact_technique_request] {
                  return engine.evaluate_technique(exact_technique_request);
                }),
            .metric_technique = std::async(
                std::launch::async, [&engine, metric_technique_request] {
                  return engine.evaluate_technique(metric_technique_request);
                }),
            .symbolic_technique = std::async(
                std::launch::async, [&engine, symbolic_technique_request] {
                  return engine.evaluate_technique(symbolic_technique_request);
                })};
      };
  const auto check_cancelled_late_wave = [&engine](LateWave& late_wave) {
    const auto cancelled_late_formula = late_wave.formula.get();
    const auto cancelled_late_range = late_wave.range.get();
    const auto cancelled_late_exact_technique = late_wave.exact_technique.get();
    const auto cancelled_late_metric_technique =
        late_wave.metric_technique.get();
    const auto cancelled_late_symbolic_technique =
        late_wave.symbolic_technique.get();

    CHECK(cancelled_late_formula.status == bruce::lhme::Status::Cancelled);
    CHECK(!cancelled_late_formula.integer.has_value());
    CHECK(!cancelled_late_formula.metric_bit_length.has_value());
    CHECK(!cancelled_late_formula.symbolic_expression.has_value());
    CHECK(cancelled_late_range.status == bruce::lhme::Status::Cancelled);
    CHECK(cancelled_late_range.candidates.empty());
    CHECK(cancelled_late_exact_technique.status == bruce::lhme::Status::Cancelled);
    CHECK(!cancelled_late_exact_technique.integer.has_value());
    CHECK(!cancelled_late_exact_technique.metric_bit_length.has_value());
    CHECK(!cancelled_late_exact_technique.symbolic_expression.has_value());
    CHECK(cancelled_late_metric_technique.status ==
          bruce::lhme::Status::Cancelled);
    CHECK(!cancelled_late_metric_technique.integer.has_value());
    CHECK(!cancelled_late_metric_technique.metric_bit_length.has_value());
    CHECK(!cancelled_late_metric_technique.symbolic_expression.has_value());
    CHECK(cancelled_late_symbolic_technique.status ==
          bruce::lhme::Status::Cancelled);
    CHECK(!cancelled_late_symbolic_technique.integer.has_value());
    CHECK(!cancelled_late_symbolic_technique.metric_bit_length.has_value());
    CHECK(!cancelled_late_symbolic_technique.symbolic_expression.has_value());
    CHECK(engine.inspect().resident_entries == 0);
  };

  engine.cancel();
  auto first_late_wave = start_late_wave();
  check_cancelled_late_wave(first_late_wave);

  engine.cancel();
  auto second_late_wave = start_late_wave();
  check_cancelled_late_wave(second_late_wave);

  engine.cancel();
  auto third_late_wave = start_late_wave();
  check_cancelled_late_wave(third_late_wave);

  struct FinalLateWave {
    std::future<bruce::lhme::EvaluationResult> formula;
    std::future<bruce::lhme::RangeResult> range;
    std::future<bruce::lhme::EvaluationResult> exact_technique;
    std::future<bruce::lhme::EvaluationResult> metric_technique;
    std::future<bruce::lhme::EvaluationResult> symbolic_technique;
    std::future<bruce::lhme::EvaluationResult> modular_technique;
    std::future<bruce::lhme::EvaluationResult> other_modular_technique;
  };
  const auto start_final_late_wave =
      [&engine, formula_request, range_request, exact_technique_request,
       metric_technique_request, symbolic_technique_request,
       modular_technique_request, other_modular_technique_request] {
        return FinalLateWave{
            .formula = std::async(std::launch::async,
                                  [&engine, formula_request] {
                                    return engine.evaluate(formula_request);
                                  }),
            .range = std::async(std::launch::async, [&engine, range_request] {
              return engine.evaluate_range(range_request, {0, 1});
            }),
            .exact_technique = std::async(
                std::launch::async, [&engine, exact_technique_request] {
                  return engine.evaluate_technique(exact_technique_request);
                }),
            .metric_technique = std::async(
                std::launch::async, [&engine, metric_technique_request] {
                  return engine.evaluate_technique(metric_technique_request);
                }),
            .symbolic_technique = std::async(
                std::launch::async, [&engine, symbolic_technique_request] {
                  return engine.evaluate_technique(symbolic_technique_request);
                }),
            .modular_technique = std::async(
                std::launch::async, [&engine, modular_technique_request] {
                  return engine.evaluate_technique(modular_technique_request);
                }),
            .other_modular_technique = std::async(
                std::launch::async,
                [&engine, other_modular_technique_request] {
                  return engine.evaluate_technique(
                      other_modular_technique_request);
                })};
      };
  const auto check_final_late_wave = [&engine](FinalLateWave& late_wave) {
    const auto cancelled_late_formula = late_wave.formula.get();
    const auto cancelled_late_range = late_wave.range.get();
    const auto cancelled_late_exact_technique = late_wave.exact_technique.get();
    const auto cancelled_late_metric_technique =
        late_wave.metric_technique.get();
    const auto cancelled_late_symbolic_technique =
        late_wave.symbolic_technique.get();
    const auto cancelled_late_modular_technique =
        late_wave.modular_technique.get();
    const auto confirmed_other_modular_technique =
        late_wave.other_modular_technique.get();

    CHECK(cancelled_late_formula.status == bruce::lhme::Status::Cancelled);
    CHECK(!cancelled_late_formula.integer.has_value());
    CHECK(!cancelled_late_formula.metric_bit_length.has_value());
    CHECK(!cancelled_late_formula.symbolic_expression.has_value());
    CHECK(cancelled_late_range.status == bruce::lhme::Status::Cancelled);
    CHECK(cancelled_late_range.candidates.empty());
    CHECK(cancelled_late_exact_technique.status == bruce::lhme::Status::Cancelled);
    CHECK(!cancelled_late_exact_technique.integer.has_value());
    CHECK(!cancelled_late_exact_technique.metric_bit_length.has_value());
    CHECK(!cancelled_late_exact_technique.symbolic_expression.has_value());
    CHECK(cancelled_late_metric_technique.status ==
          bruce::lhme::Status::Cancelled);
    CHECK(!cancelled_late_metric_technique.integer.has_value());
    CHECK(!cancelled_late_metric_technique.metric_bit_length.has_value());
    CHECK(!cancelled_late_metric_technique.symbolic_expression.has_value());
    CHECK(cancelled_late_symbolic_technique.status ==
          bruce::lhme::Status::Cancelled);
    CHECK(!cancelled_late_symbolic_technique.integer.has_value());
    CHECK(!cancelled_late_symbolic_technique.metric_bit_length.has_value());
    CHECK(!cancelled_late_symbolic_technique.symbolic_expression.has_value());
    CHECK(cancelled_late_modular_technique.status ==
          bruce::lhme::Status::Cancelled);
    CHECK(!cancelled_late_modular_technique.integer.has_value());
    CHECK(!cancelled_late_modular_technique.metric_bit_length.has_value());
    CHECK(!cancelled_late_modular_technique.symbolic_expression.has_value());
    CHECK(confirmed_other_modular_technique.status ==
          bruce::lhme::Status::Confirmed);
    CHECK(confirmed_other_modular_technique.exact_integer() == 4);
    CHECK(!confirmed_other_modular_technique.metric_bit_length.has_value());
    CHECK(!confirmed_other_modular_technique.symbolic_expression.has_value());
    CHECK(engine.inspect().resident_entries == 0);
  };

  engine.cancel();
  auto final_late_wave = start_final_late_wave();
  check_final_late_wave(final_late_wave);

  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();

  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_modular_technique = engine.evaluate_technique(
      modular_technique_request);
  CHECK(reused_modular_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_modular_technique.exact_integer() == 11);
  CHECK(!reused_modular_technique.metric_bit_length.has_value());
  CHECK(!reused_modular_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_formula = engine.evaluate(formula_request);
  CHECK(reused_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_formula.exact_integer() == 102300);
  CHECK(!reused_formula.metric_bit_length.has_value());
  CHECK(!reused_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.summary.invalid_argument_count == 0);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_exact_technique = engine.evaluate_technique(
      exact_technique_request);
  CHECK(reused_exact_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_exact_technique.exact_integer() == 102300);
  CHECK(!reused_exact_technique.metric_bit_length.has_value());
  CHECK(!reused_exact_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 2);

  const auto reused_metric_technique = engine.evaluate_technique(
      metric_technique_request);
  CHECK(reused_metric_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_metric_technique.metric_bit_length == 17);
  CHECK(!reused_metric_technique.integer.has_value());
  CHECK(!reused_metric_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 3);

  const auto reused_symbolic_technique = engine.evaluate_technique(
      symbolic_technique_request);
  CHECK(reused_symbolic_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_symbolic_technique.symbolic_expression == "100*(2^10-1)");
  CHECK(!reused_symbolic_technique.integer.has_value());
  CHECK(!reused_symbolic_technique.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 3);
}

TEST(cpp_engine_does_not_join_a_post_snapshot_cancelled_modular_noncacheable_technique_group_for_concurrent_late_cross_api_requests_with_the_same_modulus_but_a_different_canonical_computation) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const auto other_formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(3, 100, 1));
  const bruce::lhme::TechniqueRequest modular_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::TechniqueRequest other_modular_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 19};
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest exact_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest metric_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest symbolic_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest other_modular_formula_request{
      .spec = other_formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::TechniqueRequest other_modular_technique_same_modulus_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(3, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_formula;
  std::future<bruce::lhme::RangeResult> pending_range;
  std::future<bruce::lhme::EvaluationResult> pending_metric_technique;
  std::future<bruce::lhme::EvaluationResult> pending_symbolic_technique;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async,
                              [&engine, modular_technique_request] {
    return engine.evaluate_technique(modular_technique_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  bruce::lhme::test_support::EngineRangeFinalizationBarrier
      range_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueFinalizationBarrier
      technique_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueNoncacheableFinalizationBarrier
      noncacheable_technique_finalization_barrier;

  pending_formula = std::async(std::launch::async, [&engine, formula_request] {
    return engine.evaluate(formula_request);
  });
  formula_finalization_barrier.wait_until_reached();

  pending_range = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  range_finalization_barrier.wait_until_reached();

  pending_metric_technique = std::async(
      std::launch::async, [&engine, metric_technique_request] {
        return engine.evaluate_technique(metric_technique_request);
      });
  technique_finalization_barrier.wait_until_reached();

  pending_symbolic_technique = std::async(
      std::launch::async, [&engine, symbolic_technique_request] {
        return engine.evaluate_technique(symbolic_technique_request);
      });
  noncacheable_technique_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_formula = pending_formula.get();

  CHECK(cancelled_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_formula.integer.has_value());
  CHECK(!cancelled_formula.metric_bit_length.has_value());
  CHECK(!cancelled_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  range_finalization_barrier.release();
  const auto cancelled_range = pending_range.get();

  CHECK(cancelled_range.status == bruce::lhme::Status::Cancelled);
  CHECK(cancelled_range.candidates.empty());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  technique_finalization_barrier.release();
  noncacheable_technique_finalization_barrier.release();
  const auto cancelled_metric_technique = pending_metric_technique.get();
  const auto cancelled_symbolic_technique = pending_symbolic_technique.get();

  CHECK(cancelled_metric_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_metric_technique.integer.has_value());
  CHECK(!cancelled_metric_technique.metric_bit_length.has_value());
  CHECK(!cancelled_metric_technique.symbolic_expression.has_value());
  CHECK(cancelled_symbolic_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_symbolic_technique.integer.has_value());
  CHECK(!cancelled_symbolic_technique.metric_bit_length.has_value());
  CHECK(!cancelled_symbolic_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  struct LateWave {
    std::future<bruce::lhme::EvaluationResult> formula;
    std::future<bruce::lhme::RangeResult> range;
    std::future<bruce::lhme::EvaluationResult> exact_technique;
    std::future<bruce::lhme::EvaluationResult> metric_technique;
    std::future<bruce::lhme::EvaluationResult> symbolic_technique;
  };
  const auto start_late_wave =
      [&engine, formula_request, range_request, exact_technique_request,
       metric_technique_request, symbolic_technique_request] {
        return LateWave{
            .formula = std::async(std::launch::async,
                                  [&engine, formula_request] {
                                    return engine.evaluate(formula_request);
                                  }),
            .range = std::async(std::launch::async, [&engine, range_request] {
              return engine.evaluate_range(range_request, {0, 1});
            }),
            .exact_technique = std::async(
                std::launch::async, [&engine, exact_technique_request] {
                  return engine.evaluate_technique(exact_technique_request);
                }),
            .metric_technique = std::async(
                std::launch::async, [&engine, metric_technique_request] {
                  return engine.evaluate_technique(metric_technique_request);
                }),
            .symbolic_technique = std::async(
                std::launch::async, [&engine, symbolic_technique_request] {
                  return engine.evaluate_technique(symbolic_technique_request);
                })};
      };
  const auto check_cancelled_late_wave = [&engine](LateWave& late_wave) {
    const auto cancelled_late_formula = late_wave.formula.get();
    const auto cancelled_late_range = late_wave.range.get();
    const auto cancelled_late_exact_technique = late_wave.exact_technique.get();
    const auto cancelled_late_metric_technique =
        late_wave.metric_technique.get();
    const auto cancelled_late_symbolic_technique =
        late_wave.symbolic_technique.get();

    CHECK(cancelled_late_formula.status == bruce::lhme::Status::Cancelled);
    CHECK(!cancelled_late_formula.integer.has_value());
    CHECK(!cancelled_late_formula.metric_bit_length.has_value());
    CHECK(!cancelled_late_formula.symbolic_expression.has_value());
    CHECK(cancelled_late_range.status == bruce::lhme::Status::Cancelled);
    CHECK(cancelled_late_range.candidates.empty());
    CHECK(cancelled_late_exact_technique.status ==
          bruce::lhme::Status::Cancelled);
    CHECK(!cancelled_late_exact_technique.integer.has_value());
    CHECK(!cancelled_late_exact_technique.metric_bit_length.has_value());
    CHECK(!cancelled_late_exact_technique.symbolic_expression.has_value());
    CHECK(cancelled_late_metric_technique.status ==
          bruce::lhme::Status::Cancelled);
    CHECK(!cancelled_late_metric_technique.integer.has_value());
    CHECK(!cancelled_late_metric_technique.metric_bit_length.has_value());
    CHECK(!cancelled_late_metric_technique.symbolic_expression.has_value());
    CHECK(cancelled_late_symbolic_technique.status ==
          bruce::lhme::Status::Cancelled);
    CHECK(!cancelled_late_symbolic_technique.integer.has_value());
    CHECK(!cancelled_late_symbolic_technique.metric_bit_length.has_value());
    CHECK(!cancelled_late_symbolic_technique.symbolic_expression.has_value());
    CHECK(engine.inspect().resident_entries == 0);
  };

  engine.cancel();
  auto first_late_wave = start_late_wave();
  check_cancelled_late_wave(first_late_wave);

  engine.cancel();
  auto second_late_wave = start_late_wave();
  check_cancelled_late_wave(second_late_wave);

  engine.cancel();
  auto third_late_wave = start_late_wave();
  check_cancelled_late_wave(third_late_wave);

  struct FinalLateWave {
    std::future<bruce::lhme::EvaluationResult> formula;
    std::future<bruce::lhme::RangeResult> range;
    std::future<bruce::lhme::EvaluationResult> exact_technique;
    std::future<bruce::lhme::EvaluationResult> metric_technique;
    std::future<bruce::lhme::EvaluationResult> symbolic_technique;
    std::future<bruce::lhme::EvaluationResult> modular_technique;
    std::future<bruce::lhme::EvaluationResult> other_modular_technique;
    std::future<bruce::lhme::EvaluationResult> other_modular_formula;
    std::future<bruce::lhme::EvaluationResult>
        other_modular_technique_same_modulus;
  };
  const auto start_final_late_wave =
      [&engine, formula_request, range_request, exact_technique_request,
       metric_technique_request, symbolic_technique_request,
       modular_technique_request, other_modular_technique_request,
       other_modular_formula_request,
       other_modular_technique_same_modulus_request] {
        return FinalLateWave{
            .formula = std::async(std::launch::async,
                                  [&engine, formula_request] {
                                    return engine.evaluate(formula_request);
                                  }),
            .range = std::async(std::launch::async, [&engine, range_request] {
              return engine.evaluate_range(range_request, {0, 1});
            }),
            .exact_technique = std::async(
                std::launch::async, [&engine, exact_technique_request] {
                  return engine.evaluate_technique(exact_technique_request);
                }),
            .metric_technique = std::async(
                std::launch::async, [&engine, metric_technique_request] {
                  return engine.evaluate_technique(metric_technique_request);
                }),
            .symbolic_technique = std::async(
                std::launch::async, [&engine, symbolic_technique_request] {
                  return engine.evaluate_technique(symbolic_technique_request);
                }),
            .modular_technique = std::async(
                std::launch::async, [&engine, modular_technique_request] {
                  return engine.evaluate_technique(modular_technique_request);
                }),
            .other_modular_technique = std::async(
                std::launch::async,
                [&engine, other_modular_technique_request] {
                  return engine.evaluate_technique(
                      other_modular_technique_request);
                }),
            .other_modular_formula = std::async(
                std::launch::async, [&engine, other_modular_formula_request] {
                  return engine.evaluate(other_modular_formula_request);
                }),
            .other_modular_technique_same_modulus = std::async(
                std::launch::async,
                [&engine, other_modular_technique_same_modulus_request] {
                  return engine.evaluate_technique(
                      other_modular_technique_same_modulus_request);
                })};
      };
  const auto check_final_late_wave = [&engine](FinalLateWave& late_wave) {
    const auto cancelled_late_formula = late_wave.formula.get();
    const auto cancelled_late_range = late_wave.range.get();
    const auto cancelled_late_exact_technique = late_wave.exact_technique.get();
    const auto cancelled_late_metric_technique =
        late_wave.metric_technique.get();
    const auto cancelled_late_symbolic_technique =
        late_wave.symbolic_technique.get();
    const auto cancelled_late_modular_technique =
        late_wave.modular_technique.get();
    const auto confirmed_other_modular_technique =
        late_wave.other_modular_technique.get();
    const auto confirmed_other_modular_formula =
        late_wave.other_modular_formula.get();
    const auto confirmed_other_modular_technique_same_modulus =
        late_wave.other_modular_technique_same_modulus.get();

    CHECK(cancelled_late_formula.status == bruce::lhme::Status::Cancelled);
    CHECK(!cancelled_late_formula.integer.has_value());
    CHECK(!cancelled_late_formula.metric_bit_length.has_value());
    CHECK(!cancelled_late_formula.symbolic_expression.has_value());
    CHECK(cancelled_late_range.status == bruce::lhme::Status::Cancelled);
    CHECK(cancelled_late_range.candidates.empty());
    CHECK(cancelled_late_exact_technique.status ==
          bruce::lhme::Status::Cancelled);
    CHECK(!cancelled_late_exact_technique.integer.has_value());
    CHECK(!cancelled_late_exact_technique.metric_bit_length.has_value());
    CHECK(!cancelled_late_exact_technique.symbolic_expression.has_value());
    CHECK(cancelled_late_metric_technique.status ==
          bruce::lhme::Status::Cancelled);
    CHECK(!cancelled_late_metric_technique.integer.has_value());
    CHECK(!cancelled_late_metric_technique.metric_bit_length.has_value());
    CHECK(!cancelled_late_metric_technique.symbolic_expression.has_value());
    CHECK(cancelled_late_symbolic_technique.status ==
          bruce::lhme::Status::Cancelled);
    CHECK(!cancelled_late_symbolic_technique.integer.has_value());
    CHECK(!cancelled_late_symbolic_technique.metric_bit_length.has_value());
    CHECK(!cancelled_late_symbolic_technique.symbolic_expression.has_value());
    CHECK(cancelled_late_modular_technique.status ==
          bruce::lhme::Status::Cancelled);
    CHECK(!cancelled_late_modular_technique.integer.has_value());
    CHECK(!cancelled_late_modular_technique.metric_bit_length.has_value());
    CHECK(!cancelled_late_modular_technique.symbolic_expression.has_value());
    CHECK(confirmed_other_modular_technique.status ==
          bruce::lhme::Status::Confirmed);
    CHECK(confirmed_other_modular_technique.exact_integer() == 4);
    CHECK(!confirmed_other_modular_technique.metric_bit_length.has_value());
    CHECK(!confirmed_other_modular_technique.symbolic_expression.has_value());
    CHECK(confirmed_other_modular_formula.status ==
          bruce::lhme::Status::Confirmed);
    CHECK(confirmed_other_modular_formula.exact_integer() == 3);
    CHECK(!confirmed_other_modular_formula.metric_bit_length.has_value());
    CHECK(!confirmed_other_modular_formula.symbolic_expression.has_value());
    CHECK(confirmed_other_modular_technique_same_modulus.status ==
          bruce::lhme::Status::Confirmed);
    CHECK(confirmed_other_modular_technique_same_modulus.exact_integer() == 3);
    CHECK(!confirmed_other_modular_technique_same_modulus.metric_bit_length
               .has_value());
    CHECK(!confirmed_other_modular_technique_same_modulus.symbolic_expression
               .has_value());
    CHECK(engine.inspect().resident_entries == 0);
  };

  engine.cancel();
  auto final_late_wave = start_final_late_wave();
  check_final_late_wave(final_late_wave);

  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();

  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_modular_technique = engine.evaluate_technique(
      modular_technique_request);
  CHECK(reused_modular_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_modular_technique.exact_integer() == 11);
  CHECK(!reused_modular_technique.metric_bit_length.has_value());
  CHECK(!reused_modular_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_formula = engine.evaluate(formula_request);
  CHECK(reused_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_formula.exact_integer() == 102300);
  CHECK(!reused_formula.metric_bit_length.has_value());
  CHECK(!reused_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.summary.invalid_argument_count == 0);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_exact_technique = engine.evaluate_technique(
      exact_technique_request);
  CHECK(reused_exact_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_exact_technique.exact_integer() == 102300);
  CHECK(!reused_exact_technique.metric_bit_length.has_value());
  CHECK(!reused_exact_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 2);

  const auto reused_metric_technique = engine.evaluate_technique(
      metric_technique_request);
  CHECK(reused_metric_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_metric_technique.metric_bit_length == 17);
  CHECK(!reused_metric_technique.integer.has_value());
  CHECK(!reused_metric_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 3);

  const auto reused_symbolic_technique = engine.evaluate_technique(
      symbolic_technique_request);
  CHECK(reused_symbolic_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_symbolic_technique.symbolic_expression == "100*(2^10-1)");
  CHECK(!reused_symbolic_technique.integer.has_value());
  CHECK(!reused_symbolic_technique.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 3);
}

TEST(cpp_engine_keeps_a_post_snapshot_cancelled_modular_noncacheable_technique_group_cancelled_for_concurrent_late_cross_api_joiners_after_all_inflight_joiner_lease_endings_and_repeated_cancels) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::TechniqueRequest modular_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest exact_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest metric_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest symbolic_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_formula;
  std::future<bruce::lhme::RangeResult> pending_range;
  std::future<bruce::lhme::EvaluationResult> pending_metric_technique;
  std::future<bruce::lhme::EvaluationResult> pending_symbolic_technique;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async,
                              [&engine, modular_technique_request] {
    return engine.evaluate_technique(modular_technique_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  bruce::lhme::test_support::EngineRangeFinalizationBarrier
      range_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueFinalizationBarrier
      technique_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueNoncacheableFinalizationBarrier
      noncacheable_technique_finalization_barrier;

  pending_formula = std::async(std::launch::async, [&engine, formula_request] {
    return engine.evaluate(formula_request);
  });
  formula_finalization_barrier.wait_until_reached();

  pending_range = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  range_finalization_barrier.wait_until_reached();

  pending_metric_technique = std::async(
      std::launch::async, [&engine, metric_technique_request] {
        return engine.evaluate_technique(metric_technique_request);
      });
  technique_finalization_barrier.wait_until_reached();

  pending_symbolic_technique = std::async(
      std::launch::async, [&engine, symbolic_technique_request] {
        return engine.evaluate_technique(symbolic_technique_request);
      });
  noncacheable_technique_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_formula = pending_formula.get();

  CHECK(cancelled_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_formula.integer.has_value());
  CHECK(!cancelled_formula.metric_bit_length.has_value());
  CHECK(!cancelled_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  range_finalization_barrier.release();
  const auto cancelled_range = pending_range.get();

  CHECK(cancelled_range.status == bruce::lhme::Status::Cancelled);
  CHECK(cancelled_range.candidates.empty());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  technique_finalization_barrier.release();
  noncacheable_technique_finalization_barrier.release();
  const auto cancelled_metric_technique = pending_metric_technique.get();
  const auto cancelled_symbolic_technique = pending_symbolic_technique.get();

  CHECK(cancelled_metric_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_metric_technique.integer.has_value());
  CHECK(!cancelled_metric_technique.metric_bit_length.has_value());
  CHECK(!cancelled_metric_technique.symbolic_expression.has_value());
  CHECK(cancelled_symbolic_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_symbolic_technique.integer.has_value());
  CHECK(!cancelled_symbolic_technique.metric_bit_length.has_value());
  CHECK(!cancelled_symbolic_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  auto pending_late_formula = std::async(
      std::launch::async, [&engine, formula_request] {
        return engine.evaluate(formula_request);
      });
  auto pending_late_range = std::async(
      std::launch::async, [&engine, range_request] {
        return engine.evaluate_range(range_request, {0, 1});
      });
  auto pending_late_exact_technique = std::async(
      std::launch::async, [&engine, exact_technique_request] {
        return engine.evaluate_technique(exact_technique_request);
      });
  auto pending_late_metric_technique = std::async(
      std::launch::async, [&engine, metric_technique_request] {
        return engine.evaluate_technique(metric_technique_request);
      });
  auto pending_late_symbolic_technique = std::async(
      std::launch::async, [&engine, symbolic_technique_request] {
        return engine.evaluate_technique(symbolic_technique_request);
      });
  const auto cancelled_late_formula = pending_late_formula.get();
  const auto cancelled_late_range = pending_late_range.get();
  const auto cancelled_late_exact_technique = pending_late_exact_technique.get();
  const auto cancelled_late_metric_technique = pending_late_metric_technique.get();
  const auto cancelled_late_symbolic_technique =
      pending_late_symbolic_technique.get();

  CHECK(cancelled_late_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_late_formula.integer.has_value());
  CHECK(!cancelled_late_formula.metric_bit_length.has_value());
  CHECK(!cancelled_late_formula.symbolic_expression.has_value());
  CHECK(cancelled_late_range.status == bruce::lhme::Status::Cancelled);
  CHECK(cancelled_late_range.candidates.empty());
  CHECK(cancelled_late_exact_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_late_exact_technique.integer.has_value());
  CHECK(!cancelled_late_exact_technique.metric_bit_length.has_value());
  CHECK(!cancelled_late_exact_technique.symbolic_expression.has_value());
  CHECK(cancelled_late_metric_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_late_metric_technique.integer.has_value());
  CHECK(!cancelled_late_metric_technique.metric_bit_length.has_value());
  CHECK(!cancelled_late_metric_technique.symbolic_expression.has_value());
  CHECK(cancelled_late_symbolic_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_late_symbolic_technique.integer.has_value());
  CHECK(!cancelled_late_symbolic_technique.metric_bit_length.has_value());
  CHECK(!cancelled_late_symbolic_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();

  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_formula = engine.evaluate(formula_request);
  CHECK(reused_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_formula.exact_integer() == 102300);
  CHECK(!reused_formula.metric_bit_length.has_value());
  CHECK(!reused_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.summary.invalid_argument_count == 0);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_exact_technique = engine.evaluate_technique(
      exact_technique_request);
  CHECK(reused_exact_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_exact_technique.exact_integer() == 102300);
  CHECK(!reused_exact_technique.metric_bit_length.has_value());
  CHECK(!reused_exact_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 2);

  const auto reused_metric_technique = engine.evaluate_technique(
      metric_technique_request);
  CHECK(reused_metric_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_metric_technique.metric_bit_length == 17);
  CHECK(!reused_metric_technique.integer.has_value());
  CHECK(!reused_metric_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 3);

  const auto reused_symbolic_technique = engine.evaluate_technique(
      symbolic_technique_request);
  CHECK(reused_symbolic_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_symbolic_technique.symbolic_expression == "100*(2^10-1)");
  CHECK(!reused_symbolic_technique.integer.has_value());
  CHECK(!reused_symbolic_technique.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 3);
}

TEST(cpp_engine_keeps_a_post_snapshot_cancelled_modular_noncacheable_technique_group_cancelled_for_concurrent_late_exact_metric_and_symbolic_technique_joiners_after_all_inflight_joiner_lease_endings_and_repeated_cancels) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::TechniqueRequest modular_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest exact_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest metric_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest symbolic_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_formula;
  std::future<bruce::lhme::RangeResult> pending_range;
  std::future<bruce::lhme::EvaluationResult> pending_metric_technique;
  std::future<bruce::lhme::EvaluationResult> pending_symbolic_technique;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async,
                              [&engine, modular_technique_request] {
    return engine.evaluate_technique(modular_technique_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  bruce::lhme::test_support::EngineRangeFinalizationBarrier
      range_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueFinalizationBarrier
      technique_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueNoncacheableFinalizationBarrier
      noncacheable_technique_finalization_barrier;

  pending_formula = std::async(std::launch::async, [&engine, formula_request] {
    return engine.evaluate(formula_request);
  });
  formula_finalization_barrier.wait_until_reached();

  pending_range = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  range_finalization_barrier.wait_until_reached();

  pending_metric_technique = std::async(
      std::launch::async, [&engine, metric_technique_request] {
        return engine.evaluate_technique(metric_technique_request);
      });
  technique_finalization_barrier.wait_until_reached();

  pending_symbolic_technique = std::async(
      std::launch::async, [&engine, symbolic_technique_request] {
        return engine.evaluate_technique(symbolic_technique_request);
      });
  noncacheable_technique_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_formula = pending_formula.get();

  CHECK(cancelled_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_formula.integer.has_value());
  CHECK(!cancelled_formula.metric_bit_length.has_value());
  CHECK(!cancelled_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  range_finalization_barrier.release();
  const auto cancelled_range = pending_range.get();

  CHECK(cancelled_range.status == bruce::lhme::Status::Cancelled);
  CHECK(cancelled_range.candidates.empty());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  technique_finalization_barrier.release();
  noncacheable_technique_finalization_barrier.release();
  const auto cancelled_metric_technique = pending_metric_technique.get();
  const auto cancelled_symbolic_technique = pending_symbolic_technique.get();

  CHECK(cancelled_metric_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_metric_technique.integer.has_value());
  CHECK(!cancelled_metric_technique.metric_bit_length.has_value());
  CHECK(!cancelled_metric_technique.symbolic_expression.has_value());
  CHECK(cancelled_symbolic_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_symbolic_technique.integer.has_value());
  CHECK(!cancelled_symbolic_technique.metric_bit_length.has_value());
  CHECK(!cancelled_symbolic_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  auto pending_late_exact_technique = std::async(
      std::launch::async, [&engine, exact_technique_request] {
        return engine.evaluate_technique(exact_technique_request);
      });
  auto pending_late_metric_technique = std::async(
      std::launch::async, [&engine, metric_technique_request] {
        return engine.evaluate_technique(metric_technique_request);
      });
  auto pending_late_symbolic_technique = std::async(
      std::launch::async, [&engine, symbolic_technique_request] {
        return engine.evaluate_technique(symbolic_technique_request);
      });
  const auto cancelled_late_exact_technique = pending_late_exact_technique.get();
  const auto cancelled_late_metric_technique = pending_late_metric_technique.get();
  const auto cancelled_late_symbolic_technique =
      pending_late_symbolic_technique.get();

  CHECK(cancelled_late_exact_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_late_exact_technique.integer.has_value());
  CHECK(!cancelled_late_exact_technique.metric_bit_length.has_value());
  CHECK(!cancelled_late_exact_technique.symbolic_expression.has_value());
  CHECK(cancelled_late_metric_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_late_metric_technique.integer.has_value());
  CHECK(!cancelled_late_metric_technique.metric_bit_length.has_value());
  CHECK(!cancelled_late_metric_technique.symbolic_expression.has_value());
  CHECK(cancelled_late_symbolic_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_late_symbolic_technique.integer.has_value());
  CHECK(!cancelled_late_symbolic_technique.metric_bit_length.has_value());
  CHECK(!cancelled_late_symbolic_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();

  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_formula = engine.evaluate(formula_request);
  CHECK(reused_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_formula.exact_integer() == 102300);
  CHECK(!reused_formula.metric_bit_length.has_value());
  CHECK(!reused_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.summary.invalid_argument_count == 0);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_exact_technique = engine.evaluate_technique(
      exact_technique_request);
  CHECK(reused_exact_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_exact_technique.exact_integer() == 102300);
  CHECK(!reused_exact_technique.metric_bit_length.has_value());
  CHECK(!reused_exact_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 2);

  const auto reused_metric_technique = engine.evaluate_technique(
      metric_technique_request);
  CHECK(reused_metric_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_metric_technique.metric_bit_length == 17);
  CHECK(!reused_metric_technique.integer.has_value());
  CHECK(!reused_metric_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 3);

  const auto reused_symbolic_technique = engine.evaluate_technique(
      symbolic_technique_request);
  CHECK(reused_symbolic_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_symbolic_technique.symbolic_expression == "100*(2^10-1)");
  CHECK(!reused_symbolic_technique.integer.has_value());
  CHECK(!reused_symbolic_technique.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 3);
}

TEST(cpp_engine_keeps_a_post_snapshot_cancelled_modular_noncacheable_technique_group_cancelled_for_a_late_symbolic_technique_joiner_after_all_inflight_joiner_lease_endings_and_repeated_cancels) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::TechniqueRequest modular_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest exact_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest metric_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest symbolic_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_formula;
  std::future<bruce::lhme::RangeResult> pending_range;
  std::future<bruce::lhme::EvaluationResult> pending_metric_technique;
  std::future<bruce::lhme::EvaluationResult> pending_symbolic_technique;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async,
                              [&engine, modular_technique_request] {
    return engine.evaluate_technique(modular_technique_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  bruce::lhme::test_support::EngineRangeFinalizationBarrier
      range_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueFinalizationBarrier
      technique_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueNoncacheableFinalizationBarrier
      noncacheable_technique_finalization_barrier;

  pending_formula = std::async(std::launch::async, [&engine, formula_request] {
    return engine.evaluate(formula_request);
  });
  formula_finalization_barrier.wait_until_reached();

  pending_range = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  range_finalization_barrier.wait_until_reached();

  pending_metric_technique = std::async(
      std::launch::async, [&engine, metric_technique_request] {
        return engine.evaluate_technique(metric_technique_request);
      });
  technique_finalization_barrier.wait_until_reached();

  pending_symbolic_technique = std::async(
      std::launch::async, [&engine, symbolic_technique_request] {
        return engine.evaluate_technique(symbolic_technique_request);
      });
  noncacheable_technique_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_formula = pending_formula.get();

  CHECK(cancelled_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_formula.integer.has_value());
  CHECK(!cancelled_formula.metric_bit_length.has_value());
  CHECK(!cancelled_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  range_finalization_barrier.release();
  const auto cancelled_range = pending_range.get();

  CHECK(cancelled_range.status == bruce::lhme::Status::Cancelled);
  CHECK(cancelled_range.candidates.empty());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  technique_finalization_barrier.release();
  noncacheable_technique_finalization_barrier.release();
  const auto cancelled_metric_technique = pending_metric_technique.get();
  const auto cancelled_symbolic_technique = pending_symbolic_technique.get();

  CHECK(cancelled_metric_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_metric_technique.integer.has_value());
  CHECK(!cancelled_metric_technique.metric_bit_length.has_value());
  CHECK(!cancelled_metric_technique.symbolic_expression.has_value());
  CHECK(cancelled_symbolic_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_symbolic_technique.integer.has_value());
  CHECK(!cancelled_symbolic_technique.metric_bit_length.has_value());
  CHECK(!cancelled_symbolic_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  const auto cancelled_late_symbolic_technique = engine.evaluate_technique(
      symbolic_technique_request);

  CHECK(cancelled_late_symbolic_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_late_symbolic_technique.integer.has_value());
  CHECK(!cancelled_late_symbolic_technique.metric_bit_length.has_value());
  CHECK(!cancelled_late_symbolic_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();

  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_formula = engine.evaluate(formula_request);
  CHECK(reused_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_formula.exact_integer() == 102300);
  CHECK(!reused_formula.metric_bit_length.has_value());
  CHECK(!reused_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.summary.invalid_argument_count == 0);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_exact_technique = engine.evaluate_technique(
      exact_technique_request);
  CHECK(reused_exact_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_exact_technique.exact_integer() == 102300);
  CHECK(!reused_exact_technique.metric_bit_length.has_value());
  CHECK(!reused_exact_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 2);

  const auto reused_metric_technique = engine.evaluate_technique(
      metric_technique_request);
  CHECK(reused_metric_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_metric_technique.metric_bit_length == 17);
  CHECK(!reused_metric_technique.integer.has_value());
  CHECK(!reused_metric_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 3);

  const auto reused_symbolic_technique = engine.evaluate_technique(
      symbolic_technique_request);
  CHECK(reused_symbolic_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_symbolic_technique.symbolic_expression == "100*(2^10-1)");
  CHECK(!reused_symbolic_technique.integer.has_value());
  CHECK(!reused_symbolic_technique.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 3);
}

TEST(cpp_engine_keeps_a_post_snapshot_cancelled_modular_noncacheable_technique_group_cancelled_for_a_late_metric_technique_joiner_after_all_inflight_joiner_lease_endings_and_repeated_cancels) {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8,
       .cache_max_bytes = 4096,
       .worker_count = 1,
       .tile_size = 2});
  const auto kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, 1));
  const bruce::lhme::TechniqueRequest modular_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Modular,
      .budget = bruce::lhme::Budget::unlimited(),
      .modulus = 17};
  const bruce::lhme::EvaluationRequest formula_request{
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::EvaluationRequest range_request{
      .spec = formula,
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest exact_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest metric_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Metric,
      .budget = bruce::lhme::Budget::unlimited()};
  const bruce::lhme::TechniqueRequest symbolic_technique_request{
      .identity = kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Symbolic,
      .budget = bruce::lhme::Budget::unlimited()};

  std::future<bruce::lhme::EvaluationResult> pending_origin;
  std::future<bruce::lhme::EvaluationResult> pending_formula;
  std::future<bruce::lhme::RangeResult> pending_range;
  std::future<bruce::lhme::EvaluationResult> pending_metric_technique;
  std::future<bruce::lhme::EvaluationResult> pending_symbolic_technique;
  bruce::lhme::test_support::EngineTechniqueNoncacheablePostFinalizationBarrier
      origin_post_finalization_barrier{
          bruce::lhme::test_support::
              EngineTechniqueNoncacheablePostFinalizationBarrierMode::
                  FirstArrivalOnly};
  pending_origin = std::async(std::launch::async,
                              [&engine, modular_technique_request] {
    return engine.evaluate_technique(modular_technique_request);
  });
  origin_post_finalization_barrier.wait_until_reached();

  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      formula_finalization_barrier;
  bruce::lhme::test_support::EngineRangeFinalizationBarrier
      range_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueFinalizationBarrier
      technique_finalization_barrier;
  bruce::lhme::test_support::EngineTechniqueNoncacheableFinalizationBarrier
      noncacheable_technique_finalization_barrier;

  pending_formula = std::async(std::launch::async, [&engine, formula_request] {
    return engine.evaluate(formula_request);
  });
  formula_finalization_barrier.wait_until_reached();

  pending_range = std::async(std::launch::async, [&engine, range_request] {
    return engine.evaluate_range(range_request, {0, 1});
  });
  range_finalization_barrier.wait_until_reached();

  pending_metric_technique = std::async(
      std::launch::async, [&engine, metric_technique_request] {
        return engine.evaluate_technique(metric_technique_request);
      });
  technique_finalization_barrier.wait_until_reached();

  pending_symbolic_technique = std::async(
      std::launch::async, [&engine, symbolic_technique_request] {
        return engine.evaluate_technique(symbolic_technique_request);
      });
  noncacheable_technique_finalization_barrier.wait_until_reached();

  engine.cancel();
  formula_finalization_barrier.release();
  const auto cancelled_formula = pending_formula.get();

  CHECK(cancelled_formula.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_formula.integer.has_value());
  CHECK(!cancelled_formula.metric_bit_length.has_value());
  CHECK(!cancelled_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  range_finalization_barrier.release();
  const auto cancelled_range = pending_range.get();

  CHECK(cancelled_range.status == bruce::lhme::Status::Cancelled);
  CHECK(cancelled_range.candidates.empty());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  technique_finalization_barrier.release();
  noncacheable_technique_finalization_barrier.release();
  const auto cancelled_metric_technique = pending_metric_technique.get();
  const auto cancelled_symbolic_technique = pending_symbolic_technique.get();

  CHECK(cancelled_metric_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_metric_technique.integer.has_value());
  CHECK(!cancelled_metric_technique.metric_bit_length.has_value());
  CHECK(!cancelled_metric_technique.symbolic_expression.has_value());
  CHECK(cancelled_symbolic_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_symbolic_technique.integer.has_value());
  CHECK(!cancelled_symbolic_technique.metric_bit_length.has_value());
  CHECK(!cancelled_symbolic_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  engine.cancel();
  const auto cancelled_late_metric_technique = engine.evaluate_technique(
      metric_technique_request);

  CHECK(cancelled_late_metric_technique.status == bruce::lhme::Status::Cancelled);
  CHECK(!cancelled_late_metric_technique.integer.has_value());
  CHECK(!cancelled_late_metric_technique.metric_bit_length.has_value());
  CHECK(!cancelled_late_metric_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  origin_post_finalization_barrier.release();
  const auto finalized_origin = pending_origin.get();

  CHECK(finalized_origin.status == bruce::lhme::Status::Confirmed);
  CHECK(finalized_origin.exact_integer() == 11);
  CHECK(!finalized_origin.metric_bit_length.has_value());
  CHECK(!finalized_origin.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 0);

  const auto reused_formula = engine.evaluate(formula_request);
  CHECK(reused_formula.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_formula.exact_integer() == 102300);
  CHECK(!reused_formula.metric_bit_length.has_value());
  CHECK(!reused_formula.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_range = engine.evaluate_range(range_request, {0, 1});
  CHECK(reused_range.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_range.summary.checked_count == 2);
  CHECK(reused_range.summary.confirmed_count == 2);
  CHECK(reused_range.summary.invalid_argument_count == 0);
  CHECK(reused_range.candidates.size() == 2);
  CHECK(reused_range.candidates[0].exact_integer() == 0);
  CHECK(reused_range.candidates[1].exact_integer() == 100);
  CHECK(engine.inspect().resident_entries == 1);

  const auto reused_exact_technique = engine.evaluate_technique(
      exact_technique_request);
  CHECK(reused_exact_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_exact_technique.exact_integer() == 102300);
  CHECK(!reused_exact_technique.metric_bit_length.has_value());
  CHECK(!reused_exact_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 2);

  const auto reused_metric_technique = engine.evaluate_technique(
      metric_technique_request);
  CHECK(reused_metric_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_metric_technique.metric_bit_length == 17);
  CHECK(!reused_metric_technique.integer.has_value());
  CHECK(!reused_metric_technique.symbolic_expression.has_value());
  CHECK(engine.inspect().resident_entries == 3);

  const auto reused_symbolic_technique = engine.evaluate_technique(
      symbolic_technique_request);
  CHECK(reused_symbolic_technique.status == bruce::lhme::Status::Confirmed);
  CHECK(reused_symbolic_technique.symbolic_expression == "100*(2^10-1)");
  CHECK(!reused_symbolic_technique.integer.has_value());
  CHECK(!reused_symbolic_technique.metric_bit_length.has_value());
  CHECK(engine.inspect().resident_entries == 3);
}
