#include "test.hpp"

#include "engine_technique_cache_hit_finalization_test_barrier.hpp"

#include "bruce/lhme/engine.hpp"
#include "bruce/lhme/kernel_sdk.hpp"
#include "bruce/lhme/power_offset_kernel.hpp"
#include "bruce/lhme/trace.hpp"

#include <atomic>
#include <barrier>
#include <future>
#include <memory>
#include <string>
#include <string_view>

namespace {

using bruce::lhme::Budget;
using bruce::lhme::Engine;
using bruce::lhme::EngineConfig;
using bruce::lhme::EvaluationMode;
using bruce::lhme::EvaluationRequest;
using bruce::lhme::EvaluationResult;
using bruce::lhme::ExactFormulaTechniqueCacheBinding;
using bruce::lhme::FormulaSpec;
using bruce::lhme::GateScope;
using bruce::lhme::Status;
using bruce::lhme::TechniqueIdentity;
using bruce::lhme::TechniqueKernel;
using bruce::lhme::TechniqueRequest;

constexpr std::string_view kParameters = "formula=1:1";

FormulaSpec formula(std::uint32_t version = 1) {
  return bruce::lhme::make_power_offset_spec(2, 100, version);
}

EvaluationRequest formula_request(const FormulaSpec& specification,
                                  std::int64_t index = 10,
                                  Budget budget = Budget::unlimited()) {
  return {.spec = specification,
          .address = {.index = index},
          .mode = EvaluationMode::Exact,
          .budget = budget};
}

class EquivalentFormulaKernel final : public TechniqueKernel {
 public:
  EquivalentFormulaKernel(TechniqueIdentity identity, FormulaSpec formula,
                          bool claims_formula_computation = false)
      : identity_(std::move(identity)), formula_(std::move(formula)),
        claims_formula_computation_(claims_formula_computation) {}

  [[nodiscard]] TechniqueIdentity identity() const override { return identity_; }

  [[nodiscard]] std::optional<bruce::lhme::ComputationIdentity>
  canonical_computation_identity(const TechniqueRequest&) const override {
    if (!claims_formula_computation_) return std::nullopt;
    return bruce::lhme::canonical_computation_identity(formula_);
  }

  [[nodiscard]] EvaluationResult evaluate(
      const TechniqueRequest& request) const override {
    ++calls;
    if (request.identity != identity_ || request.parameters != kParameters ||
        (request.mode != EvaluationMode::Exact &&
         request.mode != EvaluationMode::Metric)) {
      return bruce::lhme::invalid_technique_request_result(
          request.mode, "unexpected equivalent technique request");
    }
    auto result = bruce::lhme::evaluate({.spec = formula_,
                                         .address = request.address,
                                         .mode = request.mode,
                                         .budget = request.budget,
                                         .modulus = request.modulus,
                                         .cancellation_signal =
                                             request.cancellation_signal});
    if (result.status == Status::Confirmed) {
      result.trace = bruce::lhme::make_trace({{
          .scope = GateScope::Publish,
          .status = Status::Confirmed,
          .reason = "equivalent technique producer",
      }});
    }
    return result;
  }

  mutable std::atomic_uint calls{0};

 private:
  TechniqueIdentity identity_;
  FormulaSpec formula_;
  bool claims_formula_computation_ = false;
};

TechniqueRequest technique_request(const TechniqueIdentity& identity,
                                   std::int64_t index = 10,
                                   Budget budget = Budget::unlimited(),
                                   std::string parameters = std::string{kParameters}) {
  return {.identity = identity,
          .parameters = std::move(parameters),
          .address = {.index = index},
          .mode = EvaluationMode::Exact,
          .budget = budget};
}

ExactFormulaTechniqueCacheBinding binding_for(const TechniqueIdentity& identity,
                                              const FormulaSpec& specification) {
  return {.technique = identity,
          .accepted_parameters = std::string{kParameters},
          .formula = specification};
}

Engine configured_engine() {
  return Engine::create({.cache_max_entries = 32, .cache_max_bytes = 1U << 20U});
}

void check_neutral_cross_api_trace(const EvaluationResult& result) {
  CHECK(result.status == Status::Confirmed);
  CHECK(result.trace != nullptr);
  CHECK(result.trace->events.size() == 1);
  CHECK(result.trace->events.front().scope == GateScope::Cache);
  CHECK(result.trace->events.front().status == Status::Confirmed);
  CHECK(result.trace->events.front().reason == "cross-api exact cache hit");
}

}  // namespace

TEST(power_offset_exact_cache_binding_helper_uses_canonical_power_offset_values) {
  const auto binding = bruce::lhme::make_power_offset_exact_cache_binding(2, 100, 1);
  const bruce::lhme::PowerOffsetKernel kernel;

  CHECK(binding.technique == kernel.identity());
  CHECK(binding.accepted_parameters ==
        bruce::lhme::encode_power_offset_parameters(2, 100));
  CHECK(binding.formula.version == 1);
  CHECK(binding.formula.base == 2);
  CHECK(binding.formula.scale == 100);

  bool rejected_mismatched_version = false;
  try {
    static_cast<void>(
        bruce::lhme::make_power_offset_exact_cache_binding(2, 100, 2));
  } catch (const std::invalid_argument&) {
    rejected_mismatched_version = true;
  }
  CHECK(rejected_mismatched_version);

  auto engine = configured_engine();
  const auto specification = engine.register_formula(formula());
  const auto registered_kernel = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  CHECK(engine.register_technique(registered_kernel));
  CHECK(engine.register_exact_formula_cache_binding(
      bruce::lhme::make_power_offset_exact_cache_binding(2, 100, 1)));
  const auto formula_result = engine.evaluate(formula_request(specification));
  const auto technique_result = engine.evaluate_technique({
      .identity = registered_kernel->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = EvaluationMode::Exact,
      .budget = Budget::unlimited(),
  });
  CHECK(technique_result.exact_integer() == formula_result.exact_integer());
  check_neutral_cross_api_trace(technique_result);
}

TEST(exact_cache_binding_reuses_a_formula_entry_for_a_technique_request) {
  auto engine = configured_engine();
  const auto specification = engine.register_formula(formula());
  const auto kernel = std::make_shared<EquivalentFormulaKernel>(
      TechniqueIdentity{.id = "test.bound_formula", .version = 1}, specification);
  CHECK(engine.register_technique(kernel));
  CHECK(engine.register_exact_formula_cache_binding(
      binding_for(kernel->identity(), specification)));

  const auto formula_result = engine.evaluate(formula_request(specification));
  CHECK(formula_result.status == Status::Confirmed);
  const auto first = engine.evaluate_technique(technique_request(kernel->identity()));
  const auto second = engine.evaluate_technique(technique_request(kernel->identity()));

  CHECK(kernel->calls.load() == 0);
  CHECK(first.exact_integer() == formula_result.exact_integer());
  check_neutral_cross_api_trace(first);
  check_neutral_cross_api_trace(second);
  CHECK(first.trace->id == second.trace->id);
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(exact_cache_binding_reuses_a_technique_entry_for_a_formula_request) {
  auto engine = configured_engine();
  const auto specification = engine.register_formula(formula());
  const auto kernel = std::make_shared<EquivalentFormulaKernel>(
      TechniqueIdentity{.id = "test.technique_formula", .version = 1}, specification);
  CHECK(engine.register_technique(kernel));
  CHECK(engine.register_exact_formula_cache_binding(
      binding_for(kernel->identity(), specification)));

  const auto technique = engine.evaluate_technique(technique_request(kernel->identity()));
  const auto formula_hit = engine.evaluate(formula_request(specification));

  CHECK(technique.status == Status::Confirmed);
  CHECK(kernel->calls.load() == 1);
  CHECK(formula_hit.exact_integer() == technique.exact_integer());
  check_neutral_cross_api_trace(formula_hit);
  CHECK(engine.inspect().resident_entries == 1);
}

TEST(non_opt_in_techniques_keep_their_independent_cache_key) {
  auto engine = configured_engine();
  const auto specification = engine.register_formula(formula());
  const auto kernel = std::make_shared<EquivalentFormulaKernel>(
      TechniqueIdentity{.id = "test.not_bound", .version = 1}, specification);
  CHECK(engine.register_technique(kernel));

  CHECK(engine.evaluate(formula_request(specification)).status == Status::Confirmed);
  const auto first = engine.evaluate_technique(technique_request(kernel->identity()));
  const auto second = engine.evaluate_technique(technique_request(kernel->identity()));

  CHECK(first.status == Status::Confirmed);
  CHECK(second.status == Status::Confirmed);
  CHECK(kernel->calls.load() == 1);
  CHECK(second.trace->events.back().reason == "cache hit");
  CHECK(engine.inspect().resident_entries == 2);
}

TEST(exact_cache_bindings_reject_invalid_unregistered_and_conflicting_declarations) {
  auto engine = configured_engine();
  const auto specification = engine.register_formula(formula());
  const auto next_specification = engine.register_formula(formula(2));
  const auto kernel = std::make_shared<EquivalentFormulaKernel>(
      TechniqueIdentity{.id = "test.binding_validation", .version = 1}, specification);

  CHECK(!engine.register_exact_formula_cache_binding(
      binding_for(kernel->identity(), specification)));
  CHECK(engine.register_technique(kernel));
  CHECK(!engine.register_exact_formula_cache_binding({
      .technique = kernel->identity(), .accepted_parameters = "", .formula = specification}));
  CHECK(!engine.register_exact_formula_cache_binding({
      .technique = {},
      .accepted_parameters = std::string{kParameters},
      .formula = specification}));
  CHECK(!engine.register_exact_formula_cache_binding({
      .technique = kernel->identity(),
      .accepted_parameters = std::string(4097, 'p'),
      .formula = specification}));
  CHECK(!engine.register_exact_formula_cache_binding({
      .technique = kernel->identity(),
      .accepted_parameters = std::string{kParameters},
      .formula = formula(3)}));
  CHECK(engine.register_exact_formula_cache_binding(
      binding_for(kernel->identity(), specification)));
  CHECK(engine.register_exact_formula_cache_binding(
      binding_for(kernel->identity(), specification)));
  CHECK(!engine.register_exact_formula_cache_binding(
      binding_for(kernel->identity(), next_specification)));
}

TEST(exact_cache_bindings_separate_parameters_versions_formulae_and_indices) {
  auto engine = configured_engine();
  const auto first_formula = engine.register_formula(formula(1));
  const auto second_formula = engine.register_formula(formula(2));
  const auto first_kernel = std::make_shared<EquivalentFormulaKernel>(
      TechniqueIdentity{.id = "test.binding_separation", .version = 1}, first_formula);
  const auto second_kernel = std::make_shared<EquivalentFormulaKernel>(
      TechniqueIdentity{.id = "test.binding_separation", .version = 2}, first_formula);
  CHECK(engine.register_technique(first_kernel));
  CHECK(engine.register_technique(second_kernel));
  CHECK(engine.register_exact_formula_cache_binding(
      binding_for(first_kernel->identity(), first_formula)));

  CHECK(engine.evaluate(formula_request(first_formula, 10)).status == Status::Confirmed);
  CHECK(engine.evaluate(formula_request(second_formula, 10)).status == Status::Confirmed);
  CHECK(engine.evaluate_technique(technique_request(first_kernel->identity(), 11)).status ==
        Status::Confirmed);
  CHECK(engine.evaluate_technique(technique_request(
            first_kernel->identity(), 10, Budget::unlimited(), "different"))
            .status == Status::InvalidArgument);
  CHECK(engine.evaluate_technique(technique_request(second_kernel->identity())).status ==
        Status::Confirmed);

  CHECK(first_kernel->calls.load() == 2);
  CHECK(second_kernel->calls.load() == 1);
  CHECK(engine.inspect().resident_entries == 4);
}

TEST(computation_identity_cannot_authorize_a_foreign_cache_hit) {
  auto engine = configured_engine();
  const auto specification = engine.register_formula(formula());
  const auto kernel = std::make_shared<EquivalentFormulaKernel>(
      TechniqueIdentity{.id = "test.identity_only", .version = 1}, specification, true);
  CHECK(engine.register_technique(kernel));

  CHECK(engine.evaluate(formula_request(specification)).status == Status::Confirmed);
  const auto technique = engine.evaluate_technique(technique_request(kernel->identity()));

  CHECK(technique.status == Status::Confirmed);
  CHECK(kernel->calls.load() == 1);
  CHECK(engine.inspect().resident_entries == 2);
}

TEST(exact_cache_binding_rechecks_the_callers_budget_on_a_cross_api_hit) {
  auto engine = configured_engine();
  const auto specification = engine.register_formula(formula());
  const auto kernel = std::make_shared<EquivalentFormulaKernel>(
      TechniqueIdentity{.id = "test.binding_budget", .version = 1}, specification);
  CHECK(engine.register_technique(kernel));
  CHECK(engine.register_exact_formula_cache_binding(
      binding_for(kernel->identity(), specification)));
  CHECK(engine.evaluate(formula_request(specification, 10)).status == Status::Confirmed);

  Budget small_budget = Budget::unlimited();
  small_budget.max_bit_length = 1;
  const auto result = engine.evaluate_technique(
      technique_request(kernel->identity(), 10, small_budget));

  CHECK(result.status == Status::BudgetExceeded);
  CHECK(kernel->calls.load() == 0);
  CHECK(result.trace->events.back().reason == "cache rejected by caller budget");
}

TEST(exact_cache_binding_does_not_share_metric_cache_entries) {
  auto engine = configured_engine();
  const auto specification = engine.register_formula(formula());
  const auto kernel = std::make_shared<EquivalentFormulaKernel>(
      TechniqueIdentity{.id = "test.binding_metric", .version = 1}, specification);
  CHECK(engine.register_technique(kernel));
  CHECK(engine.register_exact_formula_cache_binding(
      binding_for(kernel->identity(), specification)));
  const auto formula_metric = engine.evaluate({.spec = specification,
                                                .address = {.index = 10},
                                                .mode = EvaluationMode::Metric,
                                                .budget = Budget::unlimited()});
  const auto technique_metric = engine.evaluate_technique({
      .identity = kernel->identity(),
      .parameters = std::string{kParameters},
      .address = {.index = 10},
      .mode = EvaluationMode::Metric,
      .budget = Budget::unlimited(),
  });

  CHECK(formula_metric.status == Status::Confirmed);
  CHECK(technique_metric.status == Status::Confirmed);
  CHECK(kernel->calls.load() == 1);
  CHECK(engine.inspect().resident_entries == 2);
}

TEST(cancellation_of_a_cross_api_cache_hit_keeps_the_cache_usable) {
  auto engine = configured_engine();
  const auto specification = engine.register_formula(formula());
  const auto kernel = std::make_shared<EquivalentFormulaKernel>(
      TechniqueIdentity{.id = "test.binding_cancel", .version = 1}, specification);
  CHECK(engine.register_technique(kernel));
  CHECK(engine.register_exact_formula_cache_binding(
      binding_for(kernel->identity(), specification)));
  CHECK(engine.evaluate(formula_request(specification)).status == Status::Confirmed);

  bruce::lhme::test_support::EngineTechniqueCacheHitFinalizationBarrier barrier;
  auto pending = std::async(std::launch::async, [&] {
    return engine.evaluate_technique(technique_request(kernel->identity()));
  });
  barrier.wait_until_reached();
  engine.cancel();
  barrier.release();

  const auto cancelled = pending.get();
  const auto reused = engine.evaluate_technique(technique_request(kernel->identity()));
  CHECK(cancelled.status == Status::Cancelled);
  CHECK(cancelled.trace->events.front().reason == "cross-api exact cache hit");
  CHECK(reused.status == Status::Confirmed);
  check_neutral_cross_api_trace(reused);
  CHECK(kernel->calls.load() == 0);
}

TEST(simultaneous_formula_and_bound_technique_misses_remain_correct) {
  for (int iteration = 0; iteration < 16; ++iteration) {
    auto engine = configured_engine();
    const auto specification = engine.register_formula(formula());
    const auto kernel = std::make_shared<EquivalentFormulaKernel>(
        TechniqueIdentity{.id = "test.binding_race", .version = 1}, specification);
    CHECK(engine.register_technique(kernel));
    CHECK(engine.register_exact_formula_cache_binding(
        binding_for(kernel->identity(), specification)));

    std::barrier start{2};
    auto formula_future = std::async(std::launch::async, [&] {
      start.arrive_and_wait();
      return engine.evaluate(formula_request(specification));
    });
    auto technique_future = std::async(std::launch::async, [&] {
      start.arrive_and_wait();
      return engine.evaluate_technique(technique_request(kernel->identity()));
    });
    const auto formula_result = formula_future.get();
    const auto technique_result = technique_future.get();

    CHECK(formula_result.status == Status::Confirmed);
    CHECK(technique_result.status == Status::Confirmed);
    CHECK(formula_result.exact_integer() == technique_result.exact_integer());
    CHECK(engine.inspect().resident_entries == 1);
  }
}
