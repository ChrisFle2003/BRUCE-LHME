#include "test.hpp"

#include "bruce/lhme/formula_override_store.hpp"
#include "bruce/lhme/power_offset_kernel.hpp"
#include "bruce/lhme/trace.hpp"
#include "engine_formula_finalization_test_barrier.hpp"
#include "engine_technique_cache_hit_finalization_test_barrier.hpp"
#include "engine_technique_finalization_test_barrier.hpp"

#include <atomic>
#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace bruce::lhme;

FormulaBinding power_binding(Engine& engine) {
  return engine.register_formula(make_power_offset_spec(2, 100, 1));
}

FormulaStreamingRequest streaming_request(const FormulaBinding& binding,
                                          Budget budget = Budget::unlimited()) {
  return {.binding = binding, .budget = budget};
}

StreamingOptions modular_options(std::uint32_t worker_count = 1) {
  return {.worker_count = worker_count,
          .tile_size = 4,
          .retention_mode = RetentionMode::TopK,
          .max_retained_candidates = 4,
          .retention_memory_bytes = 4096,
          .filter = {.mode = FilterMode::ModularEquals,
                     .modulus = 17,
                     .expected_residue = 11}};
}

std::vector<std::int64_t> addresses(
    const FormulaStreamingRangeResult& result) {
  std::vector<std::int64_t> output;
  for (const auto& candidate : result.retained_candidates) {
    output.push_back(candidate.address.index);
  }
  return output;
}

std::vector<FormulaValueSource> sources(
    const FormulaStreamingRangeResult& result) {
  std::vector<FormulaValueSource> output;
  for (const auto& candidate : result.retained_candidates) {
    output.push_back(candidate.value.source);
  }
  return output;
}

std::vector<BigInt> exact_values(const FormulaStreamingRangeResult& result) {
  std::vector<BigInt> output;
  for (const auto& candidate : result.retained_candidates) {
    output.push_back(candidate.value.result.exact_integer());
  }
  return output;
}

std::uint64_t retained_result_size(const EvaluationResult& result) {
  std::uint64_t bytes = sizeof(EvaluationResult);
  if (result.integer.has_value()) {
    bytes += result.integer->convert_to<std::string>().size();
  }
  if (result.trace != nullptr) {
    for (const auto& event : result.trace->events) bytes += event.reason.size();
  }
  return bytes;
}

}  // namespace

TEST(formula_override_streaming_keeps_addresses_values_and_sources) {
  auto engine = Engine::create({});
  const auto binding = power_binding(engine);
  FormulaOverrideStore store{engine};
  CHECK(store.set_override(binding, {.index = 10}, BigInt{-40}) ==
        Status::Confirmed);

  const auto before = store.inspect().computed_exact_evaluation_count;
  const auto result = store.evaluate_streaming_range(
      streaming_request(binding), {0, 30}, modular_options());
  const auto after = store.inspect().computed_exact_evaluation_count;

  CHECK(result.status == Status::Confirmed);
  CHECK(result.summary.checked_count == 31);
  CHECK(result.summary.confirmed_count == 4);
  CHECK(result.summary.invalid_argument_count == 0);
  CHECK(result.filtered_out_count == 27);
  CHECK(result.exact_evaluation_count == 4);
  CHECK(result.effective_retention_mode == RetentionMode::TopK);
  CHECK(addresses(result) == std::vector<std::int64_t>({2, 10, 18, 26}));
  CHECK(sources(result) == std::vector<FormulaValueSource>(
                               {FormulaValueSource::Computed,
                                FormulaValueSource::Override,
                                FormulaValueSource::Computed,
                                FormulaValueSource::Computed}));
  CHECK(exact_values(result) ==
        std::vector<BigInt>({BigInt{300}, BigInt{-40}, BigInt{26214300},
                             BigInt{6710886300LL}}));
  CHECK(after - before == 3);
}

TEST(formula_override_streaming_filters_on_the_effective_override_value) {
  auto engine = Engine::create({});
  const auto binding = power_binding(engine);
  FormulaOverrideStore store{engine};
  CHECK(store.set_override(binding, {.index = 10}, BigInt{-42}) ==
        Status::Confirmed);

  auto options = modular_options();
  options.filter.expected_residue = 9;
  const auto result = store.evaluate_streaming_range(
      streaming_request(binding), {0, 10}, options);

  CHECK(result.status == Status::Confirmed);
  CHECK(result.filtered_out_count == 10);
  CHECK(result.exact_evaluation_count == 1);
  CHECK(result.summary.confirmed_count == 1);
  CHECK(addresses(result) == std::vector<std::int64_t>({10}));
  CHECK(result.retained_candidates.front().value.source ==
        FormulaValueSource::Override);
  CHECK(result.retained_candidates.front().value.result.exact_integer() == -42);
}

TEST(formula_override_streaming_retains_filtered_candidates_with_mixed_sources) {
  auto engine = Engine::create({});
  const auto binding = power_binding(engine);
  FormulaOverrideStore store{engine};
  CHECK(store.set_override(binding, {.index = 10}, BigInt{-40}) ==
        Status::Confirmed);

  auto options = modular_options();
  options.retention_mode = RetentionMode::FilteredCandidates;
  const auto result = store.evaluate_streaming_range(
      streaming_request(binding), {0, 30}, options);

  CHECK(result.status == Status::Confirmed);
  CHECK(result.effective_retention_mode == RetentionMode::FilteredCandidates);
  CHECK(addresses(result) == std::vector<std::int64_t>({2, 10, 18, 26}));
  CHECK(sources(result) == std::vector<FormulaValueSource>(
                               {FormulaValueSource::Computed,
                                FormulaValueSource::Override,
                                FormulaValueSource::Computed,
                                FormulaValueSource::Computed}));
}

TEST(formula_override_streaming_adaptive_transitions_to_top_k_with_mixed_sources) {
  auto engine = Engine::create({});
  const auto binding = power_binding(engine);
  FormulaOverrideStore store{engine};
  CHECK(store.set_override(binding, {.index = 10}, BigInt{-40}) ==
        Status::Confirmed);

  auto options = modular_options();
  options.retention_mode = RetentionMode::Adaptive;
  options.max_retained_candidates = 3;
  const auto result = store.evaluate_streaming_range(
      streaming_request(binding), {0, 30}, options);

  CHECK(result.status == Status::Confirmed);
  CHECK(result.effective_retention_mode == RetentionMode::TopK);
  CHECK(addresses(result) == std::vector<std::int64_t>({2, 10, 18}));
  CHECK(sources(result) == std::vector<FormulaValueSource>(
                               {FormulaValueSource::Computed,
                                FormulaValueSource::Override,
                                FormulaValueSource::Computed}));
}

TEST(formula_override_streaming_zero_retention_preserves_summary) {
  auto engine = Engine::create({});
  const auto binding = power_binding(engine);
  FormulaOverrideStore store{engine};
  auto options = modular_options();
  options.filter = {};
  options.max_retained_candidates = 0;

  const auto result = store.evaluate_streaming_range(
      streaming_request(binding), {0, 5}, options);

  CHECK(result.status == Status::Confirmed);
  CHECK(result.effective_retention_mode == RetentionMode::SummaryOnly);
  CHECK(result.retained_candidates.empty());
  CHECK(result.summary.checked_count == 6);
  CHECK(result.summary.confirmed_count == 6);
  CHECK(result.exact_evaluation_count == 6);
}

TEST(formula_override_streaming_summarizes_candidates_that_exceed_memory_limit) {
  auto engine = Engine::create({});
  const auto binding = power_binding(engine);
  FormulaOverrideStore store{engine};
  auto options = modular_options();
  options.filter = {};
  options.max_retained_candidates = 16;
  options.retention_memory_bytes = 1;

  const auto result = store.evaluate_streaming_range(
      streaming_request(binding), {20, 20}, options);

  CHECK(result.status == Status::Confirmed);
  CHECK(result.summary.checked_count == 1);
  CHECK(result.summary.confirmed_count == 1);
  CHECK(result.exact_evaluation_count == 1);
  CHECK(result.retained_candidates.empty());
}

TEST(formula_override_streaming_charges_address_and_source_beyond_legacy_result_budget) {
  auto engine = Engine::create({});
  const auto binding = power_binding(engine);
  FormulaOverrideStore store{engine};
  CHECK(store.set_override(binding, {.index = 10}, BigInt{-40}) ==
        Status::Confirmed);

  const auto value = store.evaluate({.binding = binding,
                                     .address = {.index = 10},
                                     .mode = EvaluationMode::Exact});
  CHECK(value.result.status == Status::Confirmed);
  CHECK(value.source == FormulaValueSource::Override);

  auto options = modular_options();
  options.filter = {};
  options.max_retained_candidates = 1;
  const auto legacy_bytes = retained_result_size(value.result);
  options.retention_memory_bytes = legacy_bytes;
  const auto without_store_metadata = store.evaluate_streaming_range(
      streaming_request(binding), {10, 10}, options);

  CHECK(without_store_metadata.status == Status::Confirmed);
  CHECK(without_store_metadata.summary.confirmed_count == 1);
  CHECK(without_store_metadata.retained_candidates.empty());

  options.retention_memory_bytes =
      legacy_bytes + sizeof(FormulaValueAtAddress) - sizeof(EvaluationResult);
  const auto with_store_metadata = store.evaluate_streaming_range(
      streaming_request(binding), {10, 10}, options);

  CHECK(with_store_metadata.status == Status::Confirmed);
  CHECK(with_store_metadata.retained_candidates.size() == 1);
  CHECK(with_store_metadata.retained_candidates.front().address.index == 10);
  CHECK(with_store_metadata.retained_candidates.front().value.source ==
        FormulaValueSource::Override);
  CHECK(with_store_metadata.retained_candidates.front().value.result.exact_integer() ==
        -40);
}

TEST(formula_override_streaming_is_deterministic_across_worker_counts) {
  auto one_engine = Engine::create({});
  const auto one_binding = power_binding(one_engine);
  FormulaOverrideStore one_store{one_engine};
  CHECK(one_store.set_override(one_binding, {.index = 10}, BigInt{-40}) ==
        Status::Confirmed);

  auto four_engine = Engine::create({});
  const auto four_binding = power_binding(four_engine);
  FormulaOverrideStore four_store{four_engine};
  CHECK(four_store.set_override(four_binding, {.index = 10}, BigInt{-40}) ==
        Status::Confirmed);

  const auto one = one_store.evaluate_streaming_range(
      streaming_request(one_binding), {0, 30}, modular_options(1));
  const auto four = four_store.evaluate_streaming_range(
      streaming_request(four_binding), {0, 30}, modular_options(4));

  CHECK(one.status == Status::Confirmed);
  CHECK(one.summary == four.summary);
  CHECK(one.filtered_out_count == four.filtered_out_count);
  CHECK(one.exact_evaluation_count == four.exact_evaluation_count);
  CHECK(one.effective_retention_mode == four.effective_retention_mode);
  CHECK(addresses(one) == addresses(four));
  CHECK(sources(one) == sources(four));
  CHECK(exact_values(one) == exact_values(four));
}

TEST(formula_override_streaming_honours_pre_cancelled_request_flag_and_signal) {
  auto engine = Engine::create({});
  const auto binding = power_binding(engine);
  FormulaOverrideStore store{engine};
  auto request = streaming_request(binding);
  request.cancellation_requested = true;
  const auto flagged = store.evaluate_streaming_range(
      request, {0, 30}, modular_options());

  CHECK(flagged.status == Status::Cancelled);
  CHECK(flagged.summary.checked_count == 0);
  CHECK(flagged.exact_evaluation_count == 0);
  CHECK(flagged.retained_candidates.empty());

  auto signal = std::make_shared<std::atomic_bool>(true);
  request.cancellation_requested = false;
  request.cancellation_signal = signal;
  const auto signalled = store.evaluate_streaming_range(
      request, {0, 30}, modular_options());

  CHECK(signalled.status == Status::Cancelled);
  CHECK(signalled.summary.checked_count == 0);
  CHECK(signalled.exact_evaluation_count == 0);
  CHECK(signalled.retained_candidates.empty());
}

TEST(formula_override_streaming_cancels_before_stream_result_publication) {
  auto engine = Engine::create({});
  const auto binding = power_binding(engine);
  FormulaOverrideStore store{engine};
  auto signal = std::make_shared<std::atomic_bool>(false);
  auto request = streaming_request(binding);
  request.cancellation_signal = signal;
  auto options = modular_options();
  options.filter = {};
  options.tile_size = 1;

  // This barrier is after the Formula core returns. The test intentionally
  // verifies the Store stream's publication boundary, not core preemption.
  test_support::EngineFormulaFinalizationBarrier barrier;
  auto pending = std::async(std::launch::async, [&] {
    return store.evaluate_streaming_range(request, {0, 30}, options);
  });
  barrier.wait_until_reached();
  signal->store(true, std::memory_order_release);
  signal->notify_all();
  barrier.release();
  const auto result = pending.get();

  CHECK(result.status == Status::Cancelled);
  CHECK(result.retained_candidates.empty());
}

TEST(formula_override_streaming_technique_cancels_before_stream_result_publication) {
  auto engine = Engine::create({});
  const auto kernel = std::make_shared<PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const TechniqueBinding binding{
      .identity = kernel->identity(),
      .canonical_parameters = encode_power_offset_parameters(2, 100)};
  FormulaOverrideStore store{engine};
  auto signal = std::make_shared<std::atomic_bool>(false);
  auto request = streaming_request(binding);
  request.cancellation_signal = signal;
  auto options = modular_options();
  options.filter = {};
  options.tile_size = 1;

  // This barrier is after the TechniqueKernel returns. The test intentionally
  // verifies the Store stream's publication boundary, not kernel preemption.
  test_support::EngineTechniqueFinalizationBarrier barrier;
  auto pending = std::async(std::launch::async, [&] {
    return store.evaluate_streaming_range(request, {0, 0}, options);
  });
  barrier.wait_until_reached();
  signal->store(true, std::memory_order_release);
  signal->notify_all();
  barrier.release();
  const auto result = pending.get();

  CHECK(result.status == Status::Cancelled);
  CHECK(result.retained_candidates.empty());
}

TEST(formula_override_streaming_cached_technique_cancels_before_stream_result_publication) {
  auto engine = Engine::create({.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto kernel = std::make_shared<PowerOffsetKernel>();
  CHECK(engine.register_technique(kernel));
  const TechniqueBinding binding{
      .identity = kernel->identity(),
      .canonical_parameters = encode_power_offset_parameters(2, 100)};
  CHECK(engine.evaluate_technique({.identity = binding.identity,
                                   .parameters = binding.canonical_parameters,
                                   .address = {.index = 0},
                                   .mode = EvaluationMode::Exact,
                                   .budget = Budget::unlimited()})
            .status == Status::Confirmed);
  CHECK(engine.inspect().resident_entries == 1);

  FormulaOverrideStore store{engine};
  auto signal = std::make_shared<std::atomic_bool>(false);
  auto request = streaming_request(binding);
  request.cancellation_signal = signal;
  auto options = modular_options();
  options.filter = {};
  options.tile_size = 1;

  // This barrier is on the cached Technique result after cache lookup. The
  // test verifies the Store stream's publication boundary, not kernel work.
  test_support::EngineTechniqueCacheHitFinalizationBarrier barrier;
  auto pending = std::async(std::launch::async, [&] {
    return store.evaluate_streaming_range(request, {0, 0}, options);
  });
  barrier.wait_until_reached();
  signal->store(true, std::memory_order_release);
  signal->notify_all();
  barrier.release();
  const auto result = pending.get();

  CHECK(result.status == Status::Cancelled);
  CHECK(result.retained_candidates.empty());
}

TEST(formula_override_streaming_rejects_invalid_requests_and_enforces_budget) {
  auto engine = Engine::create({});
  const auto binding = power_binding(engine);
  FormulaOverrideStore store{engine};

  auto invalid_formula = make_power_offset_spec(2, 100, 1);
  invalid_formula.nonnegative_index = false;
  const auto invalid_binding = store.evaluate_streaming_range(
      streaming_request(invalid_formula), {0, 5}, modular_options());
  CHECK(invalid_binding.status == Status::InvalidArgument);
  CHECK(invalid_binding.exact_evaluation_count == 0);

  auto invalid_filter = modular_options();
  invalid_filter.filter.modulus = 0;
  const auto invalid_options = store.evaluate_streaming_range(
      streaming_request(binding), {0, 5}, invalid_filter);
  CHECK(invalid_options.status == Status::InvalidArgument);
  CHECK(invalid_options.exact_evaluation_count == 0);

  Budget range_budget = Budget::unlimited();
  range_budget.max_candidate_count = 5;
  const auto range_limited = store.evaluate_streaming_range(
      streaming_request(binding, range_budget), {0, 5}, modular_options());
  CHECK(range_limited.status == Status::BudgetExceeded);
  CHECK(range_limited.exact_evaluation_count == 0);

  Budget exact_budget = Budget::unlimited();
  exact_budget.max_bit_length = 1;
  auto accept_all = modular_options();
  accept_all.filter = {};
  const auto exact_limited = store.evaluate_streaming_range(
      streaming_request(binding, exact_budget), {2, 2}, accept_all);
  CHECK(exact_limited.status == Status::BudgetExceeded);
  CHECK(exact_limited.retained_candidates.empty());
}

TEST(formula_override_streaming_rejects_a_valid_but_unregistered_formula) {
  auto engine = Engine::create({});
  const FormulaBinding binding{make_power_offset_spec(2, 100, 1)};
  FormulaOverrideStore store{engine};

  const auto result = store.evaluate_streaming_range(
      streaming_request(binding), {0, 30}, modular_options(4));

  CHECK(result.status == Status::InvalidArgument);
  CHECK(result.summary.checked_count == 0);
  CHECK(result.exact_evaluation_count == 0);
  CHECK(result.retained_candidates.empty());
}

TEST(formula_override_streaming_rejects_a_valid_but_unregistered_technique) {
  auto engine = Engine::create({});
  const TechniqueBinding binding{
      .identity = {.id = "bruce.power_offset", .version = 1},
      .canonical_parameters = encode_power_offset_parameters(2, 100)};
  FormulaOverrideStore store{engine};

  const auto result = store.evaluate_streaming_range(
      streaming_request(binding), {0, 30}, modular_options(4));

  CHECK(result.status == Status::InvalidArgument);
  CHECK(result.summary.checked_count == 0);
  CHECK(result.exact_evaluation_count == 0);
  CHECK(result.retained_candidates.empty());
}
