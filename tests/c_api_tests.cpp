#include "test.hpp"
#include "engine_formula_cache_hit_finalization_test_barrier.hpp"
#include "engine_formula_cache_hit_post_finalization_test_barrier.hpp"
#include "engine_formula_cache_miss_post_finalization_test_barrier.hpp"
#include "engine_formula_finalization_test_barrier.hpp"

#include <atomic>
#include <future>
#include <string>
#include <thread>

#include "bruce/lhme/c_api.h"

TEST(c_api_uses_opaque_handles_and_releases_everything) {
  const lhme_engine_config config{.cache_max_entries = 8,
                                  .cache_max_bytes = 4096,
                                  .worker_count = 1,
                                  .tile_size = 4};
  lhme_engine* engine = lhme_engine_create(&config);
  CHECK(engine != nullptr);

  const lhme_power_offset_spec spec{.base_decimal = "2",
                                    .scale_decimal = "100",
                                    .version = 1};
  lhme_formula* formula = lhme_formula_register(engine, &spec);
  CHECK(formula != nullptr);

  const lhme_evaluation_request request{
      .index = 10,
      .mode = LHME_MODE_EXACT,
      .max_bit_length = UINT64_MAX,
      .modulus_decimal = nullptr};
  lhme_result* result = lhme_evaluate(engine, formula, &request);
  CHECK(result != nullptr);
  CHECK(lhme_result_status(result) == LHME_STATUS_CONFIRMED);
  CHECK(std::string(lhme_result_exact_decimal(result)) == "102300");

  lhme_result_destroy(result);
  lhme_formula_destroy(formula);
  lhme_engine_destroy(engine);
}

TEST(c_api_rejects_an_unknown_mode_instead_of_silently_changing_it) {
  lhme_engine* engine = lhme_engine_create(nullptr);
  const lhme_power_offset_spec spec{.base_decimal = "2",
                                    .scale_decimal = "100",
                                    .version = 1};
  lhme_formula* formula = lhme_formula_register(engine, &spec);
  const lhme_evaluation_request request{
      .index = 10,
      .mode = static_cast<lhme_mode>(99),
      .max_bit_length = UINT64_MAX,
      .modulus_decimal = nullptr};

  lhme_result* result = lhme_evaluate(engine, formula, &request);

  CHECK(result != nullptr);
  CHECK(lhme_result_status(result) == LHME_STATUS_INVALID_ARGUMENT);
  lhme_result_destroy(result);
  lhme_formula_destroy(formula);
  lhme_engine_destroy(engine);
}

TEST(c_api_cancel_without_an_active_evaluation_does_not_poison_a_later_call) {
  lhme_engine* engine = lhme_engine_create(nullptr);
  CHECK(engine != nullptr);
  const lhme_power_offset_spec spec{.base_decimal = "2",
                                    .scale_decimal = "100",
                                    .version = 1};
  lhme_formula* formula = lhme_formula_register(engine, &spec);
  CHECK(formula != nullptr);
  const lhme_evaluation_request request{
      .index = 10,
      .mode = LHME_MODE_EXACT,
      .max_bit_length = UINT64_MAX,
      .modulus_decimal = nullptr};

  lhme_cancel(engine);
  lhme_result* result = lhme_evaluate(engine, formula, &request);

  CHECK(result != nullptr);
  CHECK(lhme_result_status(result) == LHME_STATUS_CONFIRMED);
  CHECK(std::string(lhme_result_exact_decimal(result)) == "102300");
  lhme_result_destroy(result);
  lhme_formula_destroy(formula);
  lhme_engine_destroy(engine);
}

TEST(c_api_cancels_an_active_evaluation_and_reuses_the_engine) {
  constexpr std::int64_t kLargeIndex = 8 * 1024 * 1024;
  const lhme_engine_config config{.cache_max_entries = 0,
                                  .cache_max_bytes = 0,
                                  .worker_count = 1,
                                  .tile_size = 1};
  lhme_engine* engine = lhme_engine_create(&config);
  CHECK(engine != nullptr);
  const lhme_power_offset_spec spec{.base_decimal = "2",
                                    .scale_decimal = "1",
                                    .version = 1};
  lhme_formula* formula = lhme_formula_register(engine, &spec);
  CHECK(formula != nullptr);
  const lhme_evaluation_request active_request{
      .index = kLargeIndex,
      .mode = LHME_MODE_EXACT,
      .max_bit_length = static_cast<std::uint64_t>(kLargeIndex),
      .modulus_decimal = nullptr};
  std::atomic_bool worker_started{false};
  std::atomic_bool worker_finished{false};
  auto pending = std::async(std::launch::async, [&] {
    worker_started.store(true, std::memory_order_release);
    lhme_result* result = lhme_evaluate(engine, formula, &active_request);
    worker_finished.store(true, std::memory_order_release);
    return result;
  });

  while (!worker_started.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  while (!worker_finished.load(std::memory_order_acquire)) {
    lhme_cancel(engine);
    std::this_thread::yield();
  }

  lhme_result* cancelled = pending.get();
  CHECK(cancelled != nullptr);
  CHECK(lhme_result_status(cancelled) == LHME_STATUS_CANCELLED);
  lhme_result_destroy(cancelled);

  const lhme_evaluation_request reuse_request{
      .index = 10,
      .mode = LHME_MODE_EXACT,
      .max_bit_length = UINT64_MAX,
      .modulus_decimal = nullptr};
  lhme_result* reused = lhme_evaluate(engine, formula, &reuse_request);
  CHECK(reused != nullptr);
  CHECK(lhme_result_status(reused) == LHME_STATUS_CONFIRMED);
  CHECK(lhme_result_exact_decimal(reused) != nullptr);
  CHECK(std::string(lhme_result_exact_decimal(reused)) == "1023");
  lhme_result_destroy(reused);
  lhme_formula_destroy(formula);
  lhme_engine_destroy(engine);
}

TEST(c_api_cancels_a_formula_joining_a_post_snapshot_cancelled_group_without_mutating_the_finalized_result) {
  const lhme_engine_config config{.cache_max_entries = 8,
                                  .cache_max_bytes = 4096,
                                  .worker_count = 1,
                                  .tile_size = 1};
  lhme_engine* engine = lhme_engine_create(&config);
  CHECK(engine != nullptr);
  const lhme_power_offset_spec spec{.base_decimal = "2",
                                    .scale_decimal = "100",
                                    .version = 1};
  lhme_formula* formula = lhme_formula_register(engine, &spec);
  CHECK(formula != nullptr);
  const lhme_evaluation_request request{
      .index = 10,
      .mode = LHME_MODE_EXACT,
      .max_bit_length = UINT64_MAX,
      .modulus_decimal = nullptr};

  std::future<lhme_result*> pending;
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&] {
    return lhme_evaluate(engine, formula, &request);
  });
  post_finalization_barrier.wait_until_reached();

  lhme_cancel(engine);
  lhme_result* joining = lhme_evaluate(engine, formula, &request);
  CHECK(joining != nullptr);
  CHECK(lhme_result_status(joining) == LHME_STATUS_CANCELLED);
  CHECK(lhme_result_exact_decimal(joining) == nullptr);
  lhme_result_destroy(joining);

  post_finalization_barrier.release();
  lhme_result* finalized = pending.get();
  CHECK(finalized != nullptr);
  CHECK(lhme_result_status(finalized) == LHME_STATUS_CONFIRMED);
  CHECK(lhme_result_exact_decimal(finalized) != nullptr);
  CHECK(std::string(lhme_result_exact_decimal(finalized)) == "102300");
  lhme_result_destroy(finalized);

  lhme_result* reused = lhme_evaluate(engine, formula, &request);
  CHECK(reused != nullptr);
  CHECK(lhme_result_status(reused) == LHME_STATUS_CONFIRMED);
  CHECK(lhme_result_exact_decimal(reused) != nullptr);
  CHECK(std::string(lhme_result_exact_decimal(reused)) == "102300");
  lhme_result_destroy(reused);
  lhme_formula_destroy(formula);
  lhme_engine_destroy(engine);
}

TEST(c_api_cancels_a_formula_cache_hit_joining_a_post_snapshot_cancelled_group_without_mutating_the_finalized_result) {
  const lhme_engine_config config{.cache_max_entries = 8,
                                  .cache_max_bytes = 4096,
                                  .worker_count = 1,
                                  .tile_size = 1};
  lhme_engine* engine = lhme_engine_create(&config);
  CHECK(engine != nullptr);
  const lhme_power_offset_spec spec{.base_decimal = "2",
                                    .scale_decimal = "100",
                                    .version = 1};
  lhme_formula* formula = lhme_formula_register(engine, &spec);
  CHECK(formula != nullptr);
  const lhme_evaluation_request request{
      .index = 10,
      .mode = LHME_MODE_EXACT,
      .max_bit_length = UINT64_MAX,
      .modulus_decimal = nullptr};

  lhme_result* warmed = lhme_evaluate(engine, formula, &request);
  CHECK(warmed != nullptr);
  CHECK(lhme_result_status(warmed) == LHME_STATUS_CONFIRMED);
  CHECK(lhme_result_exact_decimal(warmed) != nullptr);
  CHECK(std::string(lhme_result_exact_decimal(warmed)) == "102300");
  lhme_result_destroy(warmed);

  std::future<lhme_result*> pending;
  bruce::lhme::test_support::EngineFormulaCacheHitPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&] {
    return lhme_evaluate(engine, formula, &request);
  });
  post_finalization_barrier.wait_until_reached();

  lhme_cancel(engine);
  lhme_result* joining = lhme_evaluate(engine, formula, &request);
  CHECK(joining != nullptr);
  CHECK(lhme_result_status(joining) == LHME_STATUS_CANCELLED);
  CHECK(lhme_result_exact_decimal(joining) == nullptr);
  lhme_result_destroy(joining);

  post_finalization_barrier.release();
  lhme_result* finalized = pending.get();
  CHECK(finalized != nullptr);
  CHECK(lhme_result_status(finalized) == LHME_STATUS_CONFIRMED);
  CHECK(lhme_result_exact_decimal(finalized) != nullptr);
  CHECK(std::string(lhme_result_exact_decimal(finalized)) == "102300");
  lhme_result_destroy(finalized);

  lhme_result* reused = lhme_evaluate(engine, formula, &request);
  CHECK(reused != nullptr);
  CHECK(lhme_result_status(reused) == LHME_STATUS_CONFIRMED);
  CHECK(lhme_result_exact_decimal(reused) != nullptr);
  CHECK(std::string(lhme_result_exact_decimal(reused)) == "102300");
  lhme_result_destroy(reused);
  lhme_formula_destroy(formula);
  lhme_engine_destroy(engine);
}

TEST(c_api_cancels_a_modular_formula_joining_a_post_snapshot_cancelled_group_without_mutating_the_finalized_result) {
  const lhme_engine_config config{.cache_max_entries = 8,
                                  .cache_max_bytes = 4096,
                                  .worker_count = 1,
                                  .tile_size = 1};
  lhme_engine* engine = lhme_engine_create(&config);
  CHECK(engine != nullptr);
  const lhme_power_offset_spec spec{.base_decimal = "2",
                                    .scale_decimal = "100",
                                    .version = 1};
  lhme_formula* formula = lhme_formula_register(engine, &spec);
  CHECK(formula != nullptr);
  const lhme_evaluation_request request{
      .index = 10,
      .mode = LHME_MODE_MODULAR,
      .max_bit_length = UINT64_MAX,
      .modulus_decimal = "17"};

  std::future<lhme_result*> pending;
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&] {
    return lhme_evaluate(engine, formula, &request);
  });
  post_finalization_barrier.wait_until_reached();

  lhme_cancel(engine);
  lhme_result* joining = lhme_evaluate(engine, formula, &request);
  CHECK(joining != nullptr);
  CHECK(lhme_result_status(joining) == LHME_STATUS_CANCELLED);
  CHECK(lhme_result_exact_decimal(joining) == nullptr);
  lhme_result_destroy(joining);

  post_finalization_barrier.release();
  lhme_result* finalized = pending.get();
  CHECK(finalized != nullptr);
  CHECK(lhme_result_status(finalized) == LHME_STATUS_CONFIRMED);
  CHECK(lhme_result_exact_decimal(finalized) != nullptr);
  CHECK(std::string(lhme_result_exact_decimal(finalized)) == "11");
  lhme_result_destroy(finalized);

  lhme_result* reused = lhme_evaluate(engine, formula, &request);
  CHECK(reused != nullptr);
  CHECK(lhme_result_status(reused) == LHME_STATUS_CONFIRMED);
  CHECK(lhme_result_exact_decimal(reused) != nullptr);
  CHECK(std::string(lhme_result_exact_decimal(reused)) == "11");
  lhme_result_destroy(reused);
  lhme_formula_destroy(formula);
  lhme_engine_destroy(engine);
}

TEST(c_api_cancels_a_symbolic_formula_joining_a_post_snapshot_cancelled_group_without_mutating_the_finalized_result) {
  const lhme_engine_config config{.cache_max_entries = 8,
                                  .cache_max_bytes = 4096,
                                  .worker_count = 1,
                                  .tile_size = 1};
  lhme_engine* engine = lhme_engine_create(&config);
  CHECK(engine != nullptr);
  const lhme_power_offset_spec spec{.base_decimal = "2",
                                    .scale_decimal = "100",
                                    .version = 1};
  lhme_formula* formula = lhme_formula_register(engine, &spec);
  CHECK(formula != nullptr);
  const lhme_evaluation_request request{
      .index = 10,
      .mode = LHME_MODE_SYMBOLIC,
      .max_bit_length = UINT64_MAX,
      .modulus_decimal = nullptr};

  std::future<lhme_result*> pending;
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&] {
    return lhme_evaluate(engine, formula, &request);
  });
  post_finalization_barrier.wait_until_reached();

  lhme_cancel(engine);
  lhme_result* joining = lhme_evaluate(engine, formula, &request);
  CHECK(joining != nullptr);
  CHECK(lhme_result_status(joining) == LHME_STATUS_CANCELLED);
  CHECK(lhme_result_exact_decimal(joining) == nullptr);
  lhme_result_destroy(joining);

  post_finalization_barrier.release();
  lhme_result* finalized = pending.get();
  CHECK(finalized != nullptr);
  CHECK(lhme_result_status(finalized) == LHME_STATUS_CONFIRMED);
  CHECK(lhme_result_exact_decimal(finalized) == nullptr);
  lhme_result_destroy(finalized);

  lhme_result* reused = lhme_evaluate(engine, formula, &request);
  CHECK(reused != nullptr);
  CHECK(lhme_result_status(reused) == LHME_STATUS_CONFIRMED);
  CHECK(lhme_result_exact_decimal(reused) == nullptr);
  lhme_result_destroy(reused);
  lhme_formula_destroy(formula);
  lhme_engine_destroy(engine);
}

TEST(c_api_cancels_a_metric_formula_cache_miss_joining_a_post_snapshot_cancelled_group_without_mutating_the_finalized_result) {
  const lhme_engine_config config{.cache_max_entries = 8,
                                  .cache_max_bytes = 4096,
                                  .worker_count = 1,
                                  .tile_size = 1};
  lhme_engine* engine = lhme_engine_create(&config);
  CHECK(engine != nullptr);
  const lhme_power_offset_spec spec{.base_decimal = "2",
                                    .scale_decimal = "100",
                                    .version = 1};
  lhme_formula* formula = lhme_formula_register(engine, &spec);
  CHECK(formula != nullptr);
  const lhme_evaluation_request request{
      .index = 10,
      .mode = LHME_MODE_METRIC,
      .max_bit_length = UINT64_MAX,
      .modulus_decimal = nullptr};

  std::future<lhme_result*> pending;
  bruce::lhme::test_support::EngineFormulaCacheMissPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&] {
    return lhme_evaluate(engine, formula, &request);
  });
  post_finalization_barrier.wait_until_reached();

  lhme_cancel(engine);
  lhme_result* joining = lhme_evaluate(engine, formula, &request);
  CHECK(joining != nullptr);
  CHECK(lhme_result_status(joining) == LHME_STATUS_CANCELLED);
  CHECK(lhme_result_exact_decimal(joining) == nullptr);
  lhme_result_destroy(joining);

  post_finalization_barrier.release();
  lhme_result* finalized = pending.get();
  CHECK(finalized != nullptr);
  CHECK(lhme_result_status(finalized) == LHME_STATUS_CONFIRMED);
  CHECK(lhme_result_exact_decimal(finalized) == nullptr);
  lhme_result_destroy(finalized);

  lhme_result* reused = lhme_evaluate(engine, formula, &request);
  CHECK(reused != nullptr);
  CHECK(lhme_result_status(reused) == LHME_STATUS_CONFIRMED);
  CHECK(lhme_result_exact_decimal(reused) == nullptr);
  lhme_result_destroy(reused);
  lhme_formula_destroy(formula);
  lhme_engine_destroy(engine);
}

TEST(c_api_cancels_a_metric_formula_cache_hit_joining_a_post_snapshot_cancelled_group_without_mutating_the_finalized_result) {
  const lhme_engine_config config{.cache_max_entries = 8,
                                  .cache_max_bytes = 4096,
                                  .worker_count = 1,
                                  .tile_size = 1};
  lhme_engine* engine = lhme_engine_create(&config);
  CHECK(engine != nullptr);
  const lhme_power_offset_spec spec{.base_decimal = "2",
                                    .scale_decimal = "100",
                                    .version = 1};
  lhme_formula* formula = lhme_formula_register(engine, &spec);
  CHECK(formula != nullptr);
  const lhme_evaluation_request request{
      .index = 10,
      .mode = LHME_MODE_METRIC,
      .max_bit_length = UINT64_MAX,
      .modulus_decimal = nullptr};

  lhme_result* warmed = lhme_evaluate(engine, formula, &request);
  CHECK(warmed != nullptr);
  CHECK(lhme_result_status(warmed) == LHME_STATUS_CONFIRMED);
  CHECK(lhme_result_exact_decimal(warmed) == nullptr);
  lhme_result_destroy(warmed);

  std::future<lhme_result*> pending;
  bruce::lhme::test_support::EngineFormulaCacheHitPostFinalizationBarrier
      post_finalization_barrier;
  pending = std::async(std::launch::async, [&] {
    return lhme_evaluate(engine, formula, &request);
  });
  post_finalization_barrier.wait_until_reached();

  lhme_cancel(engine);
  lhme_result* joining = lhme_evaluate(engine, formula, &request);
  CHECK(joining != nullptr);
  CHECK(lhme_result_status(joining) == LHME_STATUS_CANCELLED);
  CHECK(lhme_result_exact_decimal(joining) == nullptr);
  lhme_result_destroy(joining);

  post_finalization_barrier.release();
  lhme_result* finalized = pending.get();
  CHECK(finalized != nullptr);
  CHECK(lhme_result_status(finalized) == LHME_STATUS_CONFIRMED);
  CHECK(lhme_result_exact_decimal(finalized) == nullptr);
  lhme_result_destroy(finalized);

  lhme_result* reused = lhme_evaluate(engine, formula, &request);
  CHECK(reused != nullptr);
  CHECK(lhme_result_status(reused) == LHME_STATUS_CONFIRMED);
  CHECK(lhme_result_exact_decimal(reused) == nullptr);
  lhme_result_destroy(reused);
  lhme_formula_destroy(formula);
  lhme_engine_destroy(engine);
}

TEST(c_api_cancels_a_metric_formula_cache_hit_before_its_completion_snapshot_and_reuses_after_completion) {
  const lhme_engine_config config{.cache_max_entries = 8,
                                  .cache_max_bytes = 4096,
                                  .worker_count = 1,
                                  .tile_size = 1};
  lhme_engine* engine = lhme_engine_create(&config);
  CHECK(engine != nullptr);
  const lhme_power_offset_spec spec{.base_decimal = "2",
                                    .scale_decimal = "100",
                                    .version = 1};
  lhme_formula* formula = lhme_formula_register(engine, &spec);
  CHECK(formula != nullptr);
  const lhme_evaluation_request request{
      .index = 10,
      .mode = LHME_MODE_METRIC,
      .max_bit_length = UINT64_MAX,
      .modulus_decimal = nullptr};

  lhme_result* warmed = lhme_evaluate(engine, formula, &request);
  CHECK(warmed != nullptr);
  CHECK(lhme_result_status(warmed) == LHME_STATUS_CONFIRMED);
  CHECK(lhme_result_exact_decimal(warmed) == nullptr);
  lhme_result_destroy(warmed);

  std::future<lhme_result*> pending;
  bruce::lhme::test_support::EngineFormulaCacheHitFinalizationBarrier
      finalization_barrier;
  pending = std::async(std::launch::async, [&] {
    return lhme_evaluate(engine, formula, &request);
  });
  finalization_barrier.wait_until_reached();

  lhme_cancel(engine);
  finalization_barrier.release();
  lhme_result* cancelled = pending.get();
  CHECK(cancelled != nullptr);
  CHECK(lhme_result_status(cancelled) == LHME_STATUS_CANCELLED);
  CHECK(lhme_result_exact_decimal(cancelled) == nullptr);
  lhme_result_destroy(cancelled);

  lhme_result* reused = lhme_evaluate(engine, formula, &request);
  CHECK(reused != nullptr);
  CHECK(lhme_result_status(reused) == LHME_STATUS_CONFIRMED);
  CHECK(lhme_result_exact_decimal(reused) == nullptr);
  lhme_result_destroy(reused);
  lhme_formula_destroy(formula);
  lhme_engine_destroy(engine);
}

TEST(c_api_cancels_a_metric_formula_cache_miss_before_its_completion_snapshot_and_reuses_after_completion) {
  const lhme_engine_config config{.cache_max_entries = 8,
                                  .cache_max_bytes = 4096,
                                  .worker_count = 1,
                                  .tile_size = 1};
  lhme_engine* engine = lhme_engine_create(&config);
  CHECK(engine != nullptr);
  const lhme_power_offset_spec spec{.base_decimal = "2",
                                    .scale_decimal = "100",
                                    .version = 1};
  lhme_formula* formula = lhme_formula_register(engine, &spec);
  CHECK(formula != nullptr);
  const lhme_evaluation_request request{
      .index = 10,
      .mode = LHME_MODE_METRIC,
      .max_bit_length = UINT64_MAX,
      .modulus_decimal = nullptr};

  std::future<lhme_result*> pending;
  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      finalization_barrier;
  pending = std::async(std::launch::async, [&] {
    return lhme_evaluate(engine, formula, &request);
  });
  finalization_barrier.wait_until_reached();

  lhme_cancel(engine);
  finalization_barrier.release();
  lhme_result* cancelled = pending.get();
  CHECK(cancelled != nullptr);
  CHECK(lhme_result_status(cancelled) == LHME_STATUS_CANCELLED);
  CHECK(lhme_result_exact_decimal(cancelled) == nullptr);
  lhme_result_destroy(cancelled);

  lhme_result* reused = lhme_evaluate(engine, formula, &request);
  CHECK(reused != nullptr);
  CHECK(lhme_result_status(reused) == LHME_STATUS_CONFIRMED);
  CHECK(lhme_result_exact_decimal(reused) == nullptr);
  lhme_result_destroy(reused);
  lhme_formula_destroy(formula);
  lhme_engine_destroy(engine);
}

TEST(c_api_cancels_an_exact_formula_cache_miss_before_its_completion_snapshot_and_reuses_after_completion) {
  const lhme_engine_config config{.cache_max_entries = 8,
                                  .cache_max_bytes = 4096,
                                  .worker_count = 1,
                                  .tile_size = 1};
  lhme_engine* engine = lhme_engine_create(&config);
  CHECK(engine != nullptr);
  const lhme_power_offset_spec spec{.base_decimal = "2",
                                    .scale_decimal = "100",
                                    .version = 1};
  lhme_formula* formula = lhme_formula_register(engine, &spec);
  CHECK(formula != nullptr);
  const lhme_evaluation_request request{
      .index = 10,
      .mode = LHME_MODE_EXACT,
      .max_bit_length = UINT64_MAX,
      .modulus_decimal = nullptr};

  std::future<lhme_result*> pending;
  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      finalization_barrier;
  pending = std::async(std::launch::async, [&] {
    return lhme_evaluate(engine, formula, &request);
  });
  finalization_barrier.wait_until_reached();

  lhme_cancel(engine);
  finalization_barrier.release();
  lhme_result* cancelled = pending.get();
  CHECK(cancelled != nullptr);
  CHECK(lhme_result_status(cancelled) == LHME_STATUS_CANCELLED);
  CHECK(lhme_result_exact_decimal(cancelled) == nullptr);
  lhme_result_destroy(cancelled);

  lhme_result* reused = lhme_evaluate(engine, formula, &request);
  CHECK(reused != nullptr);
  CHECK(lhme_result_status(reused) == LHME_STATUS_CONFIRMED);
  CHECK(lhme_result_exact_decimal(reused) != nullptr);
  CHECK(std::string(lhme_result_exact_decimal(reused)) == "102300");
  lhme_result_destroy(reused);
  lhme_formula_destroy(formula);
  lhme_engine_destroy(engine);
}

TEST(c_api_cancels_an_exact_formula_cache_hit_before_its_completion_snapshot_and_reuses_after_completion) {
  const lhme_engine_config config{.cache_max_entries = 8,
                                  .cache_max_bytes = 4096,
                                  .worker_count = 1,
                                  .tile_size = 1};
  lhme_engine* engine = lhme_engine_create(&config);
  CHECK(engine != nullptr);
  const lhme_power_offset_spec spec{.base_decimal = "2",
                                    .scale_decimal = "100",
                                    .version = 1};
  lhme_formula* formula = lhme_formula_register(engine, &spec);
  CHECK(formula != nullptr);
  const lhme_evaluation_request request{
      .index = 10,
      .mode = LHME_MODE_EXACT,
      .max_bit_length = UINT64_MAX,
      .modulus_decimal = nullptr};

  lhme_result* warmed = lhme_evaluate(engine, formula, &request);
  CHECK(warmed != nullptr);
  CHECK(lhme_result_status(warmed) == LHME_STATUS_CONFIRMED);
  CHECK(lhme_result_exact_decimal(warmed) != nullptr);
  CHECK(std::string(lhme_result_exact_decimal(warmed)) == "102300");
  lhme_result_destroy(warmed);

  std::future<lhme_result*> pending;
  bruce::lhme::test_support::EngineFormulaCacheHitFinalizationBarrier
      finalization_barrier;
  pending = std::async(std::launch::async, [&] {
    return lhme_evaluate(engine, formula, &request);
  });
  finalization_barrier.wait_until_reached();

  lhme_cancel(engine);
  finalization_barrier.release();
  lhme_result* cancelled = pending.get();
  CHECK(cancelled != nullptr);
  CHECK(lhme_result_status(cancelled) == LHME_STATUS_CANCELLED);
  CHECK(lhme_result_exact_decimal(cancelled) == nullptr);
  lhme_result_destroy(cancelled);

  lhme_result* reused = lhme_evaluate(engine, formula, &request);
  CHECK(reused != nullptr);
  CHECK(lhme_result_status(reused) == LHME_STATUS_CONFIRMED);
  CHECK(lhme_result_exact_decimal(reused) != nullptr);
  CHECK(std::string(lhme_result_exact_decimal(reused)) == "102300");
  lhme_result_destroy(reused);
  lhme_formula_destroy(formula);
  lhme_engine_destroy(engine);
}

TEST(c_api_cancels_a_modular_formula_before_its_completion_snapshot_and_reuses_after_completion) {
  const lhme_engine_config config{.cache_max_entries = 8,
                                  .cache_max_bytes = 4096,
                                  .worker_count = 1,
                                  .tile_size = 1};
  lhme_engine* engine = lhme_engine_create(&config);
  CHECK(engine != nullptr);
  const lhme_power_offset_spec spec{.base_decimal = "2",
                                    .scale_decimal = "100",
                                    .version = 1};
  lhme_formula* formula = lhme_formula_register(engine, &spec);
  CHECK(formula != nullptr);
  const lhme_evaluation_request request{
      .index = 10,
      .mode = LHME_MODE_MODULAR,
      .max_bit_length = UINT64_MAX,
      .modulus_decimal = "17"};

  std::future<lhme_result*> pending;
  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      finalization_barrier;
  pending = std::async(std::launch::async, [&] {
    return lhme_evaluate(engine, formula, &request);
  });
  finalization_barrier.wait_until_reached();

  lhme_cancel(engine);
  finalization_barrier.release();
  lhme_result* cancelled = pending.get();
  CHECK(cancelled != nullptr);
  CHECK(lhme_result_status(cancelled) == LHME_STATUS_CANCELLED);
  CHECK(lhme_result_exact_decimal(cancelled) == nullptr);
  lhme_result_destroy(cancelled);

  lhme_result* reused = lhme_evaluate(engine, formula, &request);
  CHECK(reused != nullptr);
  CHECK(lhme_result_status(reused) == LHME_STATUS_CONFIRMED);
  CHECK(lhme_result_exact_decimal(reused) != nullptr);
  CHECK(std::string(lhme_result_exact_decimal(reused)) == "11");
  lhme_result_destroy(reused);
  lhme_formula_destroy(formula);
  lhme_engine_destroy(engine);
}

TEST(c_api_cancels_a_symbolic_formula_before_its_completion_snapshot_and_reuses_after_completion) {
  const lhme_engine_config config{.cache_max_entries = 8,
                                  .cache_max_bytes = 4096,
                                  .worker_count = 1,
                                  .tile_size = 1};
  lhme_engine* engine = lhme_engine_create(&config);
  CHECK(engine != nullptr);
  const lhme_power_offset_spec spec{.base_decimal = "2",
                                    .scale_decimal = "100",
                                    .version = 1};
  lhme_formula* formula = lhme_formula_register(engine, &spec);
  CHECK(formula != nullptr);
  const lhme_evaluation_request request{
      .index = 10,
      .mode = LHME_MODE_SYMBOLIC,
      .max_bit_length = UINT64_MAX,
      .modulus_decimal = nullptr};

  std::future<lhme_result*> pending;
  bruce::lhme::test_support::EngineFormulaFinalizationBarrier
      finalization_barrier;
  pending = std::async(std::launch::async, [&] {
    return lhme_evaluate(engine, formula, &request);
  });
  finalization_barrier.wait_until_reached();

  lhme_cancel(engine);
  finalization_barrier.release();
  lhme_result* cancelled = pending.get();
  CHECK(cancelled != nullptr);
  CHECK(lhme_result_status(cancelled) == LHME_STATUS_CANCELLED);
  CHECK(lhme_result_exact_decimal(cancelled) == nullptr);
  lhme_result_destroy(cancelled);

  lhme_result* reused = lhme_evaluate(engine, formula, &request);
  CHECK(reused != nullptr);
  CHECK(lhme_result_status(reused) == LHME_STATUS_CONFIRMED);
  CHECK(lhme_result_exact_decimal(reused) == nullptr);
  lhme_result_destroy(reused);
  lhme_formula_destroy(formula);
  lhme_engine_destroy(engine);
}
