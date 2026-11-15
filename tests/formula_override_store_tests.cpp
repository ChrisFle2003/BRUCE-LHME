#include "test.hpp"

#include "bruce/lhme/formula_override_store.hpp"

#include <array>
#include <atomic>
#include <future>
#include <thread>
#include <vector>

namespace {

using namespace bruce::lhme;

FormulaBinding power_binding(Engine& engine, BigInt base = 2) {
  return engine.register_formula(make_power_offset_spec(base, 100, 1));
}

FormulaOverrideRequest request_for(const FormulaBinding& binding,
                                   std::int64_t index,
                                   EvaluationMode mode = EvaluationMode::Exact) {
  return {.binding = binding,
          .address = {.index = index},
          .mode = mode,
          .budget = Budget::unlimited()};
}

}  // namespace

TEST(formula_override_store_evaluates_a_registered_formula_without_kevin) {
  auto engine = Engine::create({});
  const auto binding = power_binding(engine);
  FormulaOverrideStore store{engine};

  CHECK(store.set_override(binding, {.index = 10}, BigInt{7}) == Status::Confirmed);
  const auto overridden = store.evaluate(request_for(binding, 10));
  const auto computed = store.evaluate(request_for(binding, 2));

  CHECK(overridden.source == FormulaValueSource::Override);
  CHECK(overridden.result.status == Status::Confirmed);
  CHECK(overridden.result.exact_integer() == 7);
  CHECK(computed.source == FormulaValueSource::Computed);
  CHECK(computed.result.status == Status::Confirmed);
  CHECK(computed.result.exact_integer() == 300);
}

TEST(formula_override_store_never_crosses_formula_binding_identities) {
  auto engine = Engine::create({});
  const auto first = power_binding(engine, 2);
  const auto second = power_binding(engine, 3);
  FormulaOverrideStore store{engine};

  CHECK(store.set_override(first, {.index = 2}, BigInt{17}) == Status::Confirmed);
  const auto first_result = store.evaluate(request_for(first, 2));
  const auto second_result = store.evaluate(request_for(second, 2));

  CHECK(first_result.source == FormulaValueSource::Override);
  CHECK(first_result.result.exact_integer() == 17);
  CHECK(second_result.source == FormulaValueSource::Computed);
  CHECK(second_result.result.exact_integer() == 800);
}

TEST(formula_override_store_rejects_formula_specs_outside_the_v0_1_contract) {
  auto engine = Engine::create({});
  auto unsupported = make_power_offset_spec(2, 100, 1);
  unsupported.nonnegative_index = false;
  FormulaOverrideStore store{engine};

  CHECK(store.set_override(unsupported, {.index = 2}, BigInt{17}) ==
        Status::InvalidArgument);
  CHECK(store.evaluate(request_for(unsupported, 2)).result.status ==
        Status::InvalidArgument);
}

TEST(formula_override_store_rejects_new_records_at_the_count_limit) {
  auto engine = Engine::create({});
  const auto binding = power_binding(engine);
  FormulaOverrideStore store{
      engine, {.max_override_count = 1}};

  CHECK(store.set_override(binding, {.index = 2}, BigInt{17}) ==
        Status::Confirmed);
  CHECK(store.set_override(binding, {.index = 3}, BigInt{19}) ==
        Status::BudgetExceeded);
  CHECK(store.inspect().override_count == 1);
  CHECK(store.evaluate(request_for(binding, 2)).result.exact_integer() == 17);
  CHECK(store.evaluate(request_for(binding, 3)).source == FormulaValueSource::Computed);

  CHECK(store.set_override(binding, {.index = 2}, BigInt{23}) ==
        Status::Confirmed);
  CHECK(store.evaluate(request_for(binding, 2)).result.exact_integer() == 23);
  CHECK(store.remove_override(binding, {.index = 2}));
  CHECK(store.set_override(binding, {.index = 3}, BigInt{19}) ==
        Status::Confirmed);
}

TEST(formula_override_store_enforces_value_and_storage_limits_without_replacing) {
  auto engine = Engine::create({});
  const auto binding = power_binding(engine);
  FormulaOverrideStore value_limited{
      engine, {.max_override_value_bytes = 1}};

  CHECK(value_limited.set_override(binding, {.index = 2}, BigInt{255}) ==
        Status::Confirmed);
  CHECK(value_limited.set_override(binding, {.index = 2}, BigInt{256}) ==
        Status::BudgetExceeded);
  CHECK(value_limited.evaluate(request_for(binding, 2)).result.exact_integer() == 255);

  FormulaOverrideStore baseline{engine};
  CHECK(baseline.set_override(binding, {.index = 4}, BigInt{255}) ==
        Status::Confirmed);
  const auto record_bytes = baseline.inspect().override_storage_bytes;
  CHECK(record_bytes > 0);

  FormulaOverrideStore storage_limited{
      engine, {.max_override_count = 2,
               .max_override_storage_bytes = record_bytes}};
  CHECK(storage_limited.set_override(binding, {.index = 4}, BigInt{255}) ==
        Status::Confirmed);
  CHECK(storage_limited.inspect().override_storage_bytes == record_bytes);
  CHECK(storage_limited.set_override(binding, {.index = 5}, BigInt{255}) ==
        Status::BudgetExceeded);
  CHECK(storage_limited.inspect().override_count == 1);
  CHECK(storage_limited.evaluate(request_for(binding, 4)).result.exact_integer() ==
        255);
}

TEST(formula_override_store_capacity_remains_safe_during_parallel_updates) {
  auto engine = Engine::create({});
  const auto binding = power_binding(engine);
  FormulaOverrideStore store{
      engine, {.max_override_count = 1}};
  std::atomic_bool start = false;
  std::array<Status, 2> statuses{Status::InternalError, Status::InternalError};

  std::thread first([&] {
    start.wait(false);
    statuses[0] = store.set_override(binding, {.index = 2}, BigInt{17});
  });
  std::thread second([&] {
    start.wait(false);
    statuses[1] = store.set_override(binding, {.index = 3}, BigInt{19});
  });
  start.store(true);
  start.notify_all();
  first.join();
  second.join();

  const auto confirmed = static_cast<std::size_t>(statuses[0] == Status::Confirmed) +
                         static_cast<std::size_t>(statuses[1] == Status::Confirmed);
  const auto exhausted =
      static_cast<std::size_t>(statuses[0] == Status::BudgetExceeded) +
      static_cast<std::size_t>(statuses[1] == Status::BudgetExceeded);
  CHECK(confirmed == 1);
  CHECK(exhausted == 1);
  CHECK(store.inspect().override_count == 1);
}

TEST(formula_override_store_parallel_exact_reads_are_deterministic) {
  auto engine = Engine::create({});
  const auto binding = power_binding(engine);
  FormulaOverrideStore store{engine};
  CHECK(store.set_override(binding, {.index = 4}, BigInt{29}) == Status::Confirmed);

  std::vector<std::future<FormulaValue>> reads;
  for (int count = 0; count < 8; ++count) {
    reads.push_back(std::async(std::launch::async, [&store, &binding] {
      return store.evaluate(request_for(binding, 4));
    }));
  }
  for (auto& read : reads) {
    const auto value = read.get();
    CHECK(value.source == FormulaValueSource::Override);
    CHECK(value.result.exact_integer() == 29);
  }
}

TEST(formula_override_store_rejects_invalid_modulus_and_too_small_override_budget) {
  auto engine = Engine::create({});
  const auto binding = power_binding(engine);
  FormulaOverrideStore store{engine};
  CHECK(store.set_override(binding, {.index = 4}, BigInt{1024}) == Status::Confirmed);

  auto modular = request_for(binding, 4, EvaluationMode::Modular);
  modular.modulus = 0;
  auto limited = request_for(binding, 4);
  limited.budget.max_bit_length = 10;

  CHECK(store.evaluate(modular).result.status == Status::InvalidArgument);
  CHECK(store.evaluate(limited).result.status == Status::BudgetExceeded);
}

TEST(formula_override_store_rejects_an_override_that_exceeds_its_trace_budget) {
  auto engine = Engine::create({});
  const auto binding = power_binding(engine);
  FormulaOverrideStore store{engine};
  CHECK(store.set_override(binding, {.index = 4}, BigInt{1024}) == Status::Confirmed);
  auto trace_limited = request_for(binding, 4);
  trace_limited.budget.max_trace_bytes = 0;

  const auto result = store.evaluate(trace_limited);

  CHECK(result.source == FormulaValueSource::Override);
  CHECK(result.result.status == Status::BudgetExceeded);
  CHECK(!result.result.integer.has_value());
  CHECK(result.result.trace == nullptr);
}

TEST(formula_override_store_ignores_idle_engine_cancellation_for_computed_values) {
  auto engine = Engine::create({});
  const auto binding = power_binding(engine);
  FormulaOverrideStore store{engine};
  engine.cancel();

  const auto result = store.evaluate(request_for(binding, 4));

  CHECK(result.source == FormulaValueSource::Computed);
  CHECK(result.result.status == Status::Confirmed);
}

TEST(formula_override_store_keeps_reads_valid_during_override_updates) {
  auto engine = Engine::create({.cache_max_entries = 32, .cache_max_bytes = 4096});
  const auto binding = power_binding(engine);
  FormulaOverrideStore store{engine};
  constexpr FormulaAddress address{.index = 6};
  std::atomic_bool start = false;
  std::atomic<std::uint32_t> failures = 0;

  std::thread writer([&] {
    start.wait(false);
    for (std::uint32_t iteration = 1; iteration <= 256; ++iteration) {
      if (store.set_override(binding, address, BigInt{iteration}) != Status::Confirmed) {
        ++failures;
      }
      if (!store.remove_override(binding, address)) ++failures;
    }
  });
  std::vector<std::thread> readers;
  for (std::uint32_t reader = 0; reader < 4; ++reader) {
    readers.emplace_back([&] {
      start.wait(false);
      for (std::uint32_t iteration = 0; iteration < 256; ++iteration) {
        const auto value = store.evaluate(request_for(binding, address.index));
        if (value.result.status != Status::Confirmed ||
            !value.result.integer.has_value() ||
            (value.source != FormulaValueSource::Computed &&
             value.source != FormulaValueSource::Override)) {
          ++failures;
        }
      }
    });
  }

  start.store(true);
  start.notify_all();
  writer.join();
  for (auto& reader : readers) reader.join();

  CHECK(failures.load() == 0);
  CHECK(store.set_override(binding, address, BigInt{41}) == Status::Confirmed);
  const auto final_value = store.evaluate(request_for(binding, address.index));
  CHECK(final_value.source == FormulaValueSource::Override);
  CHECK(final_value.result.exact_integer() == 41);
}
