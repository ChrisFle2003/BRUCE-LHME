#include "test.hpp"

#include "bruce/lhme/cache.hpp"
#include "bruce/lhme/technique.hpp"

#include <barrier>
#include <cstddef>
#include <future>
#include <vector>

namespace {

bruce::lhme::FormulaSpec spec(std::uint32_t version = 1) {
  return bruce::lhme::make_power_offset_spec(2, 100, version);
}

bruce::lhme::EvaluationResult confirmed(std::int64_t index) {
  return bruce::lhme::evaluate({.spec = spec(),
                                .address = {.index = index},
                                .mode = bruce::lhme::EvaluationMode::Exact,
                                .budget = bruce::lhme::Budget::unlimited()});
}

bruce::lhme::CacheKey key_with_all_fields() {
  bruce::lhme::CacheKey key{
      .domain = "formula",
      .formula_identity = "power_offset(base=2,scale=100)",
      .specification_version = 7,
      .technique_id = "bruce.test",
      .technique_version = 11,
      .technique_parameters = "base=2;scale=100",
      .index = 13,
      .mode = bruce::lhme::EvaluationMode::Exact,
  };
  for (std::size_t index = 0; index < key.specification_hash.size(); ++index) {
    key.specification_hash[index] = static_cast<std::byte>(index + 1U);
  }
  return key;
}

}  // namespace

TEST(cache_round_trip_preserves_confirmed_result) {
  bruce::lhme::ResultCache cache({.max_entries = 4, .max_bytes = 1024});
  const auto key = bruce::lhme::CacheKey::exact(spec(), 10);
  const auto original = confirmed(10);
  cache.admit(key, original);
  const auto cached = cache.lookup(key);

  CHECK(cached.has_value());
  CHECK(cached->exact_integer() == original.exact_integer());
  CHECK(cached->status == original.status);
}

TEST(mode_or_version_mismatch_is_a_cache_miss) {
  bruce::lhme::ResultCache cache({.max_entries = 4, .max_bytes = 1024});
  cache.admit(bruce::lhme::CacheKey::exact(spec(), 10), confirmed(10));

  CHECK(!cache.lookup(bruce::lhme::CacheKey::metric(spec(), 10)).has_value());
  CHECK(!cache.lookup(bruce::lhme::CacheKey::exact(spec(2), 10)).has_value());
}

TEST(cache_never_exceeds_configured_byte_limit) {
  bruce::lhme::ResultCache cache({.max_entries = 8, .max_bytes = 1});
  cache.admit(bruce::lhme::CacheKey::exact(spec(), 10), confirmed(10));
  CHECK(cache.statistics().resident_bytes <= 1);
}

TEST(cache_byte_budget_counts_technique_parameter_bytes) {
  bruce::lhme::ResultCache cache({.max_entries = 4, .max_bytes = 512});
  const auto metric = bruce::lhme::evaluate(
      {.spec = spec(),
       .address = {.index = 10},
       .mode = bruce::lhme::EvaluationMode::Metric,
       .budget = bruce::lhme::Budget::unlimited()});

  cache.admit(
      bruce::lhme::CacheKey::technique(
          {.id = "bruce.test", .version = 1}, std::string(1024, 'p'), 1,
          bruce::lhme::EvaluationMode::Metric),
      metric);

  CHECK(cache.statistics().resident_entries == 0);
  CHECK(cache.statistics().resident_bytes == 0);
}

TEST(cache_replacing_an_entry_keeps_resident_bytes_accurate) {
  bruce::lhme::ResultCache cache({.max_entries = 4, .max_bytes = 4096});
  const auto key = bruce::lhme::CacheKey::exact(spec(), 5);

  cache.admit(key, confirmed(5));
  const auto first_bytes = cache.statistics().resident_bytes;
  cache.admit(key, confirmed(5));

  CHECK(cache.statistics().resident_entries == 1);
  CHECK(first_bytes > 0);
  CHECK(cache.statistics().resident_bytes == first_bytes);
}

TEST(cache_hit_keeps_the_entry_that_was_most_recently_used) {
  bruce::lhme::ResultCache cache({.max_entries = 2, .max_bytes = 4096});
  const auto first = bruce::lhme::CacheKey::exact(spec(), 1);
  const auto second = bruce::lhme::CacheKey::exact(spec(), 2);
  const auto third = bruce::lhme::CacheKey::exact(spec(), 3);

  cache.admit(first, confirmed(1));
  cache.admit(second, confirmed(2));
  CHECK(cache.lookup(first).has_value());
  cache.admit(third, confirmed(3));

  CHECK(cache.lookup(first).has_value());
  CHECK(!cache.lookup(second).has_value());
  CHECK(cache.lookup(third).has_value());
}

TEST(cache_miss_does_not_change_recency) {
  bruce::lhme::ResultCache cache({.max_entries = 2, .max_bytes = 4096});
  const auto first = bruce::lhme::CacheKey::exact(spec(), 1);
  const auto second = bruce::lhme::CacheKey::exact(spec(), 2);
  const auto missing = bruce::lhme::CacheKey::exact(spec(), 99);
  const auto third = bruce::lhme::CacheKey::exact(spec(), 3);

  cache.admit(first, confirmed(1));
  cache.admit(second, confirmed(2));
  CHECK(!cache.lookup(missing).has_value());
  cache.admit(third, confirmed(3));

  CHECK(!cache.lookup(first).has_value());
  CHECK(cache.lookup(second).has_value());
  CHECK(cache.lookup(third).has_value());
}

TEST(cache_replacement_counts_as_the_most_recent_use) {
  bruce::lhme::ResultCache cache({.max_entries = 2, .max_bytes = 4096});
  const auto first = bruce::lhme::CacheKey::exact(spec(), 1);
  const auto second = bruce::lhme::CacheKey::exact(spec(), 2);
  const auto third = bruce::lhme::CacheKey::exact(spec(), 3);

  cache.admit(first, confirmed(1));
  cache.admit(second, confirmed(2));
  cache.admit(first, confirmed(1));
  cache.admit(third, confirmed(3));

  CHECK(cache.lookup(first).has_value());
  CHECK(!cache.lookup(second).has_value());
  CHECK(cache.lookup(third).has_value());
}

TEST(cache_byte_budget_evicts_the_least_recently_used_entry) {
  const auto first = bruce::lhme::CacheKey::exact(spec(), 1);
  const auto second = bruce::lhme::CacheKey::exact(spec(), 2);
  const auto third = bruce::lhme::CacheKey::exact(spec(), 3);
  const auto result = confirmed(1);
  bruce::lhme::ResultCache measurement({.max_entries = 4, .max_bytes = 4096});
  measurement.admit(first, result);
  const auto entry_bytes = measurement.statistics().resident_bytes;
  bruce::lhme::ResultCache cache({.max_entries = 3, .max_bytes = entry_bytes * 2});

  cache.admit(first, result);
  cache.admit(second, result);
  CHECK(cache.lookup(first).has_value());
  cache.admit(third, result);

  CHECK(cache.lookup(first).has_value());
  CHECK(!cache.lookup(second).has_value());
  CHECK(cache.lookup(third).has_value());
}

TEST(cache_does_not_admit_non_confirmed_results) {
  bruce::lhme::ResultCache cache({.max_entries = 2, .max_bytes = 4096});
  const auto key = bruce::lhme::CacheKey::exact(spec(), 1);
  auto result = confirmed(1);
  result.status = bruce::lhme::Status::Cancelled;

  cache.admit(key, result);

  CHECK(!cache.lookup(key).has_value());
  CHECK(cache.statistics().resident_entries == 0);
}

TEST(cache_parallel_admission_and_lookups_preserve_confirmed_results) {
  bruce::lhme::ResultCache cache({.max_entries = 4, .max_bytes = 4096});
  std::barrier start{4};
  std::vector<std::future<bool>> workers;
  workers.reserve(4);
  for (std::int64_t worker = 0; worker < 4; ++worker) {
    workers.push_back(std::async(std::launch::async, [&cache, &start, worker] {
      const auto key = bruce::lhme::CacheKey::exact(spec(), worker);
      const auto result = confirmed(worker);
      start.arrive_and_wait();
      for (int iteration = 0; iteration < 100; ++iteration) {
        cache.admit(key, result);
        const auto cached = cache.lookup(key);
        if (!cached.has_value() || cached->status != bruce::lhme::Status::Confirmed ||
            cached->exact_integer() != result.exact_integer()) {
          return false;
        }
      }
      return true;
    }));
  }

  for (auto& worker : workers) CHECK(worker.get());
  const auto statistics = cache.statistics();
  CHECK(statistics.resident_entries == 4);
  CHECK(statistics.resident_bytes <= 4096);
}

TEST(cache_repeated_evictions_preserve_the_least_recently_used_entry) {
  constexpr std::int64_t capacity = 16;
  bruce::lhme::ResultCache cache({.max_entries = capacity, .max_bytes = 1U << 20U});

  for (std::int64_t index = 0; index < capacity * 4; ++index) {
    cache.admit(bruce::lhme::CacheKey::exact(spec(), index), confirmed(index));
  }

  CHECK(cache.statistics().resident_entries == capacity);
  for (std::int64_t index = 0; index < capacity * 3; ++index) {
    CHECK(!cache.lookup(bruce::lhme::CacheKey::exact(spec(), index)).has_value());
  }
  for (std::int64_t index = capacity * 3; index < capacity * 4; ++index) {
    CHECK(cache.lookup(bruce::lhme::CacheKey::exact(spec(), index)).has_value());
  }

  cache.admit(bruce::lhme::CacheKey::exact(spec(), capacity * 4),
              confirmed(capacity * 4));

  CHECK(!cache.lookup(bruce::lhme::CacheKey::exact(spec(), capacity * 3)).has_value());
  CHECK(cache.lookup(bruce::lhme::CacheKey::exact(spec(), capacity * 4)).has_value());
}

TEST(cache_copy_preserves_independent_lru_order) {
  bruce::lhme::ResultCache original({.max_entries = 2, .max_bytes = 4096});
  const auto first = bruce::lhme::CacheKey::exact(spec(), 1);
  const auto second = bruce::lhme::CacheKey::exact(spec(), 2);
  const auto third = bruce::lhme::CacheKey::exact(spec(), 3);
  original.admit(first, confirmed(1));
  original.admit(second, confirmed(2));
  auto copy = original;

  CHECK(copy.lookup(first).has_value());
  copy.admit(third, confirmed(3));

  CHECK(copy.lookup(first).has_value());
  CHECK(!copy.lookup(second).has_value());
  CHECK(copy.lookup(third).has_value());
  CHECK(original.lookup(first).has_value());
  CHECK(original.lookup(second).has_value());
  CHECK(!original.lookup(third).has_value());
}

TEST(cache_move_preserves_lru_order) {
  bruce::lhme::ResultCache source({.max_entries = 2, .max_bytes = 4096});
  const auto first = bruce::lhme::CacheKey::exact(spec(), 1);
  const auto second = bruce::lhme::CacheKey::exact(spec(), 2);
  const auto third = bruce::lhme::CacheKey::exact(spec(), 3);
  source.admit(first, confirmed(1));
  source.admit(second, confirmed(2));

  auto moved = std::move(source);
  CHECK(moved.lookup(first).has_value());
  moved.admit(third, confirmed(3));

  CHECK(moved.lookup(first).has_value());
  CHECK(!moved.lookup(second).has_value());
  CHECK(moved.lookup(third).has_value());
}

TEST(cache_with_a_zero_byte_budget_rejects_metric_entries_too) {
  bruce::lhme::ResultCache cache({.max_entries = 4, .max_bytes = 0});
  const auto value = bruce::lhme::evaluate({.spec = spec(),
                                             .address = {.index = 10},
                                             .mode = bruce::lhme::EvaluationMode::Metric,
                                             .budget = bruce::lhme::Budget::unlimited()});

  cache.admit(bruce::lhme::CacheKey::metric(spec(), 10), value);

  CHECK(cache.statistics().resident_entries == 0);
}

TEST(technique_cache_key_separates_length_ambiguous_parameters) {
  const auto first = bruce::lhme::CacheKey::technique(
      {.id = "example", .version = 1}, "base=1:1;scale=2:23", 10,
      bruce::lhme::EvaluationMode::Exact);
  const auto second = bruce::lhme::CacheKey::technique(
      {.id = "example", .version = 1}, "base=2:12;scale=1:3", 10,
      bruce::lhme::EvaluationMode::Exact);

  CHECK(first != second);
}

TEST(cache_keys_with_different_hash_relevant_fields_remain_distinct) {
  bruce::lhme::ResultCache cache({.max_entries = 16, .max_bytes = 1U << 20U});
  const auto base = key_with_all_fields();
  std::vector<bruce::lhme::CacheKey> variants;

  auto domain = base;
  domain.domain = "technique";
  variants.push_back(std::move(domain));

  auto specification_hash = base;
  specification_hash.specification_hash[0] = std::byte{0xFF};
  variants.push_back(std::move(specification_hash));

  auto formula_identity = base;
  formula_identity.formula_identity = "another_formula";
  variants.push_back(std::move(formula_identity));

  auto specification_version = base;
  ++specification_version.specification_version;
  variants.push_back(std::move(specification_version));

  auto technique_id = base;
  technique_id.technique_id = "bruce.other";
  variants.push_back(std::move(technique_id));

  auto technique_version = base;
  ++technique_version.technique_version;
  variants.push_back(std::move(technique_version));

  auto technique_parameters = base;
  technique_parameters.technique_parameters = "base=3;scale=100";
  variants.push_back(std::move(technique_parameters));

  auto index = base;
  ++index.index;
  variants.push_back(std::move(index));

  auto mode = base;
  mode.mode = bruce::lhme::EvaluationMode::Metric;
  variants.push_back(std::move(mode));

  const auto base_result = confirmed(1);
  cache.admit(base, base_result);
  for (std::size_t position = 0; position < variants.size(); ++position) {
    cache.admit(variants[position], confirmed(static_cast<std::int64_t>(position + 2U)));
  }

  CHECK(cache.statistics().resident_entries == variants.size() + 1U);
  CHECK(cache.lookup(base)->exact_integer() == base_result.exact_integer());
  for (std::size_t position = 0; position < variants.size(); ++position) {
    const auto cached = cache.lookup(variants[position]);
    CHECK(cached.has_value());
    CHECK(cached->exact_integer() ==
          confirmed(static_cast<std::int64_t>(position + 2U)).exact_integer());
  }
}

TEST(cache_preserves_lru_order_through_larger_deterministic_churn) {
  constexpr std::int64_t capacity = 128;
  constexpr std::int64_t total_entries = capacity * 4;
  bruce::lhme::ResultCache cache({.max_entries = capacity, .max_bytes = 1U << 20U});

  for (std::int64_t index = 0; index < total_entries; ++index) {
    cache.admit(bruce::lhme::CacheKey::exact(spec(), index), confirmed(index));
  }

  CHECK(cache.statistics().resident_entries == capacity);
  for (std::int64_t index = total_entries - capacity; index < total_entries; ++index) {
    CHECK(cache.lookup(bruce::lhme::CacheKey::exact(spec(), index)).has_value());
  }

  cache.admit(bruce::lhme::CacheKey::exact(spec(), total_entries),
              confirmed(total_entries));

  CHECK(!cache.lookup(bruce::lhme::CacheKey::exact(spec(), total_entries - capacity)).has_value());
  CHECK(cache.lookup(bruce::lhme::CacheKey::exact(spec(), total_entries - capacity + 1)).has_value());
  CHECK(cache.lookup(bruce::lhme::CacheKey::exact(spec(), total_entries)).has_value());
}
