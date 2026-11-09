#pragma once

#include <compare>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

#include "bruce/lhme/formula.hpp"
#include "bruce/lhme/technique.hpp"

namespace bruce::lhme {

struct CacheKey {
  std::string domain;
  std::array<std::byte, 32> specification_hash{};
  std::string formula_identity;
  std::uint32_t specification_version = 0;
  std::string technique_id;
  std::uint32_t technique_version = 0;
  std::string technique_parameters;
  std::int64_t index = 0;
  EvaluationMode mode = EvaluationMode::Symbolic;

  auto operator<=>(const CacheKey&) const = default;

  [[nodiscard]] static CacheKey exact(const FormulaSpec& spec, std::int64_t index);
  [[nodiscard]] static CacheKey metric(const FormulaSpec& spec, std::int64_t index);
  [[nodiscard]] static CacheKey technique(const TechniqueIdentity& identity,
                                          std::string_view parameters,
                                          std::int64_t index,
                                          EvaluationMode mode);
};

struct CacheConfig { std::size_t max_entries = 0; std::size_t max_bytes = 0; };
struct CacheStatistics { std::size_t resident_entries = 0; std::size_t resident_bytes = 0; };

enum class CacheEntryProducer {
  Unknown,
  FormulaApi,
  TechniqueApi,
};

struct CacheLookup {
  EvaluationResult result;
  CacheEntryProducer producer = CacheEntryProducer::Unknown;
};

class ResultCache {
 public:
  explicit ResultCache(CacheConfig config);
  ResultCache(const ResultCache& other);
  ResultCache& operator=(const ResultCache& other);
  ResultCache(ResultCache&& other) noexcept = default;
  ResultCache& operator=(ResultCache&& other) noexcept = default;
  [[nodiscard]] std::optional<EvaluationResult> lookup(const CacheKey& key) const;
  [[nodiscard]] std::optional<CacheLookup> lookup_with_producer(
      const CacheKey& key) const;
  void admit(const CacheKey& key, const EvaluationResult& result);
  void admit(const CacheKey& key, const EvaluationResult& result,
             CacheEntryProducer producer);
  [[nodiscard]] CacheStatistics statistics() const;

 private:
  using RecencyList = std::list<CacheKey>;
  struct CacheKeyHash {
    [[nodiscard]] std::size_t operator()(const CacheKey& key) const noexcept;
  };
  struct Entry {
    EvaluationResult result;
    CacheEntryProducer producer = CacheEntryProducer::Unknown;
    std::size_t bytes = 0;
    RecencyList::iterator recency_position;
  };
  using EntryMap = std::unordered_map<CacheKey, Entry, CacheKeyHash>;
  void copy_from_unlocked(const ResultCache& other);

  CacheConfig config_;
  mutable RecencyList recency_;
  mutable EntryMap entries_;
  std::size_t resident_bytes_ = 0;
  std::shared_ptr<std::mutex> mutex_;
};
}  // namespace bruce::lhme
