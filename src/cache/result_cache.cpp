#include "bruce/lhme/cache.hpp"
#include "bruce/lhme/trace.hpp"

namespace bruce::lhme {
namespace {
constexpr std::uint64_t fnv_offset_basis = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

void hash_byte(std::uint64_t& state, std::uint8_t value) noexcept {
  state ^= value;
  state *= fnv_prime;
}

void hash_unsigned(std::uint64_t& state, std::uint64_t value) noexcept {
  for (unsigned int shift = 0; shift < 64U; shift += 8U) {
    hash_byte(state, static_cast<std::uint8_t>(value >> shift));
  }
}

void hash_string(std::uint64_t& state, const std::string& value) noexcept {
  hash_unsigned(state, value.size());
  for (const auto character : value) {
    hash_byte(state, static_cast<std::uint8_t>(static_cast<unsigned char>(character)));
  }
}

CacheKey make_key(const FormulaSpec& spec, std::int64_t index, EvaluationMode mode) {
  return {.domain = "formula",
          .specification_hash = spec.specification_hash,
          .formula_identity = canonical_formula_identity(spec),
          .specification_version = spec.version,
          .index = index,
          .mode = mode};
}

CacheKey make_technique_key(const TechniqueIdentity& identity,
                            std::string_view parameters, std::int64_t index,
                            EvaluationMode mode) {
  return {.domain = "technique",
          .technique_id = identity.id,
          .technique_version = identity.version,
          .technique_parameters = std::string{parameters},
          .index = index,
          .mode = mode};
}

class RecencyAppendRollback {
 public:
  explicit RecencyAppendRollback(std::list<CacheKey>& recency)
      : recency_(recency) {}

  ~RecencyAppendRollback() {
    if (active_) recency_.pop_back();
  }

  void release() noexcept { active_ = false; }

 private:
  std::list<CacheKey>& recency_;
  bool active_ = true;
};

std::size_t dynamic_string_bytes(const std::string& value) {
  return value.capacity();
}

std::size_t trace_bytes(const std::shared_ptr<const TraceRecord>& trace) {
  if (trace == nullptr) return 0;

  std::size_t bytes = sizeof(TraceRecord) + dynamic_string_bytes(trace->id) +
                      trace->events.capacity() * sizeof(GateEvent);
  for (const auto& event : trace->events) {
    bytes += dynamic_string_bytes(event.reason);
  }
  return bytes;
}

std::size_t result_bytes(const EvaluationResult& result) {
  std::size_t bytes = sizeof(EvaluationResult) + trace_bytes(result.trace);
  if (result.integer.has_value() && *result.integer != 0) {
    const auto magnitude = *result.integer < 0 ? -*result.integer : *result.integer;
    bytes += static_cast<std::size_t>(boost::multiprecision::msb(magnitude) / 8U) + 1U;
  }
  if (result.symbolic_expression.has_value()) {
    bytes += dynamic_string_bytes(*result.symbolic_expression);
  }
  return bytes;
}

std::size_t key_bytes(const CacheKey& key) {
  return sizeof(CacheKey) + dynamic_string_bytes(key.domain) +
         dynamic_string_bytes(key.formula_identity) +
         dynamic_string_bytes(key.technique_id) +
         dynamic_string_bytes(key.technique_parameters);
}

std::size_t entry_bytes(const CacheKey& key, const EvaluationResult& result) {
  constexpr std::size_t map_node_overhead = sizeof(void*) * 4U;
  return key_bytes(key) + result_bytes(result) + sizeof(std::uint64_t) +
         sizeof(std::size_t) + sizeof(CacheEntryProducer) + map_node_overhead;
}
}  // namespace

std::size_t ResultCache::CacheKeyHash::operator()(const CacheKey& key) const noexcept {
  auto state = fnv_offset_basis;
  hash_string(state, key.domain);
  for (const auto value : key.specification_hash) {
    hash_byte(state, static_cast<std::uint8_t>(value));
  }
  hash_string(state, key.formula_identity);
  hash_unsigned(state, key.specification_version);
  hash_string(state, key.technique_id);
  hash_unsigned(state, key.technique_version);
  hash_string(state, key.technique_parameters);
  hash_unsigned(state, static_cast<std::uint64_t>(key.index));
  hash_unsigned(state, static_cast<std::uint64_t>(key.mode));
  return static_cast<std::size_t>(state);
}

CacheKey CacheKey::exact(const FormulaSpec& spec, std::int64_t index) { return make_key(spec, index, EvaluationMode::Exact); }
CacheKey CacheKey::metric(const FormulaSpec& spec, std::int64_t index) { return make_key(spec, index, EvaluationMode::Metric); }
CacheKey CacheKey::technique(const TechniqueIdentity& identity,
                             std::string_view parameters, std::int64_t index,
                             EvaluationMode mode) {
  return make_technique_key(identity, parameters, index, mode);
}
ResultCache::ResultCache(CacheConfig config)
    : config_(config), mutex_(std::make_shared<std::mutex>()) {}
ResultCache::ResultCache(const ResultCache& other)
    : mutex_(other.mutex_) {
  const std::lock_guard lock(*other.mutex_);
  copy_from_unlocked(other);
}
ResultCache& ResultCache::operator=(const ResultCache& other) {
  if (this == &other) return *this;

  ResultCache copied(other);
  const auto destination_mutex = mutex_;
  {
    const std::lock_guard lock(*destination_mutex);
    config_ = copied.config_;
    recency_.swap(copied.recency_);
    entries_.swap(copied.entries_);
    resident_bytes_ = copied.resident_bytes_;
  }
  mutex_ = copied.mutex_;
  return *this;
}
void ResultCache::copy_from_unlocked(const ResultCache& other) {
  config_ = other.config_;
  resident_bytes_ = other.resident_bytes_;
  recency_ = other.recency_;
  for (auto position = recency_.begin(); position != recency_.end(); ++position) {
    const auto source = other.entries_.find(*position);
    entries_.emplace(*position,
                     Entry{.result = source->second.result,
                           .producer = source->second.producer,
                           .bytes = source->second.bytes,
                           .recency_position = position});
  }
}
std::optional<EvaluationResult> ResultCache::lookup(const CacheKey& key) const {
  const auto cached = lookup_with_producer(key);
  return cached.has_value() ? std::optional<EvaluationResult>{cached->result}
                            : std::nullopt;
}

std::optional<CacheLookup> ResultCache::lookup_with_producer(
    const CacheKey& key) const {
  const std::lock_guard lock(*mutex_);
  const auto it = entries_.find(key);
  if (it == entries_.end()) return std::nullopt;
  recency_.splice(recency_.end(), recency_, it->second.recency_position);
  return CacheLookup{.result = it->second.result, .producer = it->second.producer};
}
void ResultCache::admit(const CacheKey& key, const EvaluationResult& result) {
  admit(key, result, CacheEntryProducer::Unknown);
}
void ResultCache::admit(const CacheKey& key, const EvaluationResult& result,
                        CacheEntryProducer producer) {
  const std::lock_guard lock(*mutex_);
  if (result.status != Status::Confirmed || result.trace == nullptr) return;
  const auto bytes = entry_bytes(key, result);
  if (config_.max_entries == 0 || config_.max_bytes == 0 ||
      bytes > config_.max_bytes) return;
  if (const auto existing = entries_.find(key); existing != entries_.end()) {
    resident_bytes_ -= existing->second.bytes;
    recency_.erase(existing->second.recency_position);
    entries_.erase(existing);
  }
  while (!entries_.empty() &&
         (entries_.size() >= config_.max_entries ||
          bytes > config_.max_bytes - resident_bytes_)) {
    const auto victim = entries_.find(recency_.front());
    resident_bytes_ -= victim->second.bytes;
    recency_.pop_front();
    entries_.erase(victim);
  }
  recency_.push_back(key);
  RecencyAppendRollback rollback(recency_);
  entries_.emplace(key, Entry{.result = result,
                              .producer = producer,
                              .bytes = bytes,
                              .recency_position = std::prev(recency_.end())});
  rollback.release();
  resident_bytes_ += bytes;
}
CacheStatistics ResultCache::statistics() const {
  const std::lock_guard lock(*mutex_);
  return {.resident_entries = entries_.size(), .resident_bytes = resident_bytes_};
}
}  // namespace bruce::lhme
