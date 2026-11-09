#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "bruce/lhme/types.hpp"

namespace bruce::lhme {

enum class GateScope {
  Branch,
  Tile,
  Run,
  Cache,
  Publish,
  Projection,
};

struct GateEvent {
  GateScope scope = GateScope::Branch;
  Status status = Status::InternalError;
  std::string reason;
};

struct TraceRecord {
  std::string id;
  std::vector<GateEvent> events;

  [[nodiscard]] bool contains(GateScope scope) const;
};

[[nodiscard]] std::shared_ptr<const TraceRecord> make_trace(
    std::vector<GateEvent> events);

// Returns the platform-independent trace-budget size. It counts an 8-byte ID
// length, each byte of the ID, an 8-byte event count, then for every event a
// 1-byte scope, 1-byte status, 8-byte reason length, and each reason byte.
[[nodiscard]] std::uint64_t trace_budget_bytes(const TraceRecord& trace) noexcept;

// Replaces an over-budget result with an untraced BudgetExceeded result so a
// caller never receives a partial successful trace.
void enforce_trace_budget(EvaluationResult& result,
                          std::uint64_t max_trace_bytes) noexcept;

}  // namespace bruce::lhme
