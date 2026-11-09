#include "bruce/lhme/trace.hpp"

#include <cstdint>
#include <limits>
#include <utility>

namespace bruce::lhme {
namespace {

constexpr std::uint64_t kLengthFieldBytes = 8;
constexpr std::uint64_t kEventTagBytes = 2;

std::uint64_t saturating_add(std::uint64_t left, std::uint64_t right) {
  return left > std::numeric_limits<std::uint64_t>::max() - right
             ? std::numeric_limits<std::uint64_t>::max()
             : left + right;
}

}  // namespace

bool TraceRecord::contains(GateScope scope) const {
  for (const auto& event : events) {
    if (event.scope == scope) {
      return true;
    }
  }
  return false;
}

std::shared_ptr<const TraceRecord> make_trace(std::vector<GateEvent> events) {
  std::uint64_t hash = 14695981039346656037ULL;
  for (const auto& event : events) {
    hash ^= static_cast<std::uint64_t>(event.scope);
    hash *= 1099511628211ULL;
    hash ^= static_cast<std::uint64_t>(event.status);
    hash *= 1099511628211ULL;
    for (const auto character : event.reason) {
      hash ^= static_cast<unsigned char>(character);
      hash *= 1099511628211ULL;
    }
  }
  return std::make_shared<const TraceRecord>(
      TraceRecord{.id = std::to_string(hash), .events = std::move(events)});
}

std::uint64_t trace_budget_bytes(const TraceRecord& trace) noexcept {
  auto bytes = saturating_add(kLengthFieldBytes,
                              static_cast<std::uint64_t>(trace.id.size()));
  bytes = saturating_add(bytes, kLengthFieldBytes);
  for (const auto& event : trace.events) {
    bytes = saturating_add(bytes, kEventTagBytes);
    bytes = saturating_add(bytes, kLengthFieldBytes);
    bytes = saturating_add(bytes,
                           static_cast<std::uint64_t>(event.reason.size()));
  }
  return bytes;
}

void enforce_trace_budget(EvaluationResult& result,
                          std::uint64_t max_trace_bytes) noexcept {
  if (result.trace == nullptr ||
      trace_budget_bytes(*result.trace) <= max_trace_bytes) {
    return;
  }
  result.status = Status::BudgetExceeded;
  result.integer.reset();
  result.metric_bit_length.reset();
  result.symbolic_expression.reset();
  result.trace.reset();
}

}  // namespace bruce::lhme
