#pragma once

#include <atomic>
#include <stdexcept>

namespace bruce::lhme::test_support {
namespace detail {

enum class RangeEvaluationBarrierState : unsigned int {
  Idle,
  Preparing,
  Armed,
};

inline std::atomic<RangeEvaluationBarrierState>& range_barrier_state() {
  static std::atomic<RangeEvaluationBarrierState> state{
      RangeEvaluationBarrierState::Idle};
  return state;
}

inline std::atomic_bool& range_barrier_reached() {
  static std::atomic_bool reached{false};
  return reached;
}

inline std::atomic_bool& range_barrier_released() {
  static std::atomic_bool released{true};
  return released;
}

}  // namespace detail

class RangeEvaluationBarrier final {
 public:
  RangeEvaluationBarrier() {
    auto& state = detail::range_barrier_state();
    auto expected = detail::RangeEvaluationBarrierState::Idle;
    if (!state.compare_exchange_strong(
            expected, detail::RangeEvaluationBarrierState::Preparing,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
      throw std::logic_error("a range evaluation test barrier is already active");
    }

    detail::range_barrier_reached().store(false, std::memory_order_relaxed);
    detail::range_barrier_released().store(false, std::memory_order_relaxed);
    state.store(detail::RangeEvaluationBarrierState::Armed,
                std::memory_order_release);
  }

  RangeEvaluationBarrier(const RangeEvaluationBarrier&) = delete;
  RangeEvaluationBarrier& operator=(const RangeEvaluationBarrier&) = delete;

  ~RangeEvaluationBarrier() {
    release();
    detail::range_barrier_state().store(
        detail::RangeEvaluationBarrierState::Idle, std::memory_order_release);
    detail::range_barrier_state().notify_all();
  }

  void wait_until_reached() const noexcept {
    auto& reached = detail::range_barrier_reached();
    while (!reached.load(std::memory_order_acquire)) {
      reached.wait(false, std::memory_order_relaxed);
    }
  }

  void release() const noexcept {
    auto& released = detail::range_barrier_released();
    released.store(true, std::memory_order_release);
    released.notify_all();
  }
};

inline void wait_for_range_evaluation_test_barrier() noexcept {
  if (detail::range_barrier_state().load(std::memory_order_acquire) !=
      detail::RangeEvaluationBarrierState::Armed) {
    return;
  }

  auto& reached = detail::range_barrier_reached();
  reached.store(true, std::memory_order_release);
  reached.notify_all();

  auto& released = detail::range_barrier_released();
  while (!released.load(std::memory_order_acquire)) {
    released.wait(false, std::memory_order_relaxed);
  }
}

}  // namespace bruce::lhme::test_support
