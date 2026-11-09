#pragma once

#include <atomic>
#include <stdexcept>

namespace bruce::lhme::test_support {
namespace detail {

enum class SchedulerCandidateBarrierState : unsigned int {
  Idle,
  Preparing,
  Armed,
};

inline std::atomic<SchedulerCandidateBarrierState>&
scheduler_candidate_barrier_state() {
  static std::atomic<SchedulerCandidateBarrierState> state{
      SchedulerCandidateBarrierState::Idle};
  return state;
}

inline std::atomic_bool& scheduler_candidate_barrier_reached() {
  static std::atomic_bool reached{false};
  return reached;
}

inline std::atomic_bool& scheduler_candidate_barrier_released() {
  static std::atomic_bool released{true};
  return released;
}

}  // namespace detail

class SchedulerCandidateBarrier final {
 public:
  SchedulerCandidateBarrier() {
    auto& state = detail::scheduler_candidate_barrier_state();
    auto expected = detail::SchedulerCandidateBarrierState::Idle;
    if (!state.compare_exchange_strong(
            expected, detail::SchedulerCandidateBarrierState::Preparing,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
      throw std::logic_error(
          "a scheduler candidate test barrier is already active");
    }

    detail::scheduler_candidate_barrier_reached().store(
        false, std::memory_order_relaxed);
    detail::scheduler_candidate_barrier_released().store(
        false, std::memory_order_relaxed);
    state.store(detail::SchedulerCandidateBarrierState::Armed,
                std::memory_order_release);
  }

  SchedulerCandidateBarrier(const SchedulerCandidateBarrier&) = delete;
  SchedulerCandidateBarrier& operator=(const SchedulerCandidateBarrier&) =
      delete;

  ~SchedulerCandidateBarrier() {
    release();
    detail::scheduler_candidate_barrier_state().store(
        detail::SchedulerCandidateBarrierState::Idle, std::memory_order_release);
    detail::scheduler_candidate_barrier_state().notify_all();
  }

  void wait_until_reached() const noexcept {
    auto& reached = detail::scheduler_candidate_barrier_reached();
    while (!reached.load(std::memory_order_acquire)) {
      reached.wait(false, std::memory_order_relaxed);
    }
  }

  void release() const noexcept {
    auto& released = detail::scheduler_candidate_barrier_released();
    released.store(true, std::memory_order_release);
    released.notify_all();
  }
};

inline void wait_for_scheduler_candidate_test_barrier() noexcept {
  if (detail::scheduler_candidate_barrier_state().load(
          std::memory_order_acquire) !=
      detail::SchedulerCandidateBarrierState::Armed) {
    return;
  }

  auto& reached = detail::scheduler_candidate_barrier_reached();
  reached.store(true, std::memory_order_release);
  reached.notify_all();

  auto& released = detail::scheduler_candidate_barrier_released();
  while (!released.load(std::memory_order_acquire)) {
    released.wait(false, std::memory_order_relaxed);
  }
}

}  // namespace bruce::lhme::test_support
