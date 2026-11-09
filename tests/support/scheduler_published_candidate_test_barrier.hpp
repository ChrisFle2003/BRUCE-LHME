#pragma once

#include <atomic>
#include <stdexcept>

namespace bruce::lhme::test_support {
namespace detail {

enum class SchedulerPublishedCandidateBarrierState : unsigned int {
  Idle,
  Preparing,
  Armed,
};

inline std::atomic<SchedulerPublishedCandidateBarrierState>&
scheduler_published_candidate_barrier_state() {
  static std::atomic<SchedulerPublishedCandidateBarrierState> state{
      SchedulerPublishedCandidateBarrierState::Idle};
  return state;
}

inline std::atomic_bool& scheduler_published_candidate_barrier_reached() {
  static std::atomic_bool reached{false};
  return reached;
}

inline std::atomic_bool& scheduler_published_candidate_barrier_released() {
  static std::atomic_bool released{true};
  return released;
}

}  // namespace detail

class SchedulerPublishedCandidateBarrier final {
 public:
  SchedulerPublishedCandidateBarrier() {
    auto& state = detail::scheduler_published_candidate_barrier_state();
    auto expected = detail::SchedulerPublishedCandidateBarrierState::Idle;
    if (!state.compare_exchange_strong(
            expected,
            detail::SchedulerPublishedCandidateBarrierState::Preparing,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
      throw std::logic_error(
          "a scheduler published-candidate test barrier is already active");
    }

    detail::scheduler_published_candidate_barrier_reached().store(
        false, std::memory_order_relaxed);
    detail::scheduler_published_candidate_barrier_released().store(
        false, std::memory_order_relaxed);
    state.store(detail::SchedulerPublishedCandidateBarrierState::Armed,
                std::memory_order_release);
  }

  SchedulerPublishedCandidateBarrier(
      const SchedulerPublishedCandidateBarrier&) = delete;
  SchedulerPublishedCandidateBarrier& operator=(
      const SchedulerPublishedCandidateBarrier&) = delete;

  ~SchedulerPublishedCandidateBarrier() {
    release();
    detail::scheduler_published_candidate_barrier_state().store(
        detail::SchedulerPublishedCandidateBarrierState::Idle,
        std::memory_order_release);
    detail::scheduler_published_candidate_barrier_state().notify_all();
  }

  void wait_until_reached() const noexcept {
    auto& reached = detail::scheduler_published_candidate_barrier_reached();
    while (!reached.load(std::memory_order_acquire)) {
      reached.wait(false, std::memory_order_relaxed);
    }
  }

  void release() const noexcept {
    auto& released = detail::scheduler_published_candidate_barrier_released();
    released.store(true, std::memory_order_release);
    released.notify_all();
  }
};

inline void wait_for_scheduler_published_candidate_test_barrier() noexcept {
  if (detail::scheduler_published_candidate_barrier_state().load(
          std::memory_order_acquire) !=
      detail::SchedulerPublishedCandidateBarrierState::Armed) {
    return;
  }

  auto& reached = detail::scheduler_published_candidate_barrier_reached();
  reached.store(true, std::memory_order_release);
  reached.notify_all();

  auto& released = detail::scheduler_published_candidate_barrier_released();
  while (!released.load(std::memory_order_acquire)) {
    released.wait(false, std::memory_order_relaxed);
  }
}

}  // namespace bruce::lhme::test_support
