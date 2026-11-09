#pragma once

#include <atomic>
#include <stdexcept>

namespace bruce::lhme::test_support {
namespace detail {

enum class SchedulerTileBarrierState : unsigned int {
  Idle,
  Preparing,
  Armed,
};

inline std::atomic<SchedulerTileBarrierState>& scheduler_tile_barrier_state() {
  static std::atomic<SchedulerTileBarrierState> state{
      SchedulerTileBarrierState::Idle};
  return state;
}

inline std::atomic_bool& scheduler_tile_barrier_reached() {
  static std::atomic_bool reached{false};
  return reached;
}

inline std::atomic_bool& scheduler_tile_barrier_released() {
  static std::atomic_bool released{true};
  return released;
}

}  // namespace detail

class SchedulerTileBarrier final {
 public:
  SchedulerTileBarrier() {
    auto& state = detail::scheduler_tile_barrier_state();
    auto expected = detail::SchedulerTileBarrierState::Idle;
    if (!state.compare_exchange_strong(
            expected, detail::SchedulerTileBarrierState::Preparing,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
      throw std::logic_error("a scheduler tile test barrier is already active");
    }

    detail::scheduler_tile_barrier_reached().store(false,
                                                   std::memory_order_relaxed);
    detail::scheduler_tile_barrier_released().store(false,
                                                    std::memory_order_relaxed);
    state.store(detail::SchedulerTileBarrierState::Armed,
                std::memory_order_release);
  }

  SchedulerTileBarrier(const SchedulerTileBarrier&) = delete;
  SchedulerTileBarrier& operator=(const SchedulerTileBarrier&) = delete;

  ~SchedulerTileBarrier() {
    release();
    detail::scheduler_tile_barrier_state().store(
        detail::SchedulerTileBarrierState::Idle, std::memory_order_release);
    detail::scheduler_tile_barrier_state().notify_all();
  }

  void wait_until_reached() const noexcept {
    auto& reached = detail::scheduler_tile_barrier_reached();
    while (!reached.load(std::memory_order_acquire)) {
      reached.wait(false, std::memory_order_relaxed);
    }
  }

  void release() const noexcept {
    auto& released = detail::scheduler_tile_barrier_released();
    released.store(true, std::memory_order_release);
    released.notify_all();
  }
};

inline void wait_for_scheduler_tile_test_barrier() noexcept {
  if (detail::scheduler_tile_barrier_state().load(std::memory_order_acquire) !=
      detail::SchedulerTileBarrierState::Armed) {
    return;
  }

  auto& reached = detail::scheduler_tile_barrier_reached();
  reached.store(true, std::memory_order_release);
  reached.notify_all();

  auto& released = detail::scheduler_tile_barrier_released();
  while (!released.load(std::memory_order_acquire)) {
    released.wait(false, std::memory_order_relaxed);
  }
}

}  // namespace bruce::lhme::test_support
