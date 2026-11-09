#pragma once

#include <atomic>
#include <stdexcept>

namespace bruce::lhme::test_support {
namespace detail {

enum class SchedulerFinalTileAggregationBarrierState : unsigned int {
  Idle,
  Preparing,
  Armed,
};

inline std::atomic<SchedulerFinalTileAggregationBarrierState>&
scheduler_final_tile_aggregation_barrier_state() {
  static std::atomic<SchedulerFinalTileAggregationBarrierState> state{
      SchedulerFinalTileAggregationBarrierState::Idle};
  return state;
}

inline std::atomic_bool& scheduler_final_tile_aggregation_barrier_reached() {
  static std::atomic_bool reached{false};
  return reached;
}

inline std::atomic_bool& scheduler_final_tile_aggregation_barrier_released() {
  static std::atomic_bool released{true};
  return released;
}

}  // namespace detail

class SchedulerFinalTileAggregationBarrier final {
 public:
  SchedulerFinalTileAggregationBarrier() {
    auto& state = detail::scheduler_final_tile_aggregation_barrier_state();
    auto expected = detail::SchedulerFinalTileAggregationBarrierState::Idle;
    if (!state.compare_exchange_strong(
            expected,
            detail::SchedulerFinalTileAggregationBarrierState::Preparing,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
      throw std::logic_error(
          "a scheduler final-tile aggregation test barrier is already active");
    }

    detail::scheduler_final_tile_aggregation_barrier_reached().store(
        false, std::memory_order_relaxed);
    detail::scheduler_final_tile_aggregation_barrier_released().store(
        false, std::memory_order_relaxed);
    state.store(detail::SchedulerFinalTileAggregationBarrierState::Armed,
                std::memory_order_release);
  }

  SchedulerFinalTileAggregationBarrier(
      const SchedulerFinalTileAggregationBarrier&) = delete;
  SchedulerFinalTileAggregationBarrier& operator=(
      const SchedulerFinalTileAggregationBarrier&) = delete;

  ~SchedulerFinalTileAggregationBarrier() {
    release();
    detail::scheduler_final_tile_aggregation_barrier_state().store(
        detail::SchedulerFinalTileAggregationBarrierState::Idle,
        std::memory_order_release);
    detail::scheduler_final_tile_aggregation_barrier_state().notify_all();
  }

  void wait_until_reached() const noexcept {
    auto& reached = detail::scheduler_final_tile_aggregation_barrier_reached();
    while (!reached.load(std::memory_order_acquire)) {
      reached.wait(false, std::memory_order_relaxed);
    }
  }

  void release() const noexcept {
    auto& released = detail::scheduler_final_tile_aggregation_barrier_released();
    released.store(true, std::memory_order_release);
    released.notify_all();
  }
};

inline void wait_for_scheduler_final_tile_aggregation_test_barrier() noexcept {
  if (detail::scheduler_final_tile_aggregation_barrier_state().load(
          std::memory_order_acquire) !=
      detail::SchedulerFinalTileAggregationBarrierState::Armed) {
    return;
  }

  auto& reached = detail::scheduler_final_tile_aggregation_barrier_reached();
  reached.store(true, std::memory_order_release);
  reached.notify_all();

  auto& released = detail::scheduler_final_tile_aggregation_barrier_released();
  while (!released.load(std::memory_order_acquire)) {
    released.wait(false, std::memory_order_relaxed);
  }
}

}  // namespace bruce::lhme::test_support
