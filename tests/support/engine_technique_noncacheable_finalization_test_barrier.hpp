#pragma once

#include <atomic>
#include <stdexcept>

namespace bruce::lhme::test_support {
namespace detail {

enum class EngineTechniqueNoncacheableFinalizationBarrierState : unsigned int {
  Idle,
  Preparing,
  Armed,
};

inline std::atomic<EngineTechniqueNoncacheableFinalizationBarrierState>&
engine_technique_noncacheable_finalization_barrier_state() {
  static std::atomic<EngineTechniqueNoncacheableFinalizationBarrierState> state{
      EngineTechniqueNoncacheableFinalizationBarrierState::Idle};
  return state;
}

inline std::atomic_bool& engine_technique_noncacheable_finalization_barrier_reached() {
  static std::atomic_bool reached{false};
  return reached;
}

inline std::atomic_bool& engine_technique_noncacheable_finalization_barrier_released() {
  static std::atomic_bool released{true};
  return released;
}

}  // namespace detail

class EngineTechniqueNoncacheableFinalizationBarrier final {
 public:
  EngineTechniqueNoncacheableFinalizationBarrier() {
    auto& state = detail::engine_technique_noncacheable_finalization_barrier_state();
    auto expected = detail::EngineTechniqueNoncacheableFinalizationBarrierState::Idle;
    if (!state.compare_exchange_strong(
            expected,
            detail::EngineTechniqueNoncacheableFinalizationBarrierState::Preparing,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
      throw std::logic_error(
          "an engine technique non-cacheable finalization test barrier is already active");
    }

    detail::engine_technique_noncacheable_finalization_barrier_reached().store(
        false, std::memory_order_relaxed);
    detail::engine_technique_noncacheable_finalization_barrier_released().store(
        false, std::memory_order_relaxed);
    state.store(detail::EngineTechniqueNoncacheableFinalizationBarrierState::Armed,
                std::memory_order_release);
  }

  EngineTechniqueNoncacheableFinalizationBarrier(
      const EngineTechniqueNoncacheableFinalizationBarrier&) = delete;
  EngineTechniqueNoncacheableFinalizationBarrier& operator=(
      const EngineTechniqueNoncacheableFinalizationBarrier&) = delete;

  ~EngineTechniqueNoncacheableFinalizationBarrier() {
    release();
    detail::engine_technique_noncacheable_finalization_barrier_state().store(
        detail::EngineTechniqueNoncacheableFinalizationBarrierState::Idle,
        std::memory_order_release);
    detail::engine_technique_noncacheable_finalization_barrier_state().notify_all();
  }

  void wait_until_reached() const noexcept {
    auto& reached = detail::engine_technique_noncacheable_finalization_barrier_reached();
    while (!reached.load(std::memory_order_acquire)) {
      reached.wait(false, std::memory_order_relaxed);
    }
  }

  void release() const noexcept {
    auto& released = detail::engine_technique_noncacheable_finalization_barrier_released();
    released.store(true, std::memory_order_release);
    released.notify_all();
  }
};

inline void wait_for_engine_technique_noncacheable_finalization_test_barrier() noexcept {
  if (detail::engine_technique_noncacheable_finalization_barrier_state().load(
          std::memory_order_acquire) !=
      detail::EngineTechniqueNoncacheableFinalizationBarrierState::Armed) {
    return;
  }

  auto& reached = detail::engine_technique_noncacheable_finalization_barrier_reached();
  reached.store(true, std::memory_order_release);
  reached.notify_all();

  auto& released = detail::engine_technique_noncacheable_finalization_barrier_released();
  while (!released.load(std::memory_order_acquire)) {
    released.wait(false, std::memory_order_relaxed);
  }
}

}  // namespace bruce::lhme::test_support
