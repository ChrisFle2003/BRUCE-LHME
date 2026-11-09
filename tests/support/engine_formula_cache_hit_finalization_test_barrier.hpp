#pragma once

#include <atomic>
#include <stdexcept>

namespace bruce::lhme::test_support {
namespace detail {

enum class EngineFormulaCacheHitFinalizationBarrierState : unsigned int {
  Idle,
  Preparing,
  Armed,
};

inline std::atomic<EngineFormulaCacheHitFinalizationBarrierState>&
engine_formula_cache_hit_finalization_barrier_state() {
  static std::atomic<EngineFormulaCacheHitFinalizationBarrierState> state{
      EngineFormulaCacheHitFinalizationBarrierState::Idle};
  return state;
}

inline std::atomic_bool& engine_formula_cache_hit_finalization_barrier_reached() {
  static std::atomic_bool reached{false};
  return reached;
}

inline std::atomic_bool& engine_formula_cache_hit_finalization_barrier_released() {
  static std::atomic_bool released{true};
  return released;
}

}  // namespace detail

class EngineFormulaCacheHitFinalizationBarrier final {
 public:
  EngineFormulaCacheHitFinalizationBarrier() {
    auto& state = detail::engine_formula_cache_hit_finalization_barrier_state();
    auto expected = detail::EngineFormulaCacheHitFinalizationBarrierState::Idle;
    if (!state.compare_exchange_strong(
            expected,
            detail::EngineFormulaCacheHitFinalizationBarrierState::Preparing,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
      throw std::logic_error(
          "an engine formula cache-hit finalization test barrier is already active");
    }

    detail::engine_formula_cache_hit_finalization_barrier_reached().store(
        false, std::memory_order_relaxed);
    detail::engine_formula_cache_hit_finalization_barrier_released().store(
        false, std::memory_order_relaxed);
    state.store(detail::EngineFormulaCacheHitFinalizationBarrierState::Armed,
                std::memory_order_release);
  }

  EngineFormulaCacheHitFinalizationBarrier(
      const EngineFormulaCacheHitFinalizationBarrier&) = delete;
  EngineFormulaCacheHitFinalizationBarrier& operator=(
      const EngineFormulaCacheHitFinalizationBarrier&) = delete;

  ~EngineFormulaCacheHitFinalizationBarrier() {
    release();
    detail::engine_formula_cache_hit_finalization_barrier_state().store(
        detail::EngineFormulaCacheHitFinalizationBarrierState::Idle,
        std::memory_order_release);
    detail::engine_formula_cache_hit_finalization_barrier_state().notify_all();
  }

  void wait_until_reached() const noexcept {
    auto& reached = detail::engine_formula_cache_hit_finalization_barrier_reached();
    while (!reached.load(std::memory_order_acquire)) {
      reached.wait(false, std::memory_order_relaxed);
    }
  }

  void release() const noexcept {
    auto& released = detail::engine_formula_cache_hit_finalization_barrier_released();
    released.store(true, std::memory_order_release);
    released.notify_all();
  }
};

inline void wait_for_engine_formula_cache_hit_finalization_test_barrier() noexcept {
  if (detail::engine_formula_cache_hit_finalization_barrier_state().load(
          std::memory_order_acquire) !=
      detail::EngineFormulaCacheHitFinalizationBarrierState::Armed) {
    return;
  }

  auto& reached = detail::engine_formula_cache_hit_finalization_barrier_reached();
  reached.store(true, std::memory_order_release);
  reached.notify_all();

  auto& released = detail::engine_formula_cache_hit_finalization_barrier_released();
  while (!released.load(std::memory_order_acquire)) {
    released.wait(false, std::memory_order_relaxed);
  }
}

}  // namespace bruce::lhme::test_support
