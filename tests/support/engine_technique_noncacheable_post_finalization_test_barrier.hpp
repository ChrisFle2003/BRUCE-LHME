#pragma once

#include <atomic>
#include <stdexcept>

namespace bruce::lhme::test_support {

enum class EngineTechniqueNoncacheablePostFinalizationBarrierMode : unsigned int {
  AllArrivals,
  FirstArrivalOnly,
};

namespace detail {

enum class EngineTechniqueNoncacheablePostFinalizationBarrierState : unsigned int {
  Idle,
  Preparing,
  Armed,
};

inline std::atomic<EngineTechniqueNoncacheablePostFinalizationBarrierState>&
engine_technique_noncacheable_post_finalization_barrier_state() {
  static std::atomic<EngineTechniqueNoncacheablePostFinalizationBarrierState> state{
      EngineTechniqueNoncacheablePostFinalizationBarrierState::Idle};
  return state;
}

inline std::atomic_bool&
engine_technique_noncacheable_post_finalization_barrier_reached() {
  static std::atomic_bool reached{false};
  return reached;
}

inline std::atomic_bool&
engine_technique_noncacheable_post_finalization_barrier_released() {
  static std::atomic_bool released{true};
  return released;
}

inline std::atomic<EngineTechniqueNoncacheablePostFinalizationBarrierMode>&
engine_technique_noncacheable_post_finalization_barrier_mode() {
  static std::atomic<EngineTechniqueNoncacheablePostFinalizationBarrierMode> mode{
      EngineTechniqueNoncacheablePostFinalizationBarrierMode::AllArrivals};
  return mode;
}

inline std::atomic_uint&
engine_technique_noncacheable_post_finalization_barrier_arrivals() {
  static std::atomic_uint arrivals{0};
  return arrivals;
}

}  // namespace detail

class EngineTechniqueNoncacheablePostFinalizationBarrier final {
 public:
  explicit EngineTechniqueNoncacheablePostFinalizationBarrier(
      EngineTechniqueNoncacheablePostFinalizationBarrierMode mode =
          EngineTechniqueNoncacheablePostFinalizationBarrierMode::AllArrivals) {
    auto& state = detail::engine_technique_noncacheable_post_finalization_barrier_state();
    auto expected =
        detail::EngineTechniqueNoncacheablePostFinalizationBarrierState::Idle;
    if (!state.compare_exchange_strong(
            expected,
            detail::EngineTechniqueNoncacheablePostFinalizationBarrierState::Preparing,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
      throw std::logic_error(
          "an engine technique non-cacheable post-finalization test barrier is already active");
    }

    detail::engine_technique_noncacheable_post_finalization_barrier_reached().store(
        false, std::memory_order_relaxed);
    detail::engine_technique_noncacheable_post_finalization_barrier_released().store(
        false, std::memory_order_relaxed);
    detail::engine_technique_noncacheable_post_finalization_barrier_mode().store(
        mode, std::memory_order_relaxed);
    detail::engine_technique_noncacheable_post_finalization_barrier_arrivals().store(
        0, std::memory_order_relaxed);
    state.store(
        detail::EngineTechniqueNoncacheablePostFinalizationBarrierState::Armed,
        std::memory_order_release);
  }

  EngineTechniqueNoncacheablePostFinalizationBarrier(
      const EngineTechniqueNoncacheablePostFinalizationBarrier&) = delete;
  EngineTechniqueNoncacheablePostFinalizationBarrier& operator=(
      const EngineTechniqueNoncacheablePostFinalizationBarrier&) = delete;

  ~EngineTechniqueNoncacheablePostFinalizationBarrier() {
    release();
    detail::engine_technique_noncacheable_post_finalization_barrier_state().store(
        detail::EngineTechniqueNoncacheablePostFinalizationBarrierState::Idle,
        std::memory_order_release);
    detail::engine_technique_noncacheable_post_finalization_barrier_state().notify_all();
  }

  void wait_until_reached() const noexcept {
    auto& reached = detail::engine_technique_noncacheable_post_finalization_barrier_reached();
    while (!reached.load(std::memory_order_acquire)) {
      reached.wait(false, std::memory_order_relaxed);
    }
  }

  void release() const noexcept {
    auto& released = detail::engine_technique_noncacheable_post_finalization_barrier_released();
    released.store(true, std::memory_order_release);
    released.notify_all();
  }
};

inline void
wait_for_engine_technique_noncacheable_post_finalization_test_barrier() noexcept {
  if (detail::engine_technique_noncacheable_post_finalization_barrier_state().load(
          std::memory_order_acquire) !=
      detail::EngineTechniqueNoncacheablePostFinalizationBarrierState::Armed) {
    return;
  }

  const auto arrival =
      detail::engine_technique_noncacheable_post_finalization_barrier_arrivals()
          .fetch_add(1, std::memory_order_acq_rel);
  if (detail::engine_technique_noncacheable_post_finalization_barrier_mode().load(
          std::memory_order_acquire) ==
          EngineTechniqueNoncacheablePostFinalizationBarrierMode::FirstArrivalOnly &&
      arrival != 0) {
    return;
  }

  auto& reached = detail::engine_technique_noncacheable_post_finalization_barrier_reached();
  reached.store(true, std::memory_order_release);
  reached.notify_all();

  auto& released = detail::engine_technique_noncacheable_post_finalization_barrier_released();
  while (!released.load(std::memory_order_acquire)) {
    released.wait(false, std::memory_order_relaxed);
  }
}

}  // namespace bruce::lhme::test_support
