#include <atomic>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "bruce/lhme/engine.hpp"
#include "bruce/lhme/kernel_sdk.hpp"
#include "bruce/lhme/power_offset_kernel.hpp"
#include "bruce/lhme/trace.hpp"

namespace {

class AffineKernel final : public bruce::lhme::TechniqueKernel {
 public:
  [[nodiscard]] bruce::lhme::TechniqueIdentity identity() const override {
    return {.id = "example.affine", .version = 1};
  }

  [[nodiscard]] bruce::lhme::EvaluationResult evaluate(
      const bruce::lhme::TechniqueRequest& request) const override {
    if (bruce::lhme::cancellation_requested(request)) {
      return bruce::lhme::cancelled_technique_result(request.mode,
                                                      "kernel cancellation requested");
    }
    if (request.identity != identity() ||
        request.parameters != "factor=1:2;offset=1:3" ||
        request.address.index < 0 ||
        request.mode != bruce::lhme::EvaluationMode::Exact) {
      return bruce::lhme::invalid_technique_request_result(
          request.mode, "expected exact affine request");
    }

    return bruce::lhme::confirmed_exact_result(
        bruce::lhme::BigInt{request.address.index} * 2 + 3,
        bruce::lhme::make_trace({}));
  }
};

// Deliberately overrides only the long-standing TechniqueKernel operations:
// identity() and evaluate().  It does not provide a ComputationIdentity;
// cache equivalence is declared explicitly through Engine below.
class FormulaBackedKernel final : public bruce::lhme::TechniqueKernel {
 public:
  explicit FormulaBackedKernel(bruce::lhme::FormulaSpec formula)
      : formula_(std::move(formula)) {}

  [[nodiscard]] bruce::lhme::TechniqueIdentity identity() const override {
    return {.id = "example.formula_backed", .version = 1};
  }

  [[nodiscard]] bruce::lhme::EvaluationResult evaluate(
      const bruce::lhme::TechniqueRequest& request) const override {
    if (bruce::lhme::cancellation_requested(request)) {
      return bruce::lhme::cancelled_technique_result(request.mode,
                                                      "kernel cancellation requested");
    }
    if (request.identity != identity() || request.parameters != kParameters ||
        request.address.index < 0 ||
        request.mode != bruce::lhme::EvaluationMode::Exact) {
      return bruce::lhme::invalid_technique_request_result(
          request.mode, "expected exact formula-backed request");
    }

    ++calls;
    return bruce::lhme::evaluate({
        .spec = formula_,
        .address = request.address,
        .mode = request.mode,
        .budget = request.budget,
        .modulus = request.modulus,
        .cancellation_signal = request.cancellation_signal,
    });
  }

  mutable std::atomic_uint calls{0};

  static constexpr std::string_view kParameters = "base=1:2;scale=3:100";

 private:
  bruce::lhme::FormulaSpec formula_;
};

[[nodiscard]] bool is_neutral_cross_api_hit(
    const bruce::lhme::EvaluationResult& result) {
  return result.trace != nullptr && result.trace->events.size() == 1 &&
         result.trace->events.front().scope == bruce::lhme::GateScope::Cache &&
         result.trace->events.front().status == bruce::lhme::Status::Confirmed &&
         result.trace->events.front().reason == "cross-api exact cache hit";
}

}  // namespace

int main() {
  auto engine = bruce::lhme::Engine::create(
      {.cache_max_entries = 8, .cache_max_bytes = 4096});
  const auto kernel = std::make_shared<AffineKernel>();
  if (!engine.register_technique(kernel)) return 1;

  const auto result = engine.evaluate_technique({
      .identity = kernel->identity(),
      .parameters = "factor=1:2;offset=1:3",
      .address = {.index = 21},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited(),
  });

  // This block is intentionally built only through find_package() by the
  // installed-package consumer test.  It proves the public exact-cache
  // equivalence API without relying on the BRUCE source-tree include path.
  const bruce::lhme::PowerOffsetKernel identity_source;
  const auto version = identity_source.identity().version;
  const auto formula = engine.register_formula(
      bruce::lhme::make_power_offset_spec(2, 100, version));
  const auto power_offset = std::make_shared<bruce::lhme::PowerOffsetKernel>();
  if (!engine.register_technique(power_offset)) return 3;
  const bruce::lhme::ExactFormulaTechniqueCacheBinding binding =
      bruce::lhme::make_power_offset_exact_cache_binding(2, 100, version);
  if (!engine.register_exact_formula_cache_binding(binding)) return 4;

  const auto formula_result = engine.evaluate({
      .spec = formula,
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited(),
  });
  const auto technique_result = engine.evaluate_technique({
      .identity = power_offset->identity(),
      .parameters = bruce::lhme::encode_power_offset_parameters(2, 100),
      .address = {.index = 10},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited(),
  });
  const bool neutral_cross_api_trace =
      is_neutral_cross_api_hit(technique_result);

  // A third-party kernel can make the same explicit declaration without a new
  // TechniqueKernel virtual method or a ComputationIdentity override.
  const auto formula_backed = std::make_shared<FormulaBackedKernel>(formula);
  if (!engine.register_technique(formula_backed)) return 5;
  const bruce::lhme::ExactFormulaTechniqueCacheBinding formula_backed_binding{
      .technique = formula_backed->identity(),
      .accepted_parameters = std::string{FormulaBackedKernel::kParameters},
      .formula = formula,
  };
  if (!engine.register_exact_formula_cache_binding(formula_backed_binding)) return 6;

  // Formula -> Technique: the kernel must not run, and the trace is neutral.
  const auto formula_producer = engine.evaluate({
      .spec = formula,
      .address = {.index = 12},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited(),
  });
  const auto technique_from_formula = engine.evaluate_technique({
      .identity = formula_backed->identity(),
      .parameters = std::string{FormulaBackedKernel::kParameters},
      .address = {.index = 12},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited(),
  });

  // Technique -> Formula: use another index so the Technique produces first.
  const auto technique_producer = engine.evaluate_technique({
      .identity = formula_backed->identity(),
      .parameters = std::string{FormulaBackedKernel::kParameters},
      .address = {.index = 13},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited(),
  });
  const auto formula_from_technique = engine.evaluate({
      .spec = formula,
      .address = {.index = 13},
      .mode = bruce::lhme::EvaluationMode::Exact,
      .budget = bruce::lhme::Budget::unlimited(),
  });

  return result.status == bruce::lhme::Status::Confirmed &&
                 result.integer.has_value() && *result.integer == 45
                 && formula_result.status == bruce::lhme::Status::Confirmed
                 && technique_result.status == bruce::lhme::Status::Confirmed
                 && formula_result.integer == technique_result.integer
                 && neutral_cross_api_trace
                 && formula_producer.status == bruce::lhme::Status::Confirmed
                 && technique_from_formula.status == bruce::lhme::Status::Confirmed
                 && formula_producer.integer == technique_from_formula.integer
                 && is_neutral_cross_api_hit(technique_from_formula)
                 && formula_backed->calls.load() == 1
                 && technique_producer.status == bruce::lhme::Status::Confirmed
                 && formula_from_technique.status == bruce::lhme::Status::Confirmed
                 && technique_producer.integer == formula_from_technique.integer
                 && is_neutral_cross_api_hit(formula_from_technique)
             ? 0
             : 2;
}
