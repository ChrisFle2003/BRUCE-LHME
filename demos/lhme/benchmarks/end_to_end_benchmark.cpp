#include "benchmark_statistics.hpp"

#include "bruce/lhme/engine.hpp"
#include "bruce/lhme/fibonacci_kernel.hpp"
#include "bruce/lhme/formula_override_snapshot.hpp"
#include "bruce/lhme/formula_override_store.hpp"
#include "bruce/lhme/gguf_inspector.hpp"
#include "bruce/lhme/gguf_pattern_scan.hpp"
#include "bruce/lhme/gguf_report.hpp"
#include "bruce/lhme/gguf_storage_plan.hpp"
#include "bruce/lhme/kevin_memory_store.hpp"
#include "bruce/lhme/power_offset_kernel.hpp"
#include "bruce/lhme/streaming.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using namespace bruce::lhme;

struct BenchmarkCase {
  std::string id;
  std::string subsystem;
  std::string workload;
  std::string variant;
  std::string input;
  std::string unit;
  std::string notes;
  std::size_t repetitions = 1;
  std::function<bool()> setup;
  std::function<bool()> run;
  std::function<bool()> verify;
};

class CsvWriter {
 public:
  explicit CsvWriter(const std::filesystem::path& path) : output_(path) {
    output_ << "case_id,subsystem,workload,variant,input,iteration,unit,value,validation,notes\n";
  }

  [[nodiscard]] bool good() const { return output_.good(); }

  void row(const BenchmarkCase& benchmark, std::size_t iteration, double value) {
    output_ << quote(benchmark.id) << ',' << quote(benchmark.subsystem) << ','
            << quote(benchmark.workload) << ',' << quote(benchmark.variant) << ','
            << quote(benchmark.input) << ',' << iteration << ',' << quote(benchmark.unit)
            << ',' << std::fixed << std::setprecision(3) << value << ",confirmed,"
            << quote(benchmark.notes) << '\n';
  }

 private:
  [[nodiscard]] static std::string quote(std::string_view text) {
    std::string escaped{"\""};
    for (const auto character : text) {
      if (character == '\"') escaped += "\"\"";
      else escaped += character;
    }
    escaped += '\"';
    return escaped;
  }

  std::ofstream output_;
};

class RegistrationKernel final : public TechniqueKernel {
 public:
  explicit RegistrationKernel(std::uint32_t ordinal) : ordinal_(ordinal) {}

  [[nodiscard]] TechniqueIdentity identity() const override {
    return {.id = "bruce.benchmark.registration." + std::to_string(ordinal_),
            .version = 1};
  }

  [[nodiscard]] EvaluationResult evaluate(const TechniqueRequest& request) const override {
    return {.status = Status::InvalidArgument, .mode = request.mode};
  }

 private:
  std::uint32_t ordinal_ = 0;
};

[[nodiscard]] bool confirmed_integer(const EvaluationResult& result) {
  return result.status == Status::Confirmed && result.integer.has_value();
}

[[nodiscard]] bool confirmed_metric(const EvaluationResult& result) {
  return result.status == Status::Confirmed && result.metric_bit_length.has_value();
}

[[nodiscard]] std::filesystem::path gguf_fixture_path() {
  return std::filesystem::temp_directory_path() /
      ("bruce-lhme-benchmark-gguf-" +
       std::to_string(Clock::now().time_since_epoch().count()) + ".gguf");
}

void append_u32(std::vector<std::byte>& bytes, std::uint32_t value) {
  for (std::size_t index = 0; index < 4; ++index) {
    bytes.push_back(static_cast<std::byte>((value >> (index * 8U)) & 0xffU));
  }
}

void append_u64(std::vector<std::byte>& bytes, std::uint64_t value) {
  for (std::size_t index = 0; index < 8; ++index) {
    bytes.push_back(static_cast<std::byte>((value >> (index * 8U)) & 0xffU));
  }
}

void append_string(std::vector<std::byte>& bytes, std::string_view value) {
  append_u64(bytes, value.size());
  for (const auto character : value) {
    bytes.push_back(static_cast<std::byte>(static_cast<unsigned char>(character)));
  }
}

[[nodiscard]] bool write_gguf_fixture(const std::filesystem::path& path) {
  // One valid GGUF V3 F32 tensor (4,096 elements / 16 KiB) with a constant payload.
  std::vector<std::byte> bytes;
  append_u32(bytes, 0x46554747U);  // GGUF, little endian
  append_u32(bytes, 3U);
  append_u64(bytes, 1U);           // tensor count
  append_u64(bytes, 0U);           // metadata count
  append_string(bytes, "weight");
  append_u32(bytes, 1U);           // dimensions
  append_u64(bytes, 4'096U);
  append_u32(bytes, 0U);           // GGML_TYPE_F32
  append_u64(bytes, 0U);           // relative tensor offset
  while (bytes.size() % 32U != 0U) bytes.push_back(std::byte{0});
  bytes.insert(bytes.end(), 4'096U * sizeof(float), std::byte{0x5a});

  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  return output.good();
}

[[nodiscard]] std::vector<GridStep> benchmark_grid_steps() {
  return {{.x = 1, .y = 1}};
}

[[nodiscard]] std::uint32_t benchmark_workers() {
  const auto hardware = std::thread::hardware_concurrency();
  return hardware > 1U ? 2U : 1U;
}

template <typename Context>
[[nodiscard]] std::shared_ptr<Context> shared_context() {
  return std::make_shared<Context>();
}

std::vector<BenchmarkCase> core_cases() {
  const auto spec = make_power_offset_spec(2, 100, 1);
  std::vector<BenchmarkCase> cases;
  cases.push_back({
      .id = "core.exact.small",
      .subsystem = "core",
      .workload = "PowerOffset direct evaluation",
      .variant = "Exact",
      .input = "base=2;scale=100;index=32",
      .unit = "ns/op",
      .notes = "direct formula API; 256 operations per timed sample",
      .repetitions = 256,
      .setup = [] { return true; },
      .run = [spec] {
        return confirmed_integer(evaluate({.spec = spec, .address = {.index = 32},
                                           .mode = EvaluationMode::Exact}));
      },
  });
  cases.push_back({
      .id = "core.exact.medium",
      .subsystem = "core",
      .workload = "PowerOffset direct evaluation",
      .variant = "Exact",
      .input = "base=2;scale=100;index=4096",
      .unit = "ns/op",
      .notes = "direct formula API; 16 operations per timed sample",
      .repetitions = 16,
      .setup = [] { return true; },
      .run = [spec] {
        const auto result = evaluate({.spec = spec, .address = {.index = 4096},
                                      .mode = EvaluationMode::Exact});
        return confirmed_integer(result) && result.integer->backend().size() > 1U;
      },
  });
  cases.push_back({
      .id = "core.metric.large",
      .subsystem = "core",
      .workload = "PowerOffset direct evaluation",
      .variant = "Metric",
      .input = "base=2;scale=100;index=1000000",
      .unit = "ns/op",
      .notes = "direct formula API; no Exact materialization requested",
      .repetitions = 256,
      .setup = [] { return true; },
      .run = [spec] {
        return confirmed_metric(evaluate({.spec = spec, .address = {.index = 1'000'000},
                                          .mode = EvaluationMode::Metric}));
      },
  });
  cases.push_back({
      .id = "core.modular.large",
      .subsystem = "core",
      .workload = "PowerOffset direct evaluation",
      .variant = "Modular",
      .input = "base=2;scale=100;index=1000000;modulus=65537",
      .unit = "ns/op",
      .notes = "direct formula API; modular output range validated",
      .repetitions = 256,
      .setup = [] { return true; },
      .run = [spec] {
        const auto result = evaluate({.spec = spec, .address = {.index = 1'000'000},
                                      .mode = EvaluationMode::Modular, .modulus = BigInt{65'537}});
        return confirmed_integer(result) && *result.integer >= 0 &&
               *result.integer < BigInt{65'537};
      },
  });
  cases.push_back({
      .id = "core.symbolic.large",
      .subsystem = "core",
      .workload = "PowerOffset direct evaluation",
      .variant = "Symbolic",
      .input = "base=2;scale=100;index=1000000",
      .unit = "ns/op",
      .notes = "direct formula API; symbolic expression validated as nonempty",
      .repetitions = 256,
      .setup = [] { return true; },
      .run = [spec] {
        const auto result = evaluate({.spec = spec, .address = {.index = 1'000'000},
                                      .mode = EvaluationMode::Symbolic});
        return result.status == Status::Confirmed && result.symbolic_expression.has_value() &&
               !result.symbolic_expression->empty();
      },
  });
  cases.push_back({
      .id = "core.budget_rejection",
      .subsystem = "core",
      .workload = "PowerOffset direct evaluation",
      .variant = "Exact budget rejection",
      .input = "base=2;scale=100;index=4096;max_bit_length=1",
      .unit = "ns/op",
      .notes = "rejection path; not comparable to successful Exact evaluation",
      .repetitions = 256,
      .setup = [] { return true; },
      .run = [spec] {
        return evaluate({.spec = spec, .address = {.index = 4096},
                         .mode = EvaluationMode::Exact,
                         .budget = {.max_bit_length = 1}}).status == Status::BudgetExceeded;
      },
  });
  cases.push_back({
      .id = "core.pre_cancelled",
      .subsystem = "core",
      .workload = "PowerOffset direct evaluation",
      .variant = "pre-request cancellation",
      .input = "base=2;scale=100;index=4096",
      .unit = "ns/op",
      .notes = "deterministic cancellation path; not comparable to successful evaluation",
      .repetitions = 256,
      .setup = [] { return true; },
      .run = [spec] {
        return evaluate({.spec = spec, .address = {.index = 4096},
                         .mode = EvaluationMode::Exact,
                         .cancellation_requested = true}).status == Status::Cancelled;
      },
  });
  return cases;
}

std::vector<BenchmarkCase> engine_cases() {
  const auto spec = make_power_offset_spec(2, 100, 1);
  const auto parameters = encode_power_offset_parameters(2, 100);
  std::vector<BenchmarkCase> cases;

  struct RegistrationContext { std::unique_ptr<Engine> engine; std::uint32_t next = 1; };
  const auto formula_registration = shared_context<RegistrationContext>();
  cases.push_back({
      .id = "engine.formula_registration",
      .subsystem = "engine",
      .workload = "formula registration",
      .variant = "distinct specifications",
      .input = "32 PowerOffset specifications/sample",
      .unit = "ns/registration",
      .notes = "Engine construction excluded from timed interval",
      .repetitions = 32,
      .setup = [formula_registration] {
        formula_registration->engine = std::make_unique<Engine>(Engine::create({}));
        formula_registration->next = 1;
        return true;
      },
      .run = [formula_registration] {
        const auto candidate = formula_registration->engine->register_formula(
            make_power_offset_spec(2, 100, formula_registration->next++));
        return valid_formula_spec(candidate);
      },
  });

  struct TechniqueRegistrationContext { std::unique_ptr<Engine> engine; std::uint32_t next = 1; };
  const auto technique_registration = shared_context<TechniqueRegistrationContext>();
  cases.push_back({
      .id = "engine.technique_registration",
      .subsystem = "engine",
      .workload = "technique registration",
      .variant = "distinct benchmark kernels",
      .input = "32 identity-only TechniqueKernel objects/sample",
      .unit = "ns/registration",
      .notes = "registration only; kernel evaluation is excluded",
      .repetitions = 32,
      .setup = [technique_registration] {
        technique_registration->engine = std::make_unique<Engine>(Engine::create({}));
        technique_registration->next = 1;
        return true;
      },
      .run = [technique_registration] {
        return technique_registration->engine->register_technique(
            std::make_shared<RegistrationKernel>(technique_registration->next++));
      },
  });

  struct FormulaContext {
    std::unique_ptr<Engine> engine;
    FormulaSpec registered;
    std::int64_t next_address = 0;
  };
  const auto exact_miss = shared_context<FormulaContext>();
  cases.push_back({
      .id = "engine.exact_cache_miss",
      .subsystem = "engine/cache",
      .workload = "registered FormulaSpec evaluation",
      .variant = "Exact cache miss",
      .input = "base=2;scale=100;64 distinct indices/sample",
      .unit = "ns/op",
      .notes = "formula registration and Engine construction excluded; distinct addresses force misses",
      .repetitions = 64,
      .setup = [exact_miss, spec] {
        exact_miss->engine = std::make_unique<Engine>(Engine::create(
            {.cache_max_entries = 256, .cache_max_bytes = 1U << 20U}));
        exact_miss->registered = exact_miss->engine->register_formula(spec);
        exact_miss->next_address = 128;
        return true;
      },
      .run = [exact_miss] {
        const auto result = exact_miss->engine->evaluate(
            {.spec = exact_miss->registered, .address = {.index = exact_miss->next_address++},
             .mode = EvaluationMode::Exact});
        return confirmed_integer(result);
      },
      .verify = [exact_miss] { return exact_miss->engine->inspect().resident_entries == 64U; },
  });

  const auto metric_miss = shared_context<FormulaContext>();
  cases.push_back({
      .id = "engine.metric_cache_miss",
      .subsystem = "engine/cache",
      .workload = "registered FormulaSpec evaluation",
      .variant = "Metric cache miss",
      .input = "base=2;scale=100;64 distinct indices/sample",
      .unit = "ns/op",
      .notes = "formula registration and Engine construction excluded; distinct addresses force misses",
      .repetitions = 64,
      .setup = [metric_miss, spec] {
        metric_miss->engine = std::make_unique<Engine>(Engine::create(
            {.cache_max_entries = 256, .cache_max_bytes = 1U << 20U}));
        metric_miss->registered = metric_miss->engine->register_formula(spec);
        metric_miss->next_address = 1'000'000;
        return true;
      },
      .run = [metric_miss] {
        const auto result = metric_miss->engine->evaluate(
            {.spec = metric_miss->registered,
             .address = {.index = metric_miss->next_address++},
             .mode = EvaluationMode::Metric});
        return confirmed_metric(result);
      },
      .verify = [metric_miss] { return metric_miss->engine->inspect().resident_entries == 64U; },
  });

  const auto exact_hit = shared_context<FormulaContext>();
  cases.push_back({
      .id = "engine.exact_cache_hit",
      .subsystem = "engine/cache",
      .workload = "registered FormulaSpec evaluation",
      .variant = "Exact cache hit",
      .input = "base=2;scale=100;index=1024",
      .unit = "ns/op",
      .notes = "one unscored cache-fill evaluation before each timed sample",
      .repetitions = 512,
      .setup = [exact_hit, spec] {
        exact_hit->engine = std::make_unique<Engine>(Engine::create(
            {.cache_max_entries = 8, .cache_max_bytes = 1U << 20U}));
        exact_hit->registered = exact_hit->engine->register_formula(spec);
        return confirmed_integer(exact_hit->engine->evaluate(
            {.spec = exact_hit->registered, .address = {.index = 1024},
             .mode = EvaluationMode::Exact}));
      },
      .run = [exact_hit] {
        return confirmed_integer(exact_hit->engine->evaluate(
            {.spec = exact_hit->registered, .address = {.index = 1024},
             .mode = EvaluationMode::Exact}));
      },
      .verify = [exact_hit] { return exact_hit->engine->inspect().resident_entries == 1U; },
  });

  const auto metric_hit = shared_context<FormulaContext>();
  cases.push_back({
      .id = "engine.metric_cache_hit",
      .subsystem = "engine/cache",
      .workload = "registered FormulaSpec evaluation",
      .variant = "Metric cache hit",
      .input = "base=2;scale=100;index=1000000",
      .unit = "ns/op",
      .notes = "one unscored cache-fill evaluation before each timed sample",
      .repetitions = 512,
      .setup = [metric_hit, spec] {
        metric_hit->engine = std::make_unique<Engine>(Engine::create(
            {.cache_max_entries = 8, .cache_max_bytes = 1U << 20U}));
        metric_hit->registered = metric_hit->engine->register_formula(spec);
        return confirmed_metric(metric_hit->engine->evaluate(
            {.spec = metric_hit->registered, .address = {.index = 1'000'000},
             .mode = EvaluationMode::Metric}));
      },
      .run = [metric_hit] {
        return confirmed_metric(metric_hit->engine->evaluate(
            {.spec = metric_hit->registered, .address = {.index = 1'000'000},
             .mode = EvaluationMode::Metric}));
      },
      .verify = [metric_hit] { return metric_hit->engine->inspect().resident_entries == 1U; },
  });

  const auto cache_disabled = shared_context<FormulaContext>();
  cases.push_back({
      .id = "engine.exact_cache_disabled",
      .subsystem = "engine/cache",
      .workload = "registered FormulaSpec evaluation",
      .variant = "Exact cache disabled",
      .input = "base=2;scale=100;index=1024",
      .unit = "ns/op",
      .notes = "same request repeated with zero cache capacity",
      .repetitions = 128,
      .setup = [cache_disabled, spec] {
        cache_disabled->engine = std::make_unique<Engine>(Engine::create({}));
        cache_disabled->registered = cache_disabled->engine->register_formula(spec);
        return true;
      },
      .run = [cache_disabled] {
        return confirmed_integer(cache_disabled->engine->evaluate(
            {.spec = cache_disabled->registered, .address = {.index = 1024},
             .mode = EvaluationMode::Exact}));
      },
      .verify = [cache_disabled] { return cache_disabled->engine->inspect().resident_entries == 0U; },
  });

  const auto eviction = shared_context<FormulaContext>();
  cases.push_back({
      .id = "engine.cache_eviction",
      .subsystem = "engine/cache",
      .workload = "registered FormulaSpec evaluation",
      .variant = "one-entry LRU eviction churn",
      .input = "base=2;scale=100;indices=512,513",
      .unit = "ns/op",
      .notes = "two alternating addresses with one cache entry; every call is a miss after fill",
      .repetitions = 256,
      .setup = [eviction, spec] {
        eviction->engine = std::make_unique<Engine>(Engine::create(
            {.cache_max_entries = 1, .cache_max_bytes = 1U << 20U}));
        eviction->registered = eviction->engine->register_formula(spec);
        eviction->next_address = 512;
        return confirmed_integer(eviction->engine->evaluate(
            {.spec = eviction->registered, .address = {.index = 512},
             .mode = EvaluationMode::Exact}));
      },
      .run = [eviction] {
        const auto address = eviction->next_address == 512 ? 513 : 512;
        eviction->next_address = address;
        return confirmed_integer(eviction->engine->evaluate(
            {.spec = eviction->registered, .address = {.index = address},
             .mode = EvaluationMode::Exact}));
      },
      .verify = [eviction] { return eviction->engine->inspect().resident_entries == 1U; },
  });

  struct CrossApiContext {
    std::unique_ptr<Engine> engine;
    FormulaSpec registered;
    TechniqueIdentity identity;
  };
  const auto cross_api = shared_context<CrossApiContext>();
  cases.push_back({
      .id = "engine.cross_api_exact_cache",
      .subsystem = "engine/cache",
      .workload = "formula then equivalent technique",
      .variant = "PowerOffset exact cache binding",
      .input = "base=2;scale=100;index=1024",
      .unit = "ns/op",
      .notes = "formula cache fill is unscored; timed calls use Technique API",
      .repetitions = 512,
      .setup = [cross_api, spec, parameters] {
        cross_api->engine = std::make_unique<Engine>(Engine::create(
            {.cache_max_entries = 8, .cache_max_bytes = 1U << 20U}));
        cross_api->registered = cross_api->engine->register_formula(spec);
        const auto kernel = std::make_shared<PowerOffsetKernel>();
        cross_api->identity = kernel->identity();
        if (!cross_api->engine->register_technique(kernel) ||
            !cross_api->engine->register_exact_formula_cache_binding(
                make_power_offset_exact_cache_binding(2, 100, 1))) return false;
        return confirmed_integer(cross_api->engine->evaluate(
            {.spec = cross_api->registered, .address = {.index = 1024},
             .mode = EvaluationMode::Exact}));
      },
      .run = [cross_api, parameters] {
        return confirmed_integer(cross_api->engine->evaluate_technique(
            {.identity = cross_api->identity, .parameters = parameters,
             .address = {.index = 1024}, .mode = EvaluationMode::Exact}));
      },
      .verify = [cross_api] { return cross_api->engine->inspect().resident_entries == 1U; },
  });

  const auto strict_budget = shared_context<FormulaContext>();
  cases.push_back({
      .id = "engine.cached_exact_strict_budget",
      .subsystem = "engine/cache",
      .workload = "cached Exact request",
      .variant = "caller budget rejection",
      .input = "base=2;scale=100;index=1024;max_bit_length=1",
      .unit = "ns/op",
      .notes = "a cached value is rechecked against the stricter caller budget",
      .repetitions = 256,
      .setup = [strict_budget, spec] {
        strict_budget->engine = std::make_unique<Engine>(Engine::create(
            {.cache_max_entries = 8, .cache_max_bytes = 1U << 20U}));
        strict_budget->registered = strict_budget->engine->register_formula(spec);
        return confirmed_integer(strict_budget->engine->evaluate(
            {.spec = strict_budget->registered, .address = {.index = 1024},
             .mode = EvaluationMode::Exact}));
      },
      .run = [strict_budget] {
        return strict_budget->engine->evaluate(
            {.spec = strict_budget->registered, .address = {.index = 1024},
             .mode = EvaluationMode::Exact, .budget = {.max_bit_length = 1}}).status ==
            Status::BudgetExceeded;
      },
  });

  struct TechniqueContext { std::unique_ptr<Engine> engine; TechniqueIdentity identity; };
  const auto fibonacci = shared_context<TechniqueContext>();
  cases.push_back({
      .id = "engine.fibonacci_kernel",
      .subsystem = "engine/kernel",
      .workload = "registered technique evaluation",
      .variant = "FibonacciKernel Exact",
      .input = "algorithm=fast_dbl;index=2048",
      .unit = "ns/op",
      .notes = "cache disabled; successful technique computation",
      .repetitions = 64,
      .setup = [fibonacci] {
        fibonacci->engine = std::make_unique<Engine>(Engine::create({}));
        const auto kernel = std::make_shared<FibonacciKernel>();
        fibonacci->identity = kernel->identity();
        return fibonacci->engine->register_technique(kernel);
      },
      .run = [fibonacci] {
        return confirmed_integer(fibonacci->engine->evaluate_technique(
            {.identity = fibonacci->identity,
             .parameters = kFibonacciFastDoublingParameters,
             .address = {.index = 2048}, .mode = EvaluationMode::Exact}));
      },
  });

  const auto power_kernel = shared_context<TechniqueContext>();
  cases.push_back({
      .id = "engine.power_offset_kernel",
      .subsystem = "engine/kernel",
      .workload = "registered technique evaluation",
      .variant = "PowerOffsetKernel Exact",
      .input = "base=2;scale=100;index=1024",
      .unit = "ns/op",
      .notes = "cache disabled; successful technique computation",
      .repetitions = 128,
      .setup = [power_kernel] {
        power_kernel->engine = std::make_unique<Engine>(Engine::create({}));
        const auto kernel = std::make_shared<PowerOffsetKernel>();
        power_kernel->identity = kernel->identity();
        return power_kernel->engine->register_technique(kernel);
      },
      .run = [power_kernel, parameters] {
        return confirmed_integer(power_kernel->engine->evaluate_technique(
            {.identity = power_kernel->identity, .parameters = parameters,
             .address = {.index = 1024}, .mode = EvaluationMode::Exact}));
      },
  });

  const auto unregistered = shared_context<TechniqueContext>();
  cases.push_back({
      .id = "engine.unregistered_technique",
      .subsystem = "engine/kernel",
      .workload = "technique evaluation",
      .variant = "unregistered failure",
      .input = "PowerOffset identity;base=2;scale=100;index=1024",
      .unit = "ns/op",
      .notes = "error path; not comparable to a registered-kernel computation",
      .repetitions = 256,
      .setup = [unregistered] {
        unregistered->engine = std::make_unique<Engine>(Engine::create({}));
        unregistered->identity = PowerOffsetKernel{}.identity();
        return true;
      },
      .run = [unregistered, parameters] {
        return unregistered->engine->evaluate_technique(
            {.identity = unregistered->identity, .parameters = parameters,
             .address = {.index = 1024}, .mode = EvaluationMode::Exact}).status ==
            Status::InvalidArgument;
      },
  });
  return cases;
}

std::vector<BenchmarkCase> range_cases() {
  const auto spec = make_power_offset_spec(2, 100, 1);
  const auto workers = benchmark_workers();
  std::vector<BenchmarkCase> cases;

  struct RangeContext { std::unique_ptr<Engine> engine; FormulaSpec registered; RangeResult result; };
  for (const auto worker_count : {1U, workers}) {
    if (worker_count != 1U && workers == 1U) continue;
    const auto context = shared_context<RangeContext>();
    cases.push_back({
        .id = "range.engine_exact_w" + std::to_string(worker_count),
        .subsystem = "range",
        .workload = "Engine::evaluate_range",
        .variant = "Exact; workers=" + std::to_string(worker_count),
        .input = "indices=0..511;tile_size=64",
        .unit = "us/request",
        .notes = "validated checked=512;confirmed=512;candidates=512; Engine setup excluded",
        .repetitions = 1,
        .setup = [context, spec, worker_count] {
          context->engine = std::make_unique<Engine>(Engine::create(
              {.worker_count = worker_count, .tile_size = 64}));
          context->registered = context->engine->register_formula(spec);
          return true;
        },
        .run = [context] {
          context->result = context->engine->evaluate_range(
              {.spec = context->registered, .mode = EvaluationMode::Exact}, {0, 511});
          return context->result.status == Status::Confirmed;
        },
        .verify = [context] {
          return context->result.summary.checked_count == 512U &&
                 context->result.summary.confirmed_count == 512U &&
                 context->result.candidates.size() == 512U;
        },
    });
  }

  struct StreamingContext { StreamingRangeResult result; };
  const auto add_streaming = [&cases, spec](std::string id, std::string variant,
                                            StreamingOptions options,
                                            std::function<bool(const StreamingRangeResult&)> check) {
    const auto context = shared_context<StreamingContext>();
    cases.push_back({
        .id = std::move(id),
        .subsystem = "streaming",
        .workload = "evaluate_streaming_range",
        .variant = std::move(variant),
        .input = "indices=0..1023;tile_size=" + std::to_string(options.tile_size),
        .unit = "us/request",
        .notes = "validated result counters and retention; direct streaming API; worker-pool construction is included",
        .repetitions = 1,
        .setup = [] { return true; },
        .run = [context, spec, options] {
          context->result = evaluate_streaming_range(
              {.spec = spec, .mode = EvaluationMode::Exact}, {0, 1023}, options);
          return context->result.status == Status::Confirmed;
        },
        .verify = [context, check = std::move(check)] { return check(context->result); },
    });
  };
  add_streaming("streaming.summary_w1", "SummaryOnly; AcceptAll; workers=1",
                {.worker_count = 1, .tile_size = 64,
                 .retention_mode = RetentionMode::SummaryOnly},
                [](const auto& result) {
                  return result.summary.checked_count == 1024U &&
                         result.summary.confirmed_count == 1024U &&
                         result.exact_evaluation_count == 1024U &&
                         result.filtered_out_count == 0U &&
                         result.retained_candidates.empty();
                });
  if (workers > 1U) {
    add_streaming("streaming.summary_w2", "SummaryOnly; AcceptAll; workers=2",
                  {.worker_count = workers, .tile_size = 64,
                   .retention_mode = RetentionMode::SummaryOnly},
                  [](const auto& result) {
                    return result.summary.checked_count == 1024U &&
                           result.summary.confirmed_count == 1024U &&
                           result.exact_evaluation_count == 1024U &&
                           result.filtered_out_count == 0U &&
                           result.retained_candidates.empty();
                  });
  }
  const CandidateFilter filter{.mode = FilterMode::ModularEquals,
                               .modulus = 17,
                               .expected_residue = 11};
  add_streaming("streaming.filtered_candidates", "FilteredCandidates; ModularEquals",
                {.worker_count = 1, .tile_size = 4,
                 .retention_mode = RetentionMode::FilteredCandidates,
                 .max_retained_candidates = 4, .retention_memory_bytes = 16U << 10U,
                 .filter = filter},
                [](const auto& result) {
                  return result.summary.checked_count == 1024U &&
                         result.summary.confirmed_count == 128U &&
                         result.filtered_out_count == 896U &&
                         result.exact_evaluation_count == 128U &&
                         result.retained_candidates.size() == 4U;
                });
  add_streaming("streaming.top_k", "TopK; ModularEquals",
                {.worker_count = 1, .tile_size = 4,
                 .retention_mode = RetentionMode::TopK,
                 .max_retained_candidates = 3, .retention_memory_bytes = 16U << 10U,
                 .filter = filter},
                [](const auto& result) {
                  return result.effective_retention_mode == RetentionMode::TopK &&
                         result.summary.checked_count == 1024U &&
                         result.summary.confirmed_count == 128U &&
                         result.filtered_out_count == 896U &&
                         result.exact_evaluation_count == 128U &&
                         result.retained_candidates.size() == 3U;
                });
  add_streaming("streaming.adaptive", "Adaptive; ModularEquals",
                {.worker_count = 1, .tile_size = 4,
                 .retention_mode = RetentionMode::Adaptive,
                 .max_retained_candidates = 3, .retention_memory_bytes = 16U << 10U,
                 .filter = filter},
                [](const auto& result) {
                  return result.effective_retention_mode == RetentionMode::TopK &&
                         result.summary.checked_count == 1024U &&
                         result.summary.confirmed_count == 128U &&
                         result.filtered_out_count == 896U &&
                         result.exact_evaluation_count == 128U &&
                         result.retained_candidates.size() == 3U;
                });
  return cases;
}

std::vector<BenchmarkCase> override_cases(const std::filesystem::path& snapshot_path) {
  const auto spec = make_power_offset_spec(2, 100, 1);
  const auto parameters = encode_power_offset_parameters(2, 100);
  std::vector<BenchmarkCase> cases;
  struct Context {
    std::unique_ptr<Engine> engine;
    std::unique_ptr<FormulaOverrideStore> store;
    FormulaBinding binding;
  };

  const auto formula_override = shared_context<Context>();
  cases.push_back({
      .id = "override.formula_hit",
      .subsystem = "override store",
      .workload = "FormulaBinding retrieval",
      .variant = "sparse override hit",
      .input = "PowerOffset;index=1024;override=-42",
      .unit = "ns/op",
      .notes = "cache disabled; override lookup only",
      .repetitions = 512,
      .setup = [formula_override, spec] {
        formula_override->engine = std::make_unique<Engine>(Engine::create({}));
        formula_override->binding = formula_override->engine->register_formula(spec);
        formula_override->store = std::make_unique<FormulaOverrideStore>(*formula_override->engine);
        return formula_override->store->set_override(formula_override->binding, {.index = 1024},
                                                     BigInt{-42}) == Status::Confirmed;
      },
      .run = [formula_override] {
        const auto value = formula_override->store->evaluate(
            {.binding = formula_override->binding, .address = {.index = 1024},
             .mode = EvaluationMode::Exact});
        return value.source == FormulaValueSource::Override && confirmed_integer(value.result) &&
               *value.result.integer == -42;
      },
  });

  const auto computed = shared_context<Context>();
  cases.push_back({
      .id = "override.formula_fallback",
      .subsystem = "override store",
      .workload = "FormulaBinding retrieval",
      .variant = "computed Metric fallback",
      .input = "PowerOffset;index=1000000;no override",
      .unit = "ns/op",
      .notes = "cache disabled; FormulaOverrideStore dispatches to Engine",
      .repetitions = 256,
      .setup = [computed, spec] {
        computed->engine = std::make_unique<Engine>(Engine::create({}));
        computed->binding = computed->engine->register_formula(spec);
        computed->store = std::make_unique<FormulaOverrideStore>(*computed->engine);
        return true;
      },
      .run = [computed] {
        const auto value = computed->store->evaluate(
            {.binding = computed->binding, .address = {.index = 1'000'000},
             .mode = EvaluationMode::Metric});
        return value.source == FormulaValueSource::Computed && confirmed_metric(value.result);
      },
  });

  const auto technique_override = shared_context<Context>();
  cases.push_back({
      .id = "override.technique_hit",
      .subsystem = "override store",
      .workload = "TechniqueBinding retrieval",
      .variant = "sparse override hit",
      .input = "PowerOffsetKernel;index=1024;override=-42",
      .unit = "ns/op",
      .notes = "technique registration and override insertion excluded",
      .repetitions = 512,
      .setup = [technique_override, parameters] {
        technique_override->engine = std::make_unique<Engine>(Engine::create({}));
        const auto kernel = std::make_shared<PowerOffsetKernel>();
        if (!technique_override->engine->register_technique(kernel)) return false;
        technique_override->binding = TechniqueBinding{.identity = kernel->identity(),
                                                       .canonical_parameters = parameters};
        technique_override->store = std::make_unique<FormulaOverrideStore>(*technique_override->engine);
        return technique_override->store->set_override(technique_override->binding, {.index = 1024},
                                                        BigInt{-42}) == Status::Confirmed;
      },
      .run = [technique_override] {
        const auto value = technique_override->store->evaluate(
            {.binding = technique_override->binding, .address = {.index = 1024},
             .mode = EvaluationMode::Exact});
        return value.source == FormulaValueSource::Override && confirmed_integer(value.result) &&
               *value.result.integer == -42;
      },
  });

  struct SnapshotContext {
    std::unique_ptr<Engine> engine;
    std::unique_ptr<FormulaOverrideStore> source;
    std::uint64_t bytes = 0;
  };
  const auto snapshot = shared_context<SnapshotContext>();
  cases.push_back({
      .id = "override.snapshot_save_load",
      .subsystem = "override/snapshot",
      .workload = "FormulaOverrideStore snapshot",
      .variant = "save then atomic load",
      .input = "100 FormulaBinding overrides",
      .unit = "us/request",
      .notes = "fixture construction excluded; snapshot file is deleted after each operation",
      .repetitions = 1,
      .setup = [snapshot, spec] {
        snapshot->engine = std::make_unique<Engine>(Engine::create({}));
        const auto binding = snapshot->engine->register_formula(spec);
        snapshot->source = std::make_unique<FormulaOverrideStore>(*snapshot->engine);
        for (std::int64_t index = 0; index < 100; ++index) {
          if (snapshot->source->set_override(binding, {.index = index}, BigInt{index + 1}) !=
              Status::Confirmed) return false;
        }
        return true;
      },
      .run = [snapshot, snapshot_path] {
        std::error_code error;
        std::filesystem::remove(snapshot_path, error);
        if (save_formula_override_snapshot(*snapshot->source, snapshot_path) !=
            FormulaOverrideSnapshotStatus::Confirmed) return false;
        snapshot->bytes = std::filesystem::file_size(snapshot_path, error);
        FormulaOverrideStore restored{*snapshot->engine};
        const auto loaded = load_formula_override_snapshot(restored, snapshot_path) ==
                            FormulaOverrideSnapshotStatus::Confirmed &&
                            restored.inspect().override_count == 100U;
        std::filesystem::remove(snapshot_path, error);
        return loaded && !error && snapshot->bytes > 0U;
      },
  });
  return cases;
}

std::vector<BenchmarkCase> legacy_kevin_cases() {
  const auto parameters = encode_power_offset_parameters(2, 100);
  const auto address = *GridAddress::from_steps(benchmark_grid_steps());
  std::vector<BenchmarkCase> cases;
  struct Context {
    std::unique_ptr<Engine> engine;
    std::unique_ptr<KevinMemoryStore> store;
    GridIdentity identity;
    GridAddress address = *GridAddress::from_steps({});
  };

  const auto computed = shared_context<Context>();
  cases.push_back({
      .id = "kevin.legacy_computed",
      .subsystem = "KEVIN legacy",
      .workload = "KevinMemoryStore::get",
      .variant = "registered-grid computed fallback",
      .input = "PowerOffset grid;depth=1;Exact",
      .unit = "ns/op",
      .notes = "legacy compatibility API; cache disabled",
      .repetitions = 128,
      .setup = [computed, parameters, address] {
        computed->engine = std::make_unique<Engine>(Engine::create({}));
        const auto kernel = std::make_shared<PowerOffsetKernel>();
        if (!computed->engine->register_technique(kernel)) return false;
        computed->identity = {.id = "bruce.benchmark.legacy", .version = 1};
        computed->address = address;
        computed->store = std::make_unique<KevinMemoryStore>(*computed->engine);
        return computed->store->register_grid({.identity = computed->identity,
                                               .technique = kernel->identity(),
                                               .canonical_parameters = parameters,
                                               .max_depth = 1});
      },
      .run = [computed] {
        const auto value = computed->store->get(computed->identity, computed->address,
                                                EvaluationMode::Exact, Budget::unlimited());
        return value.source == KevinValueSource::Computed && confirmed_integer(value.result);
      },
  });

  const auto overridden = shared_context<Context>();
  cases.push_back({
      .id = "kevin.legacy_override",
      .subsystem = "KEVIN legacy",
      .workload = "KevinMemoryStore::get",
      .variant = "sparse override hit",
      .input = "PowerOffset grid;depth=1;Exact;override=-42",
      .unit = "ns/op",
      .notes = "legacy compatibility API; override insertion excluded",
      .repetitions = 512,
      .setup = [overridden, parameters, address] {
        overridden->engine = std::make_unique<Engine>(Engine::create({}));
        const auto kernel = std::make_shared<PowerOffsetKernel>();
        if (!overridden->engine->register_technique(kernel)) return false;
        overridden->identity = {.id = "bruce.benchmark.legacy.override", .version = 1};
        overridden->address = address;
        overridden->store = std::make_unique<KevinMemoryStore>(*overridden->engine);
        if (!overridden->store->register_grid({.identity = overridden->identity,
                                               .technique = kernel->identity(),
                                               .canonical_parameters = parameters,
                                               .max_depth = 1})) return false;
        return overridden->store->set_override(overridden->identity, overridden->address,
                                               BigInt{-42}) == Status::Confirmed;
      },
      .run = [overridden] {
        const auto value = overridden->store->get(overridden->identity, overridden->address,
                                                  EvaluationMode::Exact, Budget::unlimited());
        return value.source == KevinValueSource::Override && confirmed_integer(value.result) &&
               *value.result.integer == -42;
      },
  });
  return cases;
}

std::vector<BenchmarkCase> gguf_cases(const std::filesystem::path& fixture) {
  std::vector<BenchmarkCase> cases;
  struct InspectionContext { GgufInspection inspection; };
  const auto inspect = shared_context<InspectionContext>();
  cases.push_back({
      .id = "gguf.inspect_valid_v3",
      .subsystem = "GGUF",
      .workload = "inspect_gguf_file",
      .variant = "valid V3 F32 fixture",
      .input = "one tensor;4096 F32 elements;16 KiB payload",
      .unit = "us/request",
      .notes = "deterministic local fixture; file creation excluded",
      .repetitions = 1,
      .setup = [] { return true; },
      .run = [inspect, fixture] {
        inspect->inspection = inspect_gguf_file(fixture);
        return inspect->inspection.status == Status::Confirmed &&
               inspect->inspection.inventory.has_value();
      },
      .verify = [inspect] { return inspect->inspection.inventory->tensors.size() == 1U; },
  });

  struct PlanContext { GgufInspection inspection; GgufStoragePlan plan; };
  const auto plan = shared_context<PlanContext>();
  cases.push_back({
      .id = "gguf.storage_plan",
      .subsystem = "GGUF",
      .workload = "make_gguf_storage_plan",
      .variant = "Budgeted",
      .input = "one tensor;budget=16384;chunk=4096",
      .unit = "us/request",
      .notes = "inspection is prepared outside the timed interval",
      .repetitions = 1,
      .setup = [plan, fixture] {
        plan->inspection = inspect_gguf_file(fixture);
        return plan->inspection.status == Status::Confirmed &&
               plan->inspection.inventory.has_value();
      },
      .run = [plan] {
        plan->plan = make_gguf_storage_plan(*plan->inspection.inventory,
                                            {.memory_budget_bytes = 16U << 10U,
                                             .chunk_size_bytes = 4U << 10U,
                                             .policy = GgufLoadingPolicy::Budgeted});
        return plan->plan.status == Status::Confirmed;
      },
      .verify = [plan] { return plan->plan.items.size() == 4U; },
  });

  struct ReportContext { GgufInspection inspection; GgufStoragePlan plan; std::string text; };
  const auto text = shared_context<ReportContext>();
  const auto setup_report = [fixture](const std::shared_ptr<ReportContext>& context) {
    context->inspection = inspect_gguf_file(fixture);
    if (context->inspection.status != Status::Confirmed || !context->inspection.inventory) return false;
    context->plan = make_gguf_storage_plan(*context->inspection.inventory,
                                           {.memory_budget_bytes = 16U << 10U,
                                            .chunk_size_bytes = 4U << 10U,
                                            .policy = GgufLoadingPolicy::AllResident});
    return context->plan.status == Status::Confirmed;
  };
  cases.push_back({
      .id = "gguf.text_report",
      .subsystem = "GGUF",
      .workload = "render_gguf_text_report",
      .variant = "text",
      .input = "valid V3 fixture;4 storage-plan chunks",
      .unit = "us/request",
      .notes = "inspection and storage plan are prepared outside the timed interval",
      .repetitions = 1,
      .setup = [text, setup_report] { return setup_report(text); },
      .run = [text] {
        text->text = render_gguf_text_report(text->inspection, text->plan);
        return !text->text.empty();
      },
  });
  const auto json = shared_context<ReportContext>();
  cases.push_back({
      .id = "gguf.json_report",
      .subsystem = "GGUF",
      .workload = "render_gguf_json_plan",
      .variant = "JSON",
      .input = "valid V3 fixture;4 storage-plan chunks",
      .unit = "us/request",
      .notes = "inspection and storage plan are prepared outside the timed interval",
      .repetitions = 1,
      .setup = [json, setup_report] { return setup_report(json); },
      .run = [json] {
        json->text = render_gguf_json_plan(json->inspection, json->plan);
        return json->text.find("\"input_fingerprint\"") != std::string::npos;
      },
  });

  struct ScanContext { GgufInspection inspection; GgufStoragePlan plan; GgufPatternScanResult scan; };
  const auto scan = shared_context<ScanContext>();
  cases.push_back({
      .id = "gguf.pattern_scan",
      .subsystem = "GGUF",
      .workload = "scan_gguf_pattern_candidates",
      .variant = "bounded constant-byte scan",
      .input = "one selected 16 KiB tensor;max_candidates=8",
      .unit = "us/request",
      .notes = "inspection and plan are prepared outside the timed interval; scan remains read-only",
      .repetitions = 1,
      .setup = [scan, fixture] {
        scan->inspection = inspect_gguf_file(fixture);
        if (scan->inspection.status != Status::Confirmed || !scan->inspection.inventory) return false;
        scan->plan = make_gguf_storage_plan(*scan->inspection.inventory,
                                            {.memory_budget_bytes = 16U << 10U,
                                             .chunk_size_bytes = 16U << 10U,
                                             .policy = GgufLoadingPolicy::AllResident});
        return scan->plan.status == Status::Confirmed;
      },
      .run = [scan, fixture] {
        scan->scan = scan_gguf_pattern_candidates(
            fixture, *scan->inspection.inventory, scan->plan.items,
            {.max_payload_bytes = 16U << 10U, .max_candidates = 8});
        return scan->scan.status == Status::Confirmed;
      },
      .verify = [scan] { return !scan->scan.candidates.empty(); },
  });
  return cases;
}

[[nodiscard]] bool run_case(const BenchmarkCase& benchmark, std::size_t iterations,
                            CsvWriter& output) {
  const auto invoke = [&benchmark] {
    for (std::size_t repetition = 0; repetition < benchmark.repetitions; ++repetition) {
      if (!benchmark.run()) return false;
    }
    return !benchmark.verify || benchmark.verify();
  };
  if (!benchmark.setup() || !invoke()) {
    std::cerr << "warm-up validation failed: " << benchmark.id << '\n';
    return false;
  }

  std::vector<double> samples;
  samples.reserve(iterations);
  for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
    if (!benchmark.setup()) {
      std::cerr << "setup failed: " << benchmark.id << " iteration " << iteration << '\n';
      return false;
    }
    const auto started = Clock::now();
    const auto valid = invoke();
    const auto elapsed = std::chrono::duration<double, std::nano>(Clock::now() - started).count();
    if (!valid) {
      std::cerr << "validation failed: " << benchmark.id << " iteration " << iteration << '\n';
      return false;
    }
    const auto normalized = benchmark.unit == "us/request"
        ? elapsed / 1'000.0
        : elapsed / static_cast<double>(benchmark.repetitions);
    samples.push_back(normalized);
    output.row(benchmark, iteration + 1U, normalized);
  }
  const auto summary = benchmark::statistics::summarize(samples);
  if (!summary.has_value()) return false;
  std::cout << benchmark.id << ": median=" << std::fixed << std::setprecision(3)
            << summary->median << ' ' << benchmark.unit << "; min=" << summary->minimum
            << "; max=" << summary->maximum << "; stddev=" << summary->standard_deviation
            << '\n';
  return output.good();
}

[[nodiscard]] bool parse_positive(std::string_view text, std::size_t& value) {
  try {
    std::size_t consumed = 0;
    const auto parsed = std::stoull(std::string{text}, &consumed);
    if (consumed != text.size() || parsed == 0U) return false;
    value = static_cast<std::size_t>(parsed);
    return true;
  } catch (...) {
    return false;
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  std::size_t iterations = 11;
  std::optional<std::filesystem::path> csv_path;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument{argv[index]};
    if (argument == "--quick") {
      iterations = 7;
    } else if (argument == "--iterations" && index + 1 < argc) {
      if (!parse_positive(argv[++index], iterations)) {
        std::cerr << "--iterations requires a positive integer\n";
        return 2;
      }
    } else if (argument == "--csv" && index + 1 < argc) {
      csv_path = std::filesystem::path{argv[++index]};
    } else {
      std::cerr << "usage: lhme_end_to_end_benchmark [--quick] [--iterations N] --csv FILE\n";
      return 2;
    }
  }
  if (!csv_path.has_value() || std::filesystem::exists(*csv_path)) {
    std::cerr << "--csv must name a new output file\n";
    return 2;
  }

  const auto fixture = gguf_fixture_path();
  const auto snapshot = fixture.string() + ".snapshot";
  if (!write_gguf_fixture(fixture)) {
    std::cerr << "could not create deterministic GGUF fixture\n";
    return 1;
  }
  CsvWriter output{*csv_path};
  if (!output.good()) {
    std::filesystem::remove(fixture);
    std::cerr << "could not open CSV output\n";
    return 1;
  }

  std::vector<BenchmarkCase> cases;
  const auto append = [&cases](std::vector<BenchmarkCase> more) {
    cases.insert(cases.end(), std::make_move_iterator(more.begin()),
                 std::make_move_iterator(more.end()));
  };
  append(core_cases());
  append(engine_cases());
  append(range_cases());
  append(override_cases(snapshot));
  append(legacy_kevin_cases());
  append(gguf_cases(fixture));

  bool success = true;
  for (const auto& benchmark : cases) {
    if (!run_case(benchmark, iterations, output)) {
      success = false;
      break;
    }
  }
  std::error_code error;
  std::filesystem::remove(fixture, error);
  std::filesystem::remove(snapshot, error);
  if (!success) return 1;
  std::cout << "validated_cases=" << cases.size() << "; iterations=" << iterations
            << "; warmups=1; raw_csv=" << csv_path->string() << '\n';
  return 0;
}
