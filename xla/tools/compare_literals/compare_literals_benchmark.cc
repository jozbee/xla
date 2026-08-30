/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdint>

#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/shape_util.h"
#include "xla/tools/compare_literals/compare_literals.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"

namespace xla::compare_literals {
namespace {

// Benchmark comparing literals that match exactly across size ranges.
void BM_CompareLiteralsExactMatch(::testing::benchmark::State& state) {
  const int64_t num_elements = state.range(0);

  Literal clean(ShapeUtil::MakeShape(F32, {num_elements}));
  clean.PopulateWithValue<float>(1.0f);

  Literal dirty(ShapeUtil::MakeShape(F32, {num_elements}));
  dirty.PopulateWithValue<float>(1.0f);

  ComparisonOptions options;

  for (auto s : state) {
    auto result = CompareLiterals(clean, dirty, options);
    ::benchmark::DoNotOptimize(result);
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          num_elements);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          num_elements * sizeof(float) * 2);
}

// Benchmark comparing literals with realistic mismatches across size ranges.
void BM_CompareLiteralsMismatches(::testing::benchmark::State& state) {
  const int64_t num_elements = state.range(0);

  Literal clean(ShapeUtil::MakeShape(F32, {num_elements}));
  clean.PopulateWithValue<float>(1.0f);

  Literal dirty(ShapeUtil::MakeShape(F32, {num_elements}));
  dirty.PopulateWithValue<float>(1.0f);

  // Perturb every 1,000th element (~0.1% mismatches exceeding tolerance).
  absl::Span<float> dirty_span = dirty.data<float>();
  for (int64_t i = 0; i < num_elements; i += 1000) {
    dirty_span[i] = 1.005f;
  }

  ComparisonOptions options;
  options.abs_error_bound = 1e-3;
  options.rel_error_bound = 1e-3;

  for (auto s : state) {
    auto result = CompareLiterals(clean, dirty, options);
    ::benchmark::DoNotOptimize(result);
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          num_elements);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          num_elements * sizeof(float) * 2);
}

// Direct benchmark on ElementComparator to measure core accumulator throughput.
void BM_ElementComparator(::testing::benchmark::State& state) {
  const int64_t num_elements = state.range(0);

  Literal clean(ShapeUtil::MakeShape(F32, {num_elements}));
  clean.PopulateWithValue<float>(1.0f);

  Literal dirty(ShapeUtil::MakeShape(F32, {num_elements}));
  dirty.PopulateWithValue<float>(1.0f);
  absl::Span<float> dirty_span = dirty.data<float>();
  for (int64_t i = 0; i < num_elements; i += 1000) {
    dirty_span[i] = 1.005f;
  }

  absl::Span<const float> clean_span = clean.data<float>();

  ComparisonOptions options;
  options.abs_error_bound = 1e-3;
  options.rel_error_bound = 1e-3;

  for (auto s : state) {
    ElementComparator<float> comparator(options, num_elements);
    for (int64_t i = 0; i < num_elements; ++i) {
      comparator.RecordElement(i, clean_span[i], dirty_span[i]);
    }
    ComparisonResult result = comparator.Finalize();
    ::benchmark::DoNotOptimize(result);
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          num_elements);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          num_elements * sizeof(float) * 2);
}

BENCHMARK(BM_CompareLiteralsExactMatch)
    ->RangeMultiplier(10)
    ->Range(1'000, 10'000'000)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_CompareLiteralsMismatches)
    ->RangeMultiplier(10)
    ->Range(1'000, 10'000'000)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_ElementComparator)
    ->RangeMultiplier(10)
    ->Range(1'000, 10'000'000)
    ->Unit(benchmark::kMillisecond);

}  // namespace
}  // namespace xla::compare_literals
