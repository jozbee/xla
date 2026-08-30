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

#include "xla/tools/compare_literals/compare_literals.h"

#include <limits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/literal.h"
#include "xla/literal_util.h"

namespace xla::compare_literals {
namespace {

using ::testing::HasSubstr;

TEST(CompareLiteralsTest, ExactMatch) {
  Literal lit1 = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f});
  Literal lit2 = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f});

  ComparisonOptions options;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(lit1, lit2, options));

  EXPECT_TRUE(result.passed);
  EXPECT_EQ(result.element_type, "f32");
  EXPECT_EQ(result.shape_str, "f32[4]");
  EXPECT_EQ(result.total_elements, 4);
  EXPECT_EQ(result.exact_matches, 4);
  EXPECT_EQ(result.mismatches, 0);
  EXPECT_DOUBLE_EQ(result.max_abs_error, 0.0);
  EXPECT_DOUBLE_EQ(result.max_rel_error, 0.0);
  EXPECT_THAT(result.SummaryToString(), HasSubstr("Element Type: f32"));
  EXPECT_THAT(result.SummaryToString(), HasSubstr("Shape: f32[4]"));
}

TEST(CompareLiteralsTest, WithinTolerance) {
  Literal lit1 = LiteralUtil::CreateR1<float>({1.0f, 10.0f, 100.0f});
  Literal lit2 = LiteralUtil::CreateR1<float>({1.0001f, 10.001f, 100.01f});

  ComparisonOptions options;
  options.abs_error_bound = 1e-3;
  options.rel_error_bound = 1e-3;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(lit1, lit2, options));

  EXPECT_TRUE(result.passed);
  EXPECT_EQ(result.total_elements, 3);
  EXPECT_EQ(result.mismatches, 0);
  EXPECT_GT(result.max_abs_error, 0.0);
}

TEST(CompareLiteralsTest, ExceedsTolerance) {
  Literal lit1 = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f});
  Literal lit2 = LiteralUtil::CreateR1<float>({1.0f, 2.5f, 3.0f});

  ComparisonOptions options;
  options.abs_error_bound = 1e-3;
  options.rel_error_bound = 1e-3;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(lit1, lit2, options));

  EXPECT_FALSE(result.passed);
  EXPECT_EQ(result.total_elements, 3);
  EXPECT_EQ(result.exact_matches, 2);
  EXPECT_EQ(result.mismatches, 1);
  EXPECT_NEAR(result.max_abs_error, 0.5, 1e-5);
  EXPECT_NEAR(result.max_rel_error, 0.25, 1e-5);
  ASSERT_EQ(result.top_mismatches.size(), 1);
  EXPECT_EQ(result.top_mismatches[0].linear_index, 1);
  EXPECT_EQ(result.top_mismatches[0].clean_str, "2");
  EXPECT_EQ(result.top_mismatches[0].dirty_str, "2.5");
}

TEST(CompareLiteralsTest, NaNHandling) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  Literal lit1 = LiteralUtil::CreateR1<float>({1.0f, kNaN});
  Literal lit2 = LiteralUtil::CreateR1<float>({1.0f, kNaN});

  ComparisonOptions options;
  options.all_nans_are_equivalent = true;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(lit1, lit2, options));

  EXPECT_TRUE(result.passed);
  EXPECT_EQ(result.exact_matches, 2);
}

TEST(CompareLiteralsTest, NaNMismatch) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  Literal lit1 = LiteralUtil::CreateR1<float>({1.0f, 2.0f});
  Literal lit2 = LiteralUtil::CreateR1<float>({1.0f, kNaN});

  ComparisonOptions options;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(lit1, lit2, options));

  EXPECT_FALSE(result.passed);
  EXPECT_EQ(result.nan_mismatches, 1);
  EXPECT_EQ(result.mismatches, 1);
}

TEST(CompareLiteralsTest, ShapeMismatch) {
  Literal lit1 = LiteralUtil::CreateR1<float>({1.0f, 2.0f});
  Literal lit2 = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f});

  ComparisonOptions options;
  auto status_or = CompareLiterals(lit1, lit2, options);
  EXPECT_FALSE(status_or.ok());
}

TEST(CompareLiteralsTest, HistogramAndHeatmapOutput) {
  Literal lit1 = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 4.0f});
  Literal lit2 = LiteralUtil::CreateR1<float>({1.01f, 2.02f, 4.04f});

  ComparisonOptions options;
  options.abs_error_bound = 1e-3;
  options.rel_error_bound = 1e-3;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(lit1, lit2, options));

  std::string hist_str = result.histogram.ToString();
  EXPECT_THAT(hist_str, HasSubstr("Summary: min ="));

  std::string heatmap_str = result.heatmap.ToString(/*use_color=*/false);
  EXPECT_THAT(heatmap_str, HasSubstr("2D Error Heatmap"));
  EXPECT_THAT(heatmap_str, HasSubstr("Legend:"));
}

TEST(ElementComparatorTest, RecordIndividualElements) {
  ComparisonOptions options;
  options.abs_error_bound = 1e-3;
  options.rel_error_bound = 1e-3;

  ElementComparator<float> comparator(options, /*total_elements=*/3);
  comparator.RecordElement(0, 1.0f, 1.0f);      // exact match
  comparator.RecordElement(1, 10.0f, 10.005f);  // within tolerance
  comparator.RecordElement(2, 2.0f, 2.5f);      // mismatch

  ComparisonResult result = comparator.Finalize();
  EXPECT_FALSE(result.passed);
  EXPECT_EQ(result.total_elements, 3);
  EXPECT_EQ(result.exact_matches, 1);
  EXPECT_EQ(result.mismatches, 1);
  EXPECT_NEAR(result.max_abs_error, 0.5, 1e-5);
  EXPECT_NEAR(result.max_rel_error, 0.25, 1e-5);
  ASSERT_EQ(result.top_mismatches.size(), 1);
  EXPECT_EQ(result.top_mismatches[0].clean_str, "2");
  EXPECT_EQ(result.top_mismatches[0].dirty_str, "2.5");
}

TEST(ElementComparatorTest, ComplexNumbers) {
  ComparisonOptions options;
  options.abs_error_bound = 1e-2;
  options.rel_error_bound = 1e-2;

  ElementComparator<std::complex<float>> comparator(options,
                                                    /*total_elements=*/2);
  comparator.RecordElement(0, {1.0f, 2.0f}, {1.0f, 2.0f});
  comparator.RecordElement(1, {1.0f, 0.0f}, {2.0f, 0.0f});

  ComparisonResult result = comparator.Finalize();
  EXPECT_FALSE(result.passed);
  EXPECT_EQ(result.exact_matches, 1);
  EXPECT_EQ(result.mismatches, 1);
  EXPECT_DOUBLE_EQ(result.max_abs_error, 1.0);
}

TEST(CompareLiteralsTest, NegativeValues) {
  Literal lit1 = LiteralUtil::CreateR1<float>({-10.0f, -100.0f});
  Literal lit2 = LiteralUtil::CreateR1<float>({-10.005f, -99.95f});

  ComparisonOptions options;
  options.abs_error_bound = 1e-1;
  options.rel_error_bound = 1e-3;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(lit1, lit2, options));

  EXPECT_TRUE(result.passed);
  EXPECT_NEAR(result.max_abs_error, 0.05, 1e-5);
  EXPECT_NEAR(result.max_rel_error, 5e-4, 1e-6);

  // Exceeding tolerance on negative numbers
  Literal lit3 = LiteralUtil::CreateR1<float>({-2.0f});
  Literal lit4 = LiteralUtil::CreateR1<float>({-2.5f});
  ASSERT_OK_AND_ASSIGN(ComparisonResult result2,
                       CompareLiterals(lit3, lit4, options));
  EXPECT_FALSE(result2.passed);
  EXPECT_NEAR(result2.max_abs_error, 0.5, 1e-5);
  EXPECT_NEAR(result2.max_rel_error, 0.25, 1e-5);
}

TEST(CompareLiteralsTest, RelaxedNaNs) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  // Expected NaN, actual finite: passes under relaxed_nans.
  Literal expected_nan = LiteralUtil::CreateR1<float>({kNaN});
  Literal actual_finite = LiteralUtil::CreateR1<float>({42.0f});

  ComparisonOptions options;
  options.relaxed_nans = true;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(expected_nan, actual_finite, options));
  EXPECT_TRUE(result.passed);

  // Unexpected NaN, actual NaN when expecting finite: fails under relaxed_nans.
  Literal expected_finite = LiteralUtil::CreateR1<float>({42.0f});
  Literal actual_nan = LiteralUtil::CreateR1<float>({kNaN});
  ASSERT_OK_AND_ASSIGN(ComparisonResult result2,
                       CompareLiterals(expected_finite, actual_nan, options));
  EXPECT_FALSE(result2.passed);
  EXPECT_EQ(result2.nan_mismatches, 1);
}

TEST(CompareLiteralsTest, ExactMatchesIncludedInWelford) {
  // 3 exact matches, 1 element with 0.04 relative error.
  Literal lit1 = LiteralUtil::CreateR1<float>({10.0f, 10.0f, 10.0f, 10.0f});
  Literal lit2 = LiteralUtil::CreateR1<float>({10.0f, 10.0f, 10.0f, 10.4f});

  ComparisonOptions options;
  options.abs_error_bound = 1.0;
  options.rel_error_bound = 0.1;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(lit1, lit2, options));

  EXPECT_TRUE(result.passed);
  // Mean relative error across 4 elements: (0 + 0 + 0 + 0.04) / 4 = 0.01.
  EXPECT_NEAR(result.histogram.mean_rel_error, 0.01, 1e-5);
}

TEST(CompareLiteralsTest, LargeInt64Comparison) {
  // Values above 2^53 that differ by 1.
  constexpr int64_t kBase = 9007199254740992LL;  // 2^53
  Literal lit1 = LiteralUtil::CreateR1<int64_t>({kBase, kBase + 1});
  Literal lit2 = LiteralUtil::CreateR1<int64_t>({kBase, kBase});

  ComparisonOptions options;
  options.abs_error_bound = 0.0;
  options.rel_error_bound = 0.0;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(lit1, lit2, options));

  EXPECT_FALSE(result.passed);
  EXPECT_EQ(result.exact_matches, 1);
  EXPECT_EQ(result.mismatches, 1);
  EXPECT_DOUBLE_EQ(result.max_abs_error, 1.0);
}

TEST(CompareLiteralsTest, SingleElementMedian) {
  Literal lit1 = LiteralUtil::CreateR1<float>({100.0f});
  Literal lit2 = LiteralUtil::CreateR1<float>({102.0f});  // +2% rel error

  ComparisonOptions options;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(lit1, lit2, options));

  EXPECT_GT(result.histogram.median_bin_index, 0);
  EXPECT_EQ(result.histogram.bins[result.histogram.median_bin_index].count, 1);
}

TEST(CompareLiteralsTest, SuggestedErrorSpecFailingComparison) {
  Literal lit1 = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f});
  Literal lit2 = LiteralUtil::CreateR1<float>({1.0f, 2.5f, 3.0f});

  ComparisonOptions options;
  options.abs_error_bound = 1e-3;
  options.rel_error_bound = 1e-3;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(lit1, lit2, options));

  EXPECT_FALSE(result.passed);
  ASSERT_TRUE(result.suggested_error_spec.has_value());
  const auto& spec = *result.suggested_error_spec;

  EXPECT_GE(spec.pure_abs_bound, 0.5);
  EXPECT_GE(spec.pure_rel_bound, 0.25);
  EXPECT_GE(spec.margin_abs_bound, spec.abs_bound);
  EXPECT_GE(spec.margin_rel_bound, spec.rel_bound);

  // Crucial verification: running CompareLiterals with suggested balanced
  // bounds MUST pass!
  ComparisonOptions passing_options;
  passing_options.abs_error_bound = spec.abs_bound;
  passing_options.rel_error_bound = spec.rel_bound;
  ASSERT_OK_AND_ASSIGN(ComparisonResult passing_result,
                       CompareLiterals(lit1, lit2, passing_options));
  EXPECT_TRUE(passing_result.passed);
  EXPECT_EQ(passing_result.mismatches, 0);

  // Output formatting verification
  EXPECT_THAT(result.SummaryToString(), HasSubstr("Suggested ErrorSpec"));
  EXPECT_THAT(result.SummaryToString(), HasSubstr("Balanced:"));
  EXPECT_THAT(result.SummaryToString(), HasSubstr("Pure Absolute:"));
  EXPECT_THAT(result.SummaryToString(), HasSubstr("Pure Relative:"));
}

TEST(CompareLiteralsTest, SuggestedErrorSpecNulloptOnNanMismatches) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  Literal lit1 = LiteralUtil::CreateR1<float>({1.0f, 2.0f});
  Literal lit2 = LiteralUtil::CreateR1<float>({1.0f, kNaN});

  ComparisonOptions options;
  ASSERT_OK_AND_ASSIGN(ComparisonResult result,
                       CompareLiterals(lit1, lit2, options));

  EXPECT_FALSE(result.passed);
  EXPECT_EQ(result.nan_mismatches, 1);
  EXPECT_FALSE(result.suggested_error_spec.has_value());
}

}  // namespace
}  // namespace xla::compare_literals
