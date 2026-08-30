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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_comparison.h"
#include "xla/primitive_util.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/xla_data.pb.h"

namespace xla::compare_literals {
namespace {

template <typename NativeT>
void CompareArrayValues(const LiteralSlice& clean, const LiteralSlice& dirty,
                        const ComparisonOptions& options,
                        ComparisonResult* result) {
  ElementComparator<NativeT> comparator(options,
                                        ShapeUtil::ElementsIn(clean.shape()));

  if (LayoutUtil::Equal(dirty.shape().layout(), clean.shape().layout()) &&
      clean.shape().is_static() && dirty.shape().is_static()) {
    absl::Span<const NativeT> clean_span = clean.data<NativeT>();
    absl::Span<const NativeT> dirty_span = dirty.data<NativeT>();
    const int64_t num_elements = clean_span.size();
    for (int64_t i = 0; i < num_elements; ++i) {
      comparator.RecordElement(i, clean_span[i], dirty_span[i]);
    }
  } else {
    std::vector<int64_t> multi_index(clean.shape().dimensions_size(), 0);
    const int64_t num_elements = ShapeUtil::ElementsIn(clean.shape());
    for (int64_t i = 0; i < num_elements; ++i) {
      comparator.RecordElement(i, clean.Get<NativeT>(multi_index),
                               dirty.Get<NativeT>(multi_index));
      for (int d = multi_index.size() - 1; d >= 0; --d) {
        if (++multi_index[d] < clean.shape().dimensions(d)) {
          break;
        }
        multi_index[d] = 0;
      }
    }
  }

  *result = comparator.Finalize();
}

std::string FormatCompactSci(double v) {
  if (std::isnan(v)) return "nan";
  if (std::isinf(v)) return v < 0 ? "-inf" : "inf";
  if (v == 0.0) return "0";
  bool negative = v < 0.0;
  if (negative) v = -v;

  int exp = static_cast<int>(std::floor(std::log10(v) + 1e-9));
  double mantissa = std::round((v / std::pow(10.0, exp)) * 1e6) / 1e6;
  if (mantissa >= 10.0) {
    mantissa /= 10.0;
    exp += 1;
  }

  std::string s;
  if (negative) s += "-";
  absl::StrAppendFormat(&s, "%ge%d", mantissa, exp);
  return s;
}

}  // namespace

std::string RelErrorHistogram::ToString(int max_bar_width) const {
  std::string out =
      "1D Signed Relative Error Distribution ((actual - expected) / "
      "expected):\n";
  int64_t max_bin_count = 0;
  for (const auto& b : bins) {
    max_bin_count = std::max(max_bin_count, b.count);
  }

  for (size_t i = 0; i < bins.size(); ++i) {
    const auto& b = bins[i];
    int bar_len =
        (max_bin_count > 0)
            ? static_cast<int>(b.count * max_bar_width / max_bin_count)
            : 0;
    std::string bar(bar_len, '*');

    std::string markers;
    if (static_cast<int>(i) == median_bin_index) {
      markers += " <--- median";
    }
    if (b.is_exact_zero) {
      markers += " <--- exact match (zero)";
    } else if (mean_rel_error >= b.lower && mean_rel_error < b.upper) {
      markers += " <--- mean";
    }

    std::string range_str;
    if (b.is_exact_zero) {
      range_str = "       [   0  ]";
    } else if (std::isinf(b.lower)) {
      range_str = absl::StrFormat(" (-inf, %5s)", FormatCompactSci(b.upper));
    } else if (std::isinf(b.upper)) {
      range_str = absl::StrFormat(" (%5s, +inf)", FormatCompactSci(b.lower));
    } else {
      range_str = absl::StrFormat(" [%5s, %5s)", FormatCompactSci(b.lower),
                                  FormatCompactSci(b.upper));
    }

    absl::StrAppendFormat(&out, "  %2d: %s %8d %s%s\n", i, range_str, b.count,
                          bar, markers);
  }

  absl::StrAppendFormat(
      &out,
      "  Summary: min = %1.3e | max = %1.3e | mean = %1.3e | std_dev = %1.3e\n",
      min_rel_error, max_rel_error, mean_rel_error, std_dev_rel_error);
  return out;
}

std::string ErrorHeatmap::ToString(bool use_color) const {
  if (abs_thresholds.empty() || rel_thresholds.empty()) {
    return "";
  }

  const int T_a = target_abs_idx >= 0 ? target_abs_idx : 0;
  const int T_r = target_rel_idx >= 0 ? target_rel_idx : 0;
  const int num_cols_total = static_cast<int>(abs_thresholds.size());
  const int num_rows_total = static_cast<int>(rel_thresholds.size());

  // Find the first column >= T_a that has all zeros across all elements so the
  // leftmost column has all zeros.
  int col_zero = num_cols_total - 1;
  for (int c = T_a; c < num_cols_total; ++c) {
    bool all_zero = true;
    for (int r = 0; r < num_rows_total; ++r) {
      if (mismatch_counts[r][c] != 0) {
        all_zero = false;
        break;
      }
    }
    if (all_zero) {
      col_zero = c;
      break;
    }
  }

  // Target up to 13 columns to comfortably fit a 120-column terminal.
  constexpr int kMaxCols = 13;
  int col_end = std::min(num_cols_total - 1, col_zero);
  int needed_cols_above = col_end - T_a + 1;
  int col_start = 0;
  if (needed_cols_above >= kMaxCols) {
    col_start = std::max(0, T_a - 1);
    col_end = std::min(num_cols_total - 1, col_start + kMaxCols - 1);
  } else {
    int remaining = kMaxCols - needed_cols_above;
    col_start = std::max(0, T_a - remaining);
    col_end = std::min(num_cols_total - 1, col_start + kMaxCols - 1);
  }

  // Find the first row >= T_r that has all zeros across all elements so the
  // bottom-most row has all zeros.
  int row_zero = num_rows_total - 1;
  for (int r = T_r; r < num_rows_total; ++r) {
    bool all_zero = true;
    for (int c = 0; c < num_cols_total; ++c) {
      if (mismatch_counts[r][c] != 0) {
        all_zero = false;
        break;
      }
    }
    if (all_zero) {
      row_zero = r;
      break;
    }
  }

  // Target up to 25 rows to accommodate the full failure envelope down to zero.
  constexpr int kMaxRows = 25;
  int row_end = std::min(num_rows_total - 1, row_zero);
  int needed_rows_above = row_end - T_r + 1;
  int row_start = 0;
  if (needed_rows_above >= kMaxRows) {
    row_start = std::max(0, T_r - 2);
    row_end = std::min(num_rows_total - 1, row_start + kMaxRows - 1);
  } else {
    int remaining = kMaxRows - needed_rows_above;
    row_start = std::max(0, T_r - remaining);
    row_end = std::min(num_rows_total - 1, row_start + kMaxRows - 1);
  }

  std::string out =
      "2D Error Heatmap (Count of elements failing: abs_diff > X AND rel_diff "
      "> Y):\n";

  // Table header (from largest abs_threshold down to smallest)
  absl::StrAppend(&out, "      Rel \\ Abs |");
  for (int a = col_end; a >= col_start; --a) {
    std::string col_hdr = FormatCompactSci(abs_thresholds[a]);
    if (a == target_abs_idx) {
      col_hdr = absl::StrCat("*", col_hdr);
    }
    absl::StrAppendFormat(&out, " %6s |", col_hdr);
  }
  absl::StrAppend(&out, "\n    ------------+");
  for (int a = col_end; a >= col_start; --a) {
    absl::StrAppend(&out, "-------+");
  }
  absl::StrAppend(&out, "\n");

  // Rows from smallest rel_threshold up to largest
  for (int r = row_start; r <= row_end; ++r) {
    std::string row_hdr = FormatCompactSci(rel_thresholds[r]);
    if (r == target_rel_idx) {
      row_hdr = absl::StrCat("*", row_hdr);
    }
    absl::StrAppendFormat(&out, "    %9s |", row_hdr);

    for (int a = col_end; a >= col_start; --a) {
      int64_t count = mismatch_counts[r][a];
      double ratio = total_elements > 0
                         ? static_cast<double>(count) / total_elements
                         : 0.0;
      bool is_target = (r == target_rel_idx && a == target_abs_idx);

      std::string cell_str;
      if (is_target) {
        cell_str = absl::StrFormat("[%4d]", count);
      } else {
        cell_str = absl::StrFormat("%6d", count);
      }

      if (use_color) {
        if (count == 0) {
          cell_str = absl::StrCat("\033[32m", cell_str, "\033[0m");
        } else if (ratio < 0.001) {
          cell_str = absl::StrCat("\033[33m", cell_str, "\033[0m");
        } else {
          cell_str = absl::StrCat("\033[31m", cell_str, "\033[0m");
        }
        if (is_target) {
          cell_str = absl::StrCat("\033[1m", cell_str);
        }
      }
      absl::StrAppendFormat(&out, " %s |", cell_str);
    }
    absl::StrAppend(&out, "\n");
  }

  absl::StrAppend(&out, "    ------------+");
  for (int a = col_end; a >= col_start; --a) {
    absl::StrAppend(&out, "-------+");
  }
  absl::StrAppend(&out, "\n");
  absl::StrAppend(&out,
                  "    Legend: [*] Target Tolerance (abs, rel)  |  Green (0) = "
                  "100% elements within bounds\n");

  return out;
}

std::string ComparisonResult::SummaryToString() const {
  std::string out;
  double mismatch_pct =
      total_elements > 0 ? (100.0 * mismatches / total_elements) : 0.0;
  absl::StrAppendFormat(
      &out,
      "Verdict: %s\n"
      "  Element Type: %s\n"
      "  Shape: %s\n"
      "  Total Elements: %d\n"
      "  Exact Bitwise Matches: %d (%.2f%%)\n"
      "  Mismatches (exceeding tolerance): %d (%.4f%%)\n"
      "  NaN Mismatches: %d\n"
      "  Inf Mismatches: %d\n"
      "  Max Absolute Error: %1.4e\n"
      "  Max Relative Error: %1.4e\n",
      passed ? "PASS (MATCH)" : "FAIL (MISMATCH)",
      element_type.empty() ? "unknown" : element_type,
      shape_str.empty() ? "unknown" : shape_str, total_elements, exact_matches,
      total_elements > 0 ? (100.0 * exact_matches / total_elements) : 0.0,
      mismatches, mismatch_pct, nan_mismatches, inf_mismatches, max_abs_error,
      max_rel_error);

  if (!top_mismatches.empty()) {
    absl::StrAppend(&out, "\nFirst Mismatches:\n");
    for (const auto& m : top_mismatches) {
      absl::StrAppendFormat(
          &out,
          "  [Index %8d] Clean: %12s | Dirty: %12s | Abs Diff: %1.3e | "
          "Rel Diff: %1.3e\n",
          m.linear_index, m.clean_str, m.dirty_str, m.abs_diff, m.rel_diff);
    }
  }

  if (!passed && suggested_error_spec.has_value()) {
    absl::StrAppend(&out, "\n", suggested_error_spec->ToString(), "\n");
  }

  return out;
}

std::string SuggestedErrorSpec::ToString() const {
  std::string s;
  absl::StrAppend(&s, "Suggested ErrorSpec (to pass all elements):\n");
  absl::StrAppendFormat(
      &s,
      "  Balanced:       abs = %s, rel = %s  (with 2x margin: abs = %s, rel = "
      "%s)\n",
      FormatCompactSci(abs_bound), FormatCompactSci(rel_bound),
      FormatCompactSci(margin_abs_bound), FormatCompactSci(margin_rel_bound));
  absl::StrAppendFormat(
      &s,
      "  Pure Absolute:  abs = %s              (with 2x margin: abs = %s)\n",
      FormatCompactSci(pure_abs_bound),
      FormatCompactSci(margin_pure_abs_bound));
  absl::StrAppendFormat(
      &s, "  Pure Relative:  rel = %s              (with 2x margin: rel = %s)",
      FormatCompactSci(pure_rel_bound),
      FormatCompactSci(margin_pure_rel_bound));
  return s;
}

absl::StatusOr<ComparisonResult> CompareLiterals(
    const LiteralSlice& clean, const LiteralSlice& dirty,
    const ComparisonOptions& options) {
  ABSL_RETURN_IF_ERROR(
      literal_comparison::EqualShapes(clean.shape(), dirty.shape()));

  if (!clean.shape().IsArray()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Only array literals are supported; got: ",
                     ShapeUtil::HumanString(clean.shape())));
  }

  if (!primitive_util::IsArrayType(clean.shape().element_type())) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported element type for literal comparison: ",
                     primitive_util::LowercasePrimitiveTypeName(
                         clean.shape().element_type())));
  }

  ComparisonResult result;
  primitive_util::ArrayTypeSwitch(
      [&](auto type_constant) {
        using NativeT = primitive_util::NativeTypeOf<type_constant>;
        CompareArrayValues<NativeT>(clean, dirty, options, &result);
      },
      clean.shape().element_type());

  result.element_type =
      primitive_util::LowercasePrimitiveTypeName(clean.shape().element_type());
  result.shape_str = ShapeUtil::HumanString(clean.shape());
  return result;
}

absl::StatusOr<ComparisonResult> CompareLiteralProtos(
    const LiteralProto& clean_proto, const LiteralProto& dirty_proto,
    const ComparisonOptions& options) {
  ABSL_ASSIGN_OR_RETURN(Literal clean, Literal::CreateFromProto(clean_proto));
  ABSL_ASSIGN_OR_RETURN(Literal dirty, Literal::CreateFromProto(dirty_proto));
  return CompareLiterals(clean, dirty, options);
}

absl::StatusOr<ComparisonResult> CompareLiteralFiles(
    absl::string_view clean_file, absl::string_view dirty_file,
    const ComparisonOptions& options) {
  LiteralProto clean_proto;
  ABSL_RETURN_IF_ERROR(tsl::ReadBinaryProto(tsl::Env::Default(),
                                       std::string(clean_file), &clean_proto));

  LiteralProto dirty_proto;
  ABSL_RETURN_IF_ERROR(tsl::ReadBinaryProto(tsl::Env::Default(),
                                       std::string(dirty_file), &dirty_proto));

  return CompareLiteralProtos(clean_proto, dirty_proto, options);
}

}  // namespace xla::compare_literals
