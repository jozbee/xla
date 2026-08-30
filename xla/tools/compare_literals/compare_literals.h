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

#ifndef XLA_TOOLS_COMPARE_LITERALS_COMPARE_LITERALS_H_
#define XLA_TOOLS_COMPARE_LITERALS_COMPARE_LITERALS_H_

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/literal.h"
#include "xla/xla_data.pb.h"

namespace xla::compare_literals {

struct ComparisonOptions {
  double abs_error_bound = 1e-3;
  double rel_error_bound = 1e-3;
  bool relaxed_nans = false;
  bool all_nans_are_equivalent = true;
  int max_mismatches_to_record = 10;
};

// Represents a bin in the 1D relative error distribution.
struct RelErrorBin {
  double lower = 0.0;
  double upper = 0.0;
  int64_t count = 0;
  bool is_exact_zero = false;
};

// 1D Relative Error Histogram with ASCII formatting.
struct RelErrorHistogram {
  std::vector<RelErrorBin> bins;
  int64_t total_samples = 0;
  double min_rel_error = 0.0;
  double max_rel_error = 0.0;
  double mean_rel_error = 0.0;
  double std_dev_rel_error = 0.0;
  int median_bin_index = -1;

  // Formats as an ASCII bar chart similar to dot_algorithms_test.cc.
  std::string ToString(int max_bar_width = 40) const;
};

// 2D Heatmap of element mismatches for pairs of (abs_threshold, rel_threshold).
struct ErrorHeatmap {
  // Sorted threshold boundaries.
  std::vector<double> abs_thresholds;
  std::vector<double> rel_thresholds;

  // 2D grid: mismatch_counts[rel_idx][abs_idx] is the number of elements
  // having abs_diff > abs_thresholds[abs_idx] AND
  // rel_diff > rel_thresholds[rel_idx].
  std::vector<std::vector<int64_t>> mismatch_counts;

  // The user's target tolerance parameters.
  double target_abs = 0.0;
  double target_rel = 0.0;
  int target_abs_idx = -1;
  int target_rel_idx = -1;
  int64_t total_elements = 0;

  // Formats the 2D matrix into a terminal-friendly table with ANSI colors.
  std::string ToString(bool use_color = true) const;
};

// Detailed info for an individual element mismatch.
struct MismatchDetail {
  int64_t linear_index = 0;
  std::string clean_str;
  std::string dirty_str;
  double abs_diff = 0.0;
  double rel_diff = 0.0;
};

// Suggested error specification (abs and rel bounds) to make comparison pass.
struct SuggestedErrorSpec {
  // Balanced point on Pareto frontier (knee in log-log space).
  double abs_bound = 0.0;
  double rel_bound = 0.0;
  double margin_abs_bound = 0.0;
  double margin_rel_bound = 0.0;

  // Pure absolute bound (ignoring relative error).
  double pure_abs_bound = 0.0;
  double margin_pure_abs_bound = 0.0;

  // Pure relative bound (ignoring absolute error).
  double pure_rel_bound = 0.0;
  double margin_pure_rel_bound = 0.0;

  std::string ToString() const;
};

// Result of comparing two literals.
struct ComparisonResult {
  bool passed = false;
  std::string element_type;
  std::string shape_str;
  int64_t total_elements = 0;
  int64_t exact_matches = 0;
  int64_t mismatches = 0;
  int64_t nan_mismatches = 0;
  int64_t inf_mismatches = 0;
  double max_abs_error = 0.0;
  double max_rel_error = 0.0;

  std::vector<MismatchDetail> top_mismatches;
  RelErrorHistogram histogram;
  ErrorHeatmap heatmap;
  std::optional<SuggestedErrorSpec> suggested_error_spec;

  std::string SummaryToString() const;
};

// Type trait helpers for floating-point, integer, and complex values.
template <typename T>
struct is_complex : std::false_type {};
template <typename T>
struct is_complex<std::complex<T>> : std::true_type {};
template <typename T>
inline constexpr bool is_complex_v = is_complex<T>::value;

template <typename T, typename Enable = void>
struct ValueTraits {
  static bool IsNan(T val) {
    if constexpr (std::is_floating_point_v<T> || !std::is_integral_v<T>) {
      return std::isnan(static_cast<double>(val));
    }
    return false;
  }
  static bool IsInf(T val) {
    if constexpr (std::is_floating_point_v<T> || !std::is_integral_v<T>) {
      return std::isinf(static_cast<double>(val));
    }
    return false;
  }
  static double AbsDiff(T a, T b) {
    if constexpr (std::is_integral_v<T>) {
      if (a == b) return 0.0;
      uint64_t diff = (a > b)
                          ? static_cast<uint64_t>(a) - static_cast<uint64_t>(b)
                          : static_cast<uint64_t>(b) - static_cast<uint64_t>(a);
      return std::max(1.0, static_cast<double>(diff));
    } else {
      return std::abs(static_cast<double>(a) - static_cast<double>(b));
    }
  }
  static double Magnitude(T val) { return std::abs(static_cast<double>(val)); }
  static double ToDouble(T val) { return Magnitude(val); }
  static std::string Format(T val) {
    if constexpr (std::is_same_v<T, bool>) {
      return val ? "true" : "false";
    } else if constexpr (std::is_integral_v<T>) {
      return absl::StrCat(val);
    } else {
      return absl::StrFormat("%g", static_cast<double>(val));
    }
  }
};

template <typename U>
struct ValueTraits<std::complex<U>> {
  static bool IsNan(const std::complex<U>& val) {
    return std::isnan(val.real()) || std::isnan(val.imag());
  }
  static bool IsInf(const std::complex<U>& val) {
    return std::isinf(val.real()) || std::isinf(val.imag());
  }
  static double AbsDiff(const std::complex<U>& a, const std::complex<U>& b) {
    return std::abs(std::complex<double>(a.real(), a.imag()) -
                    std::complex<double>(b.real(), b.imag()));
  }
  static double Magnitude(const std::complex<U>& val) {
    return std::abs(std::complex<double>(val.real(), val.imag()));
  }
  static double ToDouble(const std::complex<U>& val) { return Magnitude(val); }
  static std::string Format(const std::complex<U>& val) {
    return absl::StrFormat("(%g, %g)", val.real(), val.imag());
  }
};

// Builds a wide grid of thresholds covering 10^-7 to 10^2 with {1, 2, 3, 4, 5}
// steps per decade, ensuring the target tolerance is always included.
inline std::vector<double> BuildThresholds(double target) {
  std::vector<double> thresholds;
  thresholds.reserve(55);

  const double multipliers[] = {1.0, 2.0, 3.0, 4.0, 5.0};
  for (int e = -7; e <= 2; ++e) {
    double base = std::pow(10.0, e);
    for (double m : multipliers) {
      thresholds.push_back(m * base);
    }
  }
  thresholds.push_back(1e3);

  if (target > 0.0 && !absl::c_linear_search(thresholds, target)) {
    thresholds.push_back(target);
  }
  absl::c_sort(thresholds);
  thresholds.erase(std::unique(thresholds.begin(), thresholds.end()),
                   thresholds.end());
  return thresholds;
}

// Subdivided boundaries with {1, 2, 3, 4, 5} steps per decade.
inline std::vector<RelErrorBin> CreateDefaultRelBins() {
  constexpr double kInfinity = std::numeric_limits<double>::infinity();
  std::vector<RelErrorBin> bins;
  bins.reserve(75);

  const double multipliers[] = {1.0, 2.0, 3.0, 4.0, 5.0};

  // Positive boundaries: 0.0, 2e-6, 3e-6, 4e-6, 5e-6, 1e-5, ..., 10.0, +inf
  std::vector<double> pos_bounds;
  pos_bounds.push_back(0.0);
  for (int e = -6; e <= 0; ++e) {
    double base = std::pow(10.0, e);
    for (double m : multipliers) {
      if (e == -6 && m == 1.0) continue;
      pos_bounds.push_back(m * base);
    }
  }
  pos_bounds.push_back(10.0);
  pos_bounds.push_back(kInfinity);

  // Negative bins: symmetric, ordered ascending from -inf to 0
  bins.push_back({-kInfinity, -pos_bounds[pos_bounds.size() - 2], 0, false});
  for (int i = static_cast<int>(pos_bounds.size()) - 2; i >= 1; --i) {
    bins.push_back({-pos_bounds[i], -pos_bounds[i - 1], 0, false});
  }

  // Exact zero bin
  bins.push_back({0.0, 0.0, 0, true});

  // Positive bins: ordered ascending from 0 to +inf
  for (size_t i = 1; i < pos_bounds.size(); ++i) {
    bins.push_back({pos_bounds[i - 1], pos_bounds[i], 0, false});
  }

  return bins;
}

inline int FindRelBin(double rel_err, const std::vector<RelErrorBin>& bins) {
  if (rel_err == 0.0) {
    for (size_t i = 0; i < bins.size(); ++i) {
      if (bins[i].is_exact_zero) return static_cast<int>(i);
    }
  }
  for (size_t i = 0; i < bins.size(); ++i) {
    if (bins[i].is_exact_zero) continue;
    if (rel_err >= bins[i].lower && rel_err < bins[i].upper) {
      return static_cast<int>(i);
    }
  }
  if (rel_err < bins.front().upper) return 0;
  return static_cast<int>(bins.size() - 1);
}

// Computes suggested ErrorSpec (balanced on Pareto frontier, pure abs, pure
// rel) based on the 2D heatmap suffix sums and max recorded errors.
inline std::optional<SuggestedErrorSpec> ComputeSuggestedErrorSpec(
    const ErrorHeatmap& heatmap, double max_abs_error, double max_rel_error,
    int64_t nan_mismatches, int64_t inf_mismatches) {
  if (nan_mismatches > 0 || inf_mismatches > 0) {
    return std::nullopt;
  }
  if (heatmap.abs_thresholds.empty() || heatmap.rel_thresholds.empty() ||
      heatmap.mismatch_counts.empty()) {
    return std::nullopt;
  }

  SuggestedErrorSpec spec;

  // Pure absolute bound: smallest threshold >= max_abs_error
  auto abs_pure_it =
      std::lower_bound(heatmap.abs_thresholds.begin(),
                       heatmap.abs_thresholds.end(), max_abs_error);
  spec.pure_abs_bound = (abs_pure_it != heatmap.abs_thresholds.end())
                            ? *abs_pure_it
                            : max_abs_error;
  auto m_abs_pure_it =
      std::lower_bound(heatmap.abs_thresholds.begin(),
                       heatmap.abs_thresholds.end(), 2.0 * spec.pure_abs_bound);
  spec.margin_pure_abs_bound = (m_abs_pure_it != heatmap.abs_thresholds.end())
                                   ? *m_abs_pure_it
                                   : 2.0 * spec.pure_abs_bound;

  // Pure relative bound: smallest threshold >= max_rel_error
  auto rel_pure_it =
      std::lower_bound(heatmap.rel_thresholds.begin(),
                       heatmap.rel_thresholds.end(), max_rel_error);
  spec.pure_rel_bound = (rel_pure_it != heatmap.rel_thresholds.end())
                            ? *rel_pure_it
                            : max_rel_error;
  auto m_rel_pure_it =
      std::lower_bound(heatmap.rel_thresholds.begin(),
                       heatmap.rel_thresholds.end(), 2.0 * spec.pure_rel_bound);
  spec.margin_pure_rel_bound = (m_rel_pure_it != heatmap.rel_thresholds.end())
                                   ? *m_rel_pure_it
                                   : 2.0 * spec.pure_rel_bound;

  // Pareto frontier for balanced (a, r):
  // For each abs threshold a, find smallest rel threshold r where
  // mismatch_counts[r][a] == 0.
  const int num_abs = heatmap.abs_thresholds.size();
  const int num_rel = heatmap.rel_thresholds.size();

  struct Candidate {
    double a_val;
    double r_val;
  };
  std::vector<Candidate> frontier;
  int prev_r = num_rel;
  for (int a = 0; a < num_abs; ++a) {
    double a_val = heatmap.abs_thresholds[a];
    if (a_val >= spec.pure_abs_bound) break;

    int found_r = -1;
    for (int r = 0; r < num_rel; ++r) {
      if (heatmap.mismatch_counts[r][a] == 0) {
        found_r = r;
        break;
      }
    }
    if (found_r != -1 && found_r < prev_r) {
      double r_val = heatmap.rel_thresholds[found_r];
      if (r_val < spec.pure_rel_bound) {
        frontier.push_back({a_val, r_val});
        prev_r = found_r;
      }
    }
  }

  if (frontier.empty()) {
    spec.abs_bound = spec.pure_abs_bound;
    spec.rel_bound = spec.pure_rel_bound;
    spec.margin_abs_bound = spec.margin_pure_abs_bound;
    spec.margin_rel_bound = spec.margin_pure_rel_bound;
    return spec;
  }

  if (frontier.size() == 1) {
    spec.abs_bound = frontier[0].a_val;
    spec.rel_bound = frontier[0].r_val;
  } else {
    // Find knee in normalized log-space
    double min_u = std::log10(std::max(1e-15, frontier.back().a_val));
    double max_u = std::log10(std::max(1e-15, frontier.front().a_val));
    double min_v = std::log10(std::max(1e-15, frontier.back().r_val));
    double max_v = std::log10(std::max(1e-15, frontier.front().r_val));

    if (min_u > max_u) std::swap(min_u, max_u);
    if (min_v > max_v) std::swap(min_v, max_v);

    double span_u = max_u - min_u;
    double span_v = max_v - min_v;

    double best_dist = std::numeric_limits<double>::infinity();
    int best_idx = 0;

    for (size_t i = 0; i < frontier.size(); ++i) {
      double u = std::log10(std::max(1e-15, frontier[i].a_val));
      double v = std::log10(std::max(1e-15, frontier[i].r_val));
      double norm_u = span_u > 1e-9 ? (u - min_u) / span_u : 0.0;
      double norm_v = span_v > 1e-9 ? (v - min_v) / span_v : 0.0;
      double dist = norm_u * norm_u + norm_v * norm_v;
      if (dist < best_dist) {
        best_dist = dist;
        best_idx = i;
      }
    }

    spec.abs_bound = frontier[best_idx].a_val;
    spec.rel_bound = frontier[best_idx].r_val;
  }

  auto m_abs_it =
      std::lower_bound(heatmap.abs_thresholds.begin(),
                       heatmap.abs_thresholds.end(), 2.0 * spec.abs_bound);
  spec.margin_abs_bound = (m_abs_it != heatmap.abs_thresholds.end())
                              ? *m_abs_it
                              : 2.0 * spec.abs_bound;

  auto m_rel_it =
      std::lower_bound(heatmap.rel_thresholds.begin(),
                       heatmap.rel_thresholds.end(), 2.0 * spec.rel_bound);
  spec.margin_rel_bound = (m_rel_it != heatmap.rel_thresholds.end())
                              ? *m_rel_it
                              : 2.0 * spec.rel_bound;

  return spec;
}

// Element-level comparator and metric accumulator.
template <typename NativeT>
class ElementComparator {
 public:
  ElementComparator(const ComparisonOptions& options, int64_t total_elements)
      : options_(options) {
    result_.total_elements = total_elements;

    ErrorHeatmap& heatmap = result_.heatmap;
    heatmap.total_elements = total_elements;
    heatmap.target_abs = options.abs_error_bound;
    heatmap.target_rel = options.rel_error_bound;
    heatmap.abs_thresholds = BuildThresholds(options.abs_error_bound);
    heatmap.rel_thresholds = BuildThresholds(options.rel_error_bound);

    auto abs_it = absl::c_find(heatmap.abs_thresholds, options.abs_error_bound);
    heatmap.target_abs_idx =
        abs_it != heatmap.abs_thresholds.end()
            ? std::distance(heatmap.abs_thresholds.begin(), abs_it)
            : -1;

    auto rel_it = absl::c_find(heatmap.rel_thresholds, options.rel_error_bound);
    heatmap.target_rel_idx =
        rel_it != heatmap.rel_thresholds.end()
            ? std::distance(heatmap.rel_thresholds.begin(), rel_it)
            : -1;

    const int num_abs_thresh = heatmap.abs_thresholds.size();
    const int num_rel_thresh = heatmap.rel_thresholds.size();
    hist_2d_.assign(num_rel_thresh + 1,
                    std::vector<int64_t>(num_abs_thresh + 1, 0));

    result_.histogram.bins = CreateDefaultRelBins();
  }

  // Compares an individual element pair at linear index `idx`.
  void RecordElement(int64_t idx, NativeT clean_val, NativeT dirty_val) {
    constexpr double kInfinity = std::numeric_limits<double>::infinity();
    double abs_diff = 0.0;
    double rel_diff = 0.0;
    double signed_rel = 0.0;
    bool is_nan = false;
    bool is_inf = false;
    bool has_rel_error = false;

    if (ValueTraits<NativeT>::IsNan(clean_val) ||
        ValueTraits<NativeT>::IsNan(dirty_val)) {
      is_nan = true;
      if (options_.all_nans_are_equivalent &&
          ValueTraits<NativeT>::IsNan(clean_val) &&
          ValueTraits<NativeT>::IsNan(dirty_val)) {
        result_.exact_matches++;
        abs_diff = 0.0;
        rel_diff = 0.0;
        signed_rel = 0.0;
      } else if (options_.relaxed_nans &&
                 ValueTraits<NativeT>::IsNan(clean_val) &&
                 !ValueTraits<NativeT>::IsNan(dirty_val)) {
        abs_diff = 0.0;
        rel_diff = 0.0;
        signed_rel = 0.0;
      } else {
        result_.nan_mismatches++;
        result_.mismatches++;
        abs_diff = kInfinity;
        rel_diff = kInfinity;
        signed_rel = kInfinity;
      }
    } else if (ValueTraits<NativeT>::IsInf(clean_val) ||
               ValueTraits<NativeT>::IsInf(dirty_val)) {
      is_inf = true;
      if (clean_val == dirty_val) {
        result_.exact_matches++;
        abs_diff = 0.0;
        rel_diff = 0.0;
        signed_rel = 0.0;
      } else {
        result_.inf_mismatches++;
        result_.mismatches++;
        abs_diff = kInfinity;
        rel_diff = kInfinity;
        signed_rel = kInfinity;
      }
    } else {
      if (clean_val == dirty_val) {
        result_.exact_matches++;
        abs_diff = 0.0;
        rel_diff = 0.0;
        signed_rel = 0.0;

        double clean_mag = ValueTraits<NativeT>::Magnitude(clean_val);
        if (clean_mag != 0.0) {
          has_rel_error = true;
          // Online Welford update for exact matches (signed_rel = 0.0)
          finite_rel_samples_++;
          double delta = 0.0 - mean_signed_rel_;
          mean_signed_rel_ += delta / finite_rel_samples_;
          double delta2 = 0.0 - mean_signed_rel_;
          m2_signed_rel_ += delta * delta2;

          min_signed_rel_ = std::min(min_signed_rel_, 0.0);
          max_signed_rel_ = std::max(max_signed_rel_, 0.0);
        }
      } else {
        abs_diff = ValueTraits<NativeT>::AbsDiff(dirty_val, clean_val);
        result_.max_abs_error = std::max(result_.max_abs_error, abs_diff);

        double clean_mag = ValueTraits<NativeT>::Magnitude(clean_val);
        if (clean_mag != 0.0) {
          has_rel_error = true;
          if constexpr (is_complex_v<NativeT>) {
            rel_diff = abs_diff / clean_mag;
            signed_rel = rel_diff;
          } else {
            double clean_d = static_cast<double>(clean_val);
            double dirty_d = static_cast<double>(dirty_val);
            signed_rel = (dirty_d - clean_d) / clean_mag;
            rel_diff = abs_diff / clean_mag;
          }
          result_.max_rel_error = std::max(result_.max_rel_error, rel_diff);

          // Online Welford update guarded against subnormal overflow
          if (std::isfinite(signed_rel)) {
            finite_rel_samples_++;
            double delta = signed_rel - mean_signed_rel_;
            mean_signed_rel_ += delta / finite_rel_samples_;
            double delta2 = signed_rel - mean_signed_rel_;
            m2_signed_rel_ += delta * delta2;

            min_signed_rel_ = std::min(min_signed_rel_, signed_rel);
            max_signed_rel_ = std::max(max_signed_rel_, signed_rel);
          }
        }

        // Mismatch check against target bounds:
        // When clean is non-zero, both bounds must be exceeded.
        // When clean is zero, relative error is undefined so only abs bound
        // applies.
        bool is_mismatch =
            (abs_diff > options_.abs_error_bound) &&
            (!has_rel_error || rel_diff > options_.rel_error_bound);

        if (is_mismatch) {
          result_.mismatches++;
          if (result_.top_mismatches.size() <
              options_.max_mismatches_to_record) {
            result_.top_mismatches.push_back(
                {idx, ValueTraits<NativeT>::Format(clean_val),
                 ValueTraits<NativeT>::Format(dirty_val), abs_diff,
                 has_rel_error ? rel_diff : 0.0});
          }
        }
      }
    }

    // 1D histogram binning (only for finite relative errors or exact zeros)
    if (!is_nan && !is_inf && (clean_val == dirty_val || has_rel_error)) {
      int bin_idx = FindRelBin(signed_rel, result_.histogram.bins);
      result_.histogram.bins[bin_idx].count++;
      result_.histogram.total_samples++;
    }

    // 2D heatmap binning
    int a_bin =
        std::lower_bound(result_.heatmap.abs_thresholds.begin(),
                         result_.heatmap.abs_thresholds.end(), abs_diff) -
        result_.heatmap.abs_thresholds.begin();
    int r_bin =
        has_rel_error
            ? (std::lower_bound(result_.heatmap.rel_thresholds.begin(),
                                result_.heatmap.rel_thresholds.end(),
                                rel_diff) -
               result_.heatmap.rel_thresholds.begin())
            : (is_nan || is_inf || abs_diff > 0.0
                   ? static_cast<int>(result_.heatmap.rel_thresholds.size())
                   : 0);
    hist_2d_[r_bin][a_bin]++;
  }

  // Finalizes summary statistics (mean, stddev, median, 2D suffix sums)
  // and returns the final ComparisonResult.
  ComparisonResult Finalize() {
    RelErrorHistogram& histogram = result_.histogram;
    if (finite_rel_samples_ > 0) {
      histogram.min_rel_error = min_signed_rel_;
      histogram.max_rel_error = max_signed_rel_;
      histogram.mean_rel_error = mean_signed_rel_;
      histogram.std_dev_rel_error =
          finite_rel_samples_ > 1 ? std::sqrt(std::max(0.0, m2_signed_rel_) /
                                              (finite_rel_samples_ - 1))
                                  : 0.0;

      const int64_t target = (histogram.total_samples + 1) / 2;
      int64_t cumulative = 0;
      for (size_t i = 0; i < histogram.bins.size(); ++i) {
        cumulative += histogram.bins[i].count;
        if (cumulative >= target) {
          histogram.median_bin_index = static_cast<int>(i);
          break;
        }
      }
    }

    // 2D Suffix sum to compute mismatch counts
    ErrorHeatmap& heatmap = result_.heatmap;
    const int num_abs_thresh = heatmap.abs_thresholds.size();
    const int num_rel_thresh = heatmap.rel_thresholds.size();
    heatmap.mismatch_counts.assign(num_rel_thresh,
                                   std::vector<int64_t>(num_abs_thresh, 0));

    std::vector<std::vector<int64_t>> suffix(
        num_rel_thresh + 2, std::vector<int64_t>(num_abs_thresh + 2, 0));

    for (int r = num_rel_thresh; r >= 0; --r) {
      for (int a = num_abs_thresh; a >= 0; --a) {
        suffix[r][a] = hist_2d_[r][a] + suffix[r + 1][a] + suffix[r][a + 1] -
                       suffix[r + 1][a + 1];
      }
    }

    for (int r = 0; r < num_rel_thresh; ++r) {
      for (int a = 0; a < num_abs_thresh; ++a) {
        heatmap.mismatch_counts[r][a] = suffix[r + 1][a + 1];
      }
    }

    result_.passed = (result_.mismatches == 0 && result_.nan_mismatches == 0 &&
                      result_.inf_mismatches == 0);

    result_.suggested_error_spec = ComputeSuggestedErrorSpec(
        result_.heatmap, result_.max_abs_error, result_.max_rel_error,
        result_.nan_mismatches, result_.inf_mismatches);

    return result_;
  }

  const ComparisonResult& result() const { return result_; }

 private:
  ComparisonOptions options_;
  ComparisonResult result_;
  std::vector<std::vector<int64_t>> hist_2d_;

  double min_signed_rel_ = std::numeric_limits<double>::infinity();
  double max_signed_rel_ = -std::numeric_limits<double>::infinity();
  double mean_signed_rel_ = 0.0;
  double m2_signed_rel_ = 0.0;
  int64_t finite_rel_samples_ = 0;
};

// Compares two LiteralSlice objects and computes statistics, 1D histogram, and
// 2D heatmap in a single pass.
absl::StatusOr<ComparisonResult> CompareLiterals(
    const LiteralSlice& clean, const LiteralSlice& dirty,
    const ComparisonOptions& options);

// Compares two LiteralProto objects.
absl::StatusOr<ComparisonResult> CompareLiteralProtos(
    const LiteralProto& clean_proto, const LiteralProto& dirty_proto,
    const ComparisonOptions& options);

// Reads two LiteralProto binary files from disk and compares them.
absl::StatusOr<ComparisonResult> CompareLiteralFiles(
    absl::string_view clean_file, absl::string_view dirty_file,
    const ComparisonOptions& options);

}  // namespace xla::compare_literals

#endif  // XLA_TOOLS_COMPARE_LITERALS_COMPARE_LITERALS_H_
