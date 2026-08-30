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

#include <iostream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/statusor.h"
#include "xla/tools/compare_literals/compare_literals.h"
#include "tsl/platform/init_main.h"

ABSL_FLAG(std::string, clean, "",
          "Path to the clean (golden/reference) LiteralProto file.");
ABSL_FLAG(std::string, dirty, "",
          "Path to the dirty (candidate/test) LiteralProto file.");
ABSL_FLAG(double, abs_error_bound, 1e-3, "Absolute error tolerance bound.");
ABSL_FLAG(double, rel_error_bound, 1e-3, "Relative error tolerance bound.");
ABSL_FLAG(bool, show_histogram, false, "Display 1D relative error histogram.");
ABSL_FLAG(bool, show_heatmap, false, "Display 2D error heatmap.");
ABSL_FLAG(bool, suggest_error_spec, false,
          "Explicitly display suggested ErrorSpec even if comparison passes.");
ABSL_FLAG(int, max_bar_width, 40, "Maximum character width of histogram bars.");
ABSL_FLAG(bool, color, true, "Use ANSI colors in terminal output.");

namespace xla::compare_literals {
namespace {

constexpr int kExitPass = 0;
constexpr int kExitMismatch = 1;
constexpr int kExitError = 2;

int RealMain(int argc, char** argv) {
  tsl::port::InitMain(argv[0], &argc, &argv);
  std::vector<char*> positional_args = absl::ParseCommandLine(argc, argv);

  std::string clean = absl::GetFlag(FLAGS_clean);
  std::string dirty = absl::GetFlag(FLAGS_dirty);

  if (!clean.empty() && !dirty.empty()) {
    if (positional_args.size() > 1) {
      std::cerr << "Error: Do not mix --clean/--dirty flags with positional "
                   "path arguments.\n";
      return kExitError;
    }
  } else if (clean.empty() && dirty.empty()) {
    if (positional_args.size() == 3) {
      clean = positional_args[1];
      dirty = positional_args[2];
    } else {
      std::cerr << "Error: Provide either both --clean and --dirty flags, "
                   "or exactly two positional file paths.\n";
      std::cerr << "Usage:\n  " << argv[0]
                << " <clean_path> <dirty_path> [flags]\n  " << argv[0]
                << " --clean=<clean_path> --dirty=<dirty_path> [flags]\n";
      return kExitError;
    }
  } else {
    std::cerr << "Error: Both --clean and --dirty flags must be specified "
                 "together.\n";
    return kExitError;
  }

  ComparisonOptions options;
  options.abs_error_bound = absl::GetFlag(FLAGS_abs_error_bound);
  options.rel_error_bound = absl::GetFlag(FLAGS_rel_error_bound);

  absl::StatusOr<ComparisonResult> result_or =
      CompareLiteralFiles(clean, dirty, options);

  if (!result_or.ok()) {
    std::cerr << "Comparison error: " << result_or.status().message() << "\n";
    return kExitError;
  }

  const ComparisonResult& result = *result_or;

  std::cout << "Comparing:\n";
  std::cout << "  Clean: " << clean << "\n";
  std::cout << "  Dirty: " << dirty << "\n";
  std::cout << "  Bounds: abs = " << options.abs_error_bound
            << ", rel = " << options.rel_error_bound << "\n\n";

  std::cout << result.SummaryToString() << "\n";

  if (result.passed && absl::GetFlag(FLAGS_suggest_error_spec) &&
      result.suggested_error_spec.has_value()) {
    std::cout << result.suggested_error_spec->ToString() << "\n\n";
  }

  if (result.total_elements > 0) {
    if (absl::GetFlag(FLAGS_show_histogram)) {
      std::cout << result.histogram.ToString(absl::GetFlag(FLAGS_max_bar_width))
                << "\n";
    }

    if (absl::GetFlag(FLAGS_show_heatmap)) {
      std::cout << result.heatmap.ToString(absl::GetFlag(FLAGS_color)) << "\n";
    }
  }

  return result.passed ? kExitPass : kExitMismatch;
}

}  // namespace
}  // namespace xla::compare_literals

int main(int argc, char** argv) {
  return xla::compare_literals::RealMain(argc, argv);
}
