/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/pjrt/c/pjrt_c_api_cpu_internal.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_ffi_extension.h"
#include "xla/pjrt/c/pjrt_c_api_ffi_internal.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_layouts_extension.h"
#include "xla/pjrt/c/pjrt_c_api_memory_descriptions_extension.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_extension.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_internal.h"
#include "xla/pjrt/c/pjrt_c_api_shardings_extension.h"
#include "xla/pjrt/c/pjrt_c_api_status_utils.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/c/pjrt_c_api_xla_transform_extension.h"
#include "xla/pjrt/c/pjrt_c_api_xla_transform_internal.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"

namespace pjrt {
namespace cpu_plugin {

const PJRT_Api* GetCpuPjrtApi();

PJRT_Error* PJRT_Client_Create(PJRT_Client_Create_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Client_Create_Args", PJRT_Client_Create_Args_STRUCT_SIZE,
      args->struct_size));

  xla::CpuClientOptions options;
  options.cpu_device_count = 4;

  if (args->create_options != nullptr) {
    absl::flat_hash_map<std::string, xla::PjRtValueType> create_options =
        ConvertFromPjRtNamedValueList(args->create_options, args->num_options);
    const auto kExpectedOptionNameAndTypes =
        absl::flat_hash_map<std::string, PJRT_NamedValue_Type>({
            {"cpu_device_count", PJRT_NamedValue_Type::PJRT_NamedValue_kInt64},
            {"asynchronous", PJRT_NamedValue_Type::PJRT_NamedValue_kBool},
            {"max_inflight_computations",
             PJRT_NamedValue_Type::PJRT_NamedValue_kInt64},
        });
    PJRT_RETURN_IF_ERROR(
        ValidateCreateOptions(create_options, kExpectedOptionNameAndTypes));

    if (auto it = create_options.find("cpu_device_count");
        it != create_options.end()) {
      int64_t device_count_option = std::get<int64_t>(it->second);
      options.cpu_device_count = device_count_option;
      LOG(INFO) << "cpu_device_count set via create_options: "
                << device_count_option;
    }
    if (auto it = create_options.find("asynchronous");
        it != create_options.end()) {
      bool asynchronous_option = std::get<bool>(it->second);
      options.asynchronous = asynchronous_option;
      LOG(INFO) << "asynchronous set via create_options: "
                << asynchronous_option;
    }
    if (auto it = create_options.find("max_inflight_computations");
        it != create_options.end()) {
      int64_t max_inflight_option = std::get<int64_t>(it->second);
      options.max_inflight_computations_per_device =
          static_cast<int>(max_inflight_option);
      LOG(INFO) << "max_inflight_computations set via create_options: "
                << max_inflight_option;
    }
  }

  PJRT_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> client,
                        xla::GetPjRtCpuClient(std::move(options)));
  args->client = pjrt::CreateWrapperClient(GetCpuPjrtApi(), std::move(client));
  return nullptr;
}

PJRT_Error* PJRT_ExecuteContext_Create(PJRT_ExecuteContext_Create_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_ExecuteContext_Create_Args",
      PJRT_ExecuteContext_Create_Args_STRUCT_SIZE, args->struct_size));
  auto execute_context = std::make_unique<xla::ExecuteContext>();
  args->context = pjrt::CreateWrapperExecuteContext(std::move(execute_context));
  return nullptr;
}

PJRT_Error* PJRT_CpuDeviceTopology_Create(
    PJRT_TopologyDescription_Create_Args* args) {
  return StatusToPjRtError(
      absl::UnimplementedError("Topology not supported for CPU compilation."));
}

// Advertises what this build of the CPU plugin accepts, so a caller can tell a
// patched plugin from a stock one before it creates a client.
//
// This matters because the create-option surface is not discoverable
// otherwise. Since ValidateCreateOptions() above rejects unknown option names
// outright, a caller cannot simply pass an option and see whether it sticks:
// passing one this plugin does not know fails client creation. The markers
// below let a caller ask first and only send what will be accepted.
//
// `supports_synchronous_execution` reports that `asynchronous=false` is
// available, which makes XLA run computations inline on the calling thread
// instead of handing them to its dispatch pool. That is the single biggest
// structural cut to tail latency for a real-time caller, and the PJRT C API
// exposes no other route to it: PJRT_ExecuteOptions has no execution-mode
// field.
//
// `cjfc_plugin_patch_level` counts the fork's behavioural patches, so a
// caller can tell which build it has. Level 3 adds the two that need no
// option: with an intra-op pool of one thread the whole computation runs on
// the calling thread (upstream handed it to the pool's worker and waited),
// and the per-call allocations for constants and kernel arguments are gone.
// A caller can verify the first from the scheduler's own accounting rather
// than trust the marker; the marker exists so it knows to look.
PJRT_Error* PJRT_Plugin_Attributes_Cpu(PJRT_Plugin_Attributes_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Plugin_Attributes_Args", PJRT_Plugin_Attributes_Args_STRUCT_SIZE,
      args->struct_size));

  static const std::vector<PJRT_NamedValue>* const attributes = [] {
    auto* values =
        new std::vector<PJRT_NamedValue>(pjrt::GetXlaPluginCAttributes());
    auto add_marker = [values](const char* name, int64_t value) {
      // `name` is always a string literal here, so the pointer stays valid for
      // the process lifetime, which is what the C API requires of attributes.
      PJRT_NamedValue nv;
      nv.struct_size = PJRT_NamedValue_STRUCT_SIZE;
      nv.extension_start = nullptr;
      nv.name = name;
      nv.name_size = std::strlen(name);
      nv.type = PJRT_NamedValue_Type::PJRT_NamedValue_kInt64;
      nv.int64_value = value;
      nv.value_size = 1;
      values->push_back(nv);
    };
    add_marker("supports_synchronous_execution", 1);
    add_marker("supports_max_inflight_computations", 1);
    add_marker("cjfc_plugin_patch_level", 3);
    return values;
  }();

  args->num_attributes = attributes->size();
  args->attributes = attributes->data();
  return nullptr;
}

const PJRT_Api* GetCpuPjrtApi() {
  static PJRT_Layouts_Extension layouts_extension =
      pjrt::CreateLayoutsExtension(nullptr);

  static PJRT_MemoryDescriptions_Extension memory_descriptions_extension =
      pjrt::CreateMemoryDescriptionsExtension(&layouts_extension.base);

  static PJRT_FFI_Extension ffi_extension =
      pjrt::CreateFfiExtension(&memory_descriptions_extension.base);

  static PJRT_PhaseCompile_Extension phase_compile_extension =
      pjrt::CreatePhaseCompileExtension(&ffi_extension.base,
                                        /*get_compiler=*/nullptr,
                                        /*destroy_compiler=*/nullptr);

  static PJRT_Shardings_Extension shardings_extension =
      pjrt::CreateShardingsExtension(&phase_compile_extension.base);

  static PJRT_Xla_Transform_Extension xla_transform_extension =
      pjrt::CreateXlaTransformExtension(&shardings_extension.base);

  static const PJRT_Api pjrt_api = pjrt::CreatePjrtApi(
      pjrt::cpu_plugin::PJRT_Client_Create,
      pjrt::cpu_plugin::PJRT_ExecuteContext_Create,
      pjrt::cpu_plugin::PJRT_CpuDeviceTopology_Create,
      pjrt::PJRT_Plugin_Initialize_NoOp, &xla_transform_extension.base,
      pjrt::cpu_plugin::PJRT_Plugin_Attributes_Cpu);

  return &pjrt_api;
}

}  // namespace cpu_plugin
}  // namespace pjrt
