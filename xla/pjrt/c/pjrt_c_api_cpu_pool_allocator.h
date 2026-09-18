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

#ifndef XLA_PJRT_C_PJRT_C_API_CPU_POOL_ALLOCATOR_H_
#define XLA_PJRT_C_PJRT_C_API_CPU_POOL_ALLOCATOR_H_

#include <cstddef>
#include <functional>
#include <memory>

#include "absl/status/statusor.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_memory.h"

namespace pjrt {
namespace cpu_plugin {

// A block pool for the buffers XLA:CPU allocates on every execution, offered
// to the CPU PjRt client through `CpuClientOptions::allocator` and turned on by
// the `pooling_allocator` create option. It is off by default: a caller that
// does not ask for it gets stock behaviour.
//
// WHAT THE HOOK REACHES
//
// `CpuClientOptions::allocator` is consulted by `CpuDeviceMemory::Allocate` and
// `AllocateInto`, which is every output buffer and every temporary buffer of
// every execution, plus donation copies, the tuple index table and host-buffer
// allocations for `PjRtBuffer`. It does not reach the thousands of small
// allocations XLA's thunk runtime makes per call -- AsyncValue objects, thread
// pool task closures, Eigen scratch -- because those go straight to
// `::operator new`. The claim this patch can support is therefore narrow and
// worth stating exactly: it removes every `posix_memalign` the plugin makes per
// call, and nothing else. Under which conditions it removes them is a second
// question, and the drift section below answers it.
//
// WHY THE WRAPPER OBJECT IS POOLED TOO
//
// Per buffer, per execution, the stock path makes three heap allocations:
//
//   1. `CpuDeviceMemory::CreateDelayedMemory` constructs an AsyncValue with
//      `::operator new`. Not reachable from here.
//   2. The allocator returns a wrapper object -- `AlignedMemory` by default --
//      allocated with a plain `operator new`.
//   3. `tsl::port::AlignedMalloc`, which is `posix_memalign`, for the block.
//
// Installing an allocator adds a fourth: `GetPjRtCpuClient` wraps whatever this
// returns in its own `std::make_unique<CustomMemory>`
// (xla/pjrt/cpu/cpu_client.cc), which cannot be avoided without editing that
// file, and that file moved a thousand lines between two adjacent XLA revisions
// and is expensive to rebase.
//
// So a pool that serves only (3) trades one `posix_memalign` for one
// `operator new` and the allocation count barely moves. The wrapper this file
// returns therefore carries a class-level `operator new`/`operator delete`
// served from a free list of its own. The accounting is then three heap
// allocations per buffer before and two after, and the one that goes is the
// expensive one.
//
// WHY THREAD-LOCAL FREE LISTS AND NO LOCK
//
// This plugin exists to serve a `SCHED_FIFO` control loop on a `PREEMPT_RT`
// kernel. A `std::mutex` there has no priority inheritance, so a lock holder
// preempted by a higher-priority thread blocks the control loop for an
// unbounded time -- the failure this whole wrapper is built to avoid. Each
// thread therefore keeps its own array of intrusive LIFO free lists, one per
// size class, in `thread_local` storage. In steady state an acquire is a load
// and two stores: no atomic, no syscall, no lock, and no path another thread
// can hold.
//
// A block is released onto the *calling* thread's list, which need not be the
// thread that acquired it. Blocks within a size class are interchangeable, so
// that is safe; what it costs is drift, and the caps below are what bound it.
//
// WHEN THE POOL IS WORTH INSTALLING
//
// Drift is not a detail, because a block only removes an allocation when the
// release lands on the thread that acquired it. With inline execution
// (`asynchronous=false`, and also a computation XLA judges cheap) the calling
// thread allocates this execution's output and temporary buffers and later
// frees them, acquire and release meet, and the steady state takes nothing
// from the system. With asynchronous dispatch they do not meet:
// `buffer_alloc.Allocate` runs on an async-work-runner thread
// (xla/pjrt/cpu/cpu_client.cc) while the output buffers are dropped by
// whichever thread releases the `PjRtBuffer`, which is usually the caller.
// Blocks then travel one way -- the allocating thread finds its list empty on
// every acquire and pays `posix_memalign` and a first-touch walk anyway, while
// the consuming thread retains up to its cap and never spends it. Temporaries,
// acquired and dropped inside the same task, still pool.
//
// So this option belongs with `asynchronous=false`. `PJRT_Client_Create` warns
// rather than refusing when it is not, because `asynchronous` is advisory:
// XLA runs a cheap computation inline whatever the option says.
//
// WHAT IT COSTS
//
// Retained memory scales with the number of threads that ever allocated, up to
// `kMaxRetainedBytesPerThread` each, and is returned to the system only when a
// thread exits or a cap is hit. Pages are touched once on growth, so a pooled
// block is resident from then on: the pool trades resident memory for a shorter
// tail, deliberately.
class BlockPool {
 public:
  // Size classes are powers of two from 64 bytes to 256 MiB inclusive. 64 is
  // the alignment XLA:CPU wants and the smallest block that can still hold the
  // intrusive next pointer; 256 MiB is large enough for the temporary buffer of
  // any program this plugin is meant for, and a request past it bypasses the
  // pool in both directions so that one giant transient cannot be retained
  // forever.
  static constexpr size_t kMinBlockBytes = 64;
  static constexpr size_t kMaxBlockBytes = size_t{256} << 20;
  static constexpr int kNumSizeClasses = 23;  // 64 B << 22 == 256 MiB

  // Every pooled block is allocated with this alignment, and a request that
  // needs more bypasses the pool rather than being served from a list whose
  // blocks cannot satisfy it. Callers pass `xla::cpu::MinAlign()`, which is
  // Eigen's maximum alignment and is 16, 32 or 64 depending on the ISA, so the
  // bypass is a correctness valve and not a path that is expected to run.
  static constexpr size_t kBlockAlignment = 64;

  // Caps on what one thread retains. The per-class cap keeps a burst of
  // same-sized buffers from pinning memory that the steady state will never ask
  // for again; 64 is well above the number of buffers one execution of a large
  // program holds at once. The byte cap is what actually bounds the pool, since
  // the per-class caps together allow far more. Past either cap a release frees
  // the block for real instead of growing.
  //
  // The byte cap is deliberately larger than one block of the largest class,
  // and that is the whole reason it is not 256 MiB. A budget equal to the
  // largest block is exhausted by one member of the working set: a program with
  // a 200 MiB temporary retains it, and then every other buffer of the same
  // execution is freed on release and allocated again on the next call,
  // forever. Twice the largest block is the smallest value that cannot do that.
  static constexpr int kMaxBlocksPerClass = 64;
  static constexpr size_t kMaxRetainedBytesPerThread = 2 * kMaxBlockBytes;

  // The wrapper objects the allocator hands back are pooled on a list of their
  // own, and this bounds it. A burst of concurrent executions holds one wrapper
  // per live buffer, so it is generous; it exists so that a thread which
  // allocated once and will never allocate again does not hold the list
  // forever.
  static constexpr int kMaxObjectsPerThread = 256;

  // Returns a block of at least `size_bytes` aligned to at least `alignment`,
  // or nullptr if the system is out of memory. A pooled block is rounded up to
  // its size class, so the caller may not assume the block is exactly
  // `size_bytes` long -- only that it is at least that long.
  static void* Acquire(size_t size_bytes, size_t alignment);

  // Returns a block from `Acquire`. `size_bytes` and `alignment` must be the
  // values that `Acquire` was given, because they are what decides which list
  // the block belongs on and whether it was pooled at all. A null `base` is
  // ignored.
  static void Release(void* base, size_t size_bytes, size_t alignment);

  // Whether a request of this shape is served from a free list rather than
  // passed straight to `posix_memalign` and `free`. A pure function of its
  // arguments, so `Acquire` and `Release` always agree about one block.
  static bool IsPooled(size_t size_bytes, size_t alignment);

  // The block size a pooled request of `size_bytes` is rounded up to. Defined
  // for `size_bytes <= kMaxBlockBytes`, which is what `IsPooled` reports.
  static size_t ClassBytes(size_t size_bytes);

  // Introspection and reset for the unit test, over the calling thread's free
  // lists -- which, since there is no shared state, is the whole pool as that
  // thread can see it. The reset frees what this thread retained, blocks and
  // wrapper objects both, and zeroes the two counters below; a thread does the
  // freeing for itself when it exits.
  static size_t RetainedBytesForTesting();
  static int RetainedBlocksForTesting(size_t size_bytes);
  static int RetainedObjectsForTesting();
  static void ReleaseRetainedForTesting();

  // How many blocks and wrapper objects this thread has taken from the system
  // allocator since the last reset. These are what a test must assert against,
  // because pointer identity is not evidence of reuse: a pool that frees a
  // block and immediately allocates another of the same size gets the same
  // address back from glibc and looks identical from the outside. Driving
  // these to zero in the steady state is the claim this file makes.
  static size_t FreshBlocksForTesting();
  static size_t FreshObjectsForTesting();
};

// Returns an allocator for `CpuClientOptions::allocator` that serves XLA:CPU's
// per-execution buffers from `BlockPool`. The returned function is thread-safe
// and captures no state, so copying it is free and it outlives nothing.
//
// On allocation failure it returns `absl::ResourceExhaustedError` naming the
// byte count, which is what the default allocator does, rather than crashing.
// It never throws, on either half of the allocation:
// `CustomAllocator::Allocate` calls it from continuations that run on a thread
// pool, and an exception unwinding out of one of those is caught nowhere and
// takes the process with it.
std::function<absl::StatusOr<std::unique_ptr<xla::CpuMemory>>(size_t size_bytes,
                                                              size_t alignment)>
MakePoolingAllocator();

}  // namespace cpu_plugin
}  // namespace pjrt

#endif  // XLA_PJRT_C_PJRT_C_API_CPU_POOL_ALLOCATOR_H_
