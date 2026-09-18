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

#include "xla/pjrt/c/pjrt_c_api_cpu_pool_allocator.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <new>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_memory.h"

namespace pjrt {
namespace cpu_plugin {
namespace {

//===----------------------------------------------------------------------===//
// The pool itself. Standard library only, so it can be built and run under a
// sanitizer with a plain compiler, without bazel and without XLA.
//===----------------------------------------------------------------------===//

// Stride for the first-touch write below. A page smaller than this exists on no
// platform this plugin is built for, and where pages are larger -- transparent
// huge pages included -- touching more often than necessary is harmless.
constexpr size_t kPageBytes = 4096;

// Storage for one `PoolMemory`, which is four pointers wide today. The chunk is
// larger than that so a field can be added without the free list changing
// shape, and the static_assert beside the class is what catches it if one day
// it is not enough.
constexpr size_t kObjectBytes = 64;

// One block of the largest class must not be able to exhaust a thread's whole
// budget; the header says what goes wrong when it can.
static_assert(BlockPool::kMaxRetainedBytesPerThread > BlockPool::kMaxBlockBytes,
              "A thread's retained-bytes budget must exceed one block of the "
              "largest size class.");

// One thread's free lists. Blocks and wrapper objects are kept apart because
// they are different sizes and the wrapper's size is fixed.
struct FreeLists {
  void* blocks[BlockPool::kNumSizeClasses] = {};
  int block_count[BlockPool::kNumSizeClasses] = {};
  size_t retained_bytes = 0;
  void* objects = nullptr;
  int object_count = 0;

  ~FreeLists();
};

// Trivially destructible and constant-initialized, so it has no destructor to
// run and stays readable for the whole life of the thread -- including after
// `tls_lists` below has been destroyed. Reading a destroyed `thread_local` is
// undefined behaviour, and it surfaces as a crash during thread exit or at
// process teardown, which is exactly the kind of bug that is expensive to find
// inside a dlopen-ed plugin. So every entry point tests this flag before it
// touches the lists, and falls back to plain `free` once it is set.
thread_local bool tls_lists_destroyed = false;
thread_local FreeLists tls_lists;

// What this thread has taken from the system allocator, for the unit test.
// They are here rather than in `FreeLists` for the same reason the flag is:
// they are read on paths that run after the lists are gone. The cost is one
// increment on the path that just called `posix_memalign` or `malloc`.
thread_local size_t tls_fresh_blocks = 0;
thread_local size_t tls_fresh_objects = 0;

// The intrusive next pointer lives in the first bytes of a free block. A block
// is at least 64 bytes and at least 64-byte aligned, so it always has room and
// the access is aligned; `memcpy` is used rather than a cast so that the
// compiler cannot read a type into storage that has none. It compiles to one
// load or store.
void* LoadNext(void* block) {
  void* next = nullptr;
  std::memcpy(&next, block, sizeof(next));
  return next;
}

void StoreNext(void* block, void* next) {
  std::memcpy(block, &next, sizeof(next));
}

// Rounds up to a power of two at least `BlockPool::kBlockAlignment`, so that a
// caller who passes something `posix_memalign` would reject gets memory rather
// than a misleading out-of-memory error.
size_t RoundedAlignment(size_t alignment) {
  size_t rounded = BlockPool::kBlockAlignment;
  while (rounded < alignment) rounded <<= 1;
  return rounded;
}

// Index of the smallest size class that holds `size_bytes`. A bounded loop of
// at most `kNumSizeClasses` shifts; it is on the acquire path, but it is
// nanoseconds against the `posix_memalign` it exists to avoid, and keeping it
// free of compiler builtins is what lets this file be tested on its own.
int SizeClass(size_t size_bytes) {
  int index = 0;
  size_t class_bytes = BlockPool::kMinBlockBytes;
  while (class_bytes < size_bytes) {
    class_bytes <<= 1;
    ++index;
  }
  return index;
}

size_t ClassBytesFor(int size_class) {
  return BlockPool::kMinBlockBytes << size_class;
}

// Allocates a block the pool does not have. `touch` is set for blocks that can
// re-enter a free list: writing one byte per page makes the first-touch faults
// land here, in whatever call grows the pool, rather than in a later timed one.
// That is the whole reason the pool helps a latency tail, so it is not skipped
// for pooled blocks. It is skipped for blocks that bypass the pool, because
// those are used once and freed, so touching them only moves the same faults a
// few microseconds earlier inside the same call.
void* AllocateFresh(size_t bytes, size_t alignment, bool touch) {
  void* base = nullptr;
  if (::posix_memalign(&base, alignment, bytes) != 0) return nullptr;
  ++tls_fresh_blocks;
  if (touch) {
    volatile unsigned char* bytes_to_touch =
        static_cast<volatile unsigned char*>(base);
    for (size_t offset = 0; offset < bytes; offset += kPageBytes) {
      bytes_to_touch[offset] = 0;
    }
  }
  return base;
}

FreeLists::~FreeLists() {
  for (int index = 0; index < BlockPool::kNumSizeClasses; ++index) {
    void* block = blocks[index];
    while (block != nullptr) {
      void* next = LoadNext(block);
      std::free(block);
      block = next;
    }
  }
  void* object = objects;
  while (object != nullptr) {
    void* next = LoadNext(object);
    std::free(object);
    object = next;
  }
  tls_lists_destroyed = true;
}

void* AcquireObject() {
  if (!tls_lists_destroyed && tls_lists.objects != nullptr) {
    void* object = tls_lists.objects;
    tls_lists.objects = LoadNext(object);
    --tls_lists.object_count;
    return object;
  }
  void* object = std::malloc(kObjectBytes);
  if (object != nullptr) ++tls_fresh_objects;
  return object;
}

void ReleaseObject(void* object) {
  if (object == nullptr) return;
  if (tls_lists_destroyed ||
      tls_lists.object_count >= BlockPool::kMaxObjectsPerThread) {
    std::free(object);
    return;
  }
  StoreNext(object, tls_lists.objects);
  tls_lists.objects = object;
  ++tls_lists.object_count;
}

//===----------------------------------------------------------------------===//
// The adapter: the `xla::CpuMemory` XLA asks for, and the factory that makes
// them.
//===----------------------------------------------------------------------===//

class PoolMemory final : public xla::CpuMemory {
 public:
  PoolMemory(void* base, size_t size_bytes, size_t alignment)
      : base_(base), size_bytes_(size_bytes), alignment_(alignment) {}

  ~PoolMemory() override { BlockPool::Release(base_, size_bytes_, alignment_); }

  void* base() const override { return base_; }

  // The requested size, not the size class the block was rounded up to. XLA
  // reports this as the buffer's size and copies that many bytes, so rounding
  // it up here would hand a caller bytes it never asked for.
  size_t size_bytes() const override { return size_bytes_; }

  // Without these the pool would trade one `posix_memalign` per buffer for one
  // `operator new` per buffer and the allocation count would barely move; see
  // the header. The requested size is unused because the class is `final`, so
  // `sizeof(PoolMemory)` is the only size that can ever arrive.
  //
  // Only the nothrow form exists, and the throwing one is deleted so that it
  // cannot come back by accident. Whoever asks for a wrapper is already holding
  // a block that nothing owns yet, so a failure has to come back as a value
  // they can act on: an exception would leak that block -- out of the pool for
  // good, when it came from a free list -- and would leave the allocator
  // through `CustomAllocator::Allocate`, which is called from thread-pool
  // continuations where nothing catches it.
  static void* operator new(size_t) = delete;
  static void* operator new(size_t, const std::nothrow_t&) noexcept {
    return AcquireObject();
  }
  static void operator delete(void* object) { ReleaseObject(object); }
  static void operator delete(void* object, const std::nothrow_t&) noexcept {
    ReleaseObject(object);
  }

 private:
  void* base_;
  size_t size_bytes_;
  size_t alignment_;
};

// `PoolMemory` is `final` and its `operator new` always hands back a chunk of
// exactly `kObjectBytes`, so this is the check that keeps the two in step.
static_assert(sizeof(PoolMemory) <= kObjectBytes,
              "PoolMemory outgrew the object free list's chunk size.");
static_assert(alignof(PoolMemory) <= alignof(std::max_align_t),
              "PoolMemory needs more alignment than malloc guarantees.");

}  // namespace

bool BlockPool::IsPooled(size_t size_bytes, size_t alignment) {
  return size_bytes <= kMaxBlockBytes && alignment <= kBlockAlignment;
}

size_t BlockPool::ClassBytes(size_t size_bytes) {
  return ClassBytesFor(SizeClass(size_bytes));
}

void* BlockPool::Acquire(size_t size_bytes, size_t alignment) {
  const size_t rounded_alignment = RoundedAlignment(alignment);
  if (!IsPooled(size_bytes, alignment)) {
    // `posix_memalign` may return either nullptr or a unique pointer for a
    // zero-byte request, and nullptr is how this reports failure, so ask for a
    // byte instead.
    return AllocateFresh(size_bytes == 0 ? 1 : size_bytes, rounded_alignment,
                         /*touch=*/false);
  }

  // The block size depends only on the request, never on the state of this
  // thread's lists. A block that `IsPooled` accepts is therefore always exactly
  // its class size, wherever it was allocated and whichever thread later
  // releases it onto a list.
  const int size_class = SizeClass(size_bytes);
  if (!tls_lists_destroyed) {
    if (void* block = tls_lists.blocks[size_class]; block != nullptr) {
      tls_lists.blocks[size_class] = LoadNext(block);
      --tls_lists.block_count[size_class];
      tls_lists.retained_bytes -= ClassBytesFor(size_class);
      return block;
    }
  }
  return AllocateFresh(ClassBytesFor(size_class), rounded_alignment,
                       /*touch=*/true);
}

void BlockPool::Release(void* base, size_t size_bytes, size_t alignment) {
  if (base == nullptr) return;
  if (tls_lists_destroyed || !IsPooled(size_bytes, alignment)) {
    std::free(base);
    return;
  }

  const int size_class = SizeClass(size_bytes);
  const size_t class_bytes = ClassBytesFor(size_class);
  if (tls_lists.block_count[size_class] >= kMaxBlocksPerClass ||
      tls_lists.retained_bytes + class_bytes > kMaxRetainedBytesPerThread) {
    std::free(base);
    return;
  }

  StoreNext(base, tls_lists.blocks[size_class]);
  tls_lists.blocks[size_class] = base;
  ++tls_lists.block_count[size_class];
  tls_lists.retained_bytes += class_bytes;
}

size_t BlockPool::RetainedBytesForTesting() {
  return tls_lists_destroyed ? 0 : tls_lists.retained_bytes;
}

int BlockPool::RetainedBlocksForTesting(size_t size_bytes) {
  if (tls_lists_destroyed || size_bytes > kMaxBlockBytes) return 0;
  return tls_lists.block_count[SizeClass(size_bytes)];
}

int BlockPool::RetainedObjectsForTesting() {
  return tls_lists_destroyed ? 0 : tls_lists.object_count;
}

size_t BlockPool::FreshBlocksForTesting() { return tls_fresh_blocks; }

size_t BlockPool::FreshObjectsForTesting() { return tls_fresh_objects; }

void BlockPool::ReleaseRetainedForTesting() {
  tls_fresh_blocks = 0;
  tls_fresh_objects = 0;
  if (tls_lists_destroyed) return;
  for (int index = 0; index < kNumSizeClasses; ++index) {
    void* block = tls_lists.blocks[index];
    while (block != nullptr) {
      void* next = LoadNext(block);
      std::free(block);
      block = next;
    }
    tls_lists.blocks[index] = nullptr;
    tls_lists.block_count[index] = 0;
  }
  tls_lists.retained_bytes = 0;

  // The wrapper list as well, or a case that measures it would start with
  // whatever the case before it left behind.
  void* object = tls_lists.objects;
  while (object != nullptr) {
    void* next = LoadNext(object);
    std::free(object);
    object = next;
  }
  tls_lists.objects = nullptr;
  tls_lists.object_count = 0;
}

std::function<absl::StatusOr<std::unique_ptr<xla::CpuMemory>>(size_t size_bytes,
                                                              size_t alignment)>
MakePoolingAllocator() {
  return [](size_t size_bytes, size_t alignment)
             -> absl::StatusOr<std::unique_ptr<xla::CpuMemory>> {
    void* base = BlockPool::Acquire(size_bytes, alignment);
    if (base == nullptr) {
      return absl::ResourceExhaustedError(
          "Out of memory allocating " + std::to_string(size_bytes) + " bytes.");
    }
    // Nothing owns the block until the wrapper does, so a wrapper that
    // cannot be allocated has to hand it back rather than drop it.
    auto* memory = new (std::nothrow) PoolMemory(base, size_bytes, alignment);
    if (memory == nullptr) {
      BlockPool::Release(base, size_bytes, alignment);
      return absl::ResourceExhaustedError(
          "Out of memory allocating the wrapper for a " +
          std::to_string(size_bytes) + " byte buffer.");
    }
    return std::unique_ptr<xla::CpuMemory>(memory);
  };
}

}  // namespace cpu_plugin
}  // namespace pjrt
