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

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/base/config.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_memory.h"

namespace pjrt {
namespace cpu_plugin {
namespace {

bool IsAlignedTo(const void* base, size_t alignment) {
  return reinterpret_cast<uintptr_t>(base) % alignment == 0;
}

// The pool's state is per-thread and gtest runs every case on the same thread,
// so each case starts and ends with this thread's lists empty.
class BlockPoolTest : public ::testing::Test {
 protected:
  void SetUp() override { BlockPool::ReleaseRetainedForTesting(); }
  void TearDown() override { BlockPool::ReleaseRetainedForTesting(); }
};

TEST_F(BlockPoolTest, ReleasedBlockComesBackOnTheNextAcquire) {
  void* first = BlockPool::Acquire(128, 64);
  ASSERT_NE(first, nullptr);
  BlockPool::Release(first, 128, 64);
  EXPECT_EQ(BlockPool::RetainedBlocksForTesting(128), 1);

  // A request in the same class, not just the same size.
  void* second = BlockPool::Acquire(100, 64);
  EXPECT_EQ(second, first);
  EXPECT_EQ(BlockPool::RetainedBlocksForTesting(128), 0);
  BlockPool::Release(second, 100, 64);
}

TEST_F(BlockPoolTest, TheSteadyStateTakesNothingFromTheSystem) {
  // The pointer identity above is not on its own evidence of reuse: a pool that
  // freed the block and allocated another of the same size would be handed the
  // same address back by glibc and would pass. This is the assertion that a
  // pool which reuses nothing cannot pass -- it is the claim the patch makes.
  constexpr int kBlocks = 4;
  void* blocks[kBlocks];
  for (int i = 0; i < kBlocks; ++i) {
    blocks[i] = BlockPool::Acquire(4096, 64);
    ASSERT_NE(blocks[i], nullptr);
  }
  const size_t grown = BlockPool::FreshBlocksForTesting();
  EXPECT_EQ(grown, size_t{kBlocks});

  for (int i = 0; i < kBlocks; ++i) BlockPool::Release(blocks[i], 4096, 64);
  for (int i = 0; i < kBlocks; ++i) {
    blocks[i] = BlockPool::Acquire(4096, 64);
    ASSERT_NE(blocks[i], nullptr);
  }
  EXPECT_EQ(BlockPool::FreshBlocksForTesting(), grown);
  for (int i = 0; i < kBlocks; ++i) BlockPool::Release(blocks[i], 4096, 64);
}

TEST_F(BlockPoolTest, HonoursTheRequestedAlignment) {
  for (size_t alignment : {size_t{16}, size_t{64}, size_t{4096}}) {
    for (size_t size : {size_t{1}, size_t{64}, size_t{4096}, size_t{100000}}) {
      void* base = BlockPool::Acquire(size, alignment);
      ASSERT_NE(base, nullptr) << size << "/" << alignment;
      EXPECT_TRUE(IsAlignedTo(base, alignment)) << size << "/" << alignment;
      // The whole request must be writable, not just the class it rounded to.
      std::memset(base, 0xab, size);
      BlockPool::Release(base, size, alignment);
    }
  }
}

TEST_F(BlockPoolTest, AlignmentAboveTheBlockAlignmentBypassesThePool) {
  // Pooled blocks are all 64-byte aligned, so a stricter request cannot be
  // served from a list. XLA never asks for more than MinAlign(), which is at
  // most 64, so this is a valve rather than a path.
  EXPECT_TRUE(BlockPool::IsPooled(4096, 64));
  EXPECT_FALSE(BlockPool::IsPooled(4096, 4096));

  void* base = BlockPool::Acquire(4096, 4096);
  ASSERT_NE(base, nullptr);
  EXPECT_TRUE(IsAlignedTo(base, 4096));
  BlockPool::Release(base, 4096, 4096);
  EXPECT_EQ(BlockPool::RetainedBytesForTesting(), 0u);
}

TEST_F(BlockPoolTest, AdjacentSizeClassesDoNotShareBlocks) {
  EXPECT_EQ(BlockPool::ClassBytes(0), BlockPool::kMinBlockBytes);
  EXPECT_EQ(BlockPool::ClassBytes(64), 64u);
  EXPECT_EQ(BlockPool::ClassBytes(65), 128u);
  EXPECT_EQ(BlockPool::ClassBytes(BlockPool::kMaxBlockBytes),
            BlockPool::kMaxBlockBytes);

  void* small = BlockPool::Acquire(64, 64);
  ASSERT_NE(small, nullptr);
  BlockPool::Release(small, 64, 64);

  void* large = BlockPool::Acquire(65, 64);
  ASSERT_NE(large, nullptr);
  EXPECT_NE(large, small);
  EXPECT_EQ(BlockPool::RetainedBlocksForTesting(64), 1);
  EXPECT_EQ(BlockPool::RetainedBlocksForTesting(65), 0);
  BlockPool::Release(large, 65, 64);
}

TEST_F(BlockPoolTest, PerClassCapFreesRatherThanRetains) {
  const int over_the_cap = BlockPool::kMaxBlocksPerClass + 8;
  std::vector<void*> blocks;
  blocks.reserve(over_the_cap);
  for (int i = 0; i < over_the_cap; ++i) {
    blocks.push_back(BlockPool::Acquire(256, 64));
    ASSERT_NE(blocks.back(), nullptr);
  }
  for (void* base : blocks) BlockPool::Release(base, 256, 64);

  EXPECT_EQ(BlockPool::RetainedBlocksForTesting(256),
            BlockPool::kMaxBlocksPerClass);
  EXPECT_EQ(BlockPool::RetainedBytesForTesting(),
            size_t{BlockPool::kMaxBlocksPerClass} * 256);
}

TEST_F(BlockPoolTest, PerThreadByteCapFreesRatherThanRetains) {
  // A budget that one block of the largest class can exhaust is no budget: the
  // program that allocates such a block keeps it and pools nothing else, ever.
  EXPECT_GT(BlockPool::kMaxRetainedBytesPerThread, BlockPool::kMaxBlockBytes);

  // Blocks of a size that divides the cap: the ones that fit reach it exactly
  // and one more must be freed instead of retained. Filling a cap of this size
  // costs a cap's worth of touched memory, so this runs on a thread of its own
  // and hands it all back when the thread exits.
  std::thread worker([] {
    const size_t block_bytes = size_t{64} << 20;
    const int fits =
        static_cast<int>(BlockPool::kMaxRetainedBytesPerThread / block_bytes);
    std::vector<void*> blocks;
    blocks.reserve(fits + 1);
    for (int i = 0; i < fits + 1; ++i) {
      blocks.push_back(BlockPool::Acquire(block_bytes, 64));
      ASSERT_NE(blocks.back(), nullptr);
    }
    for (void* base : blocks) BlockPool::Release(base, block_bytes, 64);

    EXPECT_EQ(BlockPool::RetainedBytesForTesting(),
              BlockPool::kMaxRetainedBytesPerThread);
    EXPECT_EQ(BlockPool::RetainedBlocksForTesting(block_bytes), fits);
  });
  worker.join();
}

TEST_F(BlockPoolTest, OversizeRequestsBypassThePoolInBothDirections) {
  constexpr size_t oversize = BlockPool::kMaxBlockBytes + 1;
  EXPECT_TRUE(BlockPool::IsPooled(BlockPool::kMaxBlockBytes, 64));
  EXPECT_FALSE(BlockPool::IsPooled(oversize, 64));

  std::thread worker([] {
    void* base = BlockPool::Acquire(oversize, 64);
    ASSERT_NE(base, nullptr);
    EXPECT_TRUE(IsAlignedTo(base, 64));
    static_cast<char*>(base)[oversize - 1] = 1;
    BlockPool::Release(base, oversize, 64);
    // Nothing retained, so one giant transient cannot pin the pool.
    EXPECT_EQ(BlockPool::RetainedBytesForTesting(), 0u);
  });
  worker.join();
}

TEST_F(BlockPoolTest, BlockReleasedOnAnotherThreadIsReusableThere) {
  void* base = BlockPool::Acquire(512, 64);
  ASSERT_NE(base, nullptr);

  std::thread worker([base] {
    BlockPool::Release(base, 512, 64);
    EXPECT_EQ(BlockPool::RetainedBlocksForTesting(512), 1);
    void* again = BlockPool::Acquire(512, 64);
    EXPECT_EQ(again, base);
    BlockPool::Release(again, 512, 64);
  });
  worker.join();

  // It went onto the releasing thread's list, not this one's.
  EXPECT_EQ(BlockPool::RetainedBlocksForTesting(512), 0);
}

TEST_F(BlockPoolTest, ConcurrentAcquireAndRelease) {
  constexpr int kThreads = 8;
  constexpr int kRounds = 2000;
  const size_t sizes[] = {48, 64, 200, 1024, 5000, 70000, 300000};
  constexpr int kNumSizes = 7;

  std::atomic<int> failures{0};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([t, &sizes, &failures] {
      std::vector<void*> held;
      std::vector<size_t> held_sizes;
      for (int round = 0; round < kRounds; ++round) {
        const size_t size = sizes[(round + t) % kNumSizes];
        void* base = BlockPool::Acquire(size, 64);
        if (base == nullptr || !IsAlignedTo(base, 64)) {
          ++failures;
          continue;
        }
        // If two threads ever held the same block, one of them would see its
        // own fill pattern overwritten.
        std::memset(base, t + 1, size);
        const unsigned char* bytes = static_cast<unsigned char*>(base);
        for (size_t i = 0; i < size; ++i) {
          if (bytes[i] != static_cast<unsigned char>(t + 1)) {
            ++failures;
            break;
          }
        }
        held.push_back(base);
        held_sizes.push_back(size);
        if (held.size() >= 8) {
          for (size_t i = 0; i < held.size(); ++i) {
            BlockPool::Release(held[i], held_sizes[i], 64);
          }
          held.clear();
          held_sizes.clear();
        }
      }
      for (size_t i = 0; i < held.size(); ++i) {
        BlockPool::Release(held[i], held_sizes[i], 64);
      }
      BlockPool::ReleaseRetainedForTesting();
    });
  }
  for (std::thread& thread : threads) thread.join();
  EXPECT_EQ(failures.load(), 0);
}

TEST_F(BlockPoolTest, AllocatorReportsTheRequestedSize) {
  auto allocate = MakePoolingAllocator();

  absl::StatusOr<std::unique_ptr<xla::CpuMemory>> memory = allocate(100, 64);
  ASSERT_TRUE(memory.ok()) << memory.status();
  // The request, not the 128-byte class it was served from: XLA reports this
  // as the buffer's size.
  EXPECT_EQ((*memory)->size_bytes(), 100u);
  EXPECT_NE((*memory)->base(), nullptr);
  EXPECT_TRUE(IsAlignedTo((*memory)->base(), 64));
  std::memset((*memory)->base(), 7, 100);
}

TEST_F(BlockPoolTest, AllocatorReturnsDistinctBuffersAndRecyclesThem) {
  auto allocate = MakePoolingAllocator();
  {
    absl::StatusOr<std::unique_ptr<xla::CpuMemory>> first = allocate(100, 64);
    absl::StatusOr<std::unique_ptr<xla::CpuMemory>> second = allocate(100, 64);
    ASSERT_TRUE(first.ok());
    ASSERT_TRUE(second.ok());
    EXPECT_NE((*first)->base(), (*second)->base());
  }
  // Both destroyed, so both blocks are back on this thread's list.
  EXPECT_EQ(BlockPool::RetainedBlocksForTesting(100), 2);

  absl::StatusOr<std::unique_ptr<xla::CpuMemory>> third = allocate(128, 64);
  ASSERT_TRUE(third.ok());
  EXPECT_EQ(BlockPool::RetainedBlocksForTesting(100), 1);
}

TEST_F(BlockPoolTest, AllocatorServesAZeroByteRequest) {
  auto allocate = MakePoolingAllocator();
  absl::StatusOr<std::unique_ptr<xla::CpuMemory>> memory = allocate(0, 64);
  ASSERT_TRUE(memory.ok()) << memory.status();
  EXPECT_EQ((*memory)->size_bytes(), 0u);
  EXPECT_NE((*memory)->base(), nullptr);
}

TEST_F(BlockPoolTest, AllocatedMemoryCanBeDestroyedOnAnotherThread) {
  auto allocate = MakePoolingAllocator();
  absl::StatusOr<std::unique_ptr<xla::CpuMemory>> memory = allocate(4096, 64);
  ASSERT_TRUE(memory.ok());
  std::memset((*memory)->base(), 3, 4096);

  std::thread worker([&memory] { memory->reset(); });
  worker.join();
  EXPECT_EQ((*memory).get(), nullptr);
}

TEST_F(BlockPoolTest, TheWrapperObjectIsPooledTooAndNotJustTheBlock) {
  // This is the case that separates a patch that helps from one that does not:
  // with a plain `operator new` behind the wrapper, the pool would trade one
  // `posix_memalign` per buffer for one `operator new` per buffer, and every
  // other case in this file would still pass.
  auto allocate = MakePoolingAllocator();
  {
    absl::StatusOr<std::unique_ptr<xla::CpuMemory>> memory = allocate(100, 64);
    ASSERT_TRUE(memory.ok()) << memory.status();
  }
  EXPECT_EQ(BlockPool::RetainedObjectsForTesting(), 1);
  const size_t grown = BlockPool::FreshObjectsForTesting();
  EXPECT_EQ(grown, 1u);

  {
    absl::StatusOr<std::unique_ptr<xla::CpuMemory>> memory = allocate(100, 64);
    ASSERT_TRUE(memory.ok()) << memory.status();
    EXPECT_EQ(BlockPool::RetainedObjectsForTesting(), 0);
  }
  // The second wrapper came off the list rather than from the system.
  EXPECT_EQ(BlockPool::FreshObjectsForTesting(), grown);
  EXPECT_EQ(BlockPool::RetainedObjectsForTesting(), 1);
}

TEST_F(BlockPoolTest, TheWrapperListStopsAtItsCap) {
  auto allocate = MakePoolingAllocator();
  const int over_the_cap = BlockPool::kMaxObjectsPerThread + 8;
  std::vector<std::unique_ptr<xla::CpuMemory>> held;
  held.reserve(over_the_cap);
  for (int i = 0; i < over_the_cap; ++i) {
    absl::StatusOr<std::unique_ptr<xla::CpuMemory>> memory = allocate(64, 64);
    ASSERT_TRUE(memory.ok()) << memory.status();
    held.push_back(std::move(*memory));
  }
  held.clear();
  EXPECT_EQ(BlockPool::RetainedObjectsForTesting(),
            BlockPool::kMaxObjectsPerThread);
}

TEST_F(BlockPoolTest, AFailedAllocationIsReportedRatherThanHandedOver) {
#if defined(ABSL_HAVE_ADDRESS_SANITIZER) || \
    defined(ABSL_HAVE_THREAD_SANITIZER) || defined(ABSL_HAVE_MEMORY_SANITIZER)
  // A sanitizer aborts on an allocation this size before `posix_memalign` can
  // decline it, so these configurations cannot reach the branch. Every other
  // one can, and the plain build is where this case earns its place.
  GTEST_SKIP() << "a sanitizer aborts rather than failing an impossible "
                  "allocation";
#else
  // Without the error branch the caller gets an ok() status wrapping a buffer
  // whose base is null, and XLA writes an output or a temporary into it.
  auto allocate = MakePoolingAllocator();
  absl::StatusOr<std::unique_ptr<xla::CpuMemory>> memory =
      allocate(SIZE_MAX, 64);
  ASSERT_FALSE(memory.ok());
  EXPECT_TRUE(absl::IsResourceExhausted(memory.status())) << memory.status();
#endif
}

// A `thread_local` is destroyed in the reverse of the order it was constructed,
// so one touched before the pool's own lists outlives them. Releasing a block
// from such a destructor is the path `tls_lists_destroyed` guards: without the
// guard the block is filed onto storage that no longer exists, which is
// undefined behaviour that surfaces as a leak under the sanitizer and as a
// crash during thread exit inside a dlopen-ed plugin.
struct LateReleaser {
  ~LateReleaser() {
    BlockPool::Release(base, size_bytes, 64);
    // The acquire side runs after teardown too, and must still return memory.
    void* after = BlockPool::Acquire(size_bytes, 64);
    BlockPool::Release(after, size_bytes, 64);
  }

  void* base = nullptr;
  size_t size_bytes = 0;
};

TEST_F(BlockPoolTest, ABlockReleasedAfterThreadTeardownIsFreedNotFiled) {
  std::thread worker([] {
    // Touched before anything reaches the pool, so it is destroyed after the
    // pool's lists rather than before them.
    static thread_local LateReleaser late;
    late.size_bytes = 512;
    late.base = BlockPool::Acquire(late.size_bytes, 64);
    ASSERT_NE(late.base, nullptr);
  });
  worker.join();
}

#if defined(__linux__)
TEST_F(BlockPoolTest, APooledBlockIsResidentBeforeItIsHandedOut) {
  // The first-touch write is why the pool shortens a tail at all: it moves the
  // page faults into the call that grows the pool and out of the timed ones.
  // Nothing else in this file can tell a pool that touches from one that does
  // not, so this asks the kernel.
  const size_t block_bytes = size_t{8} << 20;
  void* base = BlockPool::Acquire(block_bytes, 64);
  ASSERT_NE(base, nullptr);

  const long page_bytes = ::sysconf(_SC_PAGESIZE);  // NOLINT(runtime/int)
  ASSERT_GT(page_bytes, 0);
  const uintptr_t page = static_cast<uintptr_t>(page_bytes);
  // mincore() rejects a start that is not page aligned, so ask about the pages
  // the block covers whole and leave its two partial ends out of it.
  const uintptr_t start = reinterpret_cast<uintptr_t>(base);
  const uintptr_t first = (start + page - 1) & ~(page - 1);
  const uintptr_t last = (start + block_bytes) & ~(page - 1);
  ASSERT_GT(last, first);

  std::vector<unsigned char> resident((last - first) / page, 0);
  ASSERT_EQ(::mincore(reinterpret_cast<void*>(first),
                      static_cast<size_t>(last - first), resident.data()),
            0)
      << std::strerror(errno);
  size_t absent = 0;
  for (unsigned char state : resident) absent += (state & 1) ? 0 : 1;
  EXPECT_EQ(absent, 0u) << absent << " of " << resident.size()
                        << " pages of a pooled block were not resident";

  BlockPool::Release(base, block_bytes, 64);
}
#endif  // defined(__linux__)

}  // namespace
}  // namespace cpu_plugin
}  // namespace pjrt
