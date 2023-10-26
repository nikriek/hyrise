#include "volatile_region.hpp"
#include <sys/mman.h>
#include <unistd.h>
#include <memory>
#include "utils/assert.hpp"

#if HYRISE_NUMA_SUPPORT
#include <numa.h>
#include <numaif.h>
#endif

namespace hyrise {

VolatileRegion::VolatileRegion(const PageSizeType size_type, std::byte* region_start, std::byte* region_end)
    : _size_type(size_type),
      _region_start(region_start),
      _region_end(region_end),
      _frames((_region_end - _region_start) / bytes_for_size_type(size_type)) {
  DebugAssert(_region_start < _region_end, "Region is too small");
  DebugAssert(_frames.size() > 0, "Not enough space for frames");
  if constexpr (ENABLE_MPROTECT) {
    if (mprotect(_region_start, _region_end - _region_start, PROT_NONE) != 0) {
      const auto error = errno;
      Fail("Failed to mprotect: " + strerror(error));
    }
  }
}

void VolatileRegion::move_page_to_numa_node(const PageID page_id, const NodeID target_memory_node) {
  DebugAssert(page_id.size_type() == _size_type, "Page does not belong to this region.");
#if HYRISE_NUMA_SUPPORT
  DebugAssert(target_memory_node != INVALID_NODE_ID, "Numa node has not been set.");
  static thread_local std::vector<void*> pages_to_move{bytes_for_size_type(_size_type) / OS_PAGE_SIZE};
  static thread_local std::vector<int> nodes{static_cast<int>(bytes_for_size_type(_size_type) / OS_PAGE_SIZE)};
  static thread_local std::vector<int> status{static_cast<int>(bytes_for_size_type(_size_type) / OS_PAGE_SIZE)};

  for (auto i = 0u; i < pages_to_move.size(); ++i) {
    pages_to_move[i] = get_page(page_id) + i * OS_PAGE_SIZE;
    nodes[i] = target_memory_node;
  }
  if (move_pages(0, pages_to_move.size(), pages_to_move.data(), nodes.data(), status.data(), MPOL_MF_MOVE) < 0) {
    const auto error = errno;
    Fail("Move pages failed: " + strerror(error));
  }
  _numa_page_movement_count.fetch_add(1, std::memory_order_relaxed);
  _frames[page_id.index].set_node_id(target_memory_node);
#endif
}

void VolatileRegion::mbind_to_numa_node(const PageID page_id, const NodeID target_memory_node) {
  DebugAssert(page_id.size_type() == _size_type, "Page does not belong to this region.");

#if HYRISE_NUMA_SUPPORT
  DebugAssert(target_memory_node != INVALID_NODE_ID, "Numa node has not been set.");

  const auto num_bytes = bytes_for_size_type(_size_type);
  auto nodes = numa_allocate_nodemask();
  numa_bitmask_setbit(nodes, target_memory_node);
  if (mbind(get_page(page_id), num_bytes, MPOL_BIND, nodes ? nodes->maskp : NULL, nodes ? nodes->size + 1 : 0,
            MPOL_MF_MOVE | MPOL_MF_STRICT) != 0) {
    const auto error = errno;
    numa_bitmask_free(nodes);
    Fail("Mbind failed: " + strerror(error) +
         " . Either no space is left or vm map count is exhausted. Try: \"sudo sysctl vm.max_map_count=X\"");
  }
  numa_bitmask_free(nodes);
  _numa_page_movement_count.fetch_add(1, std::memory_order_relaxed);
  _frames[page_id.index].set_node_id(target_memory_node);
#endif
}

void VolatileRegion::memcopy_page_to_numa_node(const PageID page_id, const NodeID target_memory_node) {
  // The intermedite buffer only exists during the lifetime of the current thread.
  constexpr auto num_numa_nodes = 4;  // TODO
  thread_local auto intermediate_buffer =
      std::make_unique_for_overwrite<uint8_t[]>(num_numa_nodes * bytes_for_size_type(MAX_PAGE_SIZE_TYPE));
  const auto page_ptr = get_page(page_id);
  const auto target_buffer =
      intermediate_buffer.get() + static_cast<uint64_t>(target_memory_node) * bytes_for_size_type(MAX_PAGE_SIZE_TYPE);
  std::memcpy(target_buffer, page_ptr, page_id.byte_count());
  free(page_id);
  mbind_to_numa_node(page_id, target_memory_node);
  std::memcpy(page_ptr, target_buffer, page_id.byte_count());
}

void VolatileRegion::free(const PageID page_id) {
  DebugAssert(page_id.size_type() == _size_type, "Page does not belong to this region.");
  // Use MADV_FREE_REUSABLE on OS X and MADV_DONTNEED on Linux
  // https://bugs.chromium.org/p/chromium/issues/detail?id=823915
#ifdef __APPLE__
  const int flags = MADV_FREE_REUSABLE;
#elif __linux__
  const int flags = MADV_DONTNEED;
#endif
  auto ptr = get_page(page_id);
  _unprotect_page(page_id);
  if (madvise(ptr, page_id.byte_count(), flags) < 0) {
    const auto error = errno;
    Fail("Failed to call madvise(MADV_DONTNEED / MADV_FREE_REUSABLE): " + strerror(error));
  }
  _protect_page(page_id);
  _madvice_free_call_count.fetch_add(1, std::memory_order_relaxed);
}

std::byte* VolatileRegion::get_page(const PageID page_id) {
  DebugAssert(page_id.size_type() == _size_type, "Page does not belong to this region.");
  const auto num_bytes = bytes_for_size_type(_size_type);
  const auto data = _region_start + page_id.index() * num_bytes;
  return data;
}

Frame* VolatileRegion::get_frame(const PageID page_id) {
  DebugAssert(page_id.size_type() == _size_type, "Page does not belong to this region.");
  return &_frames[page_id.index()];
}

size_t VolatileRegion::memory_consumption() const {
  return sizeof(*this) + sizeof(decltype(_frames)::value_type) * _frames.capacity();
}

size_t VolatileRegion::size() const {
  return _frames.size();
}

PageSizeType VolatileRegion::size_type() const {
  return _size_type;
}

void VolatileRegion::reuse(const PageID page_id) {
// On OS X, we can use MADV_FREE_REUSE to update the memory accounting.
// Source: https://bugs.chromium.org/p/chromium/issues/detail?id=823915
#ifdef __APPLE__
  const auto ptr = get_page(page_id);
  if (madvise(ptr, page_id.byte_count(), MADV_FREE_REUSE) < 0) {
    const auto error = errno;
    Fail("Failed to call madvise(MADV_FREE_REUSE): " + strerror(error));
  }
#endif
}

void VolatileRegion::_protect_page(const PageID page_id) {
  if constexpr (ENABLE_MPROTECT) {
    DebugAssert(page_id.size_type() == _size_type, "Page does not belong to this region.");
    auto data = get_page(page_id);
    if (mprotect(data, page_id.byte_count(), PROT_NONE) != 0) {
      const auto error = errno;
      Fail("Failed to mprotect: " + strerror(error));
    }
  }
}

void VolatileRegion::_unprotect_page(const PageID page_id) {
  if constexpr (ENABLE_MPROTECT) {
    DebugAssert(page_id.size_type() == _size_type, "Page does not belong to this region.");
    auto data = get_page(page_id);
    if (mprotect(data, page_id.byte_count(), PROT_READ | PROT_WRITE) != 0) {
      const auto error = errno;
      Fail("Failed to mprotect: " + strerror(error));
    }
  }
}

inline std::size_t get_os_page_size() {
  return std::size_t(sysconf(_SC_PAGESIZE));
}

constexpr size_t DEFAULT_RESERVED_VIRTUAL_MEMORY_PER_REGION =
    (VolatileRegion::DEFAULT_RESERVED_VIRTUAL_MEMORY / PAGE_SIZE_TYPES_COUNT) /
    bytes_for_size_type(MAX_PAGE_SIZE_TYPE) * bytes_for_size_type(MAX_PAGE_SIZE_TYPE);

std::byte* VolatileRegion::create_mapped_region() {
  Assert(bytes_for_size_type(MIN_PAGE_SIZE_TYPE) >= get_os_page_size(),
         "Smallest page size does not fit into an OS page: " + std::to_string(get_os_page_size()));
#ifdef __APPLE__
  const int flags = MAP_PRIVATE | MAP_ANON | MAP_NORESERVE;
#elif __linux__
  const int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE;
#endif
  const auto mapped_memory = static_cast<std::byte*>(
      mmap(NULL, VolatileRegion::DEFAULT_RESERVED_VIRTUAL_MEMORY, PROT_READ | PROT_WRITE, flags, -1, 0));

  if (mapped_memory == MAP_FAILED) {
    const auto error = errno;
    Fail("Failed to map volatile pool region: " + strerror(error));
  }

  return mapped_memory;
}

std::array<std::shared_ptr<VolatileRegion>, PAGE_SIZE_TYPES_COUNT> VolatileRegion::create_volatile_regions(
    std::byte* mapped_region) {
  DebugAssert(mapped_region != nullptr, "Region not properly mapped");
  auto array = std::array<std::shared_ptr<VolatileRegion>, PAGE_SIZE_TYPES_COUNT>{};

  // Ensure that every region has the same amount of virtual memory
  // Round to the next multiple of the largest page size
  for (auto i = size_t{0}; i < PAGE_SIZE_TYPES_COUNT; i++) {
    array[i] = std::make_shared<VolatileRegion>(magic_enum::enum_value<PageSizeType>(i),
                                                mapped_region + DEFAULT_RESERVED_VIRTUAL_MEMORY_PER_REGION * i,
                                                mapped_region + DEFAULT_RESERVED_VIRTUAL_MEMORY_PER_REGION * (i + 1));
  }

  return array;
}

void VolatileRegion::unmap_region(std::byte* region) {
  if (munmap(region, VolatileRegion::DEFAULT_RESERVED_VIRTUAL_MEMORY) < 0) {
    const auto error = errno;
    Fail("Failed to unmap volatile pool region: " + strerror(error));
  }
}

uint64_t VolatileRegion::madvice_free_call_count() const {
  return _madvice_free_call_count.load(std::memory_order_relaxed);
}

uint64_t VolatileRegion::numa_page_movement_count() const {
  return _numa_page_movement_count.load(std::memory_order_relaxed);
}

}  // namespace hyrise
