#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(HAVE_CUDA_RUNTIME)
extern "C" bool gpu_histogram_count(const std::uint32_t *values,
                                     std::size_t count,
                                     std::size_t bucketCount,
                                     std::uint64_t *outCounts);
extern "C" bool gpu_runtime_available();
#endif

namespace gpu {

inline bool is_available() {
#if defined(HAVE_CUDA_RUNTIME)
  static const bool available = gpu_runtime_available();
  return available;
#else
  return false;
#endif
}

inline bool histogram(const std::vector<std::uint32_t> &values,
                      std::size_t bucketCount,
                      std::vector<std::uint64_t> &out) {
  out.assign(bucketCount, 0);
  if (bucketCount == 0)
    return true;
  if (values.empty())
    return true;
#if defined(HAVE_CUDA_RUNTIME)
  return gpu_histogram_count(values.data(), values.size(), bucketCount,
                             out.data());
#else
  (void)values;
  return false;
#endif
}

} // namespace gpu
