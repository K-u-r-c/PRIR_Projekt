#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace {

__global__ void histogram_kernel(const std::uint32_t *values, std::size_t count,
                                 std::size_t bucketCount,
                                 unsigned long long *out) {
  std::size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;
  for (std::size_t i = gid; i < count; i += stride) {
    std::uint32_t bucket = values[i];
    if (bucket < bucketCount) {
      atomicAdd(&out[bucket], 1ULL);
    }
  }
}

inline bool check(cudaError_t err) { return err == cudaSuccess; }

} // namespace

extern "C" bool gpu_histogram_count(const std::uint32_t *values, std::size_t count,
                                     std::size_t bucketCount,
                                     std::uint64_t *outCounts) {
  if (bucketCount == 0)
    return true;
  if (count == 0) {
    for (std::size_t i = 0; i < bucketCount; ++i)
      outCounts[i] = 0;
    return true;
  }

  std::uint32_t *dValues = nullptr;
  unsigned long long *dOut = nullptr;

  std::size_t valuesBytes = count * sizeof(std::uint32_t);
  std::size_t outBytes = bucketCount * sizeof(unsigned long long);

  if (!check(cudaMalloc(reinterpret_cast<void **>(&dValues), valuesBytes)))
    goto fail;
  if (!check(cudaMalloc(reinterpret_cast<void **>(&dOut), outBytes)))
    goto fail;
  if (!check(cudaMemcpy(dValues, values, valuesBytes, cudaMemcpyHostToDevice)))
    goto fail;
  if (!check(cudaMemset(dOut, 0, outBytes)))
    goto fail;

  constexpr int blockSize = 256;
  int blocks = static_cast<int>((count + blockSize - 1) / blockSize);
  blocks = std::max(1, std::min(blocks, 4096));

  histogram_kernel<<<blocks, blockSize>>>(dValues, count, bucketCount, dOut);
  if (!check(cudaGetLastError()))
    goto fail;
  if (!check(cudaDeviceSynchronize()))
    goto fail;

  if (!check(cudaMemcpy(outCounts, dOut, outBytes, cudaMemcpyDeviceToHost)))
    goto fail;

  cudaFree(dValues);
  cudaFree(dOut);
  return true;

fail:
  if (dValues)
    cudaFree(dValues);
  if (dOut)
    cudaFree(dOut);
  return false;
}
