#include <algorithm>
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

  auto cleanup = [&]() {
    if (dValues)
      cudaFree(dValues);
    if (dOut)
      cudaFree(dOut);
  };

  bool success = false;
  do {
    if (!check(cudaMalloc(reinterpret_cast<void **>(&dValues), valuesBytes)))
      break;
    if (!check(cudaMalloc(reinterpret_cast<void **>(&dOut), outBytes)))
      break;
    if (!check(cudaMemcpy(dValues, values, valuesBytes, cudaMemcpyHostToDevice)))
      break;
    if (!check(cudaMemset(dOut, 0, outBytes)))
      break;

    constexpr int blockSize = 256;
    int blocks = static_cast<int>((count + blockSize - 1) / blockSize);
    blocks = std::max(1, std::min(blocks, 4096));

    void *kernelArgs[] = {&dValues, &count, &bucketCount, &dOut};
    if (!check(cudaLaunchKernel(reinterpret_cast<const void *>(histogram_kernel),
                                dim3(blocks), dim3(blockSize), kernelArgs, 0,
                                nullptr)))
      break;
    if (!check(cudaDeviceSynchronize()))
      break;

    if (!check(cudaMemcpy(outCounts, dOut, outBytes, cudaMemcpyDeviceToHost)))
      break;

    success = true;
  } while (false);

  cleanup();
  return success;
}

extern "C" bool gpu_runtime_available() {
  int devices = 0;
  auto status = cudaGetDeviceCount(&devices);
  if (status != cudaSuccess)
    return false;
  return devices > 0;
}
