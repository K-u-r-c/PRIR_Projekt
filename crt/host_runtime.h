#ifndef __CODEX_CUDA_SIZED_TYPES
#define __CODEX_CUDA_SIZED_TYPES
#define __need_size_t
#include <stddef.h>
#undef __need_size_t
#endif

#include "/usr/local/cuda/include/crt/host_runtime.h"

#ifdef __cudaLaunch
#undef __cudaLaunch

#define __codex_cuda_launch_pick(_1, _2, NAME, ...) NAME
#define __cudaLaunch(...)                                                     \
  __codex_cuda_launch_pick(__VA_ARGS__, __codex_cuda_launch2,                 \
                           __codex_cuda_launch1)(__VA_ARGS__)

#define __codex_cuda_launch1(fun) __codex_cuda_launch_impl(fun, false)
#define __codex_cuda_launch2(fun, isTileKernel)                               \
  __codex_cuda_launch_impl(fun, isTileKernel)

#if defined(__NV_LEGACY_LAUNCH)
#define __codex_cuda_launch_impl(fun, isTileKernel)                           \
  {                                                                           \
    volatile static char *__f __NV_ATTR_UNUSED_FOR_LAUNCH;                    \
    __f = fun;                                                                \
    dim3 __gridDim, __blockDim;                                               \
    size_t __sharedMem;                                                       \
    cudaStream_t __stream;                                                    \
    if (__cudaPopCallConfiguration(&__gridDim, &__blockDim, &__sharedMem,     \
                                   &__stream) != cudaSuccess)                 \
      return;                                                                 \
    if (isTileKernel)                                                         \
      __blockDim.x = __blockDim.y = __blockDim.z = 1;                         \
    if (__args_idx == 0) {                                                    \
      (void)cudaLaunchKernel(fun, __gridDim, __blockDim,                      \
                             &__args_arr[__args_idx], __sharedMem, __stream); \
    } else {                                                                  \
      (void)cudaLaunchKernel(fun, __gridDim, __blockDim, &__args_arr[0],      \
                             __sharedMem, __stream);                          \
    }                                                                         \
  }
#else
#define __codex_cuda_launch_impl(fun, isTileKernel)                           \
  {                                                                           \
    volatile static char *__f __NV_ATTR_UNUSED_FOR_LAUNCH;                    \
    __f = fun;                                                                \
    static cudaKernel_t __handle = 0;                                         \
    volatile static bool __tmp __NV_ATTR_UNUSED_FOR_LAUNCH =                  \
        (__cudaGetKernel(&__handle, (const void *)fun) == cudaSuccess);       \
    (void)__tmp;                                                              \
    dim3 __gridDim, __blockDim;                                               \
    size_t __sharedMem;                                                       \
    cudaStream_t __stream;                                                    \
    if (__cudaPopCallConfiguration(&__gridDim, &__blockDim, &__sharedMem,     \
                                   &__stream) != cudaSuccess)                 \
      return;                                                                 \
    if (isTileKernel)                                                         \
      __blockDim.x = __blockDim.y = __blockDim.z = 1;                         \
    if (__args_idx == 0) {                                                    \
      (void)__cudaLaunchKernel_helper(                                        \
          __handle, __gridDim, __blockDim, &__args_arr[__args_idx],           \
          __sharedMem, __stream);                                             \
    } else {                                                                  \
      (void)__cudaLaunchKernel_helper(__handle, __gridDim, __blockDim,        \
                                      &__args_arr[0], __sharedMem,            \
                                      __stream);                              \
    }                                                                         \
  }
#endif

#endif // __cudaLaunch
