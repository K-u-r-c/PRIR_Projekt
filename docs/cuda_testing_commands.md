# CUDA testing cheat sheet

Below is a curated set of shell commands that let you validate the CUDA
pipeline end-to-end: driver/toolchain health, CUDA-enabled build, and runtime
tests on a sample log. Run the commands in order; every entry describes what it
does and why it matters.

1. `nvidia-smi`
   - Shows the visible GPUs, driver version, and current utilization. If this
     fails, CUDA workloads cannot run and you need to install/enable NVIDIA
     drivers before testing the analyzer.

2. `nvcc --version`
   - Prints the CUDA toolkit version picked up by `make`. Confirm it matches
     the driver output above (major/minor compatibility) so that kernels can be
     compiled and loaded correctly.

3. `make clean`
   - Removes `build/` to eliminate objects produced without CUDA support. This
     guarantees that the subsequent compilation links the GPU objects rather
     than reusing stale CPU-only binaries.

4. `make USE_CUDA=1 NVCC=$(which nvcc)`
   - Builds `build/bin/prir` with CUDA support enabled. The flag defines
     `HAVE_CUDA_RUNTIME`, compiles `src/gpu_histogram.cu` via `nvcc`, and links
     `-lcudart`. Override `NVCC` if you want a specific toolkit instance (e.g.
     from a module environment).

5. `unzip -p web-server-access-logs.zip | head -n 200000 > /tmp/logs_gpu_test.log`
   - Materializes a realistic test log by streaming the bundled ZIP directly
     into a file. Using 200k lines keeps the workload long enough so CUDA
     acceleration is noticeable without exhausting disk space.

6. `./build/bin/prir --file /tmp/logs_gpu_test.log --phrase ERROR --phrase WARNING --stats minute --emit --threads 8`
   - CPU/OpenMP baseline run. Captures phrase counts, minute stats, and prints
     matching lines. Keep its timing (e.g. by prefixing with `time`) so you can
     compare against the CUDA run; correctness of the textual output should
     match the GPU variant.

7. `./build/bin/prir --file /tmp/logs_gpu_test.log --phrase ERROR --phrase WARNING --stats minute --emit --threads 8 --use-cuda`
   - Single-process CUDA test. The extra `--use-cuda` flag activates the GPU
     histogram kernel described in `gpu_interface.hpp`. You should observe the
     same counts as in step 6; discrepancies indicate a CUDA logic issue.

8. `mpirun -np 4 ./build/bin/prir --file /tmp/logs_gpu_test.log --phrase ERROR --phrase WARNING --stats minute --threads 4 --use-cuda`
   - Full distributed test: MPI splits the log into four ranks, each of which
     can launch CUDA kernels. Useful on multi-GPU nodes (one rank per GPU) or on
     a single GPU (kernels execute sequentially). Validate that the aggregated
     output from rank 0 matches the single-process results.

9. `CUDA_VISIBLE_DEVICES=0 ./build/bin/prir --file /tmp/logs_gpu_test.log --phrase "disk full" --phrase ALERT --use-cuda --threads 2`
   - Forces the process to a specific GPU. Handy on shared systems to ensure
     exclusivity or when debugging a single accelerator; if the command fails,
     CUDA is not seeing that GPU.

10. `nvprof ./build/bin/prir --file /tmp/logs_gpu_test.log --phrase ERROR --use-cuda --no-stats`
    - Optional profiler pass (replace with `nsys`/`ncu` on newer toolchains).
      Captures kernel timings so you can verify the GPU work actually happens
      and spot performance regressions. Requires NVIDIA profiling tools to be
      installed.

11. `make USE_CUDA=1 test ARGS="--file fixtures/small.log --phrase ERROR --use-cuda"`
    - Template for scripting regression tests. By reusing `make run` via the
      `ARGS` variable you can integrate the CUDA path into CI pipelines or
      custom shell scripts (replace the fixture path with your dataset).

Tips:
- Prefix runtime commands with `/usr/bin/time -v` if you need resource stats.
- To compare CPU vs GPU numerically, pipe both runs into `sort` + `diff`.
