# **Final Project Report**

(The .cu files show up as links, I cant stop it. They aren't real links.)

### **Single Threaded CPU Implementation**

**Average Run Time (20 iterations, DSIZE = 2048): 29445 ms**
- **Stencil_2d:**
    - For the stencil function there is no real optimal way to reduce memory accesses for single thread execution. Perhaps the input array could be duplicated and tranposed then accessed that way, however, this overhead would probably dominate any gains due to coalescence. We check the boundaries then subtract the central term as its added twice.
- **Matrix_mult:**
	- Matrix multiplcation has an optimized access pattern. The value of a is stored in a register then reused for coalesced access patterns of B. The partial sums of C are then calculated.
- **Profiling:**
    - In terms of profiling I couldn't download and use VTune. Maybe I could have on my local machine but to be honest I don't want to install anything locally and VTune required sudo permissions on the cluster. Considering the computation was 30s vs my slow.cu implementation run time of .7s, the most intensive part is almost definitely the memory access. By loop unrolling and the use of vectorized units the cpu code would like see a very significant speed increase.




### **Slow.cu**
Based on the title you can probably guess this is the 'simplest' implementation, and you'd be correct! Slow uses managed memory allocations as well as naive algorithm implementations.

**Average Run Time (20 iterations, DSIZE = 2048): 729 ms**
- **Stencil_2d_slow:**
	- This uses the same algorithm as the cpu implmentation but now wraped inside of a thread loop. Each thread is responsible for its own element of the output stencil, performing multiple operations.
- **matrix_mul_slow:**
	- Surprisingly not as slow as I thought it would be. This is the same naive implementation as the cpu version. Memory access is still coalesced around matrix B allowing for optimized memory access. This has a substantial speed up compared to the cpu case after memory transfer due to the massive parallelism.
- **Profiling:**
	- This will be discussed in a later section comparing the 3 main implementations.



### **Average.cu**
Average.cu uses the exact same kernels as Slow.cu. Average.cu differs from slow.cu by using explicit memory copies. These memory copies have a larger memory overhead. However, after the bulk transfer is completed the kernels see some noticable speed, especially for the stencil algorithm as there are substaintially less page faults. This allows the threads to continue working uninterrupted.

**Average Run Time (20 iterations, DSIZE = 2048): 655 ms**

### **Fast.cu**
This implementation pulls out all the stops, resulting in the second fastest implementation.
**Average Run Time (20 iterations, DSIZE = 2048): 575 ms**
- **Asychronous Execution:**
	- Explicit memory copies from host to device are used for all arrays and are executed on their own asynchronous streams. This allows the gpu to schedule the memory transfer for independent objects.
	- Further utilizing the streams the stencil algorithm is conducted on A and B in parallel as these are independent operations. The streams are then synchronized before performing the matrix multiplcation which is done on its own stream. After this is completed all intermediate results are copied back to the device for debugging. Realistically though we only need to return the final result. This would provide some further speed up.

- **Stencil_2d_fast:**
	- The stencil operation has been updated to use a shared memory buffer than each thread in the block can access. Unlike in the original implementation in the week4 assignment, where only some threads were loading the halo elements. I decided to split the load evenly amongst all the threads. After the shared memory buffer is filled the stencil operation is performed using the buffer.
	- I believe the halo of 3 is too small to justify the shared memory loading. Based off nsys the stencil_2d_slow kernel is more performant in average.cu.

- **Matrix_mul_fast:**
	- To fully utilize the threads I decided to break the matrices into chunks, or tiles. All the threads in the block work together to load tile A and B for calculation later. By doing this legwork early we can reduce the total number of global memory accesses by 2. Compared to the slow matrix multiplcation method this is only a small increase surprisingly, ~1.3x. This is likely due to the large caches on the gpu, signifcantly reducing the overhead in the coalesced access pattern of the earlier implementation. Instead of using BLOCK_SIZE as the tile dimension we might be able to increase this dimension to the maximum size for a shared memory allocation and see if it has a noticable effect.

	- I like to make my algorithms more general, due to this I've added some boundary checking in the tile formation. This results in some thread divergencies possible resulting in a slow down. If we known the tile size (BLOCK SIZE) is a interger multiple of DSIZE then we can remove the boundary checking allowing for more efficient buffer filling. This algorithm is robust to square matrices of any size.



### **Fastest.cu**
The kernels in Fastest.cu are exactly the same as in fast.cu. Fastest.cu is a hybrid I tested out combining explicit memory copies and managed memory. Explicit memory copies have more epxensive overhead than managed memory. This overhead gets offset only when there is *fequent* access to elements in the unified memory object. The result container for matrix_mult however is never read from in the kernel. It only has its value written once per thread! So by using managed memory for the result container we can avoid the large transfer costs of moving the container between host to device. This resulted in ~ 5% speed increase.

I also tried removing the third cuda stream to reduce the stream creation overhead but this resulted in an overall speed decrease. Personally I didn't understand why. While doing my speed testings there was normally large variance in the execution time and it likely resulted in untrust worthy results. The speeds quoted for each algorithm were produced during a golden hour where verything was stable (unstable runs could have variance upwards of 70s). At this point I had already deleted the script removing this stream.

**Average Run Time (20 iterations, DSIZE = 2048): 544 ms**



## **CUDA Performance Comparison Table**

### CUDA API Summary (cuda_api_sum)

| Operation | Slow (ns / calls) | Average (ns / calls) | Fast (ns / calls) | Fastest (ns / calls) |
|-----------|------------------|---------------------|-------------------|---------------------|
| **cudaDeviceSynchronize** | 343,598,992 / 1 | 251,007,877 / 1 | - | - |
| **cudaMallocManaged** | 214,399,543 / 5 | - | - | 78,782 / 1 |
| **cudaMalloc** | - | 206,587,344 / 5 | 868,508 / 5 | 703,694 / 4 |
| **cudaMemcpy** | - | 50,538,475 / 8 | - | - |
| **cudaMemcpyAsync** | - | - | 237,763,987 / 8 | 36,370,332 / 6 |
| **cudaStreamCreate** | - | - | 204,220,844 / 3 | 207,224,396 / 3 |
| **cudaStreamSynchronize** | - | - | 3,272,285 / 5 | 226,095,975 / 5 |
| **cudaFree** | - | 2,624,952 / 3 | 2,644,323 / 3 | 1,565,721 / 2 |
| **cudaLaunchKernel** | 276,555 / 3 | 257,285 / 3 | 252,136 / 3 | 254,675 / 3 |
| **TOTAL** | **558,276,090** | **511,016,012** | **449,023,083** | **472,618,973** |

---

### GPU Kernel Summary (cuda_gpu_kern_sum)

| Kernel | Slow (ns / inst) | Average (ns / inst) | Fast (ns / inst) | Fastest (ns / inst) |
|--------|-----------------|---------------------|------------------|---------------------|
| **matrix_mul_slow** | 214,070,706 / 1 | 248,509,595 / 1 | - | - |
| **matrix_mul_fast** | - | - | 202,209,904 / 1 | 236,587,132 / 1 |
| **stencil_2d_slow** | 129,550,610 / 2 | 2,521,464 / 2 | - | - |
| **stencil_2d_fast** | - | - | 3,279,532 / 2 | 3,903,907 / 2 |
| **TOTAL** | **343,621,316** | **251,031,059** | **205,489,436** | **240,491,039** |

---

### GPU Memory Transfer Summary (cuda_gpu_mem_time_sum)

| Operation | Slow (ns / count) | Average (ns / count) | Fast (ns / count) | Fastest (ns / count) |
|-----------|------------------|---------------------|-------------------|---------------------|
| **Host-to-Device** | 37,861,369 / 8,191* | 27,297,263 / 5 | 27,160,085 / 5 | 21,934,666 / 4 |
| **Device-to-Host** | - | 22,496,476 / 3 | 22,182,595 / 3 | 14,120,517 / 2 |
| **TOTAL** | **37,861,369** | **49,793,739** | **49,342,680** | **36,055,183** |

*Unified Memory (8,191 page faults)

---

### Observations:

1. Managed memory overall has a reduced time spent; however, as we can see from Slow, this results in thousands more calls. These calls are all page faults resulting in large stall time of the threads. Considering the time it took to run the stencil operation in Slow, we can assume the stencil operation's memory access pattern is largely responsible for the page faults as its almost 50x slower than the other algorithms.
2. One benefit of the unified memory appears to be in matrix_mul as slow had the second shortest run time Likely due to the coalesced memory pattern lining up with the read bursts from the device, providing an optimied pipeline.
3. CudaDeviceSynchronize was substantially slower in the unified memory case and why I am not entirely sure as it was only called once.
4. Once swapping to explicit memory copies the total time to trasnfer memory from host to device was reduced by 30% at the cost of additional device to host calls. THis result in more overall time conducting memory transfers. However, this overhead was immediately offset by the time spent in the stencil kernel, providing significant speed up.
5. In Average matrix_mul saw a 25% speed decrease, why I am not entirely sure. Likely due to the fact that the whole matrix was in memory rather than chunks/pages This would result in higher look up times when reading elements for computation.
6. CudaMalloc + CudaMemCpy was slightly higher than CudaMallocManaged, where the real gain was in the CudaSynchronize call.
7. For the optimized case we paid a small price for using multiple cuda streams launches, but it provided significant returns with the reduced time spent in cudaMalloc and stream synchronization. The hybrid case of fastest spent much less time than expect in the async copies, likely due to less traffic, but it came at the price of a increased synchronization penalty. Fast had more performant kernels than fast noticeably but less performant time spent in memory. Base on these metrics Fast should've outperformed fastest however that wasn't the case in practice. Likely these metric's aren't perfectly accounting for overlapping streams.


## **Alpaka**

Due to the need for generality Alpaka requires numerous wrappers for primative function calls. To convert a cuda script to Alpaka requires making the one to one comparison *mostly*. Kernel execution is a bit more verbose.

First we initialize our backend and define some alpaka types that we will be using such as the accelerator(cuda device), Dim (block dim, 2), Thread index, Queue(streams), and WorkDivision(2D blocks, with 1 index along each block dim).

In main we now begin to use the alpaka wrappers. We define our platforms which are set by the accelerator and host device. We create our 3 queues/streams. Then we initalize our host and device objects (unlike pointers in cuda). These use the same api call but are now defined by device, kind of nice. Memory copy takes the stream, destination, and source as arguments.

Once memory is allocated we can start to define our kernel executions. These are done in a similar manner defining block size and blocks in the grid. This gets wrapped into workDiv which is sent off as one arguement rather than individual arguements like in cuda kernel launches. The kernel still requires the pointer so we must use the wrapper to get the pointer to send to the kernel.

now we must rewrite our kernels. Alpaka wraps kernels as alpaka accelerator function structs. These are templated functors that allow for switching backends. After wrapping the kernel into this struct its very straight forward to adapt. The algorithm largely remains the same we just have to define the threadIdx, blockIdx, and blockDim with the alpaka getters which return the objects as arrays of the dimensions we defined rather than structs of arrays as they were 'secretly' in cuda.

Then we launch our kernels and follow the exact same pattern as we did in cuda. Wait for queueA/B to finish, launch matrix mult. Then copy back the results, resynchronize the streams, print our results during debug on small arrays to verify the results.

The runtime of the alpaka variant was roughly inbetween fast and fastest. Due to variance in runtimes its hard to confirm.

## **Environment Setup, Compilation, and Execution.**


Ensure you have NVCC compiler installed. On the wisc cluster once you ssh into a gpu node the path is set by default now. If the path cannot be located run the following from a login node:
```
ssh g38nXX # XX:01-16
export LD_LIBRARY_PATH=/usr/local/cuda/lib
export PATH=$PATH:/usr/local/cuda/bin
```
Then you can compile and run any .cu executable with:
```
nvcc my_script.cu -o my_script
./my_script
```

If you wish to use alpaka you must clone the alpaka repository, preferably in your home directory then export the paths to the libraries:
```
cd ~
git clone https://github.com/alpaka-group/alpaka.git
export ALPAKA_ROOT=$HOME/alpaka
export CPLUS_INCLUDE_PATH=$ALPAKA_ROOT/include:$CPLUS_INCLUDE_PATH
```
Then an alpaka executable with cuda backend can be compiled and ran using the following:
```
nvcc -x cu -std=c++20   -I$ALPAKA_ROOT/include   --expt-relaxed-constexpr   -DALPAKA_ACC_GPU_CUDA_ENABLED   my_script.cpp -o my_script
./my_script
```
This requires c++20.

Given all setup is done you can now run my /Project/compile_and_profile.sh which will compile all CUDA executables and profile them
```
nvcc -lineinfo slow.cu -o slow
nvcc -lineinfo average.cu -o average
nvcc -lineinfo fast.cu -o fast
nvcc -lineinfo fastest.cu -o fastest

nsys profile --stats=true --force-overwrite=true -o profile_reports/slow_profile ./slow 2>&1 | tee profile_reports/slow_full_report.log
nsys profile --stats=true --force-overwrite=true -o profile_reports/average_profile ./average 2>&1 | tee profile_reports/average_full_report.log
nsys profile --stats=true --force-overwrite=true -o profile_reports/fast_profile ./fast 2>&1 | tee profile_reports/fast_full_report.log
nsys profile --stats=true --force-overwrite=true -o profile_reports/fastest_profile ./fastest 2>&1 | tee profile_reports/fastest_full_report.log

nvcc -x cu -std=c++20   -I$ALPAKA_ROOT/include   --expt-relaxed-constexpr   -DALPAKA_ACC_GPU_CUDA_ENABLED   fastest_alpaka.cpp -o fastest_alpaka
nsys profile --stats=true --force-overwrite=true   -o profile_reports/fastest_alpaka_profile   ./fastest_alpaka 2>&1 | tee profile_reports/fastest_alpaka_full_report.log
```
This will produce the profile reports and place them in the Project/profile_reports/ directory.
