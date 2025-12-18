# **Final Project Report**

**Single Threaded CPU Implementation**
- The algorithms have been implemented in a semi-optimized way.
	- For the stencil function there is no real optimal way to reduce memory accesses for single thread execution. Perhaps the input array could be duplicated and tranposed then accessed that way, however, this overhead would probably dominate any gains due to coalescence. We check the boundaries then subtract the central term as its added twice.
	- Matrix multiplcation has an optimized access pattern. The value of a is stored in a register then reused for coalesced access patterns of B. The partial sums of C are then calculated.
- In terms of profiling I couldn't download and use VTune. Maybe I could have on my local machine but to be honest I don't want to install anything locally and VTune required sudo permissions on the cluster. Considering the computation was 30s vs my slow.cu implementation run time of .7s, the most intensive part is almost definitely the memory access. By loop unrolling and the use of vectorized units the cpu code would like see a very significant speed increase.

	Average Run Time (20 iterations, DSIZE = 2048): 29445 ms


**Slow.cu**
	Based on the title you can probably guess this is the 'simplest' implementation, and you'd be correct! Slow.cu uses managed memory allocations as well as naive algorithm implementations.
- **Stencil_2d_slow**:
	- This uses the same algorithm as the cpu implmentation but now wraped inside of a thread loop. Each thread is responsible for its own element of the output stencil, performing multiple operations.
- **matrix_mul_slow**:
	- Surprisingly not as slow as I thought it would be. This is the same naive implementation as the cpu version. Memory access is still coalesced around matrix B allowing for optimized memory access. This has a substantial speed up compared to the cpu case after memory transfer due to the massive parallelism.
- **Profiling**:
	- This will be discussed in a later section comparing the 3 main implementations.

	Average Run Time (20 iterations, DSIZE = 2048): 729 ms

**Average.cu**
	Average.cu uses the exact same kernels as Slow.cu. Average.cu differs from slow.cu by using explicit memory copies. These memory copies have a larger memory overhead. However, after the bulk transfer is completed the kernels see some noticable speed, especially for the stencil algorithm as there are substaintially less page faults. This allows the threads to continue working uninterrupted.

	Average Run Time (20 iterations, DSIZE = 2048): 655 ms

**Fast.cu**
	This implementation pulls out all the stops, resulting in the second fastest implementation.
- **Asychronous Execution**:
	- Explicit memory copies from host to device are used for all arrays and are executed on their own asynchronous streams. This allows the gpu to schedule the memory transfer for independent objects.
	- Further utilizing the streams the stencil algorithm is conducted on A and B in parallel as these are independent operations. The streams are then synchronized before performing the matrix multiplcation which is done on its own stream. After this is completed all intermediate results are copied back to the device for debugging. Realistically though we only need to return the final result. This would provide some further speed up.

- **Stencil_2d_fast**:
	- The stencil operation has been updated to use a shared memory buffer than each thread in the block can access. Unlike in the original implementation in the week4 assignment, where only some threads were loading the halo elements. I decided to split the load evenly amongst all the threads. After the shared memory buffer is filled the stencil operation is performed using the buffer.
	- I believe the halo of 3 is too small to justify the shared memory loading. Based off nsys the stencil_2d_slow kernel is more performant in average.cu.

- **Matrix_mul_fast**:
	- To fully utilize the threads I decided to break the matrices into chunks, or tiles. All the threads in the block work together to load tile A and B for calculation later. By doing this legwork early we can reduce the total number of global memory accesses by 2. Compared to the slow matrix multiplcation method this is only a small increase surprisingly, ~1.3x. This is likely due to the large caches on the gpu, signifcantly reducing the overhead in the coalesced access pattern of the earlier implementation. Instead of using BLOCK_SIZE as the tile dimension we might be able to increase this dimension to the maximum size for a shared memory allocation and see if it has a noticable effect.

	- I like to make my algorithms more general, due to this I've added some boundary checking in the tile formation. This results in some thread divergencies possible resulting in a slow down. If we known the tile size (BLOCK SIZE) is a interger multiple of DSIZE then we can remove the boundary checking allowing for more efficient buffer filling. This algorithm is robust to square matrices of any size.

	Average Run Time (20 iterations, DSIZE = 2048): 575 ms

**Fastest.cu**
	The kernels in Fastest.cu are exactly the same as in fast.cu. Fastest.cu is a hybrid I tested out combining explicit memory copies and managed memory. Explicit memory copies have more epxensive overhead than managed memory. This overhead gets offset only when there is *fequent* access to elements in the unified memory object. The result container for matrix_mult however is never read from in the kernel. It only has its value written once per thread! So by using managed memory for the result container we can avoid the large transfer costs of moving the container between host to device. This resulted in ~ 5% speed increase.

	I also tried removing the third cuda stream to reduce the stream creation overhead but this resulted in an overall speed decrease. Personally I didn't understand why. While doing my speed testings there was normally large variance in the execution time and it likely resulted in untrust worthy results. The speeds quoted for each algorithm were produced during a golden hour where verything was stable (unstable runs could have variance upwards of 70s). At this point I had already deleted the script removing this stream.

	Average Run Time (20 iterations, DSIZE = 2048): 544 ms

**NSYS Profiling**

## CUDA Performance Comparison Table

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

### Key Observations:

1. **Slow version**: Uses unified memory with 8,191 page faults causing high memory transfer time
2. **Average version**: Uses explicit synchronous memory copies
3. **Fast version**: Uses async copies and streams but has high stream overhead
4. **Fastest version**: Optimized stream usage with fewer memory transfers

