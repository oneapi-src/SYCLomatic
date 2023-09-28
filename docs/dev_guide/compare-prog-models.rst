CUDA* and SYCL* Programming Model Comparison
============================================

This section compares the CUDA* and SYCL* programming models and shows how to map
concepts and APIs from CUDA to SYCL.

Execution Model
---------------

Kernel Function
***************

In CUDA*, a kernel function is defined using the ``__global__`` declaration
specifier and is executed concurrently across multiple threads on the GPU.
Functions that are called by a CUDA kernel must be qualified with the ``__device__``
specifier. The ``__host__`` declaration specifier is used to qualify functions
that can be called from host code running on the CPU.

In contrast, a SYCL* kernel is a function that is executed on SYCL-capable devices,
such as CPUs, GPUs, or FPGAs. These kernels are launched from host code and are
executed concurrently on SYCL devices. Unlike CUDA, SYCL kernel functions do not
require special declaration specifiers and are defined using standard C++ syntax.

The following table shows CUDA and SYCL equivalencies for defining kernel functions:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - CUDA
     - SYCL
   * - ``__global__ void foo_kernel() {}``
     - ``void foo_kernel() {}``
   * - ``__device__ void foo_device() {}``
     - ``void foo_device() {}``
   * - ``__host__ void foo_host() {}``
     - ``void foo_host() {}``
   * - ``__host__ __device__ void foo_host_device() {}``
     - ``void foo_host_device() {}``

If the macro ``__CUDA_ARCH__`` is used to differentiate the code path in a CUDA
``__host__ __device__`` function, you can use macro ``__SYCL_DEVICE_ONLY__`` in
SYCL to achieve similar functionality.

The following table shows CUDA and SYCL equivalencies specifying device functions:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - CUDA
     - SYCL
   * - .. code-block::

          __host__ __device__ int foo(int i) {
          #ifdef __CUDA_ARCH__
            return i + 1;
          #else
            return 0;
          #endif
          }
          __global__ void kernel() {
            foo();
          }
          int main() {
            return foo();
          }
     - .. code-block::

          int foo(int i) {
          #ifdef __SYCL_DEVICE_ONLY__
            return i + 1;
          #else
            return 0;
          #endif
          }
          void kernel() {
            foo();
          }

          int main() {
            return foo();
          }

Execution Hierarchy
*******************

In both CUDA and SYCL programming models, the kernel execution instances are
organized hierarchically to exploit parallelism effectively. In CUDA, these
instances are called threads; in SYCL, they are referred to as work-items. CUDA
threads can be organized into blocks, which in turn can be organized into grids.
SYCL work-items can be organized into work-groups, which in turn can be organized
into ND-ranges.

From a hardware perspective, when a CUDA kernel is executed on a GPU, the Streaming
Multiprocessors (SMs) create, manage, schedule, and execute threads in groups of 32
threads, known as warps. In comparison, in SYCL a sub-group represents a collection
of related work-items within a work-group that execute concurrently.

To migrate CUDA code to SYCL code, the CUDA execution hierarchy can be mapped to
the SYCL hierarchy as shown in the following table:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - CUDA
     - SYCL
   * - thread
     - work-item
   * - warp
     - sub-group
   * - block
     - work-group
   * - grid
     - ND-range

Thread Indexing
***************

As previously discussed, CUDA threads and SYCL work-items are organized in a
hierarchical manner.

CUDA contains built-in variables to support threads:

* Thread ID: ``threadIdx.x/y/z``
* Block ID: ``blockIdx.x/y/z``
* Block dimensions: ``blockDim.x/y/z``
* Grid dimensions: ``gridDim.x/y/z``

SYCL contains equivalent built-in variables:

* Thread ID: ``sycl::nd_item.get_local_id(0/1/2)``
* Work-group ID: ``sycl::nd_item.get_group(0/1/2)``
* Work-group dimensions: ``sycl::nd_item.get_local_range().get(0/1/2)``
* ND-range dimensions: ``sycl::nd_item.get_group_range(0/1/2)``

According to the execution space index linearization sections of the
`CUDA C++ programing Guide <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>`_
and the `SYCL 2020 Specification <https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html>`_, given a block or work-group with shape *(dx, dy, dz)* and an
element with id *(x, y, z)*, then the index is *x + y \* Dx + z \* Dx \* Dy* for
CUDA, and *z + y \* Dz + x \* Dz \* Dy* for SYCL. This discrepancy is due in part
to SYCL's better alignment with C++'s multidimensional indexing. Due to the
difference in index computation methods, the CUDA execution space right-most
dimension *z* is mapped to the SYCL left-most dimension *x*, as shown in the
following table:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - CUDA
     - SYCL
   * - ``gridDim.x/y/z``
     - ``sycl::nd_item.get_group_range(2/1/0)``
   * - ``blockIdx.x/y/z``
     - ``sycl::nd_item.get_group(2/1/0)``
   * - ``blockDim.x/y/z``
     - ``sycl::nd_item.get_local_range().get(2/1/0)``
   * - ``threadIdx.x/y/z``
     - ``sycl::nd_item.get_local_id(2/1/0)``
   * - ``warpsSize``
     - ``sycl::nd_item.get_sub_group().get_local_range().get(0)``

Kernel Launch
*************

CUDA uses the ``<<<...>>>`` execution configuration syntax and the ``dim3`` type
to specify the dimensions and sizes of grids and blocks. The function call
operator, in conjunction with the execution configuration syntax, is used to
submit a kernel function to a stream for execution. When a kernel is submitted,
an index space is defined, with each thread and block receiving a unique thread
ID and block ID. These IDs are used to compute the index within the index space,
and they are accessible within the kernel through built-in variables.

SYCL uses the ``parallel_for`` member function of ``sycl::queue`` and
``sycl::range`` to specify the dimensions and sizes of ND-ranges and work-groups.
Users can also apply the SYCL kernel attribute ``[[sycl::reqd_sub_group_size(dim)]]``
to indicate that a kernel must be compiled and executed with the specified sub-group
size. Each device supports only certain sub-group sizes, as defined by ``info::device::sub_group_sizes``.

The following table shows the original CUDA code for a kernel launch example,
migrated to SYCL:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Original CUDA Code
     - Migrated SYCL Code
   * - .. code-block::

          __global__ void foo() {
            int a = threadIdx.x;
          }

          int main() {
            dim3 size_1(100, 200, 300);
            dim3 size_2(5, 10, 20);

            foo<<<size_1, size_2>>>();
          }
     - .. code-block::

          void foo(sycl::nd_item<3> item) {
            int a = item.get_local_id(2);
          }

          int main() {
            sycl::queue q;
            sycl::range<3> size_1(300, 200, 100);
            sycl::range<3> size_2(20, 10, 5);

            q.parallel_for(
              sycl::nd_range<3>(size_1 * size_2, size_2),
              [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
                foo(item);
              });
          }

Make note of the following details in the migrated SYCL code:

* In the constructor of ``sycl::nd_range``, the first parameter `global size` is
  a work-item instead of a work-group. Thus `global size` should be the product
  of ``size_1`` and ``size_2`` to align with CUDA.
* The Thread Indexing section describes that the CUDA execution space right-most
  dimension *z* should be mapped to the SYCL left-most dimension *x*. Thus in
  this example, the size of dimension needs to be reversed.

Memory Model
------------

The CUDA and SYCL memory model is a hierarchical structure. In CUDA, it comprises
multiple memory spaces, such as shared memory, global memory, constant memory, and
unified memory. Shared memory allows for efficient communication within a thread
block. Global memory is accessible to all threads across the device, providing a
larger storage space but slower access compared to shared memory. Constant memory
is a read-only space, suitable for storing unchanging data such as constants or
lookup tables. Unified memory can be accessed both on host and device.

Similarly, in SYCL, local memory is shared within a work-group, global memory is
accessible by all work-items, and shared memory is accessible on the host and
device. According to the `SYCL 2020 Specification <https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html>`_, constant memory no longer appears in the SYCL device memory
model in SYCL 2020.

Shared Memory
*************

CUDA shared memory can be mapped to SYCL local memory. To perform this migration,
declare an accessor with access target set to ``sycl::access::target::local``.
For example:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Original CUDA Code
     - Migrated SYCL Code
   * - .. code-block::

          __global__ void foo() {
            __shared__ int shm[16];
            shm[0] = 2;
          }

          int main() {
            foo<<<1, 1>>>();
          }

     - .. code-block::

          void foo(int *shm) {
           shm[0] = 2;
          }

          int main() {
            sycl::queue q;
            q.submit([&](sycl::handler &cgh) {

              sycl::local_accessor<int> shm_acc(sycl::range<1>(16), cgh);
              cgh.parallel_for(
                  sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), [=](sycl::nd_item<3> item_ct1) {
                    foo(shm_acc.get_pointer());
                  });
            });
          }

Global, Constant, and Unified Memory
************************************

CUDA global memory and constant memory can be mapped to SYCL global memory. CUDA
unified memory can be mapped to SYCL shared memory. To perform this migration,
allocate memory through ``sycl::malloc_device`` or ``sycl::malloc_shared``. For
example:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Original CUDA Code
     - Migrated SYCL Code
   * - .. code-block::

          void foo() {
            int *mem1, *mem2;

            cudaMalloc(&mem1, 10);
            cudaMallocManaged(&mem2, 10);
          }
     - .. code-block::

          void foo() {
            sycl::queue q;
            int *mem1, *mem2;

            mem1 = sycl::malloc_device<int>(10, q);
            mem2 = sycl::malloc_shared<int>(10, q);
          }

Note that in CUDA, type specifiers ``__device__``, ``__constant__``, and ``__managed__``
can be used to declare a variable resident in global memory and unified memory.
There is no direct equivalence in SYCL, but you can implement similar functionality
with helper classes. A reference implementation for ``dpct::global_memory``,
``dpct::constant_memory``, and ``dpct::shared_memory`` is provided in the
`SYCLomatic project <https://github.com/oneapi-src/SYCLomatic/blob/4893685d6f2f1c6126a1962be65dfa7688bb4162/clang/runtime/dpct-rt/include/memory.hpp.inc#L1953>`_.

CUDA Device API Mapping
-----------------------

Synchronization API
*******************

In CUDA, synchronization functions are used to coordinate the execution of
different threads in a CUDA kernel. ``__syncthreads()`` blocks the execution of
all threads in a thread block until all threads in that block have reached the
function. Additionally, all global and shared memory accesses made by these threads
prior to ``__syncthreads()`` are visible to all threads in the block. The CUDA
``__syncthreads`` function can be mapped to ``sycl::group_barrier`` (with a
``sycl::group`` object passed in). For the CUDA ``__syncthreads_and``,
``__syncthreads_or``, and ``__syncthreads_count`` function migration, an additional
group algorithm is needed after ``sycl::group_barrier``. The CUDA  ``__syncwarp``
function can be mapped to ``sycl::group_barrier``  (with a ``sycl::sub_group``
object spassed in).

The following table shows CUDA to SYCL mapping for synchronization functions:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - CUDA
     - SYCL
   * - ``__syncthreads()``
     - ``sycl::group_barrier(Group)``
   * - ``__syncthreads_and()``
     - ``sycl::group_barrier(Group)`` and ``sycl::all_of_group(Group, predicate)``
   * - ``__syncthreads_or()``
     - ``sycl::group_barrier(Group)`` and ``sycl::any_of_group(Group, predicate)``
   * - ``__syncthreads_count()``
     - ``sycl::group_barrier(Group)`` and ``sycl::reduce_over_group(Group, predicate?1:0, sycl::ext::oneapi::plus<>())``
   * - ``__syncwarp()``
     - ``sycl::group_barrier(Sub_group)``

Memory Fence API
****************

Memory fence functions can be used to enforce some ordering on memory accesses.
The memory fence functions differ in the scope in which the orderings are enforced.
CUDA memory fence functions can be mapped to ``sycl::atomic_fence`` with different
memory scope.

The following table shows CUDA to SYCL mapping for fence functions:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - CUDA
     - SYCL
   * - ``__threadfence_block()``
     - ``sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::work_group)``
   * - ``__threadfence()``
     - ``sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device)``
   * - ``__threadfence_system()``
     - ``sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system)``

Warp Intrinsic API
******************

CUDA warp intrinsic functions, including warp vote and shuffle functions, can
be mapped to the SYCL group algorithm API.

Warp Vote API
*************

The CUDA warp vote API can map to the SYCL group algorithm API, as shown in the
following table:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - CUDA
     - SYCL
   * - ``__all()/__all_sync()``
     - ``sycl::all_of_group()``
   * - ``__any()/__any_sync()``
     - ``sycl::any_of_group()``
   * - ``__ballot()/__ballot_sync()``
     - ``sycl::reduce_over_group()``

For the sync version of CUDA warp intrinsic functions, a mask is passed that
specifies the threads participating in the call. The equivalent SYCL API does not
support mask directly. Refer to the reference implementation of mask version APIs
in the `SYCLomatic project <https://github.com/oneapi-src/SYCLomatic/blob/4893685d6f2f1c6126a1962be65dfa7688bb4162/clang/runtime/dpct-rt/include/util.hpp.inc#L413>`__.

The following table shows the original CUDA code for a warp vote example, migrated to SYCL:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Original CUDA Code
     - Migrated SYCL Code
   * - .. code-block::

          __device__ void foo(){
            __all_sync(mask, predicate);
          }

     - .. code-block::

          void foo(sycl::nd_item<3> item) {
            auto g = item.get_sub_group();
            sycl::all_of_group(g, (~mask & (0x1 << g.get_local_linear_id())) || predicate);
          }

Warp Shuffle API
****************

CUDA warp shuffle functions can be mapped to the following SYCL group algorithms:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - CUDA
     - SYCL
   * - ``__shfl()/__shfl_sync()``
     - ``sycl::select_from_group()``
   * - ``__shfl_up()/__shfl_up_sync()``
     - ``sycl::shift_group_right()``
   * - ``__shfl_down()/__shfl_down_sync()``
     - ``sycl::shift_group_left()``
   * - ``__shfl_xor()/__shfl_xor_sync()``
     - ``sycl::permute_group_by_xor()``

CUDA shuffle functions support operate on a subset threads of warp. The equivalent
SYCL API does not support operations on a subset of sub_group directly. Refer to
the helper implementation in the `SYCLomatic project <https://github.com/oneapi-src/SYCLomatic>`__.

The following table shows the original CUDA code for a warp shuffle example, migrated to SYCL:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Original CUDA Code
     - Migrated SYCL Code
   * - .. code-block::

          __device__ void foo(){
            __shfl_sync(mask, val, r_id, 16);
          }
     - .. code-block::

          void foo(sycl::nd_item<3> item) {
            auto g = item.get_sub_group();
            unsigned int start_index = (g.get_local_linear_id() / 16) * 16;
            sycl::select_from_group(g, val, start_index + r_id % 16);
          }

CUDA Host API Mapping
---------------------

Device Management
*****************

The CUDA device management API can map to the SYCL device class and its member
functions as follows:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - CUDA
     - SYCL
   * - ``cudaGetDeviceCount()``
     - ``sycl::device::get_devices()``
   * - ``cudaSetDevice()``
     - ``sycl::device dev { device_selector }`` //Select sycl::device and make it ready for creating sycl::queue
   * - ``cudaGetDevice()``
     - ``sycl::queue.get_device()`` //Get active device from sycl::queue created 
   * - ``cudaGetDeviceProperties()``/``cudaDeviceGetAttribute()``
     - ``sycl::device.get_info<info type>()``

The following table shows the original CUDA code for a device management example,
migrated to SYCL:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Original CUDA Code
     - Migrated SYCL Code
   * - .. code-block::

          int device_count;
          cudaGetDeviceCount(&device_count);
          for(int i = 0; i < device_count; i++) {
            cudaSetDevice(i);
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            int warp_size = prop.warpSize;
            …
            kernel<<<size_1, size_2>>>();
          }
     - .. code-block::

          auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
          for(auto &device : devices) {
            sycl::queue q(device);
            auto sub_group_sizes = device.get_info<sycl::info::device::sub_group_sizes>();
            ...
            q.parallel_for(sycl::nd_range<3>(size_1 * size_2, size_2), [=](sycl::nd_item<3> item){
              kernel(item);
            });
          }

Stream Management
*****************

The CUDA stream management API can map to the SYCL queue class and its member
functions as follows:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - CUDA
     - SYCL
   * - ``cudaStreamCreate()``
     - Constructor of ``sycl::queue``
   * - ``cudaStreamDestroy()``
     - Destructor of ``sycl::queue``
   * - ``cudaStreamAddCallback()``
     - ``std::async()``
   * - ``cudaStreamSynchronize()``
     - ``sycl::queue.wait()``
   * - ``cudaStreamWaitEvent()``
     - ``sycl::queue.ext_oneapi_submit_barrier()``

The following table shows the original CUDA code for a stream management example,
migrated to SYCL:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Original CUDA Code
     - Migrated SYCL Code
   * - .. code-block::

          void callback(cudaStream_t st, cudaError_t status, void *vp) {…}

          void test() {
            cudaStream_t stream;
            cudaEvent_t event;

            cudaStreamCreate(&stream);
            cudaStreamAddCallback(stream, callback, 0, 0);
            cudaStreamSynchronize(stream);
            cudaStreamWaitEvent(stream, event, 0);
            cudaStreamDestroy(stream);
          }
     - .. code-block::

          void callback(sycl::queue st, int status, void *vp) {…}

          void test() {
            sycl::queue q;
            sycl::event event;
           
           std::async([&]() {
              q.wait();
              callback(q, 0, 0);
            });
            q.wait();
            q.ext_oneapi_submit_barrier({event});
          }

Note that the constructor of ``sycl::queue`` creates a queue with an out-of-order
property by default. Use ``sycl::property::queue::in_order::in_order()`` in the
construction of the queue to create an in-order queue.

Memory Management
*****************

The CUDA memory management API can map to the SYCL USM pointer-based memory
management API as follows:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - CUDA
     - SYCL
   * - ``cudaMalloc()``
     - ``sycl::malloc_device()``
   * - ``cudaMallocHost()``
     - ``sycl::malloc_host()``
   * - ``cudaMallocManaged()``
     - ``sycl::malloc_shared()``
   * - ``cudaMemcpy()``
     - ``sycl::queue.memcpy()``
   * - ``cudaMemset()``
     - ``sycl::queue.memset()``
   * - ``cudaFree()``/``cudaFreeHost()``
     - ``sycl::free()``


The following table shows the original CUDA code for a memory management example,
migrated to SYCL:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Original CUDA Code
     - Migrated SYCL Code
   * - .. code-block::

          void test() {
            int *dev_ptr, *host_ptr, *shared_ptr;
            int size;

            ...

            cudaMalloc(&dev_ptr, size);
            cudaMallocHost(&host_ptr, size);
            cudaMallocManaged(&shared_ptr, size);
            cudaMemset(dev_ptr, size, 0);
            cudaMemcpy(host_ptr, dev_ptr, size, cudaMemcpyHostToDevice);
            cudaMemcpy(shared_ptr, host_ptr, size, cudaMemcpyHostToDevice);
            ...
            int a = shared_ptr[0];
            ...
            cudaFree(dev_ptr);
            cudaFree(host_ptr);
            cudaFree(shared_ptr);
          }
     - .. code-block::

          void test() {
            sycl::queue q;
            int *dev_ptr, *host_ptr, *shared_ptr;
            int size;
            ...
            dev_ptr = (int *)sycl::malloc_device(size, q);
            host_ptr = (int *)sycl::malloc_host(size, q);
            shared_ptr = (int *)sycl::malloc_shared(size, q);
            q.memset(dev_ptr, size, 0).wait();
            q.memcpy(host_ptr, dev_ptr, size).wait();
            q.memcpy(shared_ptr, host_ptr, size).wait();
            ...
            int a = shared_ptr[0];
            ...
            sycl::free(dev_ptr, q);
            sycl::free(host_ptr, q);
            sycl::free(shared_ptr, q);
          }

Error Handling
--------------

In the CUDA runtime library error handling relies mainly on the error code
returned by the API call. In SYCL, synchronous errors are reported immediately
by the runtime throwing an exception. Use try-catch statements to catch and process
these exceptions. For example:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Original CUDA Code
     - Migrated SYCL Code
   * - .. code-block::

          void test() {
            int *ptr;
            if (cudaMalloc(&ptr, sizeof(int))) {
              std::cout << "error" << std::endl;
            }
          }
     - .. code-block::

          void test() try {
            int *ptr;
            sycl::queue q;
            ptr = sycl::malloc_device<int>(1, q);
          }
          catch (sycl::exception const &exc) {
            std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
            std::exit(1);
          }


In SYCL, asynchronous errors are not reported immediately as they occur. The queue
can optionally take an asynchronous handler at construction, with an ``exception_list``
as a parameter. Invocation of an ``async_handler`` may be triggered by the queue
member functions ``queue::wait_and_throw()``, ``queue::throw_asynchronous()``, or
automatically on destruction of a queue that contains unconsumed asynchronous errors.
When invoked, an ``async_handler`` is called and receives an ``exception_list``
argument containing a list of exception objects representing any unconsumed
asynchronous errors associated with the queue or context. The following example
shows one implementation of an asynchronous exception handler, migrated to SYCL:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Original CUDA Code
     - Migrated SYCL Code
   * - .. code-block::

          void test() {
            int *ptr;
            kernel<<<1, 1>>>();
            if (cudaDeviceSynchronize()) {
              std::cout << "error" << std::endl;
            }
          }
     - .. code-block::

          auto exception_handler = [](cl::sycl::exception_list exceptions) {
            for (std::exception_ptr const &e : exceptions) {
              try {
                std::rethrow_exception(e);
              } catch (cl::sycl::exception const &e) {
                std::cerr << "Caught asynchronous SYCL exception:" << std::endl
                          << e.what() << std::endl
                          << "Exception caught at file:" << __FILE__
                          << ", line:" << __LINE__ << std::endl;
              }
            }
          };
          void test() {
             sycl::queue q{exception_handler};
             q.parallel_for(
              sycl::nd_range<3>(size_1 * size_2, size_2),
              [=](sycl::nd_item<3> item){
                kernel(item);
              }).wait_and_throw();
          }

