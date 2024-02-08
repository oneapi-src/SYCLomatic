CodePin
===============

There are cases that the migrated SYCL program has different runtime behavior to
the original CUDA program. The reason of the inconsistency could be:
* Difference in arithmetic precision between hardware.
* Semantic difference between the CUDA and SYCL APIs
* Errors introduced during the automatic migration

CodePin is introduced as a sub-feature of |tool_name| to reduce the effort of
debugging such inconsist runtime behavior. When CodePin is enabled, |tool_name|
will not only migrate the CUDA program to SYCL but also generate instrumented
CUDA program.

The instrumented code will dump the data of related variables before/after
kernel calls or selected API calls into a report. Comparing the reports generated
from the CUDA and SYCL program helps identify the point of divergent behavior.

Command Line Option
----------------------------
CodePin can be enabled with |tool_name| command line option ``–enable-codepin``.
If ``–out-root`` is specified, the instrumented CUDA program will be put into a 
folder with ``_debug`` postfix beside the out-root folder. Otherwise, the 
instrumented CUDA program will be put in the default folder ``dpct_output_debug``.

Example
----------------------------
The following example demonstrates how CodePin works.

.. code-block:: c++

    #include <iostream>
    __global__ void vectorAdd(int3 *a, int3 *result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    result[tid].x = a[tid].x + 1;
    result[tid].y = a[tid].y + 1;
    result[tid].z = a[tid].z + 1;
    }

    int main() {
    const int vectorSize = 4;
    int3 h_a[vectorSize], h_result[vectorSize];
    int3 *d_a, *d_result;
    for (int i = 0; i < vectorSize; ++i)
        h_a[i] = make_int3(1, 2, 3);

    cudaMalloc((void **)&d_a, vectorSize * sizeof(int3));
    cudaMalloc((void **)&d_result, vectorSize * sizeof(int3));

    // Copy host vectors to device
    cudaMemcpy(d_a, h_a, vectorSize * 12, cudaMemcpyHostToDevice);

    // Launch the CUDA kernel
    vectorAdd<<<1, 4>>>(d_a, d_result);

    // Copy result from device to host
    cudaMemcpy(h_result, d_result, vectorSize * sizeof(int3),
        cudaMemcpyDeviceToHost);

    // Print the result
    for (int i = 0; i < vectorSize; ++i) {
        std::cout << "Result[" << i << "]: (" << h_result[i].x << ", "
                << h_result[i].y << ", " << h_result[i].z << ")\n";
    }
    }

    /*
    Execution Result:
    Result[0]: (2, 3, 4)
    Result[1]: (2, 3, 4)
    Result[2]: (2, 3, 4)
    Result[3]: (2, 3, 4)
    */




