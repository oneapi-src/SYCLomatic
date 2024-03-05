CodePin
===============

There are some cases where the migrated SYCL program may have runtime behavior that differs from the original CUDA program. Reasons for this inconsistency include:

* Difference in arithmetic precision between hardware
* Semantic difference between the CUDA and SYCL APIs
* Errors introduced during the automatic migration

CodePin is introduced as a sub-feature of |tool_name| in order to reduce the effort of debugging such inconsistencies in runtime behavior.
When CodePin is enabled, |tool_name| will migrate the CUDA program to SYCL, but will also generate an instrumented version of the CUDA program.

The instrumented code will dump the data of related variables, before/after selected API or kernel calls, into a report.
Comparing the reports generated from the CUDA and SYCL program can help identify the source of divergent runtime behavior.

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

    //example.cu
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
        // !! Using 12 instead of "sizeof(int3)"
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

The example CUDA code has an issue in the cudaMemcpy() before the vectorAdd kernel call:
the size to be copied is hard coded as ``vectorSize * 12`` instead of ``vectorSize * sizeof(int3)``
, which causes incorrect behavior of the migrated SYCL program. This is because ``int3`` will be
migrated to ``sycl::int3`` and the size of ``sycl::int3`` is 16 bytes, not 12 bytes.

To debug the issue, the user can migrate the CUDA program with CodePin enabled.

.. code-block:: bash

    dpct example.cu --enable-codepin

After the migration, there will be 2 files ``dpct_output/example.dp.cpp`` and ``dpct_output_debug/example.cu``.

.. code-block:: bash

    workspace
    ├── example.cu
    ├── dpct_output
    │   ├── example.dp.cpp
    │   ├── generated_schema.hpp
    │   └── MainSourceFiles.yaml
    ├── dpct_output_debug
    │   ├── example.cu
    │   └── generated_schema.hpp


``dpct_output/example.dp.cpp`` is the migrated and instrumented SYCL program:

.. code-block:: c++

    //dpct_output/example.dp.cpp
    #include <dpct/dpct.hpp>
    #include <sycl/sycl.hpp>

    #include "generated_schema.hpp"
    #include <dpct/codepin/codepin.hpp>
    #include <iostream>

    void vectorAdd(sycl::int3 *a, sycl::int3 *result,
                const sycl::nd_item<3> &item_ct1) {
        int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2);
        result[tid].x() = a[tid].x() + 1;
        result[tid].y() = a[tid].y() + 1;
        result[tid].z() = a[tid].z() + 1;
    }

    int main() {
        sycl::device dev_ct1;
        sycl::queue q_ct1(dev_ct1,
                            sycl::property_list{sycl::property::queue::in_order()});
        const int vectorSize = 4;
        sycl::int3 h_a[vectorSize], h_result[vectorSize];
        sycl::int3 *d_a, *d_result;
        for (int i = 0; i < vectorSize; ++i)
            h_a[i] = sycl::int3(1, 2, 3);

        d_a = sycl::malloc_device<sycl::int3>(vectorSize, q_ct1);
        dpct::experimental::get_ptr_size_map()[*((void **)&d_a)] =
            vectorSize * sizeof(sycl::int3);

        d_result = sycl::malloc_device<sycl::int3>(vectorSize, q_ct1);
        dpct::experimental::get_ptr_size_map()[*((void **)&d_result)] =
            vectorSize * sizeof(sycl::int3);

        // Copy host vectors to device
        q_ct1.memcpy(d_a, h_a, vectorSize * 12);

        // Launch the CUDA kernel
        dpct::experimental::gen_prolog_API_CP(
            "example.cu:38:3(SYCL)", &q_ct1,
            VAR_SCHEMA_0, (long *)&d_a, VAR_SCHEMA_1, (long *)&d_result);
        q_ct1.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, 4), sycl::range<3>(1, 1, 4)),
            [=](sycl::nd_item<3> item_ct1) { vectorAdd(d_a, d_result, item_ct1); });

        // Copy result from device to host
        dpct::experimental::gen_epilog_API_CP(
            "example.cu:38:3(SYCL)", &q_ct1,
            VAR_SCHEMA_0, (long *)&d_a, VAR_SCHEMA_1, (long *)&d_result);

        q_ct1.memcpy(h_result, d_result, vectorSize * sizeof(sycl::int3)).wait();

        // Print the result
        for (int i = 0; i < vectorSize; ++i) {
            std::cout << "Result[" << i << "]: (" << h_result[i].x() << ", "
                    << h_result[i].y() << ", " << h_result[i].z() << ")\n";
        }
    }

    /*
    Execution Result:
    Result[0]: (2, 3, 4)
    Result[1]: (2, 3, 4)
    Result[2]: (2, 3, 4)
    Result[3]: (1, 1, 1) <--- incorrect result
    */

``dpct_output_debug/example.cu`` is the instrumented CUDA program:

.. code-block:: c++

    //dpct_output_debug/example.cu
    #include "generated_schema.hpp"
    #include <dpct/codepin/codepin.hpp>
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
        dpct::experimental::get_ptr_size_map()[*((void **)&d_a)] =
            vectorSize * sizeof(int3);
        cudaMalloc((void **)&d_result, vectorSize * sizeof(int3));
        dpct::experimental::get_ptr_size_map()[*((void **)&d_result)] =
            vectorSize * sizeof(int3);

        // Copy host vectors to device
        cudaMemcpy(d_a, h_a, vectorSize * 12, cudaMemcpyHostToDevice);

        // Launch the CUDA kernel
        dpct::experimental::gen_prolog_API_CP(
            "example.cu:38:3", 0, VAR_SCHEMA_0,
            (long *)&d_a, VAR_SCHEMA_1, (long *)&d_result);
        vectorAdd<<<1, 4>>>(d_a, d_result);

        // Copy result from device to host
        dpct::experimental::gen_epilog_API_CP(
            "example.cu:38:3", 0, VAR_SCHEMA_0,
            (long *)&d_a, VAR_SCHEMA_1, (long *)&d_result);
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

After building and executing ``dpct_output/example.dp.cpp`` and ``dpct_output_debug/example.cu``, the following report will be generated.

.. figure:: /_images/codepin_example_report.png

The report helps the user to identify where the runtime behavior of the CUDA and the SYCL version start to diverge from one another.