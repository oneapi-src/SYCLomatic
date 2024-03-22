// RUN: echo "empty command"
//==============================================================
// Copyright 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include "dpct/codepin/codepin.hpp"
#include "generated_schema.hpp"
#define VECTOR_SIZE 256

void VectorAddKernel(float* A, float* B, float* C,
                     const sycl::nd_item<3> &item_ct1)
{
    A[item_ct1.get_local_id(2)] = item_ct1.get_local_id(2) + 1.0f;
    B[item_ct1.get_local_id(2)] = item_ct1.get_local_id(2) + 1.0f;
    C[item_ct1.get_local_id(2)] = A[item_ct1.get_local_id(2)] + B[item_ct1.get_local_id(2)];
}

int main() try {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    float *d_A, *d_B, *d_C;
    dpct::err0 status;

    d_A = sycl::malloc_device<float>(VECTOR_SIZE, q_ct1);
    d_B = sycl::malloc_device<float>(VECTOR_SIZE, q_ct1);
    d_C = sycl::malloc_device<float>(VECTOR_SIZE, q_ct1);
    dpct::experimental::gen_prolog_API_CP("vectorAdd:vecotr.cu:[29]:", &q_ct1, "d_A", d_A, "d_B", d_B, "d_C", d_C);
    q_ct1.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, VECTOR_SIZE), sycl::range<3>(1, 1, VECTOR_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            VectorAddKernel(d_A, d_B, d_C, item_ct1);
        });
    dpct::experimental::gen_epilog_API_CP("vectorAdd:vecotr.cu:[29]]:", &q_ct1, "d_A", d_A, "d_B", d_B, "d_C", d_C);

    float Result[VECTOR_SIZE] = {};

    // dpct::experimental::gen_prolog_API_CP("cudaMemcpy:vecotr.cu:[237]:", 0, TYPE_SHCEMA_004, (long *)&h_C, (size_t)size, TYPE_SHCEMA_007, (long *)&d_C, (size_t)size);
    status = DPCT_CHECK_ERROR(q_ct1.memcpy(Result, d_C, VECTOR_SIZE * sizeof(float)).wait());
    // dpct::experimental::gen_epilog_API_CP("cudaMemcpy:vecotr.cu:[237]:", 0, TYPE_SHCEMA_004, (long *)&h_C, (size_t)size, TYPE_SHCEMA_007, (long *)&d_C, (size_t)size);

    sycl::free(d_A, q_ct1);
    sycl::free(d_B, q_ct1);
    sycl::free(d_C, q_ct1);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        if (i % 16 == 0) {
            printf("\n");
        }
        printf("%3.0f ", Result[i]);    
    }
    printf("\n");
	
    return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
