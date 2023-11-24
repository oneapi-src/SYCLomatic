/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")

#include "../../debug_helper.hpp"
#include "generated_schema.hpp"



class TestClass {
public:
  TestClass(int a =rand() %100, int b = rand() %100): a(a), b(b) {}
  friend TestClass operator+(const TestClass input1, const TestClass input2);
  friend bool operator==(const TestClass input1, const TestClass input2);
friend bool operator!=(const TestClass input1, const TestClass input2);

  friend std::ostream& operator<<(std::ostream& os, const TestClass& dt);
  void set_a(int in) {
    a = in;
  }
  void set_b(int in) {
    b = in;
  }
  private:
    int a = 0;
    int b = 0;
};
TestClass operator+(const TestClass input1, const TestClass input2) {
  TestClass ret(0,0);
  ret.a = input1.a + input2.b;
  ret.b = input1.b + input2.b;
  return ret;
}

bool operator==(const TestClass input1, const TestClass input2) {
  if((input1.a == input2.a) && (input1.b == input2.b)) 
    return true;
  return false;
}
bool operator!=(const TestClass input1, const TestClass input2) {
  if((input1.a != input2.a) || (input1.b != input2.b)) 
    return true;
  return false;
}
std::ostream& operator<<(std::ostream& os, const TestClass& dt) {
  os << "A is " << dt.a << " B is " << dt.b << std::endl;
  return os;
}

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
void vectorAdd(const TestClass *A, const TestClass *B, TestClass *C,
                          int numElements, const sycl::nd_item<3> &item_ct1) {
  int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
          item_ct1.get_local_id(2);

  if (i < numElements) {
    C[i] = A[i] + B[i];
  }
}

/**
 * Host main routine
 */
int main(void) try {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  // Error code to check return values for CUDA calls
  dpct::err0 err = 0;

  // Print the vector length to be used, and compute its size
  int numElements = 5;
  size_t size = numElements * sizeof(TestClass);
  printf("[Vector addition of %d elements]\n", numElements);

  // Allocate the host input vector A
  TestClass *h_A = (TestClass *)malloc(size);

  // Allocate the host input vector B
  TestClass *h_B = (TestClass *)malloc(size);

  // Allocate the host output vector C
  TestClass *h_C = (TestClass *)malloc(size);
  
  // Verify that allocations succeeded
  if (h_A == NULL || h_B == NULL || h_C == NULL) {
    fprintf(stderr, "Failed to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }

  // Initialize the host input vectors
  for (int i = 0; i < numElements; ++i) {
    h_A[i].set_a(rand() % (int)1000);
    h_A[i].set_b(rand() % (int)1000);
    h_B[i].set_a(rand() % (int)1000);
    h_B[i].set_b(rand() % (int)1000);
  }

      // Allocate the device input vector A
  TestClass *d_A = NULL;
  err = DPCT_CHECK_ERROR(d_A = (TestClass *)sycl::malloc_device(size, q_ct1));

  // Allocate the device input vector B
  TestClass *d_B = NULL;
  err = DPCT_CHECK_ERROR(d_B = (TestClass *)sycl::malloc_device(size, q_ct1));

  // Allocate the device output vector C
  TestClass *d_C = NULL;
  err = DPCT_CHECK_ERROR(d_C = (TestClass *)sycl::malloc_device(size, q_ct1));

  // Copy the host input vectors A and B in host memory to the device input
  // vectors in
  // device memory
  printf("Copy input data from the host memory to the CUDA device\n");
 // target API, device pointer to host, copy then wait host.
 // Kernel function: parameter (1st value of the point.) // 
 // (d_A, 1)
 dpct::experimental::gen_prolog_API_CP(
     "cudaMemcpy:vecotr.cu:[185]:", &q_ct1, TYPE_SHCEMA_005, (long)d_A,
     (size_t)size, TYPE_SHCEMA_002, (long)h_A, (size_t)size);
 err = DPCT_CHECK_ERROR(q_ct1.memcpy(d_A, h_A, size).wait());
 dpct::experimental::gen_epilog_API_CP(
     "cudaMemcpy:vecotr.cu:[185]:", &q_ct1, TYPE_SHCEMA_005, (long)d_A,
     (size_t)size, TYPE_SHCEMA_002, (long)h_A, (size_t)size);

  dpct::experimental::gen_prolog_API_CP(
      "cudaMemcpy:vecotr.cu:[196]:", &q_ct1, TYPE_SHCEMA_006, (long)d_B,
      (size_t)size, TYPE_SHCEMA_003, (long)h_B, (size_t)size);
  err = DPCT_CHECK_ERROR(q_ct1.memcpy(d_B, h_B, size).wait());
  dpct::experimental::gen_epilog_API_CP(
      "cudaMemcpy:vecotr.cu:[196]:", &q_ct1, TYPE_SHCEMA_006, (long)d_B,
      (size_t)size, TYPE_SHCEMA_003, (long)h_B, (size_t)size);

  // Launch the Vector Add CUDA Kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);
  dpct::experimental::gen_prolog_API_CP(
      "vectorAdd:vecotr.cu:[221]:", &q_ct1, TYPE_SHCEMA_005, (long)d_A, 0,
      TYPE_SHCEMA_006, (long)d_B, 0, TYPE_SHCEMA_007, (long)d_C, 0,
      TYPE_SHCEMA_008, (long)numElements, 0);
  /*
  DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, blocksPerGrid) *
                            sycl::range<3>(1, 1, threadsPerBlock),
                        sycl::range<3>(1, 1, threadsPerBlock)),
      [=](sycl::nd_item<3> item_ct1) {
        vectorAdd(d_A, d_B, d_C, numElements, item_ct1);
      });
  dpct::experimental::gen_epilog_API_CP(
      "vectorAdd:vecotr.cu:[221]:", &q_ct1, TYPE_SHCEMA_005, (long)d_A, 0,
      TYPE_SHCEMA_006, (long)d_B, 0, TYPE_SHCEMA_007, (long)d_C, 0,
      TYPE_SHCEMA_008, (long)numElements, 0);

  /*
  DPCT1010:6: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  err = 0;

  // Copy the device result vector in device memory to the host result vector
  // in host memory.
  printf("Copy output data from the CUDA device to the host memory\n");
 dpct::experimental::gen_prolog_API_CP(
     "cudaMemcpy:vecotr.cu:[237]:", &q_ct1, TYPE_SHCEMA_004, (long)h_C,
     (size_t)size, TYPE_SHCEMA_007, (long)d_C, (size_t)size);
  err = DPCT_CHECK_ERROR(q_ct1.memcpy(h_C, d_C, size).wait());
 dpct::experimental::gen_epilog_API_CP(
     "cudaMemcpy:vecotr.cu:[237]:", &q_ct1, TYPE_SHCEMA_004, (long)h_C,
     (size_t)size, TYPE_SHCEMA_007, (long)d_C, (size_t)size);

  // Verify that the result vector is correct
  for (int i = 0; i < numElements; ++i) {
    if (h_A[i] + h_B[i] != h_C[i]) {
      std::cout << "H_A " << h_A[i] << std::endl;
      std::cout << "H_B " << h_B[i] << std::endl;
      std::cout << "H_C " << h_C[i] << std::endl;
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  printf("Test PASSED\n");

  // Free device global memory
  err = DPCT_CHECK_ERROR(sycl::free(d_A, q_ct1));

  err = DPCT_CHECK_ERROR(sycl::free(d_B, q_ct1));

  err = DPCT_CHECK_ERROR(sycl::free(d_C, q_ct1));

  // Free host memory
  free(h_A);
  free(h_B);
  free(h_C);

  printf("Done\n");
  return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
