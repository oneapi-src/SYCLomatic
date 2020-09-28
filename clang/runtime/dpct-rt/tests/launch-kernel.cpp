#include <CL/sycl.hpp>
#include <cmath>
#include <dpct/dpct.hpp>
#include <stdio.h>

void vectorAdd(const float *A, const float *B, float *C, int numElements,
               sycl::nd_item<3> item_ct1) {
  int i = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
          item_ct1.get_local_id(2);

  if (i < numElements) {
    C[i] = A[i] + B[i];
  }
}

void test_image(float* out, dpct::image_accessor_ext<float, 1> acc42, int numElements, sycl::nd_item<3> item_ct1) {
  int i = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
          item_ct1.get_local_id(2);

  if (i < numElements) {
    out[i] = acc42.read(i);
  }
}

int main(void) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  int retStatus = 0;
  int numElements = 50000;
  size_t size = numElements * sizeof(float);

  float *h_A = (float *)malloc(size);
  float *h_B = (float *)malloc(size);
  float *h_C = (float *)malloc(size);
  float *h_result = (float *)malloc(size);

  for (int i = 0; i < numElements; ++i) {
    h_A[i] = rand() / (float)RAND_MAX;
    h_B[i] = rand() / (float)RAND_MAX;
    h_C[i] = rand() / (float)RAND_MAX;
  }

  float *d_A, *d_B, *d_C;
  d_A = (float *)sycl::malloc_device(size, q_ct1);
  d_B = (float *)sycl::malloc_device(size, q_ct1);
  d_C = (float *)sycl::malloc_device(size, q_ct1);

  q_ct1.memcpy(d_A, h_A, size).wait();
  q_ct1.memcpy(d_B, h_B, size).wait();

  int threadsPerBlock = 256;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

  dpct::image_wrapper_base_p tex;
  dpct::image_data res;
  dpct::sampling_info texDesc;
  res.type = dpct::data_linear;
  res.data = h_C;
  res.chn = dpct::create_image_channel(32, 0, 0, 0, dpct::channel_float);
  res.x = numElements;
  tex = dpct::create_image_wrapper(res, texDesc);

  void **args = (void **)malloc(sizeof(void *) * 4);
  args[0] = &d_C;
  args[1] = &tex;
  args[2] = &numElements;
  q_ct1.submit([&](sycl::handler &cgh) {
    auto out_ct0 = *(float **)args[0];
    auto numElements_ct2 = *(int *)args[2];

    auto tex_acc = static_cast<dpct::image_wrapper<float, 1> *>(*(dpct::image_wrapper_base_p *)args[1])->get_access(cgh);
    auto tex_smpl = (*(dpct::image_wrapper_base_p *)args[1])->get_sampler();

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, blocksPerGrid) *
                              sycl::range<3>(1, 1, threadsPerBlock),
                          sycl::range<3>(1, 1, threadsPerBlock)),
        [=](sycl::nd_item<3> item_ct1) {
          test_image(out_ct0, dpct::image_accessor_ext<float, 1>(tex_smpl, tex_acc), numElements_ct2, item_ct1);
        });
  });

//   q_ct1.memcpy(h_result, d_C, size).wait();
//   for (int i = 0; i < numElements; ++i) {
//     if (fabs(h_C[i] - h_result[i]) > 1e-5) {
//         printf("Read Image Test FAIL!\n");
//         return -1;
//     }
//   }

  args[0] = &d_A;
  args[1] = &d_B;
  args[2] = &d_C;
  args[3] = &numElements;
  q_ct1.submit([&](sycl::handler &cgh) {
    auto A_ct0 = *(const float **)args[0];
    auto B_ct1 = *(const float **)args[1];
    auto C_ct2 = *(float **)args[2];
    auto numElements_ct3 = *(int *)args[3];

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, blocksPerGrid) *
                              sycl::range<3>(1, 1, threadsPerBlock),
                          sycl::range<3>(1, 1, threadsPerBlock)),
        [=](sycl::nd_item<3> item_ct1) {
          vectorAdd(A_ct0, B_ct1, C_ct2, numElements_ct3, item_ct1);
        });
  });

  q_ct1.memcpy(h_result, d_C, size).wait();

  for (int i = 0; i < numElements; ++i) {
    if (fabs(h_A[i] + h_B[i] - h_result[i]) > 1e-5) {
        printf("Vector Add Test FAIL!\n");
        return -1;
    }
  }

  printf("Test PASSED!\n");
  sycl::free(d_A, q_ct1);
  sycl::free(d_B, q_ct1);
  sycl::free(d_C, q_ct1);

  free(h_A);
  free(h_B);
  free(h_C);
  free(h_result);
  free(args);

  return retStatus;
}
