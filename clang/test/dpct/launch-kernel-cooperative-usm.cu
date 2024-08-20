// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/launch-kernel-cooperative-usm %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck %s --match-full-lines --input-file %T/launch-kernel-cooperative-usm/launch-kernel-cooperative-usm.dp.cpp

// CHECK: void template_device(T *d, T *s) {
template<class T>
__device__ void template_device(T *d) {
  __shared__ T s[16];
}

// CHECK: void template_kernel(T *d, const sycl::nd_item<3> &item_ct1,
// CHECK-NEXT: uint8_t *dpct_local, T *s) {
template<class T>
__global__ void template_kernel(T *d) {
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ T es[];
  template_device(d);
}

// CHECK: void kernel(int *d, dpct::image_accessor_ext<int, 1> tex, const sycl::nd_item<3> &item_ct1) {
__global__ void kernel(int *d, cudaTextureObject_t tex) {
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  tex1D(d + gtid, tex, gtid);
}

int main() {
  int *d_data;
  cudaMalloc(&d_data, sizeof(int));

  int *d_data21;
  cudaMalloc(&d_data21, sizeof(int) * 32);
  cudaTextureObject_t tex;
  cudaResourceDesc res;
  cudaTextureDesc texDesc;
  res.resType = cudaResourceTypeLinear;
  res.res.linear.devPtr = d_data21;
  res.res.linear.desc.f = cudaChannelFormatKindSigned;
  res.res.linear.desc.x = sizeof(int)*8; // bits per channel
  res.res.linear.sizeInBytes = sizeof(int)*8;
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.addressMode[2] = cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModeLinear;
  cudaCreateTextureObject(&tex, &res, &texDesc, NULL);

  void *args[2] = { &d_data, &tex };

  // CHECK: q_ct1.submit(
  // CHECK-NEXT:  [&](sycl::handler &cgh) {
  // CHECK-NEXT:    auto tex_acc = static_cast<dpct::image_wrapper<int, 1> *>(*(dpct::image_wrapper_base_p *)args[1])->get_access(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:    auto tex_smpl = (*(dpct::image_wrapper_base_p *)args[1])->get_sampler();
  // CHECK-EMPTY:
  // CHECK-NEXT:    auto d_ct0 = *(int **)args[0];
  // CHECK-EMPTY:
  // CHECK-NEXT:    cgh.parallel_for(
  // CHECK-NEXT:      sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 16), sycl::range<3>(1, 1, 16)),
  // CHECK-NEXT:      [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:        kernel(d_ct0, dpct::image_accessor_ext<int, 1>(tex_smpl, tex_acc), item_ct1);
  // CHECK-NEXT:      });
  // CHECK-NEXT:  });
  cudaLaunchCooperativeKernel((void *)&kernel, dim3(16), dim3(16), args, 0, 0);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // CHECK: stream->submit(
  // CHECK-NEXT:  [&](sycl::handler &cgh) {
  // CHECK-NEXT:    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(32), cgh);
  // CHECK-NEXT:    sycl::local_accessor<int, 1> s_acc_ct1(sycl::range<1>(16), cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:    auto d_ct0 = *(int **)args[0];
  // CHECK-EMPTY:
  // CHECK-NEXT:    cgh.parallel_for(
  // CHECK-NEXT:      sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 16), sycl::range<3>(1, 1, 16)),
  // CHECK-NEXT:      [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:        template_kernel<int>(d_ct0, item_ct1, dpct_local_acc_ct1.template get_multi_ptr<sycl::access::decorated::no>().get(), s_acc_ct1.template get_multi_ptr<sycl::access::decorated::no>().get());
  // CHECK-NEXT:      });
  // CHECK-NEXT:  });
  cudaLaunchCooperativeKernel((const void *)&template_kernel<int>, dim3(16), dim3(16), args, 32, stream);

  void *kernel_func = (void *)&kernel;

  // CHECK: /*
  // CHECK-NEXT: DPCT1123:{{[0-9]+}}: The kernel function pointer cannot be used in the device code. You need to call the kernel function with the correct argument(s) directly. According to the kernel function definition, adjusting the dimension of the sycl::nd_item may also be required.
  // CHECK-NEXT: */
  // CHECK-NEXT: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 16), sycl::range<3>(1, 1, 16)), 
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:     kernel_func();
  // CHECK-NEXT:   });
  cudaLaunchCooperativeKernel(kernel_func, dim3(16), dim3(16), args, 0, 0);

  cudaStreamDestroy(stream);
  cudaDestroyTextureObject(tex);
  cudaFree(d_data21);
  cudaFree(d_data);
}

