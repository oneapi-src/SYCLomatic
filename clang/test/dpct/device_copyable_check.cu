// RUN: dpct -out-root %T/device_copyable_check %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/device_copyable_check/device_copyable_check.dp.cpp

class CudaImage {
public:
  float *d_data;
  float f_data;
  CudaImage() {};
  ~CudaImage() {};
};
__global__ void ScaleDown(float *x, float y) { }
int main()
{
  CudaImage res;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       auto res_d_data_ct0 = res.d_data;
  // CHECK-NEXT:       auto res_f_data_ct1 = res.f_data;
  // CHECK-EMPTY:
  // CHECK-NEXT:       cgh.parallel_for(
  // CHECK-NEXT:           sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:           [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:             ScaleDown(res_d_data_ct0, res_f_data_ct1);
  // CHECK-NEXT:           });
  // CHECK-NEXT:     });
  ScaleDown<<<1, 1>>>(res.d_data, res.f_data);
  return 0;
}
