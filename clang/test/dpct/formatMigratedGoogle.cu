// RUN: cat %s > %T/formatMigratedGoogle.cu
// RUN: cd %T
// RUN: dpct --no-cl-namespace-inline -out-root %T/formatMigratedGoogle formatMigratedGoogle.cu --cuda-include-path="%cuda-path/include" --format-style=google -- -std=c++14  -x cuda --cuda-host-only
// RUN: FileCheck -strict-whitespace formatMigratedGoogle.cu --match-full-lines --input-file %T/formatMigratedGoogle/formatMigratedGoogle.dp.cpp

#include <cuda_runtime.h>
#include <cassert>

//CHECK:void testDevice(const int *K) {
//CHECK-NEXT:  int t = K[0];
//CHECK-NEXT:}
__device__ void testDevice(const int *K) {
  int t = K[0];
}

     //CHECK:void testDevice1(const int *K) { int t = K[0]; }
__device__ void testDevice1(const int *K) { int t = K[0]; }

     //CHECK:void testKernelPtr(const int *L, const int *M, int N,
//CHECK-NEXT:                   cl::sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:  testDevice(L);
//CHECK-NEXT:  int gtid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
//CHECK-NEXT:             item_ct1.get_local_id(2);
//CHECK-NEXT:}
__global__ void testKernelPtr(const int *L, const int *M, int N) {
  testDevice(L);
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
}


     //CHECK:int main() {
//CHECK-NEXT:  dpct::device_ext &dev_ct1 = dpct::get_current_device();
//CHECK-NEXT:  cl::sycl::queue &q_ct1 = dev_ct1.default_queue();
//CHECK-NEXT:  cl::sycl::range<3> griddim = cl::sycl::range<3>(1, 1, 2);
//CHECK-NEXT:  cl::sycl::range<3> threaddim = cl::sycl::range<3>(1, 1, 32);
//CHECK-NEXT:  int *karg1, *karg2;
//CHECK-NEXT:  karg1 = cl::sycl::malloc_device<int>(32, q_ct1);
//CHECK-NEXT:  karg2 = cl::sycl::malloc_device<int>(32, q_ct1);
//CHECK-NEXT:  int karg3 = 80;
//CHECK-NEXT:  /*
//CHECK-NEXT:  DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
//CHECK-NEXT:  limit. To get the device limit, query info::device::max_work_group_size.
//CHECK-NEXT:  Adjust the work-group size if needed.
//CHECK-NEXT:  */
//CHECK-NEXT:  q_ct1.parallel_for(cl::sycl::nd_range<3>(griddim * threaddim, threaddim),
//CHECK-NEXT:                     [=](cl::sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:                       testKernelPtr((const int *)karg1, karg2, karg3,
//CHECK-NEXT:                                     item_ct1);
//CHECK-NEXT:                     });
//CHECK-NEXT:}
int main() {
  dim3 griddim = 2;
  dim3 threaddim = 32;
  int *karg1, *karg2;
  cudaMalloc(&karg1, 32 * sizeof(int));
  cudaMalloc(&karg2, 32 * sizeof(int));
  int karg3 = 80;
  testKernelPtr<<<griddim, threaddim>>>((const int *)karg1, karg2, karg3);
}

     //CHECK:#define DEVICE 
#define DEVICE __device__

     //CHECK:struct SharedMemory
//CHECK-NEXT:{
//CHECK-NEXT:  unsigned int *getPointer(uint8_t *dpct_local)
//CHECK-NEXT:  {
//CHECK-NEXT:    auto s_uint = (unsigned int *)dpct_local;
//CHECK-NEXT:    return s_uint;
//CHECK-NEXT:  }
//CHECK-NEXT:};
struct SharedMemory
{
  __device__ unsigned int *getPointer()
  {
    extern __shared__ unsigned int s_uint[];
    return s_uint;
  }
};

     //CHECK:typedef struct dpct_type_{{[0-9a-z]+}}
//CHECK-NEXT:{
//CHECK-NEXT:  int SM;
//CHECK-NEXT:  int Cores;
//CHECK-NEXT:} sSMtoCores;
typedef struct
{
  int SM;
  int Cores;
} sSMtoCores;

// Currently, keep the parameters as original location.
     //CHECK:template <int EFLAG>
//CHECK-NEXT:void k_mdppp_outer_nn(const int * __restrict__ pos,
//CHECK-NEXT:                                 const float * __restrict__ q,
//CHECK-NEXT:                                 const int * __restrict__ numneigh,
//CHECK-NEXT:                                 const int * __restrict__ firstneigh,
//CHECK-NEXT:                                 const float * __restrict__ special_lj,
//CHECK-NEXT:                                 const float * __restrict__ special_coul,
//CHECK-NEXT:                                 const int * __restrict__ ljd_in,
//CHECK-NEXT:                                 const int vflag, const int ntypes,
//CHECK-NEXT:                                 const int nlocal, const int nbor_stride,
//CHECK-NEXT:                                 const float cut_coulsq, const float cut_ljsq,
//CHECK-NEXT:                                 const float cut_lj_innersq,
//CHECK-NEXT:                                 const float g_ewald, const float qqrd2e,
//CHECK-NEXT:                                 const float denom_lj_inv,
//CHECK-NEXT:                                 const int loop_trip,
//CHECK-NEXT:                                 cl::sycl::nd_item<3> item_ct1, float *sp_lj,
//CHECK-NEXT:                                 float *sp_coul, int *ljd,
//CHECK-NEXT:                                 cl::sycl::local_accessor<double, 2> la) {
template <int EFLAG>
__global__ void k_mdppp_outer_nn(const int * __restrict__ pos,
                                 const float * __restrict__ q,
                                 const int * __restrict__ numneigh,
                                 const int * __restrict__ firstneigh,
                                 const float * __restrict__ special_lj,
                                 const float * __restrict__ special_coul,
                                 const int * __restrict__ ljd_in,
                                 const int vflag, const int ntypes,
                                 const int nlocal, const int nbor_stride,
                                 const float cut_coulsq, const float cut_ljsq,
                                 const float cut_lj_innersq,
                                 const float g_ewald, const float qqrd2e,
                                 const float denom_lj_inv,
                                 const int loop_trip) {
  __shared__ float sp_lj[4];
  __shared__ float sp_coul[4];
  __shared__ int ljd[0];
  __shared__ double la[8][0];
  const int tid = threadIdx.x;
}

// In windows, only when k_mdppp_outer_nn() is instantiated, there is an AST for k_mdppp_outer_nn template.
     //CHECK:    cgh.parallel_for(cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 1),
//CHECK-NEXT:                                           cl::sycl::range<3>(1, 1, 1)),
//CHECK-NEXT:                     [=](cl::sycl::nd_item<3> item_ct1) {
void test() {
  k_mdppp_outer_nn<1><<<1, 1, 1>>>(NULL,
                                   NULL,
                                   NULL,
                                   NULL,
                                   NULL,
                                   NULL,
                                   NULL,
                                   0, 0,
                                   0, 0,
                                   0, 0,
                                   1,
                                   0, 0,
                                   0,
                                   0);
}

__device__ void sincos_1(double x, double* sptr, double* cptr) {
//CHECK:  return [&]() {
//CHECK-NEXT:    *(sptr) = cl::sycl::sincos(
//CHECK-NEXT:        x, cl::sycl::make_ptr<double,
//CHECK-NEXT:                              cl::sycl::access::address_space::global_space>(
//CHECK-NEXT:               cptr));
//CHECK-NEXT:  }();
  return ::sincos(x, sptr, cptr);
}

__device__ void sincospi_1(double x, double* sptr, double* cptr) {
//CHECK:  return [&]() {
//CHECK-NEXT:    *(sptr) = cl::sycl::sincos(
//CHECK-NEXT:        x * DPCT_PI,
//CHECK-NEXT:        cl::sycl::make_ptr<double,
//CHECK-NEXT:                           cl::sycl::access::address_space::global_space>(
//CHECK-NEXT:            cptr));
//CHECK-NEXT:  }();
  return ::sincospi (x, sptr, cptr);
}
