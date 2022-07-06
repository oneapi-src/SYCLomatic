// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --usm-level=none -out-root %T/./thrust_rawptr_noneusm %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17 -fsized-deallocation
// RUN: FileCheck --input-file %T/thrust_rawptr_noneusm/thrust_rawptr_noneusm.dp.cpp --match-full-lines %s
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <thrust/adjacent_difference.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>

struct greater_than_zero
{
  __host__ __device__
  bool operator()(int x) const
  {
    return x > 0;
  }
  typedef int argument_type;
};

int main(){
  greater_than_zero pred;

  float *host_ptr_A;
  float *host_ptr_R;
  float *host_ptr_S;
  float *device_ptr_A;
  float *device_ptr_S;
  float *device_ptr_R;


  // replace_if
  //CHECK: host_ptr_A = (float*)std::malloc(20 * sizeof(float));
  //CHECK-NEXT: host_ptr_R = (float*)std::malloc(20 * sizeof(float));
  //CHECK-NEXT: host_ptr_S = (float*)std::malloc(20 * sizeof(float));
  //CHECK-NEXT: device_ptr_A = (float *)dpct::dpct_malloc(20 * sizeof(float));
  //CHECK-NEXT: device_ptr_S = (float *)dpct::dpct_malloc(20 * sizeof(float));
  //CHECK-NEXT: device_ptr_R = (float *)dpct::dpct_malloc(20 * sizeof(float));
  //CHECK-NEXT: host_ptr_S[0]= -1;
  //CHECK-NEXT: host_ptr_S[1]= 5;
  //CHECK-NEXT: host_ptr_S[2]= 5;
  //CHECK-NEXT: host_ptr_S[3]= -395;
  //CHECK-NEXT: dpct::dpct_memcpy(device_ptr_S, host_ptr_S, 20 * sizeof(float), dpct::host_to_device);
  //CHECK-NEXT: host_ptr_A[0]= -5;
  //CHECK-NEXT: host_ptr_A[1]= 8;
  //CHECK-NEXT: host_ptr_A[2]= 50;
  //CHECK-NEXT: host_ptr_A[3]= -395;
  //CHECK-NEXT: dpct::dpct_memcpy(device_ptr_A, host_ptr_A, 20 * sizeof(float), dpct::host_to_device);
  //CHECK-NEXT: if (dpct::is_device_ptr(device_ptr_A)) {
  //CHECK-NEXT:   dpct::replace_if(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<float>(device_ptr_A), dpct::device_pointer<float>(device_ptr_A + 10), dpct::device_pointer<float>(device_ptr_S), pred, 0);
  //CHECK-NEXT: } else {
  //CHECK-NEXT:   dpct::replace_if(oneapi::dpl::execution::seq, device_ptr_A, device_ptr_A + 10, device_ptr_S, pred, 0);
  //CHECK-NEXT: };
  //CHECK-NEXT: dpct::dpct_memcpy(host_ptr_R, device_ptr_A, 20 * sizeof(float), dpct::device_to_host);
  //CHECK-NEXT: host_ptr_A[0]= -5;
  //CHECK-NEXT: host_ptr_A[1]= 8;
  //CHECK-NEXT: host_ptr_A[2]= 50;
  //CHECK-NEXT: host_ptr_A[3]= -395;
  //CHECK-NEXT: if (dpct::is_device_ptr(host_ptr_A + 10)) {
  //CHECK-NEXT:   oneapi::dpl::replace_if(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<float>(host_ptr_A), dpct::device_pointer<float>(host_ptr_A + 10), pred, 0);
  //CHECK-NEXT: } else {
  //CHECK-NEXT:   oneapi::dpl::replace_if(oneapi::dpl::execution::seq, host_ptr_A, host_ptr_A + 10, pred, 0);
  //CHECK-NEXT: };
  //CHECK-NEXT: host_ptr_A[0]= -5;
  //CHECK-NEXT: host_ptr_A[1]= 8;
  //CHECK-NEXT: host_ptr_A[2]= 50;
  //CHECK-NEXT: host_ptr_A[3]= -395;
  //CHECK-NEXT: dpct::dpct_memcpy(device_ptr_A, host_ptr_A, 20 * sizeof(float), dpct::host_to_device);
  //CHECK-NEXT: if (dpct::is_device_ptr(device_ptr_A)) {
  //CHECK-NEXT:   oneapi::dpl::replace_if(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<float>(device_ptr_A), dpct::device_pointer<float>(device_ptr_A + 10), pred, 0);
  //CHECK-NEXT: } else {
  //CHECK-NEXT:   oneapi::dpl::replace_if(oneapi::dpl::execution::seq, device_ptr_A, device_ptr_A + 10, pred, 0);
  //CHECK-NEXT: };
  //CHECK-NEXT: dpct::dpct_memcpy(host_ptr_R, device_ptr_A, 20 * sizeof(float), dpct::device_to_host);
  //CHECK-NEXT: host_ptr_A[0]= -5;
  //CHECK-NEXT: host_ptr_A[1]= 8;
  //CHECK-NEXT: host_ptr_A[2]= 50;
  //CHECK-NEXT: host_ptr_A[3]= -395;
  //CHECK-NEXT: if (dpct::is_device_ptr(host_ptr_A + 10)) {
  //CHECK-NEXT:   dpct::replace_if(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<float>(host_ptr_A), dpct::device_pointer<float>(host_ptr_A + 10), dpct::device_pointer<float>(host_ptr_S), pred, 0);
  //CHECK-NEXT: } else {
  //CHECK-NEXT:   dpct::replace_if(oneapi::dpl::execution::seq, host_ptr_A, host_ptr_A + 10, host_ptr_S, pred, 0);
  //CHECK-NEXT: };
  host_ptr_A = (float*)std::malloc(20 * sizeof(float));
  host_ptr_R = (float*)std::malloc(20 * sizeof(float));
  host_ptr_S = (float*)std::malloc(20 * sizeof(float));
  cudaMalloc(&device_ptr_A, 20 * sizeof(float));
  cudaMalloc(&device_ptr_S, 20 * sizeof(float));
  cudaMalloc(&device_ptr_R, 20 * sizeof(float));
  host_ptr_S[0]= -1;
  host_ptr_S[1]= 5;
  host_ptr_S[2]= 5;
  host_ptr_S[3]= -395;
  cudaMemcpy(device_ptr_S, host_ptr_S, 20*sizeof(float), cudaMemcpyHostToDevice);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 50;
  host_ptr_A[3]= -395;
  cudaMemcpy(device_ptr_A, host_ptr_A, 20*sizeof(float), cudaMemcpyHostToDevice);
  thrust::replace_if(thrust::device, device_ptr_A, device_ptr_A+10, device_ptr_S, pred, 0);
  cudaMemcpy(host_ptr_R, device_ptr_A, 20*sizeof(float), cudaMemcpyDeviceToHost);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 50;
  host_ptr_A[3]= -395;
  thrust::replace_if(host_ptr_A, host_ptr_A+10, pred, 0);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 50;
  host_ptr_A[3]= -395;
  cudaMemcpy(device_ptr_A, host_ptr_A, 20*sizeof(float), cudaMemcpyHostToDevice);
  thrust::replace_if(thrust::device, device_ptr_A, device_ptr_A+10, pred, 0);
  cudaMemcpy(host_ptr_R, device_ptr_A, 20*sizeof(float), cudaMemcpyDeviceToHost);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 50;
  host_ptr_A[3]= -395;
  thrust::replace_if(host_ptr_A, host_ptr_A+10, host_ptr_S, pred, 0);

  // remove_if
  //CHECK: host_ptr_A = (float*)std::malloc(20 * sizeof(float));
  //CHECK-NEXT: host_ptr_R = (float*)std::malloc(20 * sizeof(float));
  //CHECK-NEXT: host_ptr_S = (float*)std::malloc(20 * sizeof(float));
  //CHECK-NEXT: device_ptr_A = (float *)dpct::dpct_malloc(20 * sizeof(float));
  //CHECK-NEXT: device_ptr_S = (float *)dpct::dpct_malloc(20 * sizeof(float));
  //CHECK-NEXT: device_ptr_R = (float *)dpct::dpct_malloc(20 * sizeof(float));
  //CHECK-NEXT: host_ptr_S[0]= -1;
  //CHECK-NEXT: host_ptr_S[1]= 5;
  //CHECK-NEXT: host_ptr_S[2]= 5;
  //CHECK-NEXT: host_ptr_S[3]= -395;
  //CHECK-NEXT: dpct::dpct_memcpy(device_ptr_S, host_ptr_S, 20 * sizeof(float), dpct::host_to_device);
  //CHECK-NEXT: host_ptr_A[0]= -5;
  //CHECK-NEXT: host_ptr_A[1]= 8;
  //CHECK-NEXT: host_ptr_A[2]= 50;
  //CHECK-NEXT: host_ptr_A[3]= -395;
  //CHECK-NEXT: dpct::dpct_memcpy(device_ptr_A, host_ptr_A, 20 * sizeof(float), dpct::host_to_device);
  //CHECK-NEXT: if (dpct::is_device_ptr(device_ptr_A)) {
  //CHECK-NEXT:   oneapi::dpl::remove_if(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<float>(device_ptr_A), dpct::device_pointer<float>(device_ptr_A + 10), pred);
  //CHECK-NEXT: } else {
  //CHECK-NEXT:   oneapi::dpl::remove_if(oneapi::dpl::execution::seq, device_ptr_A, device_ptr_A + 10, pred);
  //CHECK-NEXT: };
  //CHECK-NEXT: dpct::dpct_memcpy(host_ptr_R, device_ptr_A, 20 * sizeof(float), dpct::device_to_host);
  //CHECK-NEXT: host_ptr_A[0]= -5;
  //CHECK-NEXT: host_ptr_A[1]= 8;
  //CHECK-NEXT: host_ptr_A[2]= 50;
  //CHECK-NEXT: host_ptr_A[3]= -395;
  //CHECK-NEXT: if (dpct::is_device_ptr(host_ptr_A + 10)) {
  //CHECK-NEXT:   oneapi::dpl::remove_if(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<float>(host_ptr_A), dpct::device_pointer<float>(host_ptr_A + 10), pred);
  //CHECK-NEXT: } else {
  //CHECK-NEXT:   oneapi::dpl::remove_if(oneapi::dpl::execution::seq, host_ptr_A, host_ptr_A + 10, pred);
  //CHECK-NEXT: };
  host_ptr_A = (float*)std::malloc(20 * sizeof(float));
  host_ptr_R = (float*)std::malloc(20 * sizeof(float));
  host_ptr_S = (float*)std::malloc(20 * sizeof(float));
  cudaMalloc(&device_ptr_A, 20 * sizeof(float));
  cudaMalloc(&device_ptr_S, 20 * sizeof(float));
  cudaMalloc(&device_ptr_R, 20 * sizeof(float));
  host_ptr_S[0]= -1;
  host_ptr_S[1]= 5;
  host_ptr_S[2]= 5;
  host_ptr_S[3]= -395;
  cudaMemcpy(device_ptr_S, host_ptr_S, 20*sizeof(float), cudaMemcpyHostToDevice);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 50;
  host_ptr_A[3]= -395;
  cudaMemcpy(device_ptr_A, host_ptr_A, 20*sizeof(float), cudaMemcpyHostToDevice);
  thrust::remove_if(thrust::device, device_ptr_A, device_ptr_A+10, pred);
  cudaMemcpy(host_ptr_R, device_ptr_A, 20*sizeof(float), cudaMemcpyDeviceToHost);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 50;
  host_ptr_A[3]= -395;
  thrust::remove_if(host_ptr_A, host_ptr_A+10, pred);
  // // error: redefinition of 'KernelInfo<oneapi::dpl::__par_backend_hetero::__pslk<dpct::RemoveIf1>>'
  // host_ptr_A[0]= -5;
  // host_ptr_A[1]= 8;
  // host_ptr_A[2]= 50;
  // host_ptr_A[3]= -395;
  // cudaMemcpy(device_ptr_A, host_ptr_A, 20*sizeof(float), cudaMemcpyHostToDevice);
  // thrust::remove_if(thrust::device, device_ptr_A, device_ptr_A+10, device_ptr_S, pred);
  // cudaMemcpy(host_ptr_R, device_ptr_A, 20*sizeof(float), cudaMemcpyDeviceToHost);
  // Report::check("remove_if", host_ptr_R[0], -5);
  // Report::check("remove_if", host_ptr_R[1], -395);
  // host_ptr_A[0]= -5;
  // host_ptr_A[1]= 8;
  // host_ptr_A[2]= 50;
  // host_ptr_A[3]= -395;
  // cudaMemcpy(device_ptr_A, host_ptr_A, 20*sizeof(float), cudaMemcpyHostToDevice);
  // thrust::remove_if(thrust::device, device_ptr_A, device_ptr_A+10, device_ptr_S, pred);
  // cudaMemcpy(host_ptr_R, device_ptr_A, 20*sizeof(float), cudaMemcpyDeviceToHost);
  // Report::check("remove_if", host_ptr_R[0], -5);
  // Report::check("remove_if", host_ptr_R[1], -395);

  // remove_copy_if
  //CHECK: host_ptr_A = (float*)std::malloc(20 * sizeof(float));
  //CHECK-NEXT: host_ptr_R = (float*)std::malloc(20 * sizeof(float));
  //CHECK-NEXT: host_ptr_S = (float*)std::malloc(20 * sizeof(float));
  //CHECK-NEXT: device_ptr_A = (float *)dpct::dpct_malloc(20 * sizeof(float));
  //CHECK-NEXT: device_ptr_S = (float *)dpct::dpct_malloc(20 * sizeof(float));
  //CHECK-NEXT: device_ptr_R = (float *)dpct::dpct_malloc(20 * sizeof(float));
  //CHECK-NEXT: host_ptr_S[0]= -1;
  //CHECK-NEXT: host_ptr_S[1]= 5;
  //CHECK-NEXT: host_ptr_S[2]= 5;
  //CHECK-NEXT: host_ptr_S[3]= -395;
  //CHECK-NEXT: dpct::dpct_memcpy(device_ptr_S, host_ptr_S, 20 * sizeof(float), dpct::host_to_device);
  //CHECK-NEXT: host_ptr_A[0]= -5;
  //CHECK-NEXT: host_ptr_A[1]= 8;
  //CHECK-NEXT: host_ptr_A[2]= 50;
  //CHECK-NEXT: host_ptr_A[3]= -395;
  //CHECK-NEXT: dpct::dpct_memcpy(device_ptr_A, host_ptr_A, 20 * sizeof(float), dpct::host_to_device);
  //CHECK-NEXT: if (dpct::is_device_ptr(device_ptr_A)) {
  //CHECK-NEXT:   oneapi::dpl::remove_copy_if(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<float>(device_ptr_A), dpct::device_pointer<float>(device_ptr_A + 10), dpct::device_pointer<float>(device_ptr_R), pred);
  //CHECK-NEXT: } else {
  //CHECK-NEXT:   oneapi::dpl::remove_copy_if(oneapi::dpl::execution::seq, device_ptr_A, device_ptr_A + 10, device_ptr_R, pred);
  //CHECK-NEXT: };
  //CHECK-NEXT: dpct::dpct_memcpy(host_ptr_R, device_ptr_R, 20 * sizeof(float), dpct::device_to_host);
  //CHECK-NEXT: host_ptr_A[0]= -5;
  //CHECK-NEXT: host_ptr_A[1]= 8;
  //CHECK-NEXT: host_ptr_A[2]= 50;
  //CHECK-NEXT: host_ptr_A[3]= -395;
  //CHECK-NEXT: if (dpct::is_device_ptr(host_ptr_A + 10)) {
  //CHECK-NEXT:   oneapi::dpl::remove_copy_if(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<float>(host_ptr_A), dpct::device_pointer<float>(host_ptr_A + 10), dpct::device_pointer<float>(host_ptr_R), pred);
  //CHECK-NEXT: } else {
  //CHECK-NEXT:   oneapi::dpl::remove_copy_if(oneapi::dpl::execution::seq, host_ptr_A, host_ptr_A + 10, host_ptr_R, pred);
  //CHECK-NEXT: };
  //CHECK-NEXT: host_ptr_A[0]= -5;
  //CHECK-NEXT: host_ptr_A[1]= 8;
  //CHECK-NEXT: host_ptr_A[2]= 50;
  //CHECK-NEXT: host_ptr_A[3]= -395;
  //CHECK-NEXT: dpct::dpct_memcpy(device_ptr_A, host_ptr_A, 20 * sizeof(float), dpct::host_to_device);
  //CHECK-NEXT: if (dpct::is_device_ptr(device_ptr_A)) {
  //CHECK-NEXT:   dpct::remove_copy_if(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<float>(device_ptr_A), dpct::device_pointer<float>(device_ptr_A + 10), dpct::device_pointer<float>(device_ptr_S), dpct::device_pointer<float>(device_ptr_R), pred);
  //CHECK-NEXT: } else {
  //CHECK-NEXT:   dpct::remove_copy_if(oneapi::dpl::execution::seq, device_ptr_A, device_ptr_A + 10, device_ptr_S, device_ptr_R, pred);
  //CHECK-NEXT: };
  //CHECK-NEXT: dpct::dpct_memcpy(host_ptr_R, device_ptr_R, 20 * sizeof(float), dpct::device_to_host);
  //CHECK-NEXT: host_ptr_A[0]= -5;
  //CHECK-NEXT: host_ptr_A[1]= 8;
  //CHECK-NEXT: host_ptr_A[2]= 50;
  //CHECK-NEXT: host_ptr_A[3]= -395;
  //CHECK-NEXT: if (dpct::is_device_ptr(host_ptr_A + 10)) {
  //CHECK-NEXT:   dpct::remove_copy_if(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<float>(host_ptr_A), dpct::device_pointer<float>(host_ptr_A + 10), dpct::device_pointer<float>(host_ptr_S), dpct::device_pointer<float>(host_ptr_R), pred);
  //CHECK-NEXT: } else {
  //CHECK-NEXT:   dpct::remove_copy_if(oneapi::dpl::execution::seq, host_ptr_A, host_ptr_A + 10, host_ptr_S, host_ptr_R, pred);
  //CHECK-NEXT: };
  host_ptr_A = (float*)std::malloc(20 * sizeof(float));
  host_ptr_R = (float*)std::malloc(20 * sizeof(float));
  host_ptr_S = (float*)std::malloc(20 * sizeof(float));
  cudaMalloc(&device_ptr_A, 20 * sizeof(float));
  cudaMalloc(&device_ptr_S, 20 * sizeof(float));
  cudaMalloc(&device_ptr_R, 20 * sizeof(float));
  host_ptr_S[0]= -1;
  host_ptr_S[1]= 5;
  host_ptr_S[2]= 5;
  host_ptr_S[3]= -395;
  cudaMemcpy(device_ptr_S, host_ptr_S, 20*sizeof(float), cudaMemcpyHostToDevice);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 50;
  host_ptr_A[3]= -395;
  cudaMemcpy(device_ptr_A, host_ptr_A, 20*sizeof(float), cudaMemcpyHostToDevice);
  thrust::remove_copy_if(thrust::device, device_ptr_A, device_ptr_A+10, device_ptr_R, pred);
  cudaMemcpy(host_ptr_R, device_ptr_R, 20*sizeof(float), cudaMemcpyDeviceToHost);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 50;
  host_ptr_A[3]= -395;
  thrust::remove_copy_if(host_ptr_A, host_ptr_A+10, host_ptr_R, pred);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 50;
  host_ptr_A[3]= -395;
  cudaMemcpy(device_ptr_A, host_ptr_A, 20*sizeof(float), cudaMemcpyHostToDevice);
  thrust::remove_copy_if(thrust::device, device_ptr_A, device_ptr_A+10, device_ptr_S, device_ptr_R, pred);
  cudaMemcpy(host_ptr_R, device_ptr_R, 20*sizeof(float), cudaMemcpyDeviceToHost);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 50;
  host_ptr_A[3]= -395;
  thrust::remove_copy_if(host_ptr_A, host_ptr_A+10, host_ptr_S, host_ptr_R, pred);

  // inclusive_scan_ptr
  //CHECK: host_ptr_A = (float*)std::malloc(20 * sizeof(float));
  //CHECK-NEXT: host_ptr_R = (float*)std::malloc(20 * sizeof(float));
  //CHECK-NEXT: host_ptr_S = (float*)std::malloc(20 * sizeof(float));
  //CHECK-NEXT: device_ptr_A = (float *)dpct::dpct_malloc(20 * sizeof(float));
  //CHECK-NEXT: device_ptr_S = (float *)dpct::dpct_malloc(20 * sizeof(float));
  //CHECK-NEXT: device_ptr_R = (float *)dpct::dpct_malloc(20 * sizeof(float));
  //CHECK-NEXT: host_ptr_A[0]= -5;
  //CHECK-NEXT: host_ptr_A[1]= 8;
  //CHECK-NEXT: host_ptr_A[2]= 396;
  //CHECK-NEXT: host_ptr_A[3]= -395;
  //CHECK-NEXT: dpct::dpct_memcpy(device_ptr_A, host_ptr_A, 20 * sizeof(float), dpct::host_to_device);
  //CHECK-NEXT: if (dpct::is_device_ptr(device_ptr_A)) {
  //CHECK-NEXT:   oneapi::dpl::inclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<float>(device_ptr_A), dpct::device_pointer<float>(device_ptr_A + 10), dpct::device_pointer<float>(device_ptr_R));
  //CHECK-NEXT: } else {
  //CHECK-NEXT:   oneapi::dpl::inclusive_scan(oneapi::dpl::execution::seq, device_ptr_A, device_ptr_A + 10, device_ptr_R);
  //CHECK-NEXT: };
  //CHECK-NEXT: dpct::dpct_memcpy(host_ptr_R, device_ptr_R, 20 * sizeof(float), dpct::device_to_host);
  //CHECK-NEXT: host_ptr_A[0]= -5;
  //CHECK-NEXT: host_ptr_A[1]= 8;
  //CHECK-NEXT: host_ptr_A[2]= 396;
  //CHECK-NEXT: host_ptr_A[3]= -395;
  //CHECK-NEXT: if (dpct::is_device_ptr(host_ptr_A + 10)) {
  //CHECK-NEXT:   oneapi::dpl::inclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<float>(host_ptr_A), dpct::device_pointer<float>(host_ptr_A + 10), dpct::device_pointer<float>(host_ptr_R), std::plus<float>());
  //CHECK-NEXT: } else {
  //CHECK-NEXT:   oneapi::dpl::inclusive_scan(oneapi::dpl::execution::seq, host_ptr_A, host_ptr_A + 10, host_ptr_R, std::plus<float>());
  //CHECK-NEXT: };
  //CHECK-NEXT: host_ptr_A[0]= -5;
  //CHECK-NEXT: host_ptr_A[1]= 8;
  //CHECK-NEXT: host_ptr_A[2]= 396;
  //CHECK-NEXT: host_ptr_A[3]= -395;
  //CHECK-NEXT: dpct::dpct_memcpy(device_ptr_A, host_ptr_A, 20 * sizeof(float), dpct::host_to_device);
  //CHECK-NEXT: if (dpct::is_device_ptr(device_ptr_A)) {
  //CHECK-NEXT:   oneapi::dpl::inclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<float>(device_ptr_A), dpct::device_pointer<float>(device_ptr_A + 10), dpct::device_pointer<float>(device_ptr_R), std::plus<float>());
  //CHECK-NEXT: } else {
  //CHECK-NEXT:   oneapi::dpl::inclusive_scan(oneapi::dpl::execution::seq, device_ptr_A, device_ptr_A + 10, device_ptr_R, std::plus<float>());
  //CHECK-NEXT: };
  //CHECK-NEXT: dpct::dpct_memcpy(host_ptr_R, device_ptr_R, 20 * sizeof(float), dpct::device_to_host);
  //CHECK-NEXT: host_ptr_A[0]= -5;
  //CHECK-NEXT: host_ptr_A[1]= 8;
  //CHECK-NEXT: host_ptr_A[2]= 396;
  //CHECK-NEXT: host_ptr_A[3]= -395;
  //CHECK-NEXT: if (dpct::is_device_ptr(host_ptr_A + 10)) {
  //CHECK-NEXT:   oneapi::dpl::inclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<float>(host_ptr_A), dpct::device_pointer<float>(host_ptr_A + 10), dpct::device_pointer<float>(host_ptr_R));
  //CHECK-NEXT: } else {
  //CHECK-NEXT:   oneapi::dpl::inclusive_scan(oneapi::dpl::execution::seq, host_ptr_A, host_ptr_A + 10, host_ptr_R);
  //CHECK-NEXT: };
  host_ptr_A = (float*)std::malloc(20 * sizeof(float));
  host_ptr_R = (float*)std::malloc(20 * sizeof(float));
  host_ptr_S = (float*)std::malloc(20 * sizeof(float));
  cudaMalloc(&device_ptr_A, 20 * sizeof(float));
  cudaMalloc(&device_ptr_S, 20 * sizeof(float));
  cudaMalloc(&device_ptr_R, 20 * sizeof(float));
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 396;
  host_ptr_A[3]= -395;
  cudaMemcpy(device_ptr_A, host_ptr_A, 20*sizeof(float), cudaMemcpyHostToDevice);
  thrust::inclusive_scan(thrust::device, device_ptr_A, device_ptr_A+10, device_ptr_R);
  cudaMemcpy(host_ptr_R, device_ptr_R, 20*sizeof(float), cudaMemcpyDeviceToHost);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 396;
  host_ptr_A[3]= -395;
  thrust::inclusive_scan(host_ptr_A, host_ptr_A+10, host_ptr_R, thrust::plus<float>());
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 396;
  host_ptr_A[3]= -395;
  cudaMemcpy(device_ptr_A, host_ptr_A, 20*sizeof(float), cudaMemcpyHostToDevice);
  thrust::inclusive_scan(thrust::device, device_ptr_A, device_ptr_A+10, device_ptr_R, thrust::plus<float>());
  cudaMemcpy(host_ptr_R, device_ptr_R, 20*sizeof(float), cudaMemcpyDeviceToHost);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 396;
  host_ptr_A[3]= -395;
  thrust::inclusive_scan(host_ptr_A, host_ptr_A+10, host_ptr_R);

  // adjacent_difference
  //CHECK: host_ptr_A = (float*)std::malloc(20 * sizeof(float));
  //CHECK-NEXT: host_ptr_R = (float*)std::malloc(20 * sizeof(float));
  //CHECK-NEXT: host_ptr_S = (float*)std::malloc(20 * sizeof(float));
  //CHECK-NEXT: device_ptr_A = (float *)dpct::dpct_malloc(20 * sizeof(float));
  //CHECK-NEXT: device_ptr_S = (float *)dpct::dpct_malloc(20 * sizeof(float));
  //CHECK-NEXT: device_ptr_R = (float *)dpct::dpct_malloc(20 * sizeof(float));
  //CHECK-NEXT: host_ptr_A[0]= -5;
  //CHECK-NEXT: host_ptr_A[1]= 8;
  //CHECK-NEXT: host_ptr_A[2]= 396;
  //CHECK-NEXT: host_ptr_A[3]= -395;
  //CHECK-NEXT: dpct::dpct_memcpy(device_ptr_A, host_ptr_A, 20 * sizeof(float), dpct::host_to_device);
  //CHECK-NEXT: if (dpct::is_device_ptr(device_ptr_A)) {
  //CHECK-NEXT:   oneapi::dpl::adjacent_difference(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<float>(device_ptr_A), dpct::device_pointer<float>(device_ptr_A + 10), dpct::device_pointer<float>(device_ptr_R));
  //CHECK-NEXT: } else {
  //CHECK-NEXT:   oneapi::dpl::adjacent_difference(oneapi::dpl::execution::seq, device_ptr_A, device_ptr_A + 10, device_ptr_R);
  //CHECK-NEXT: };
  //CHECK-NEXT: dpct::dpct_memcpy(host_ptr_R, device_ptr_R, 20 * sizeof(float), dpct::device_to_host);
  //CHECK-NEXT: host_ptr_A[0]= -5;
  //CHECK-NEXT: host_ptr_A[1]= 8;
  //CHECK-NEXT: host_ptr_A[2]= 396;
  //CHECK-NEXT: host_ptr_A[3]= -395;
  //CHECK-NEXT: if (dpct::is_device_ptr(host_ptr_A + 10)) {
  //CHECK-NEXT:   oneapi::dpl::adjacent_difference(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<float>(host_ptr_A), dpct::device_pointer<float>(host_ptr_A + 10), dpct::device_pointer<float>(host_ptr_R), std::minus<float>());
  //CHECK-NEXT: } else {
  //CHECK-NEXT:   oneapi::dpl::adjacent_difference(oneapi::dpl::execution::seq, host_ptr_A, host_ptr_A + 10, host_ptr_R, std::minus<float>());
  //CHECK-NEXT: };
  //CHECK-NEXT: host_ptr_A[0]= -5;
  //CHECK-NEXT: host_ptr_A[1]= 8;
  //CHECK-NEXT: host_ptr_A[2]= 396;
  //CHECK-NEXT: host_ptr_A[3]= -395;
  //CHECK-NEXT: dpct::dpct_memcpy(device_ptr_A, host_ptr_A, 20 * sizeof(float), dpct::host_to_device);
  //CHECK-NEXT: if (dpct::is_device_ptr(device_ptr_A)) {
  //CHECK-NEXT:   oneapi::dpl::adjacent_difference(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<float>(device_ptr_A), dpct::device_pointer<float>(device_ptr_A + 10), dpct::device_pointer<float>(device_ptr_R), std::minus<float>());
  //CHECK-NEXT: } else {
  //CHECK-NEXT:   oneapi::dpl::adjacent_difference(oneapi::dpl::execution::seq, device_ptr_A, device_ptr_A + 10, device_ptr_R, std::minus<float>());
  //CHECK-NEXT: };
  //CHECK-NEXT: dpct::dpct_memcpy(host_ptr_R, device_ptr_R, 20 * sizeof(float), dpct::device_to_host);
  //CHECK-NEXT: host_ptr_A[0]= -5;
  //CHECK-NEXT: host_ptr_A[1]= 8;
  //CHECK-NEXT: host_ptr_A[2]= 396;
  //CHECK-NEXT: host_ptr_A[3]= -395;
  //CHECK-NEXT: if (dpct::is_device_ptr(host_ptr_A + 10)) {
  //CHECK-NEXT:   oneapi::dpl::adjacent_difference(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<float>(host_ptr_A), dpct::device_pointer<float>(host_ptr_A + 10), dpct::device_pointer<float>(host_ptr_R));
  //CHECK-NEXT: } else {
  //CHECK-NEXT:   oneapi::dpl::adjacent_difference(oneapi::dpl::execution::seq, host_ptr_A, host_ptr_A + 10, host_ptr_R);
  //CHECK-NEXT: };
  host_ptr_A = (float*)std::malloc(20 * sizeof(float));
  host_ptr_R = (float*)std::malloc(20 * sizeof(float));
  host_ptr_S = (float*)std::malloc(20 * sizeof(float));
  cudaMalloc(&device_ptr_A, 20 * sizeof(float));
  cudaMalloc(&device_ptr_S, 20 * sizeof(float));
  cudaMalloc(&device_ptr_R, 20 * sizeof(float));
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 396;
  host_ptr_A[3]= -395;
  cudaMemcpy(device_ptr_A, host_ptr_A, 20*sizeof(float), cudaMemcpyHostToDevice);
  thrust::adjacent_difference(thrust::device, device_ptr_A, device_ptr_A+10, device_ptr_R);
  cudaMemcpy(host_ptr_R, device_ptr_R, 20*sizeof(float), cudaMemcpyDeviceToHost);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 396;
  host_ptr_A[3]= -395;
  thrust::adjacent_difference(host_ptr_A, host_ptr_A+10, host_ptr_R, thrust::minus<float>());
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 396;
  host_ptr_A[3]= -395;
  cudaMemcpy(device_ptr_A, host_ptr_A, 20*sizeof(float), cudaMemcpyHostToDevice);
  thrust::adjacent_difference(thrust::device, device_ptr_A, device_ptr_A+10, device_ptr_R, thrust::minus<float>());
  cudaMemcpy(host_ptr_R, device_ptr_R, 20*sizeof(float), cudaMemcpyDeviceToHost);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 396;
  host_ptr_A[3]= -395;
  thrust::adjacent_difference(host_ptr_A, host_ptr_A+10, host_ptr_R);

  // gather
  //CHECK: int *host_ptr_M;
  //CHECK-NEXT: int *device_ptr_M;
  //CHECK-NEXT: host_ptr_A = (float*)std::malloc(20 * sizeof(float));
  //CHECK-NEXT: host_ptr_R = (float*)std::malloc(20 * sizeof(float));
  //CHECK-NEXT: host_ptr_S = (float*)std::malloc(20 * sizeof(float));
  //CHECK-NEXT: host_ptr_M = (int*)std::malloc(20 * sizeof(int));
  //CHECK-NEXT: device_ptr_A = (float *)dpct::dpct_malloc(20 * sizeof(float));
  //CHECK-NEXT: device_ptr_S = (float *)dpct::dpct_malloc(20 * sizeof(float));
  //CHECK-NEXT: device_ptr_R = (float *)dpct::dpct_malloc(20 * sizeof(float));
  //CHECK-NEXT: device_ptr_M = (int *)dpct::dpct_malloc(20 * sizeof(int));
  //CHECK-NEXT: host_ptr_M[0]= 3;
  //CHECK-NEXT: host_ptr_M[1]= 2;
  //CHECK-NEXT: host_ptr_M[2]= 1;
  //CHECK-NEXT: host_ptr_M[3]= 0;
  //CHECK-NEXT: dpct::dpct_memcpy(device_ptr_M, host_ptr_M, 20 * sizeof(int), dpct::host_to_device);
  //CHECK-NEXT: host_ptr_A[0]= -5;
  //CHECK-NEXT: host_ptr_A[1]= 8;
  //CHECK-NEXT: host_ptr_A[2]= 396;
  //CHECK-NEXT: host_ptr_A[3]= -395;
  //CHECK-NEXT: dpct::dpct_memcpy(device_ptr_A, host_ptr_A, 20 * sizeof(float), dpct::host_to_device);
  //CHECK-NEXT: if (dpct::is_device_ptr(device_ptr_M)) {
  //CHECK-NEXT:   dpct::gather(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(device_ptr_M), dpct::device_pointer<int>(device_ptr_M + 4), dpct::device_pointer<float>(device_ptr_A), dpct::device_pointer<float>(device_ptr_R));
  //CHECK-NEXT: } else {
  //CHECK-NEXT:   dpct::gather(oneapi::dpl::execution::seq, device_ptr_M, device_ptr_M + 4, device_ptr_A, device_ptr_R);
  //CHECK-NEXT: };
  //CHECK-NEXT: dpct::dpct_memcpy(host_ptr_R, device_ptr_R, 20 * sizeof(float), dpct::device_to_host);
  //CHECK-NEXT: host_ptr_M[0]= 3;
  //CHECK-NEXT: host_ptr_M[1]= 2;
  //CHECK-NEXT: host_ptr_M[2]= 1;
  //CHECK-NEXT: host_ptr_M[3]= 0;
  //CHECK-NEXT: dpct::dpct_memcpy(device_ptr_M, host_ptr_M, 20 * sizeof(float), dpct::host_to_device);
  //CHECK-NEXT: host_ptr_A[0]= -5;
  //CHECK-NEXT: host_ptr_A[1]= 8;
  //CHECK-NEXT: host_ptr_A[2]= 396;
  //CHECK-NEXT: host_ptr_A[3]= -395;
  //CHECK-NEXT: if (dpct::is_device_ptr(host_ptr_M + 4)) {
  //CHECK-NEXT:   dpct::gather(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(host_ptr_M), dpct::device_pointer<int>(host_ptr_M + 4), dpct::device_pointer<float>(host_ptr_A), dpct::device_pointer<float>(host_ptr_R));
  //CHECK-NEXT: } else {
  //CHECK-NEXT:   dpct::gather(oneapi::dpl::execution::seq, host_ptr_M, host_ptr_M + 4, host_ptr_A, host_ptr_R);
  //CHECK-NEXT: };
  int *host_ptr_M;
  int *device_ptr_M;
  host_ptr_A = (float*)std::malloc(20 * sizeof(float));
  host_ptr_R = (float*)std::malloc(20 * sizeof(float));
  host_ptr_S = (float*)std::malloc(20 * sizeof(float));
  host_ptr_M = (int*)std::malloc(20 * sizeof(int));
  cudaMalloc(&device_ptr_A, 20 * sizeof(float));
  cudaMalloc(&device_ptr_S, 20 * sizeof(float));
  cudaMalloc(&device_ptr_R, 20 * sizeof(float));
  cudaMalloc(&device_ptr_M, 20 * sizeof(int));
  host_ptr_M[0]= 3;
  host_ptr_M[1]= 2;
  host_ptr_M[2]= 1;
  host_ptr_M[3]= 0;
  cudaMemcpy(device_ptr_M, host_ptr_M, 20*sizeof(int), cudaMemcpyHostToDevice);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 396;
  host_ptr_A[3]= -395;
  cudaMemcpy(device_ptr_A, host_ptr_A, 20*sizeof(float), cudaMemcpyHostToDevice);
  thrust::gather(thrust::device, device_ptr_M, device_ptr_M + 4, device_ptr_A, device_ptr_R);
  cudaMemcpy(host_ptr_R, device_ptr_R, 20*sizeof(float), cudaMemcpyDeviceToHost);
  host_ptr_M[0]= 3;
  host_ptr_M[1]= 2;
  host_ptr_M[2]= 1;
  host_ptr_M[3]= 0;
  cudaMemcpy(device_ptr_M, host_ptr_M, 20*sizeof(float), cudaMemcpyHostToDevice);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 396;
  host_ptr_A[3]= -395;
  thrust::gather(host_ptr_M, host_ptr_M + 4, host_ptr_A, host_ptr_R);

  // scatter
  //CHECK: host_ptr_A = (float*)std::malloc(20 * sizeof(float));
  //CHECK-NEXT: host_ptr_R = (float*)std::malloc(20 * sizeof(float));
  //CHECK-NEXT: host_ptr_S = (float*)std::malloc(20 * sizeof(float));
  //CHECK-NEXT: host_ptr_M = (int*)std::malloc(20 * sizeof(int));
  //CHECK-NEXT: device_ptr_A = (float *)dpct::dpct_malloc(20 * sizeof(float));
  //CHECK-NEXT: device_ptr_S = (float *)dpct::dpct_malloc(20 * sizeof(float));
  //CHECK-NEXT: device_ptr_R = (float *)dpct::dpct_malloc(20 * sizeof(float));
  //CHECK-NEXT: device_ptr_M = (int *)dpct::dpct_malloc(20 * sizeof(int));
  //CHECK-NEXT: host_ptr_M[0]= 2;
  //CHECK-NEXT: host_ptr_M[1]= 3;
  //CHECK-NEXT: host_ptr_M[2]= 1;
  //CHECK-NEXT: host_ptr_M[3]= 0;
  //CHECK-NEXT: dpct::dpct_memcpy(device_ptr_M, host_ptr_M, 20 * sizeof(int), dpct::host_to_device);
  //CHECK-NEXT: host_ptr_A[0]= -5;
  //CHECK-NEXT: host_ptr_A[1]= 8;
  //CHECK-NEXT: host_ptr_A[2]= 396;
  //CHECK-NEXT: host_ptr_A[3]= 3;
  //CHECK-NEXT: dpct::dpct_memcpy(device_ptr_A, host_ptr_A, 20 * sizeof(float), dpct::host_to_device);
  //CHECK-NEXT: if (dpct::is_device_ptr(device_ptr_A)) {
  //CHECK-NEXT:   dpct::scatter(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<float>(device_ptr_A), dpct::device_pointer<float>(device_ptr_A + 4), dpct::device_pointer<int>(device_ptr_M), dpct::device_pointer<float>(device_ptr_R));
  //CHECK-NEXT: } else {
  //CHECK-NEXT:   dpct::scatter(oneapi::dpl::execution::seq, device_ptr_A, device_ptr_A + 4, device_ptr_M, device_ptr_R);
  //CHECK-NEXT: };
  //CHECK-NEXT: dpct::dpct_memcpy(host_ptr_R, device_ptr_R, 20 * sizeof(float), dpct::device_to_host);
  //CHECK-NEXT: host_ptr_M[0]= 2;
  //CHECK-NEXT: host_ptr_M[1]= 3;
  //CHECK-NEXT: host_ptr_M[2]= 1;
  //CHECK-NEXT: host_ptr_M[3]= 0;
  //CHECK-NEXT: host_ptr_A[0]= -5;
  //CHECK-NEXT: host_ptr_A[1]= 8;
  //CHECK-NEXT: host_ptr_A[2]= 396;
  //CHECK-NEXT: host_ptr_A[3]= 3;
  //CHECK-NEXT: if (dpct::is_device_ptr(host_ptr_A + 4)) {
  //CHECK-NEXT:   dpct::scatter(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<float>(host_ptr_A), dpct::device_pointer<float>(host_ptr_A + 4), dpct::device_pointer<int>(host_ptr_M), dpct::device_pointer<float>(host_ptr_R));
  //CHECK-NEXT: } else {
  //CHECK-NEXT:   dpct::scatter(oneapi::dpl::execution::seq, host_ptr_A, host_ptr_A + 4, host_ptr_M, host_ptr_R);
  //CHECK-NEXT: };
  host_ptr_A = (float*)std::malloc(20 * sizeof(float));
  host_ptr_R = (float*)std::malloc(20 * sizeof(float));
  host_ptr_S = (float*)std::malloc(20 * sizeof(float));
  host_ptr_M = (int*)std::malloc(20 * sizeof(int));
  cudaMalloc(&device_ptr_A, 20 * sizeof(float));
  cudaMalloc(&device_ptr_S, 20 * sizeof(float));
  cudaMalloc(&device_ptr_R, 20 * sizeof(float));
  cudaMalloc(&device_ptr_M, 20 * sizeof(int));
  host_ptr_M[0]= 2;
  host_ptr_M[1]= 3;
  host_ptr_M[2]= 1;
  host_ptr_M[3]= 0;
  cudaMemcpy(device_ptr_M, host_ptr_M, 20*sizeof(int), cudaMemcpyHostToDevice);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 396;
  host_ptr_A[3]= 3;
  cudaMemcpy(device_ptr_A, host_ptr_A, 20*sizeof(float), cudaMemcpyHostToDevice);
  thrust::scatter(thrust::device, device_ptr_A, device_ptr_A + 4, device_ptr_M, device_ptr_R);
  cudaMemcpy(host_ptr_R, device_ptr_R, 20*sizeof(float), cudaMemcpyDeviceToHost);
  host_ptr_M[0]= 2;
  host_ptr_M[1]= 3;
  host_ptr_M[2]= 1;
  host_ptr_M[3]= 0;
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 396;
  host_ptr_A[3]= 3;
  thrust::scatter(host_ptr_A, host_ptr_A + 4, host_ptr_M, host_ptr_R);

  // unique_by_key_copy
  //CHECK: float *host_ptr_K;
  //CHECK-NEXT: float *device_ptr_K;
  //CHECK-NEXT: oneapi::dpl::equal_to<float> pred2;
  //CHECK-NEXT: host_ptr_A = (float*)std::malloc(20 * sizeof(float));
  //CHECK-NEXT: host_ptr_K = (float*)std::malloc(20 * sizeof(float));
  //CHECK-NEXT: host_ptr_R = (float*)std::malloc(20 * sizeof(float));
  //CHECK-NEXT: host_ptr_S = (float*)std::malloc(20 * sizeof(float));
  //CHECK-NEXT: device_ptr_A = (float *)dpct::dpct_malloc(20 * sizeof(float));
  //CHECK-NEXT: device_ptr_K = (float *)dpct::dpct_malloc(20 * sizeof(float));
  //CHECK-NEXT: device_ptr_S = (float *)dpct::dpct_malloc(20 * sizeof(float));
  //CHECK-NEXT: device_ptr_R = (float *)dpct::dpct_malloc(20 * sizeof(float));
  //CHECK-NEXT: host_ptr_K[0]= 1;
  //CHECK-NEXT: host_ptr_K[1]= 2;
  //CHECK-NEXT: host_ptr_K[2]= 2;
  //CHECK-NEXT: host_ptr_K[3]= 1;
  //CHECK-NEXT: dpct::dpct_memcpy(device_ptr_K, host_ptr_K, 20 * sizeof(float), dpct::host_to_device);
  //CHECK-NEXT: host_ptr_A[0]= -5;
  //CHECK-NEXT: host_ptr_A[1]= 8;
  //CHECK-NEXT: host_ptr_A[2]= 396;
  //CHECK-NEXT: host_ptr_A[3]= -395;
  //CHECK-NEXT: dpct::dpct_memcpy(device_ptr_A, host_ptr_A, 20 * sizeof(float), dpct::host_to_device);
  //CHECK-NEXT: if (dpct::is_device_ptr(device_ptr_K)) {
  //CHECK-NEXT:   dpct::unique_copy(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<float>(device_ptr_K), dpct::device_pointer<float>(device_ptr_K + 4), dpct::device_pointer<float>(device_ptr_A), dpct::device_pointer<float>(device_ptr_R), dpct::device_pointer<float>(device_ptr_S), pred2);
  //CHECK-NEXT: } else {
  //CHECK-NEXT:   dpct::unique_copy(oneapi::dpl::execution::seq, device_ptr_K, device_ptr_K + 4, device_ptr_A, device_ptr_R, device_ptr_S, pred2);
  //CHECK-NEXT: };
  //CHECK-NEXT: dpct::dpct_memcpy(host_ptr_S, device_ptr_S, 20 * sizeof(float), dpct::device_to_host);
  //CHECK-NEXT: host_ptr_A[0]= -5;
  //CHECK-NEXT: host_ptr_A[1]= 8;
  //CHECK-NEXT: host_ptr_A[2]= 396;
  //CHECK-NEXT: host_ptr_A[3]= -395;
  //CHECK-NEXT: if (dpct::is_device_ptr(host_ptr_K + 10)) {
  //CHECK-NEXT:   dpct::unique_copy(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<float>(host_ptr_K), dpct::device_pointer<float>(host_ptr_K + 10), dpct::device_pointer<float>(host_ptr_A), dpct::device_pointer<float>(host_ptr_R), dpct::device_pointer<float>(host_ptr_S), pred2);
  //CHECK-NEXT: } else {
  //CHECK-NEXT:   dpct::unique_copy(oneapi::dpl::execution::seq, host_ptr_K, host_ptr_K + 10, host_ptr_A, host_ptr_R, host_ptr_S, pred2);
  //CHECK-NEXT: };
  //CHECK-NEXT: host_ptr_A[0]= -5;
  //CHECK-NEXT: host_ptr_A[1]= 8;
  //CHECK-NEXT: host_ptr_A[2]= 396;
  //CHECK-NEXT: host_ptr_A[3]= -395;
  //CHECK-NEXT: dpct::dpct_memcpy(device_ptr_A, host_ptr_A, 20 * sizeof(float), dpct::host_to_device);
  //CHECK-NEXT: if (dpct::is_device_ptr(device_ptr_K)) {
  //CHECK-NEXT:   dpct::unique_copy(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<float>(device_ptr_K), dpct::device_pointer<float>(device_ptr_K + 4), dpct::device_pointer<float>(device_ptr_A), dpct::device_pointer<float>(device_ptr_R), dpct::device_pointer<float>(device_ptr_S));
  //CHECK-NEXT: } else {
  //CHECK-NEXT:   dpct::unique_copy(oneapi::dpl::execution::seq, device_ptr_K, device_ptr_K + 4, device_ptr_A, device_ptr_R, device_ptr_S);
  //CHECK-NEXT: };
  //CHECK-NEXT: dpct::dpct_memcpy(host_ptr_S, device_ptr_S, 20 * sizeof(float), dpct::device_to_host);
  //CHECK-NEXT: host_ptr_A[0]= -5;
  //CHECK-NEXT: host_ptr_A[1]= 8;
  //CHECK-NEXT: host_ptr_A[2]= 396;
  //CHECK-NEXT: host_ptr_A[3]= -395;
  //CHECK-NEXT: if (dpct::is_device_ptr(host_ptr_K + 10)) {
  //CHECK-NEXT:   dpct::unique_copy(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<float>(host_ptr_K), dpct::device_pointer<float>(host_ptr_K + 10), dpct::device_pointer<float>(host_ptr_A), dpct::device_pointer<float>(host_ptr_R), dpct::device_pointer<float>(host_ptr_S));
  //CHECK-NEXT: } else {
  //CHECK-NEXT:   dpct::unique_copy(oneapi::dpl::execution::seq, host_ptr_K, host_ptr_K + 10, host_ptr_A, host_ptr_R, host_ptr_S);
  //CHECK-NEXT: };
  float *host_ptr_K;
  float *device_ptr_K;
  thrust::equal_to<float> pred2;
  host_ptr_A = (float*)std::malloc(20 * sizeof(float));
  host_ptr_K = (float*)std::malloc(20 * sizeof(float));
  host_ptr_R = (float*)std::malloc(20 * sizeof(float));
  host_ptr_S = (float*)std::malloc(20 * sizeof(float));
  cudaMalloc(&device_ptr_A, 20 * sizeof(float));
  cudaMalloc(&device_ptr_K, 20 * sizeof(float));
  cudaMalloc(&device_ptr_S, 20 * sizeof(float));
  cudaMalloc(&device_ptr_R, 20 * sizeof(float));
  host_ptr_K[0]= 1;
  host_ptr_K[1]= 2;
  host_ptr_K[2]= 2;
  host_ptr_K[3]= 1;
  cudaMemcpy(device_ptr_K, host_ptr_K, 20*sizeof(float), cudaMemcpyHostToDevice);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 396;
  host_ptr_A[3]= -395;
  cudaMemcpy(device_ptr_A, host_ptr_A, 20*sizeof(float), cudaMemcpyHostToDevice);
  thrust::unique_by_key_copy(thrust::device, device_ptr_K, device_ptr_K + 4, device_ptr_A, device_ptr_R, device_ptr_S, pred2);
  cudaMemcpy(host_ptr_S, device_ptr_S, 20*sizeof(float), cudaMemcpyDeviceToHost);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 396;
  host_ptr_A[3]= -395;
  thrust::unique_by_key_copy(host_ptr_K, host_ptr_K+10, host_ptr_A, host_ptr_R, host_ptr_S, pred2);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 396;
  host_ptr_A[3]= -395;
  cudaMemcpy(device_ptr_A, host_ptr_A, 20*sizeof(float), cudaMemcpyHostToDevice);
  thrust::unique_by_key_copy(thrust::device, device_ptr_K, device_ptr_K + 4, device_ptr_A, device_ptr_R, device_ptr_S);
  cudaMemcpy(host_ptr_S, device_ptr_S, 20*sizeof(float), cudaMemcpyDeviceToHost);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 396;
  host_ptr_A[3]= -395;
  thrust::unique_by_key_copy(host_ptr_K, host_ptr_K+10, host_ptr_A, host_ptr_R, host_ptr_S);
  return 0;
}

void foo()
{
  const int N = 6;
  int keys[N] = {1, 4, 2, 8, 5, 7};
  char values[N] = {'a', 'b', 'c', 'd', 'e', 'f'};

  //CHECK: if (dpct::is_device_ptr(keys)) {
  //CHECK-NEXT:    dpct::stable_sort(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(keys), dpct::device_pointer<int>(keys + N), dpct::device_pointer<char>(values));
  //CHECK-NEXT:  } else {
  //CHECK-NEXT:    dpct::stable_sort(oneapi::dpl::execution::seq, keys, keys + N, values);
  //CHECK-NEXT:  };
  //CHECK-NEXT:  if (dpct::is_device_ptr(keys + N)) {
  //CHECK-NEXT:    dpct::stable_sort(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(keys), dpct::device_pointer<int>(keys + N), dpct::device_pointer<char>(values));
  //CHECK-NEXT:  } else {
  //CHECK-NEXT:    dpct::stable_sort(oneapi::dpl::execution::seq, keys, keys + N, values);
  //CHECK-NEXT:  };
  //CHECK-NEXT:  if (dpct::is_device_ptr(keys)) {
  //CHECK-NEXT:    dpct::stable_sort(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(keys), dpct::device_pointer<int>(keys + N), dpct::device_pointer<char>(values), std::greater<int>());
  //CHECK-NEXT:  } else {
  //CHECK-NEXT:    dpct::stable_sort(oneapi::dpl::execution::seq, keys, keys + N, values, std::greater<int>());
  //CHECK-NEXT:  };
  //CHECK-NEXT:  if (dpct::is_device_ptr(keys + N)) {
  //CHECK-NEXT:    dpct::stable_sort(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(keys), dpct::device_pointer<int>(keys + N), dpct::device_pointer<char>(values), std::greater<int>());
  //CHECK-NEXT:  } else {
  //CHECK-NEXT:    dpct::stable_sort(oneapi::dpl::execution::seq, keys, keys + N, values, std::greater<int>());
  //CHECK-NEXT:  };
  thrust::stable_sort_by_key(thrust::host, keys, keys + N, values);
  thrust::stable_sort_by_key(keys, keys + N, values);
  thrust::stable_sort_by_key(thrust::host, keys, keys + N, values,thrust::greater<int>());
  thrust::stable_sort_by_key(keys, keys + N, values, thrust::greater<int>());
}
