// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct -out-root %T/thrust_sort_by_key %s --cuda-include-path="%cuda-path/include" --usm-level=none
// RUN: FileCheck --input-file %T/thrust_sort_by_key/thrust_sort_by_key.dp.cpp --match-full-lines %s

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>

int main(void) {

  thrust::device_vector<int> AD(4);
  thrust::device_vector<int> BD(4);
  thrust::host_vector<int> AH(4);
  thrust::host_vector<int> BH(4);
  
  int *h_ptr;
  int *d_ptr;

  h_ptr = (int*)std::malloc(20 * sizeof(int));
  cudaMalloc(&d_ptr, 20 * sizeof(int));

/*******************************************************************************************
 1. Test sort_by_key
 2. Test four VERSIONs (with/without exec argument) AND (with/without comparator argument)
 3. Test each VERSION with (device_vector/host_vector/malloc-ed memory/cudaMalloc-ed memory)
 *******************************************************************************************/

/*********** sort_by_key ***********************************************************************************************************************************************/

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CHECK:dpct::sort(oneapi::dpl::execution::seq, AH.begin(), AH.end(), BH.begin());
// CHECK-NEXT:dpct::sort(oneapi::dpl::execution::make_device_policy(q_ct1), AD.begin(), AD.end(), BD.begin());
// CHECK-NEXT:if (dpct::is_device_ptr(h_ptr)) {
// CHECK-NEXT:  dpct::sort(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(h_ptr), dpct::device_pointer<int>(h_ptr + 4), dpct::device_pointer<>(BH.begin()));
// CHECK-NEXT:} else {
// CHECK-NEXT:  dpct::sort(oneapi::dpl::execution::seq, h_ptr, h_ptr + 4, BH.begin());
// CHECK-NEXT:};
// CHECK-NEXT:if (dpct::is_device_ptr(d_ptr)) {
// CHECK-NEXT:  dpct::sort(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(d_ptr), dpct::device_pointer<int>(d_ptr + 4), dpct::device_pointer<>(BD.begin()));
// CHECK-NEXT:} else {
// CHECK-NEXT:  dpct::sort(oneapi::dpl::execution::seq, d_ptr, d_ptr + 4, BD.begin());
// CHECK-NEXT:};
  // VERSION                                 first       last      result
  thrust::sort_by_key(                AH.begin(), AH.end(), BH.begin());
  thrust::sort_by_key(                AD.begin(), AD.end(), BD.begin());
  thrust::sort_by_key(                h_ptr,      h_ptr+4,  BH.begin());
  thrust::sort_by_key(                d_ptr,      d_ptr+4,  BD.begin());

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CHECK:dpct::sort(oneapi::dpl::execution::seq, AH.begin(), AH.end(), BH.begin(), std::greater<int>());
// CHECK-NEXT:dpct::sort(oneapi::dpl::execution::make_device_policy(q_ct1), AD.begin(), AD.end(), BD.begin(), std::greater<int>());
// CHECK-NEXT:if (dpct::is_device_ptr(h_ptr)) {
// CHECK-NEXT:  dpct::sort(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(h_ptr), dpct::device_pointer<int>(h_ptr + 4), dpct::device_pointer<>(BH.begin()), std::greater<int>());
// CHECK-NEXT:} else {
// CHECK-NEXT:  dpct::sort(oneapi::dpl::execution::seq, h_ptr, h_ptr + 4, BH.begin(), std::greater<int>());
// CHECK-NEXT:};
// CHECK-NEXT:if (dpct::is_device_ptr(d_ptr)) {
// CHECK-NEXT:  dpct::sort(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(d_ptr), dpct::device_pointer<int>(d_ptr + 4), dpct::device_pointer<>(BD.begin()), std::greater<int>());
// CHECK-NEXT:} else {
// CHECK-NEXT:  dpct::sort(oneapi::dpl::execution::seq, d_ptr, d_ptr + 4, BD.begin(), std::greater<int>());
// CHECK-NEXT:};
  // VERSION                                 first       last      result      comparator
  thrust::sort_by_key(                AH.begin(), AH.end(), BH.begin(), thrust::greater<int>());
  thrust::sort_by_key(                AD.begin(), AD.end(), BD.begin(), thrust::greater<int>());
  thrust::sort_by_key(                h_ptr,      h_ptr+4,  BH.begin(), thrust::greater<int>());
  thrust::sort_by_key(                d_ptr,      d_ptr+4,  BD.begin(), thrust::greater<int>());


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CHECK:dpct::sort(oneapi::dpl::execution::seq, AH.begin(), AH.end(), BH.begin());
// CHECK-NEXT:dpct::sort(oneapi::dpl::execution::make_device_policy(q_ct1), AD.begin(), AD.end(), BD.begin());
// CHECK-NEXT:if (dpct::is_device_ptr(h_ptr)) {
// CHECK-NEXT:  dpct::sort(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(h_ptr), dpct::device_pointer<int>(h_ptr + 4), dpct::device_pointer<>(BH.begin()));
// CHECK-NEXT:} else {
// CHECK-NEXT:  dpct::sort(oneapi::dpl::execution::seq, h_ptr, h_ptr + 4, BH.begin());
// CHECK-NEXT:};
// CHECK-NEXT:if (dpct::is_device_ptr(d_ptr)) {
// CHECK-NEXT:  dpct::sort(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(d_ptr), dpct::device_pointer<int>(d_ptr + 4), dpct::device_pointer<>(BD.begin()));
// CHECK-NEXT:} else {
// CHECK-NEXT:  dpct::sort(oneapi::dpl::execution::seq, d_ptr, d_ptr + 4, BD.begin());
// CHECK-NEXT:};
  // VERSION                 exec            first       last      result
  thrust::sort_by_key(thrust::host,   AH.begin(), AH.end(), BH.begin());
  thrust::sort_by_key(thrust::device, AD.begin(), AD.end(), BD.begin());
  thrust::sort_by_key(thrust::host,   h_ptr,      h_ptr+4,  BH.begin());
  thrust::sort_by_key(thrust::device, d_ptr,      d_ptr+4,  BD.begin());

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CHECK:dpct::sort(oneapi::dpl::execution::seq, AH.begin(), AH.end(), BH.begin(), std::greater<int>());
// CHECK-NEXT:dpct::sort(oneapi::dpl::execution::make_device_policy(q_ct1), AD.begin(), AD.end(), BD.begin(), std::greater<int>());
// CHECK-NEXT:if (dpct::is_device_ptr(h_ptr)) {
// CHECK-NEXT:  dpct::sort(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(h_ptr), dpct::device_pointer<int>(h_ptr + 4), dpct::device_pointer<>(BH.begin()), std::greater<int>());
// CHECK-NEXT:} else {
// CHECK-NEXT:  dpct::sort(oneapi::dpl::execution::seq, h_ptr, h_ptr + 4, BH.begin(), std::greater<int>());
// CHECK-NEXT:};
// CHECK-NEXT:if (dpct::is_device_ptr(d_ptr)) {
// CHECK-NEXT:  dpct::sort(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(d_ptr), dpct::device_pointer<int>(d_ptr + 4), dpct::device_pointer<>(BD.begin()), std::greater<int>());
// CHECK-NEXT:} else {
// CHECK-NEXT:  dpct::sort(oneapi::dpl::execution::seq, d_ptr, d_ptr + 4, BD.begin(), std::greater<int>());
// CHECK-NEXT:};
  // VERSION                 exec            first       last      result      comparator
  thrust::sort_by_key(thrust::host,   AH.begin(), AH.end(), BH.begin(), thrust::greater<int>());
  thrust::sort_by_key(thrust::device, AD.begin(), AD.end(), BD.begin(), thrust::greater<int>());
  thrust::sort_by_key(thrust::host,   h_ptr,      h_ptr+4,  BH.begin(), thrust::greater<int>());
  thrust::sort_by_key(thrust::device, d_ptr,      d_ptr+4,  BD.begin(), thrust::greater<int>());

  return 0;
}
