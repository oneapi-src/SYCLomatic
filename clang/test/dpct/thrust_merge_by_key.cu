// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct -out-root %T/thrust_merge_by_key %s --cuda-include-path="%cuda-path/include" --usm-level=none
// RUN: FileCheck --input-file %T/thrust_merge_by_key/thrust_merge_by_key.dp.cpp --match-full-lines %s

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>

int main(void) {

  thrust::device_vector<int> AD(4);
  thrust::device_vector<int> BD(4);
  thrust::device_vector<int> CD(4);
  thrust::device_vector<int> DD(4);
  thrust::device_vector<int> ED(8);
  thrust::device_vector<int> FD(8);
  
  thrust::host_vector<int> AH(4);
  thrust::host_vector<int> BH(4);
  thrust::host_vector<int> CH(4);
  thrust::host_vector<int> DH(4);
  thrust::host_vector<int> EH(8);
  thrust::host_vector<int> FH(8);  
  

  int *h_ptr;
  int *d_ptr;

  h_ptr = (int*)std::malloc(20 * sizeof(int));
  cudaMalloc(&d_ptr, 20 * sizeof(int));

/*******************************************************************************************
 1. Test merge_by_key
 2. Test four VERSIONs (with/without exec argument) AND (with/without comparator)
 3. Test each VERSION with (device_vector/host_vector/malloc-ed memory/cudaMalloc-ed memory)
 *******************************************************************************************/

/*********** merge_by_key ***********************************************************************************************************************************************/

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CHECK:dpct::merge(oneapi::dpl::execution::seq, AH.begin(), AH.end(), BH.begin(), BH.end(), CH.begin(), DH.begin(), EH.begin(), FH.begin());
// CHECK-NEXT:dpct::merge(oneapi::dpl::execution::make_device_policy(q_ct1), AD.begin(), AD.end(), BD.begin(), BD.end(), CD.begin(), DD.begin(), ED.begin(), FD.begin());
// CHECK-NEXT:if (dpct::is_device_ptr(h_ptr)) {
// CHECK-NEXT:  dpct::merge(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(h_ptr), dpct::device_pointer<int>(h_ptr + 4), dpct::device_pointer<>(BH.begin()), dpct::device_pointer<>(BH.end()), dpct::device_pointer<>(CH.begin()), dpct::device_pointer<>(DH.begin()), dpct::device_pointer<>(EH.begin()), dpct::device_pointer<>(FH.begin()));
// CHECK-NEXT:} else {
// CHECK-NEXT:  dpct::merge(oneapi::dpl::execution::seq, h_ptr, h_ptr + 4, BH.begin(), BH.end(), CH.begin(), DH.begin(), EH.begin(), FH.begin());
// CHECK-NEXT:};
  // VERSION                           first1      last1     first2      last2     val1        val2        keys        values
  thrust::merge_by_key(                AH.begin(), AH.end(), BH.begin(), BH.end(), CH.begin(), DH.begin(), EH.begin(), FH.begin());
  thrust::merge_by_key(                AD.begin(), AD.end(), BD.begin(), BD.end(), CD.begin(), DD.begin(), ED.begin(), FD.begin());
  thrust::merge_by_key(                h_ptr,      h_ptr+4,  BH.begin(), BH.end(), CH.begin(), DH.begin(), EH.begin(), FH.begin());
  // Overload not supported with thrust
  // thrust::merge_by_key(                d_ptr,      d_ptr+4,  BD.begin(), BD.end(), CD.begin(), DD.begin(), ED.begin(), FD.begin());

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CHECK:dpct::merge(oneapi::dpl::execution::seq, AH.begin(), AH.end(), BH.begin(), BH.end(), CH.begin(), DH.begin(), EH.begin(), FH.begin(), std::greater<int>());
// CHECK-NEXT:dpct::merge(oneapi::dpl::execution::make_device_policy(q_ct1), AD.begin(), AD.end(), BD.begin(), BD.end(), CD.begin(), DD.begin(), ED.begin(), FD.begin(), std::greater<int>());
// CHECK-NEXT:if (dpct::is_device_ptr(h_ptr)) {
// CHECK-NEXT:  dpct::merge(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(h_ptr), dpct::device_pointer<int>(h_ptr + 4), dpct::device_pointer<>(BH.begin()), dpct::device_pointer<>(BH.end()), dpct::device_pointer<>(CH.begin()), dpct::device_pointer<>(DH.begin()), dpct::device_pointer<>(EH.begin()), dpct::device_pointer<>(FH.begin()), std::greater<int>());
// CHECK-NEXT:} else {
// CHECK-NEXT:  dpct::merge(oneapi::dpl::execution::seq, h_ptr, h_ptr + 4, BH.begin(), BH.end(), CH.begin(), DH.begin(), EH.begin(), FH.begin(), std::greater<int>());
// CHECK-NEXT:};
  // VERSION                           first1      last1     first2      last2     val1        val2        keys        values      comparator
  thrust::merge_by_key(                AH.begin(), AH.end(), BH.begin(), BH.end(), CH.begin(), DH.begin(), EH.begin(), FH.begin(), thrust::greater<int>());
  thrust::merge_by_key(                AD.begin(), AD.end(), BD.begin(), BD.end(), CD.begin(), DD.begin(), ED.begin(), FD.begin(), thrust::greater<int>());
  thrust::merge_by_key(                h_ptr,      h_ptr+4,  BH.begin(), BH.end(), CH.begin(), DH.begin(), EH.begin(), FH.begin(), thrust::greater<int>());  
#ifdef ADD_BUG
  // This fails with nvcc
  thrust::merge_by_key(                d_ptr,      d_ptr+4,  BD.begin(), BD.end(), CD.begin(), DD.begin(), ED.begin(), FD.begin(), thrust::greater<int>());
#endif

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CHECK:dpct::merge(oneapi::dpl::execution::seq, AH.begin(), AH.end(), BH.begin(), BH.end(), CH.begin(), DH.begin(), EH.begin(), FH.begin());
// CHECK-NEXT:dpct::merge(oneapi::dpl::execution::make_device_policy(q_ct1), AD.begin(), AD.end(), BD.begin(), BD.end(), CD.begin(), DD.begin(), ED.begin(), FD.begin());
// CHECK-NEXT:if (dpct::is_device_ptr(h_ptr)) {
// CHECK-NEXT:  dpct::merge(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(h_ptr), dpct::device_pointer<int>(h_ptr + 4), dpct::device_pointer<>(BH.begin()), dpct::device_pointer<>(BH.end()), dpct::device_pointer<>(CH.begin()), dpct::device_pointer<>(DH.begin()), dpct::device_pointer<>(EH.begin()), dpct::device_pointer<>(FH.begin()));
// CHECK-NEXT:} else {
// CHECK-NEXT:  dpct::merge(oneapi::dpl::execution::seq, h_ptr, h_ptr + 4, BH.begin(), BH.end(), CH.begin(), DH.begin(), EH.begin(), FH.begin());
// CHECK-NEXT:};
// CHECK-NEXT:if (dpct::is_device_ptr(d_ptr)) {
// CHECK-NEXT:  dpct::merge(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(d_ptr), dpct::device_pointer<int>(d_ptr + 4), dpct::device_pointer<>(BD.begin()), dpct::device_pointer<>(BD.end()), dpct::device_pointer<>(CD.begin()), dpct::device_pointer<>(DD.begin()), dpct::device_pointer<>(ED.begin()), dpct::device_pointer<>(FD.begin()));
// CHECK-NEXT:} else {
// CHECK-NEXT:  dpct::merge(oneapi::dpl::execution::seq, d_ptr, d_ptr + 4, BD.begin(), BD.end(), CD.begin(), DD.begin(), ED.begin(), FD.begin());
// CHECK-NEXT:};
  // VERSION                           first1      last1     first2      last2     val1        val2        keys        values
  thrust::merge_by_key(thrust::host,   AH.begin(), AH.end(), BH.begin(), BH.end(), CH.begin(), DH.begin(), EH.begin(), FH.begin());
  thrust::merge_by_key(thrust::device, AD.begin(), AD.end(), BD.begin(), BD.end(), CD.begin(), DD.begin(), ED.begin(), FD.begin());
  thrust::merge_by_key(thrust::host,   h_ptr,      h_ptr+4,  BH.begin(), BH.end(), CH.begin(), DH.begin(), EH.begin(), FH.begin());
  thrust::merge_by_key(thrust::device, d_ptr,      d_ptr+4,  BD.begin(), BD.end(), CD.begin(), DD.begin(), ED.begin(), FD.begin());

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CHECK:dpct::merge(oneapi::dpl::execution::seq, AH.begin(), AH.end(), BH.begin(), BH.end(), CH.begin(), DH.begin(), EH.begin(), FH.begin(), std::greater<int>());
// CHECK-NEXT:dpct::merge(oneapi::dpl::execution::make_device_policy(q_ct1), AD.begin(), AD.end(), BD.begin(), BD.end(), CD.begin(), DD.begin(), ED.begin(), FD.begin(), std::greater<int>());
// CHECK-NEXT:if (dpct::is_device_ptr(h_ptr)) {
// CHECK-NEXT:  dpct::merge(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(h_ptr), dpct::device_pointer<int>(h_ptr + 4), dpct::device_pointer<>(BH.begin()), dpct::device_pointer<>(BH.end()), dpct::device_pointer<>(CH.begin()), dpct::device_pointer<>(DH.begin()), dpct::device_pointer<>(EH.begin()), dpct::device_pointer<>(FH.begin()), std::greater<int>());
// CHECK-NEXT:} else {
// CHECK-NEXT:  dpct::merge(oneapi::dpl::execution::seq, h_ptr, h_ptr + 4, BH.begin(), BH.end(), CH.begin(), DH.begin(), EH.begin(), FH.begin(), std::greater<int>());
// CHECK-NEXT:};
// CHECK-NEXT:if (dpct::is_device_ptr(d_ptr)) {
// CHECK-NEXT:  dpct::merge(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(d_ptr), dpct::device_pointer<int>(d_ptr + 4), dpct::device_pointer<>(BD.begin()), dpct::device_pointer<>(BD.end()), dpct::device_pointer<>(CD.begin()), dpct::device_pointer<>(DD.begin()), dpct::device_pointer<>(ED.begin()), dpct::device_pointer<>(FD.begin()), std::greater<int>());
// CHECK-NEXT:} else {
// CHECK-NEXT:  dpct::merge(oneapi::dpl::execution::seq, d_ptr, d_ptr + 4, BD.begin(), BD.end(), CD.begin(), DD.begin(), ED.begin(), FD.begin(), std::greater<int>());
// CHECK-NEXT:};
  // VERSION                           first1      last1     first2      last2     val1        val2        keys        values      comparator
  thrust::merge_by_key(thrust::host,   AH.begin(), AH.end(), BH.begin(), BH.end(), CH.begin(), DH.begin(), EH.begin(), FH.begin(), thrust::greater<int>());
  thrust::merge_by_key(thrust::device, AD.begin(), AD.end(), BD.begin(), BD.end(), CD.begin(), DD.begin(), ED.begin(), FD.begin(), thrust::greater<int>());
  thrust::merge_by_key(thrust::host,   h_ptr,      h_ptr+4,  BH.begin(), BH.end(), CH.begin(), DH.begin(), EH.begin(), FH.begin(), thrust::greater<int>());
  thrust::merge_by_key(thrust::device, d_ptr,      d_ptr+4,  BD.begin(), BD.end(), CD.begin(), DD.begin(), ED.begin(), FD.begin(), thrust::greater<int>());

  return 0;
}
