// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct -out-root %T/thrust_replace %s --cuda-include-path="%cuda-path/include" --usm-level=none
// RUN: FileCheck --input-file %T/thrust_replace/thrust_replace.dp.cpp --match-full-lines %s

#include <thrust/replace.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
struct is_less_than_zero
{
   __host__ __device__
  bool operator()(int x) const
  {
    return x < 0;
  }
};

int main(void) {

  thrust::device_vector<int> AD(4);
  thrust::device_vector<int> BD(4);
  thrust::device_vector<int> SD(4);    
  thrust::host_vector<int> AH(4);
  thrust::host_vector<int> BH(4);
  thrust::host_vector<int> SH(4);  
  
  is_less_than_zero pred;

  int *h_ptr;
  int *d_ptr;

  h_ptr = (int*)std::malloc(20 * sizeof(int));
  cudaMalloc(&d_ptr, 20 * sizeof(int));

/*******************************************************************************************
 1. Test replace_if/replace_copy_if
 2. Test four VERSIONs (with/without exec argument) AND (with/without stencil argument)
 3. Test each VERSION with (device_vector/host_vector/malloc-ed memory/cudaMalloc-ed memory)
 *******************************************************************************************/

/*********** replace_if ****************************************************************************************************************************************************/

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CHECK:oneapi::dpl::replace_if(oneapi::dpl::execution::seq, AH.begin(), AH.end(), pred, 0);
// CHECK-NEXT:oneapi::dpl::replace_if(oneapi::dpl::execution::make_device_policy(q_ct1), AD.begin(), AD.end(), pred, 0);
// CHECK-NEXT:if (dpct::is_device_ptr(h_ptr)) {
// CHECK-NEXT:  oneapi::dpl::replace_if(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(h_ptr), dpct::device_pointer<int>(h_ptr + 4), pred, 0);
// CHECK-NEXT:} else {
// CHECK-NEXT:  oneapi::dpl::replace_if(oneapi::dpl::execution::seq, h_ptr, h_ptr + 4, pred, 0);
// CHECK-NEXT:};
// CHECK-NEXT:if (dpct::is_device_ptr(d_ptr)) {
// CHECK-NEXT:  oneapi::dpl::replace_if(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(d_ptr), dpct::device_pointer<int>(d_ptr + 4), pred, 0);
// CHECK-NEXT:} else {
// CHECK-NEXT:  oneapi::dpl::replace_if(oneapi::dpl::execution::seq, d_ptr, d_ptr + 4, pred, 0);
// CHECK-NEXT:};
  // VERSION                         first       last                   pred  new_value
  thrust::replace_if(                AH.begin(), AH.end(),              pred, 0);
  thrust::replace_if(                AD.begin(), AD.end(),              pred, 0);
  thrust::replace_if(                h_ptr,      h_ptr+4,               pred, 0);
  thrust::replace_if(                d_ptr,      d_ptr+4,               pred, 0);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CHECK:dpct::replace_if(oneapi::dpl::execution::seq, AH.begin(), AH.end(), SH.begin(), pred, 0);
// CHECK-NEXT:dpct::replace_if(oneapi::dpl::execution::make_device_policy(q_ct1), AD.begin(), AD.end(), SD.begin(), pred, 0);
// CHECK-NEXT:if (dpct::is_device_ptr(h_ptr)) {
// CHECK-NEXT:  dpct::replace_if(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(h_ptr), dpct::device_pointer<int>(h_ptr + 4), dpct::device_pointer<>(SH.begin()), pred, 0);
// CHECK-NEXT:} else {
// CHECK-NEXT:  dpct::replace_if(oneapi::dpl::execution::seq, h_ptr, h_ptr + 4, SH.begin(), pred, 0);
// CHECK-NEXT:};
// CHECK-NEXT:if (dpct::is_device_ptr(d_ptr)) {
// CHECK-NEXT:  dpct::replace_if(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(d_ptr), dpct::device_pointer<int>(d_ptr + 4), dpct::device_pointer<>(SD.begin()), pred, 0);
// CHECK-NEXT:} else {
// CHECK-NEXT:  dpct::replace_if(oneapi::dpl::execution::seq, d_ptr, d_ptr + 4, SD.begin(), pred, 0);
// CHECK-NEXT:};
  // VERSION                         first       last      stencil      pred  new_value
  thrust::replace_if(                AH.begin(), AH.end(), SH.begin(),  pred, 0);
  thrust::replace_if(                AD.begin(), AD.end(), SD.begin(),  pred, 0);
  thrust::replace_if(                h_ptr,      h_ptr+4,  SH.begin(),  pred, 0);
  thrust::replace_if(                d_ptr,      d_ptr+4,  SD.begin(),  pred, 0);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CHECK:oneapi::dpl::replace_if(oneapi::dpl::execution::seq, AH.begin(), AH.end(), pred, 0);
// CHECK-NEXT:oneapi::dpl::replace_if(oneapi::dpl::execution::make_device_policy(q_ct1), AD.begin(), AD.end(), pred, 0);
// CHECK-NEXT:if (dpct::is_device_ptr(h_ptr)) {
// CHECK-NEXT:  oneapi::dpl::replace_if(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(h_ptr), dpct::device_pointer<int>(h_ptr + 4), pred, 0);
// CHECK-NEXT:} else {
// CHECK-NEXT:  oneapi::dpl::replace_if(oneapi::dpl::execution::seq, h_ptr, h_ptr + 4, pred, 0);
// CHECK-NEXT:};
// CHECK-NEXT:if (dpct::is_device_ptr(d_ptr)) {
// CHECK-NEXT:  oneapi::dpl::replace_if(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(d_ptr), dpct::device_pointer<int>(d_ptr + 4), pred, 0);
// CHECK-NEXT:} else {
// CHECK-NEXT:  oneapi::dpl::replace_if(oneapi::dpl::execution::seq, d_ptr, d_ptr + 4, pred, 0);
// CHECK-NEXT:};
  // VERSION         exec            first       last                   pred  new_value
  thrust::replace_if(thrust::host,   AH.begin(), AH.end(),              pred, 0);
  thrust::replace_if(thrust::device, AD.begin(), AD.end(),              pred, 0);
  thrust::replace_if(thrust::host,   h_ptr,      h_ptr+4,               pred, 0);
  thrust::replace_if(thrust::device, d_ptr,      d_ptr+4,               pred, 0);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CHECK:dpct::replace_if(oneapi::dpl::execution::seq, AH.begin(), AH.end(), SH.begin(), pred, 0);
// CHECK-NEXT:dpct::replace_if(oneapi::dpl::execution::make_device_policy(q_ct1), AD.begin(), AD.end(), SD.begin(), pred, 0);
// CHECK-NEXT:if (dpct::is_device_ptr(h_ptr)) {
// CHECK-NEXT:  dpct::replace_if(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(h_ptr), dpct::device_pointer<int>(h_ptr + 4), dpct::device_pointer<>(SH.begin()), pred, 0);
// CHECK-NEXT:} else {
// CHECK-NEXT:  dpct::replace_if(oneapi::dpl::execution::seq, h_ptr, h_ptr + 4, SH.begin(), pred, 0);
// CHECK-NEXT:};
// CHECK-NEXT:if (dpct::is_device_ptr(d_ptr)) {
// CHECK-NEXT:  dpct::replace_if(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(d_ptr), dpct::device_pointer<int>(d_ptr + 4), dpct::device_pointer<>(SD.begin()), pred, 0);
// CHECK-NEXT:} else {
// CHECK-NEXT:  dpct::replace_if(oneapi::dpl::execution::seq, d_ptr, d_ptr + 4, SD.begin(), pred, 0);
// CHECK-NEXT:};
  // VERSION         exec            first       last      stencil      pred  new_value
  thrust::replace_if(thrust::host,   AH.begin(), AH.end(), SH.begin(),  pred, 0);
  thrust::replace_if(thrust::device, AD.begin(), AD.end(), SD.begin(),  pred, 0);
  thrust::replace_if(thrust::host,   h_ptr,      h_ptr+4,  SH.begin(),  pred, 0);
  thrust::replace_if(thrust::device, d_ptr,      d_ptr+4,  SD.begin(),  pred, 0);
  

/*********** replace_copy_if ***********************************************************************************************************************************************/

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CHECK:oneapi::dpl::replace_copy_if(oneapi::dpl::execution::seq, AH.begin(), AH.end(), BH.begin(), pred, 0);
// CHECK-NEXT:oneapi::dpl::replace_copy_if(oneapi::dpl::execution::make_device_policy(q_ct1), AD.begin(), AD.end(), BD.begin(), pred, 0);
// CHECK-NEXT:if (dpct::is_device_ptr(h_ptr)) {
// CHECK-NEXT:  oneapi::dpl::replace_copy_if(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(h_ptr), dpct::device_pointer<int>(h_ptr + 4), dpct::device_pointer<>(BH.begin()), pred, 0);
// CHECK-NEXT:} else {
// CHECK-NEXT:  oneapi::dpl::replace_copy_if(oneapi::dpl::execution::seq, h_ptr, h_ptr + 4, BH.begin(), pred, 0);
// CHECK-NEXT:};
// CHECK-NEXT:if (dpct::is_device_ptr(d_ptr)) {
// CHECK-NEXT:  oneapi::dpl::replace_copy_if(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(d_ptr), dpct::device_pointer<int>(d_ptr + 4), dpct::device_pointer<>(BD.begin()), pred, 0);
// CHECK-NEXT:} else {
// CHECK-NEXT:  oneapi::dpl::replace_copy_if(oneapi::dpl::execution::seq, d_ptr, d_ptr + 4, BD.begin(), pred, 0);
// CHECK-NEXT:};
  // VERSION                              first       last                  result      pred  new_value
  thrust::replace_copy_if(                AH.begin(), AH.end(),             BH.begin(), pred, 0);
  thrust::replace_copy_if(                AD.begin(), AD.end(),             BD.begin(), pred, 0);
  thrust::replace_copy_if(                h_ptr,      h_ptr+4,              BH.begin(), pred, 0);
  thrust::replace_copy_if(                d_ptr,      d_ptr+4,              BD.begin(), pred, 0);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CHECK:dpct::replace_copy_if(oneapi::dpl::execution::seq, AH.begin(), AH.end(), SH.begin(), BH.begin(), pred, 0);
// CHECK-NEXT:dpct::replace_copy_if(oneapi::dpl::execution::make_device_policy(q_ct1), AD.begin(), AD.end(), SD.begin(), BD.begin(), pred, 0);
// CHECK-NEXT:if (dpct::is_device_ptr(h_ptr)) {
// CHECK-NEXT:  dpct::replace_copy_if(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(h_ptr), dpct::device_pointer<int>(h_ptr + 4), dpct::device_pointer<>(SH.begin()), dpct::device_pointer<>(BH.begin()), pred, 0);
// CHECK-NEXT:} else {
// CHECK-NEXT:  dpct::replace_copy_if(oneapi::dpl::execution::seq, h_ptr, h_ptr + 4, SH.begin(), BH.begin(), pred, 0);
// CHECK-NEXT:};
  // VERSION                              first       last      stencil     result      pred  new_value
  thrust::replace_copy_if(                AH.begin(), AH.end(), SH.begin(), BH.begin(), pred, 0);
  thrust::replace_copy_if(                AD.begin(), AD.end(), SD.begin(), BD.begin(), pred, 0);
  thrust::replace_copy_if(                h_ptr,      h_ptr+4,  SH.begin(), BH.begin(), pred, 0);  
  // Overload not supported with thrust
  // thrust::replace_copy_if(                d_ptr,      d_ptr+4,  SD.begin(), BD.begin(), pred, 0);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CHECK:oneapi::dpl::replace_copy_if(oneapi::dpl::execution::seq, AH.begin(), AH.end(), BH.begin(), pred, 0);
// CHECK-NEXT:oneapi::dpl::replace_copy_if(oneapi::dpl::execution::make_device_policy(q_ct1), AD.begin(), AD.end(), BD.begin(), pred, 0);
// CHECK-NEXT:if (dpct::is_device_ptr(h_ptr)) {
// CHECK-NEXT:  oneapi::dpl::replace_copy_if(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(h_ptr), dpct::device_pointer<int>(h_ptr + 4), dpct::device_pointer<>(BH.begin()), pred, 0);
// CHECK-NEXT:} else {
// CHECK-NEXT:  oneapi::dpl::replace_copy_if(oneapi::dpl::execution::seq, h_ptr, h_ptr + 4, BH.begin(), pred, 0);
// CHECK-NEXT:};
// CHECK-NEXT:if (dpct::is_device_ptr(d_ptr)) {
// CHECK-NEXT:  oneapi::dpl::replace_copy_if(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(d_ptr), dpct::device_pointer<int>(d_ptr + 4), dpct::device_pointer<>(BD.begin()), pred, 0);
// CHECK-NEXT:} else {
// CHECK-NEXT:  oneapi::dpl::replace_copy_if(oneapi::dpl::execution::seq, d_ptr, d_ptr + 4, BD.begin(), pred, 0);
// CHECK-NEXT:};
  // VERSION              exec            first       last                  result      pred  new_value
  thrust::replace_copy_if(thrust::host,   AH.begin(), AH.end(),             BH.begin(), pred, 0);
  thrust::replace_copy_if(thrust::device, AD.begin(), AD.end(),             BD.begin(), pred, 0);
  thrust::replace_copy_if(thrust::host,   h_ptr,      h_ptr+4,              BH.begin(), pred, 0);
  thrust::replace_copy_if(thrust::device, d_ptr,      d_ptr+4,              BD.begin(), pred, 0);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CHECK:dpct::replace_copy_if(oneapi::dpl::execution::seq, AH.begin(), AH.end(), SH.begin(), BH.begin(), pred, 0);
// CHECK-NEXT:dpct::replace_copy_if(oneapi::dpl::execution::make_device_policy(q_ct1), AD.begin(), AD.end(), SD.begin(), BD.begin(), pred, 0);
// CHECK-NEXT:if (dpct::is_device_ptr(h_ptr)) {
// CHECK-NEXT:  dpct::replace_copy_if(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(h_ptr), dpct::device_pointer<int>(h_ptr + 4), dpct::device_pointer<>(SH.begin()), dpct::device_pointer<>(BH.begin()), pred, 0);
// CHECK-NEXT:} else {
// CHECK-NEXT:  dpct::replace_copy_if(oneapi::dpl::execution::seq, h_ptr, h_ptr + 4, SH.begin(), BH.begin(), pred, 0);
// CHECK-NEXT:};
// CHECK-NEXT:if (dpct::is_device_ptr(d_ptr)) {
// CHECK-NEXT:  dpct::replace_copy_if(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(d_ptr), dpct::device_pointer<int>(d_ptr + 4), dpct::device_pointer<>(SD.begin()), dpct::device_pointer<>(BD.begin()), pred, 0);
// CHECK-NEXT:} else {
// CHECK-NEXT:  dpct::replace_copy_if(oneapi::dpl::execution::seq, d_ptr, d_ptr + 4, SD.begin(), BD.begin(), pred, 0);
// CHECK-NEXT:};
  // VERSION              exec            first       last      stencil     result      pred  new_value
  thrust::replace_copy_if(thrust::host,   AH.begin(), AH.end(), SH.begin(), BH.begin(), pred, 0);
  thrust::replace_copy_if(thrust::device, AD.begin(), AD.end(), SD.begin(), BD.begin(), pred, 0);
  thrust::replace_copy_if(thrust::host,   h_ptr,      h_ptr+4,  SH.begin(), BH.begin(), pred, 0);
  thrust::replace_copy_if(thrust::device, d_ptr,      d_ptr+4,  SD.begin(), BD.begin(), pred, 0);

  return 0;
}
