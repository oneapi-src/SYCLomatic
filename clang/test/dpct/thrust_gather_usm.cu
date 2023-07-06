// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct -out-root %T/thrust_gather_usm %s --cuda-include-path="%cuda-path/include" --usm-level=restricted
// RUN: FileCheck --input-file %T/thrust_gather_usm/thrust_gather_usm.dp.cpp --match-full-lines %s

#include <thrust/gather.h>
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
  thrust::device_vector<int> RD(4);      
  thrust::host_vector<int> AH(4);
  thrust::host_vector<int> BH(4);
  thrust::host_vector<int> SH(4);
  thrust::host_vector<int> RH(4);    
  
  is_less_than_zero pred;

  int *h_ptr;
  int *d_ptr;

  h_ptr = (int*)std::malloc(20 * sizeof(int));
  cudaMalloc(&d_ptr, 20 * sizeof(int));

/*******************************************************************************************
 1. Test gather_if
 2. Test four VERSIONs (with/without exec argument) AND (with/without predicate argument)
 3. Test each VERSION with (device_vector/host_vector/malloc-ed memory/cudaMalloc-ed memory)
 *******************************************************************************************/

/*********** gather_if ***********************************************************************************************************************************************/

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CHECK:dpct::gather_if(oneapi::dpl::execution::par_noseq, AH.begin(), AH.end(), SH.begin(), BH.begin(), RH.begin(), pred);
// CHECK-NEXT:dpct::gather_if(oneapi::dpl::execution::make_device_policy(q_ct1), AD.begin(), AD.end(), SD.begin(), BD.begin(), RD.begin(), pred);
// CHECK-NEXT:dpct::gather_if(oneapi::dpl::execution::par_noseq, h_ptr, h_ptr + 4, SH.begin(), BH.begin(), RH.begin(), pred);
  // VERSION                        first       last      stencil     input       result      pred
  thrust::gather_if(                AH.begin(), AH.end(), SH.begin(), BH.begin(), RH.begin(), pred);
  thrust::gather_if(                AD.begin(), AD.end(), SD.begin(), BD.begin(), RD.begin(), pred);
  thrust::gather_if(                h_ptr,      h_ptr+4,  SH.begin(), BH.begin(), RH.begin(), pred);
  // Overload not supported with thrust
  // thrust::gather_if(                d_ptr,      d_ptr+4,  SD.begin(), BD.begin(), RD.begin(), pred);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CHECK:dpct::gather_if(oneapi::dpl::execution::par_noseq, AH.begin(), AH.end(), SH.begin(), BH.begin(), RH.begin(), pred);
// CHECK-NEXT:dpct::gather_if(oneapi::dpl::execution::make_device_policy(q_ct1), AD.begin(), AD.end(), SD.begin(), BD.begin(), RD.begin(), pred);
// CHECK-NEXT:dpct::gather_if(oneapi::dpl::execution::par_noseq, h_ptr, h_ptr + 4, SH.begin(), BH.begin(), RH.begin(), pred);
// CHECK-NEXT:dpct::gather_if(oneapi::dpl::execution::make_device_policy(q_ct1), d_ptr, d_ptr + 4, SD.begin(), BD.begin(), RD.begin(), pred);
  // VERSION        exec            first       last      stencil     input       result      pred
  thrust::gather_if(thrust::host,   AH.begin(), AH.end(), SH.begin(), BH.begin(), RH.begin(), pred);
  thrust::gather_if(thrust::device, AD.begin(), AD.end(), SD.begin(), BD.begin(), RD.begin(), pred);
  thrust::gather_if(thrust::host,   h_ptr,      h_ptr+4,  SH.begin(), BH.begin(), RH.begin(), pred);
  thrust::gather_if(thrust::device, d_ptr,      d_ptr+4,  SD.begin(), BD.begin(), RD.begin(), pred);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// CHECK: /*
// CHECK-NEXT: DPCT1107:{{[0-9]+}}: Migration for this overload of thrust::gather_if is not supported.
// CHECK-NEXT: */
// CHECK-NEXT: thrust::gather_if(AH.begin(), AH.end(), SH.begin(), BH.begin(), RH.begin());
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1107:{{[0-9]+}}: Migration for this overload of thrust::gather_if is not supported.
// CHECK-NEXT: */
// CHECK-NEXT: thrust::gather_if(AD.begin(), AD.end(), SD.begin(), BD.begin(), RD.begin());
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1107:{{[0-9]+}}: Migration for this overload of thrust::gather_if is not supported.
// CHECK-NEXT: */
// CHECK-NEXT: thrust::gather_if(h_ptr, h_ptr + 4, SH.begin(), BH.begin(), RH.begin());
  // VERSION                        first       last      stencil     input       result
  thrust::gather_if(                AH.begin(), AH.end(), SH.begin(), BH.begin(), RH.begin());
  thrust::gather_if(                AD.begin(), AD.end(), SD.begin(), BD.begin(), RD.begin());
  thrust::gather_if(                h_ptr,      h_ptr+4,  SH.begin(), BH.begin(), RH.begin());
  // Overload not supported with thrust
  // thrust::gather_if(                d_ptr,      d_ptr+4,  SD.begin(), BD.begin(), RD.begin());

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// CHECK: /*
// CHECK-NEXT: DPCT1107:{{[0-9]+}}: Migration for this overload of thrust::gather_if is not supported.
// CHECK-NEXT: */
// CHECK-NEXT: thrust::gather_if(oneapi::dpl::execution::par_noseq, AH.begin(), AH.end(), SH.begin(), BH.begin(), RH.begin());
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1107:{{[0-9]+}}: Migration for this overload of thrust::gather_if is not supported.
// CHECK-NEXT: */
// CHECK-NEXT: thrust::gather_if(oneapi::dpl::execution::make_device_policy(q_ct1), AD.begin(), AD.end(), SD.begin(), BD.begin(), RD.begin());
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1107:{{[0-9]+}}: Migration for this overload of thrust::gather_if is not supported.
// CHECK-NEXT: */
// CHECK-NEXT: thrust::gather_if(oneapi::dpl::execution::par_noseq, h_ptr, h_ptr + 4, SH.begin(), BH.begin(), RH.begin());
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1107:{{[0-9]+}}: Migration for this overload of thrust::gather_if is not supported.
// CHECK-NEXT: */
// CHECK-NEXT: thrust::gather_if(oneapi::dpl::execution::make_device_policy(q_ct1), d_ptr, d_ptr + 4, SD.begin(), BD.begin(), RD.begin());
  // VERSION        exec            first       last      stencil     input       result
  thrust::gather_if(thrust::host,   AH.begin(), AH.end(), SH.begin(), BH.begin(), RH.begin());
  thrust::gather_if(thrust::device, AD.begin(), AD.end(), SD.begin(), BD.begin(), RD.begin());
  thrust::gather_if(thrust::host,   h_ptr,      h_ptr+4,  SH.begin(), BH.begin(), RH.begin());
  thrust::gather_if(thrust::device, d_ptr,      d_ptr+4,  SD.begin(), BD.begin(), RD.begin());
  
  return 0;
}
