// RUN: dpct --format-range=none --usm-level=none -out-root %T/cufft-different-locations %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cufft-different-locations/cufft-different-locations.dp.cpp --match-full-lines %s
#include <cstdio>
#include <cufft.h>
#include <cuda_runtime.h>

size_t* work_size;
int odist;
int ostride;
int * onembed;
int idist;
int istride;
int* inembed;
int * n;
double* odata;
double2* idata;

#define HANDLE_CUFFT_ERROR( err ) (CufftHandleError( err, __FILE__, __LINE__ ))
static void CufftHandleError( cufftResult err, const char *file, int line ) {
  if (err != CUFFT_SUCCESS) {
    fprintf(stderr, "Cufft error in file '%s' in line %i : %s.\n",
            __FILE__, __LINE__, "error" );
  }
}

int main() {
  cufftHandle plan1;
  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:int res1 = (plan1->commit(&q_ct1, 3, n, inembed, istride, idist, onembed, ostride, odist, dpct::fft::fft_type::complex_double_to_real_double, 12, nullptr), 0);
  cufftResult res1 = cufftMakePlanMany(plan1, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size);
  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:int res2 = (plan1->compute<sycl::double2, double>(idata, odata, dpct::fft::fft_direction::backward), 0);
  cufftResult res2 = cufftExecZ2D(plan1, idata, odata);

  cufftHandle plan2;
  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:res1 = (plan2->commit(&q_ct1, 3, n, inembed, istride, idist, onembed, ostride, odist, dpct::fft::fft_type::complex_double_to_real_double, 12, nullptr), 0);
  res1 = cufftMakePlanMany(plan2, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size);
  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:res2 = (plan2->compute<sycl::double2, double>(idata, odata, dpct::fft::fft_direction::backward), 0);
  res2 = cufftExecZ2D(plan2, idata, odata);

  cufftHandle plan3;
  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:HANDLE_CUFFT_ERROR((plan3->commit(&q_ct1, 3, n, inembed, istride, idist, onembed, ostride, odist, dpct::fft::fft_type::complex_double_to_real_double, 12, nullptr), 0));
  HANDLE_CUFFT_ERROR(cufftMakePlanMany(plan3, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size));
  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:HANDLE_CUFFT_ERROR((plan3->compute<sycl::double2, double>(idata, odata, dpct::fft::fft_direction::backward), 0));
  HANDLE_CUFFT_ERROR(cufftExecZ2D(plan3, idata, odata));

  cufftHandle plan4;
  cufftHandle plan5;
  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:if((plan4->commit(&q_ct1, 3, n, inembed, istride, idist, onembed, ostride, odist, dpct::fft::fft_type::complex_double_to_real_double, 12, nullptr), 0)) {
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:} else if ((plan5->commit(&q_ct1, 3, n, inembed, istride, idist, onembed, ostride, odist, dpct::fft::fft_type::complex_double_to_real_double, 12, nullptr), 0)) {
  //CHECK-NEXT:}
  if(cufftMakePlanMany(plan4, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size)) {
  } else if (cufftMakePlanMany(plan5, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size)) {
  }
  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:if ((plan4->compute<sycl::double2, double>(idata, odata, dpct::fft::fft_direction::backward), 0)) {
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:} else if((plan5->compute<sycl::double2, double>(idata, odata, dpct::fft::fft_direction::backward), 0)) {
  //CHECK-NEXT:}
  if (cufftExecZ2D(plan4, idata, odata)) {
  } else if(cufftExecZ2D(plan5, idata, odata)) {
  }

  cufftHandle plan6;
  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:if(int res = (plan6->commit(&q_ct1, 3, n, inembed, istride, idist, onembed, ostride, odist, dpct::fft::fft_type::complex_double_to_real_double, 12, nullptr), 0)) {
  //CHECK-NEXT:}
  if(cufftResult res = cufftMakePlanMany(plan6, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size)) {
  }
  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:if(int res = (plan6->compute<sycl::double2, double>(idata, odata, dpct::fft::fft_direction::backward), 0)) {
  //CHECK-NEXT:}
  if(cufftResult res = cufftExecZ2D(plan6, idata, odata)) {
  }

  cufftHandle plan7;
  //CHECK:for (plan7->commit(&q_ct1, 3, n, inembed, istride, idist, onembed, ostride, odist, dpct::fft::fft_type::complex_double_to_real_double, 12, nullptr);;) {
  //CHECK-NEXT:}
  for (cufftMakePlanMany(plan7, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size);;) {
  }
  //CHECK:for (plan7->compute<sycl::double2, double>(idata, odata, dpct::fft::fft_direction::backward);;) {
  //CHECK-NEXT:}
  for (cufftExecZ2D(plan7, idata, odata);;) {
  }

  cufftHandle plan8;
  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:for (;(plan8->commit(&q_ct1, 3, n, inembed, istride, idist, onembed, ostride, odist, dpct::fft::fft_type::complex_double_to_real_double, 12, nullptr), 0);) {
  //CHECK-NEXT:}
  for (;cufftMakePlanMany(plan8, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size);) {
  }

  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:for (;(plan8->compute<sycl::double2, double>(idata, odata, dpct::fft::fft_direction::backward), 0);) {
  //CHECK-NEXT:}
  for (;cufftExecZ2D(plan8, idata, odata);) {
  }

  cufftHandle plan9;
  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:while ((plan9->commit(&q_ct1, 3, n, inembed, istride, idist, onembed, ostride, odist, dpct::fft::fft_type::complex_double_to_real_double, 12, nullptr), 0) != 0) {
  //CHECK-NEXT:}
  while (cufftMakePlanMany(plan9, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size) != 0) {
  }

  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:while ((plan9->compute<sycl::double2, double>(idata, odata, dpct::fft::fft_direction::backward), 0) != 0) {
  //CHECK-NEXT:}
  while (cufftExecZ2D(plan9, idata, odata) != 0) {
  }

  cufftHandle plan10;
  //CHECK:do {
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:} while ((plan10->commit(&q_ct1, 3, n, inembed, istride, idist, onembed, ostride, odist, dpct::fft::fft_type::complex_double_to_real_double, 12, nullptr), 0));
  do {
  } while (cufftMakePlanMany(plan10, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size));
  //CHECK:do {
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:} while ((plan10->compute<sycl::double2, double>(idata, odata, dpct::fft::fft_direction::backward), 0));
  do {
  } while (cufftExecZ2D(plan10, idata, odata));

  cufftHandle plan11;
  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:switch (int stat = (plan11->commit(&q_ct1, 3, n, inembed, istride, idist, onembed, ostride, odist, dpct::fft::fft_type::complex_double_to_real_double, 12, nullptr), 0)){
  //CHECK-NEXT:}
  switch (int stat = cufftMakePlanMany(plan11, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size)){
  }

  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:switch (int stat = (plan11->compute<sycl::double2, double>(idata, odata, dpct::fft::fft_direction::backward), 0)){
  //CHECK-NEXT:}
  switch (int stat = cufftExecZ2D(plan11, idata, odata)){
  }
  return 0;
}

cufftResult foo1(cufftHandle plan) {
  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:return (plan->commit(&dpct::get_default_queue(), 3, n, inembed, istride, idist, onembed, ostride, odist, dpct::fft::fft_type::complex_double_to_real_double, 12, nullptr), 0);
  return cufftMakePlanMany(plan, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size);
}

cufftResult foo2(cufftHandle plan) {
  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:return (plan->compute<sycl::double2, double>(idata, odata, dpct::fft::fft_direction::backward), 0);
  return cufftExecZ2D(plan, idata, odata);
}

cufftResult foo3(cufftHandle plan) {
  //CHECK:plan->commit(&dpct::get_default_queue(), 3, n, inembed, istride, idist, onembed, ostride, odist, dpct::fft::fft_type::complex_double_to_real_double, 12, nullptr);
  cufftMakePlanMany(plan, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size);
}

cufftResult foo4(cufftHandle plan) {
  //CHECK:plan->compute<sycl::double2, double>(idata, odata, dpct::fft::fft_direction::backward);
  cufftExecZ2D(plan, idata, odata);
}

