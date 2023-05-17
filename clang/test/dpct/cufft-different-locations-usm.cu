// RUN: dpct --format-range=none -out-root %T/cufft-different-locations-usm %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cufft-different-locations-usm/cufft-different-locations-usm.dp.cpp --match-full-lines %s
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
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:int res1 = DPCT_CHECK_ERROR(plan1->commit(&q_ct1, 3, n, inembed, istride, idist, onembed, ostride, odist, dpct::fft::fft_type::complex_double_to_real_double, 12, work_size));
  cufftResult res1 = cufftMakePlanMany(plan1, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size);
  //CHECK:int res2 = DPCT_CHECK_ERROR(plan1->compute<sycl::double2, double>(idata, odata, dpct::fft::fft_direction::backward));
  cufftResult res2 = cufftExecZ2D(plan1, idata, odata);

  cufftHandle plan2;
  //CHECK:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:res1 = DPCT_CHECK_ERROR(plan2->commit(&q_ct1, 3, n, inembed, istride, idist, onembed, ostride, odist, dpct::fft::fft_type::complex_double_to_real_double, 12, work_size));
  res1 = cufftMakePlanMany(plan2, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size);
  //CHECK:res2 = DPCT_CHECK_ERROR(plan2->compute<sycl::double2, double>(idata, odata, dpct::fft::fft_direction::backward));
  res2 = cufftExecZ2D(plan2, idata, odata);

  cufftHandle plan3;
  //CHECK:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:HANDLE_CUFFT_ERROR(DPCT_CHECK_ERROR(plan3->commit(&q_ct1, 3, n, inembed, istride, idist, onembed, ostride, odist, dpct::fft::fft_type::complex_double_to_real_double, 12, work_size)));
  HANDLE_CUFFT_ERROR(cufftMakePlanMany(plan3, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size));
  //CHECK:HANDLE_CUFFT_ERROR(DPCT_CHECK_ERROR(plan3->compute<sycl::double2, double>(idata, odata, dpct::fft::fft_direction::backward)));
  HANDLE_CUFFT_ERROR(cufftExecZ2D(plan3, idata, odata));

  cufftHandle plan4;
  cufftHandle plan5;
  //CHECK:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:if(DPCT_CHECK_ERROR(plan4->commit(&q_ct1, 3, n, inembed, istride, idist, onembed, ostride, odist, dpct::fft::fft_type::complex_double_to_real_double, 12, work_size))) {
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:} else if (DPCT_CHECK_ERROR(plan5->commit(&q_ct1, 3, n, inembed, istride, idist, onembed, ostride, odist, dpct::fft::fft_type::complex_double_to_real_double, 12, work_size))) {
  //CHECK-NEXT:}
  if(cufftMakePlanMany(plan4, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size)) {
  } else if (cufftMakePlanMany(plan5, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size)) {
  }
  //CHECK:if (DPCT_CHECK_ERROR(plan4->compute<sycl::double2, double>(idata, odata, dpct::fft::fft_direction::backward))) {
  //CHECK-NEXT:} else if(DPCT_CHECK_ERROR(plan5->compute<sycl::double2, double>(idata, odata, dpct::fft::fft_direction::backward))) {
  //CHECK-NEXT:}
  if (cufftExecZ2D(plan4, idata, odata)) {
  } else if(cufftExecZ2D(plan5, idata, odata)) {
  }

  cufftHandle plan6;
  //CHECK:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:if(int res = DPCT_CHECK_ERROR(plan6->commit(&q_ct1, 3, n, inembed, istride, idist, onembed, ostride, odist, dpct::fft::fft_type::complex_double_to_real_double, 12, work_size))) {
  //CHECK-NEXT:}
  if(cufftResult res = cufftMakePlanMany(plan6, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size)) {
  }
  //CHECK:if(int res = DPCT_CHECK_ERROR(plan6->compute<sycl::double2, double>(idata, odata, dpct::fft::fft_direction::backward))) {
  //CHECK-NEXT:}
  if(cufftResult res = cufftExecZ2D(plan6, idata, odata)) {
  }

  cufftHandle plan7;
  //CHECK:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:for (plan7->commit(&q_ct1, 3, n, inembed, istride, idist, onembed, ostride, odist, dpct::fft::fft_type::complex_double_to_real_double, 12, work_size);;) {
  //CHECK-NEXT:}
  for (cufftMakePlanMany(plan7, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size);;) {
  }
  //CHECK:for (plan7->compute<sycl::double2, double>(idata, odata, dpct::fft::fft_direction::backward);;) {
  //CHECK-NEXT:}
  for (cufftExecZ2D(plan7, idata, odata);;) {
  }

  cufftHandle plan8;
  //CHECK:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:for (;DPCT_CHECK_ERROR(plan8->commit(&q_ct1, 3, n, inembed, istride, idist, onembed, ostride, odist, dpct::fft::fft_type::complex_double_to_real_double, 12, work_size));) {
  //CHECK-NEXT:}
  for (;cufftMakePlanMany(plan8, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size);) {
  }

  //CHECK:for (;DPCT_CHECK_ERROR(plan8->compute<sycl::double2, double>(idata, odata, dpct::fft::fft_direction::backward));) {
  //CHECK-NEXT:}
  for (;cufftExecZ2D(plan8, idata, odata);) {
  }

  cufftHandle plan9;
  //CHECK:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:while (DPCT_CHECK_ERROR(plan9->commit(&q_ct1, 3, n, inembed, istride, idist, onembed, ostride, odist, dpct::fft::fft_type::complex_double_to_real_double, 12, work_size)) != 0) {
  //CHECK-NEXT:}
  while (cufftMakePlanMany(plan9, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size) != 0) {
  }

  //CHECK:while (DPCT_CHECK_ERROR(plan9->compute<sycl::double2, double>(idata, odata, dpct::fft::fft_direction::backward)) != 0) {
  //CHECK-NEXT:}
  while (cufftExecZ2D(plan9, idata, odata) != 0) {
  }

  cufftHandle plan10;
  //CHECK:do {
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:} while (DPCT_CHECK_ERROR(plan10->commit(&q_ct1, 3, n, inembed, istride, idist, onembed, ostride, odist, dpct::fft::fft_type::complex_double_to_real_double, 12, work_size)));
  do {
  } while (cufftMakePlanMany(plan10, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size));
  //CHECK:do {
  //CHECK-NEXT:} while (DPCT_CHECK_ERROR(plan10->compute<sycl::double2, double>(idata, odata, dpct::fft::fft_direction::backward)));
  do {
  } while (cufftExecZ2D(plan10, idata, odata));

  cufftHandle plan11;
  //CHECK:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:switch (int stat = DPCT_CHECK_ERROR(plan11->commit(&q_ct1, 3, n, inembed, istride, idist, onembed, ostride, odist, dpct::fft::fft_type::complex_double_to_real_double, 12, work_size))){
  //CHECK-NEXT:}
  switch (int stat = cufftMakePlanMany(plan11, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size)){
  }

  //CHECK:switch (int stat = DPCT_CHECK_ERROR(plan11->compute<sycl::double2, double>(idata, odata, dpct::fft::fft_direction::backward))){
  //CHECK-NEXT:}
  switch (int stat = cufftExecZ2D(plan11, idata, odata)){
  }
  return 0;
}

cufftResult foo1(cufftHandle plan) {
  //CHECK:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:return DPCT_CHECK_ERROR(plan->commit(&dpct::get_default_queue(), 3, n, inembed, istride, idist, onembed, ostride, odist, dpct::fft::fft_type::complex_double_to_real_double, 12, work_size));
  return cufftMakePlanMany(plan, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size);
}

cufftResult foo2(cufftHandle plan) {
  //CHECK:return DPCT_CHECK_ERROR(plan->compute<sycl::double2, double>(idata, odata, dpct::fft::fft_direction::backward));
  return cufftExecZ2D(plan, idata, odata);
}

cufftResult foo3(cufftHandle plan) {
  //CHECK:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan->commit(&dpct::get_default_queue(), 3, n, inembed, istride, idist, onembed, ostride, odist, dpct::fft::fft_type::complex_double_to_real_double, 12, work_size);
  cufftMakePlanMany(plan, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size);
}

cufftResult foo4(cufftHandle plan) {
  //CHECK:plan->compute<sycl::double2, double>(idata, odata, dpct::fft::fft_direction::backward);
  cufftExecZ2D(plan, idata, odata);
}

