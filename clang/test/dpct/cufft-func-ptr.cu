// RUN: cat %s > %T/cufft-func-ptr.cu
// RUN: cd %T
// RUN: dpct -out-root %T/cufft-func-ptr cufft-func-ptr.cu --cuda-include-path="%cuda-path/include"  -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cufft-func-ptr/cufft-func-ptr.dp.cpp --match-full-lines cufft-func-ptr.cu
#include <cufft.h>

//CHECK:auto static pt2CufftExec =
//CHECK-NEXT:    [](std::shared_ptr<oneapi::mkl::dft::descriptor<
//CHECK-NEXT:           oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>
//CHECK-NEXT:           desc,
//CHECK-NEXT:       sycl::double2 *in_data, double *out_data) {
//CHECK-NEXT:      /*
//CHECK-NEXT:      DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
//CHECK-NEXT:      */
//CHECK-NEXT:      desc->commit(dpct::get_default_queue());
//CHECK-NEXT:      if ((void *)in_data == (void *)out_data) {
//CHECK-NEXT:        oneapi::mkl::dft::compute_backward(*desc, (double *)in_data);
//CHECK-NEXT:      } else {
//CHECK-NEXT:        oneapi::mkl::dft::compute_backward(*desc, (double *)in_data, out_data);
//CHECK-NEXT:      }
//CHECK-NEXT:      return 0;
//CHECK-NEXT:    };
static cufftResult (*pt2CufftExec)(cufftHandle, cufftDoubleComplex *,
                                    double *) = &cufftExecZ2D;

int main() {
//CHECK:  std::shared_ptr<oneapi::mkl::dft::descriptor<
//CHECK-NEXT:      oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>
//CHECK-NEXT:      plan1;
//CHECK-NEXT:  /*
//CHECK-NEXT:  DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be
//CHECK-NEXT:  deduced, so it is assumed out-of-place. You may need to adjust the code.
//CHECK-NEXT:  */
//CHECK-NEXT:  plan1 = std::make_shared<oneapi::mkl::dft::descriptor<
//CHECK-NEXT:      oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(10);
//CHECK-NEXT:  plan1->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
//CHECK-NEXT:                   DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
//CHECK-NEXT:  std::int64_t input_stride_ct0[2] = {0, 1};
//CHECK-NEXT:  plan1->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
//CHECK-NEXT:                   input_stride_ct0);
  cufftHandle plan1;
  cufftPlan1d(&plan1, 10, CUFFT_Z2D, 1);

  double* odata;
  double2* idata;

  pt2CufftExec(plan1, idata, odata);

  return 0;
}

int foo1() {
//CHECK:  typedef int (*Func_t)(
//CHECK-NEXT:      std::shared_ptr<oneapi::mkl::dft::descriptor<
//CHECK-NEXT:          oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>,
//CHECK-NEXT:      sycl::double2 *, double *);
  typedef cufftResult (*Func_t)(cufftHandle, cufftDoubleComplex *, double *);

//CHECK:  auto static FuncPtr =
//CHECK-NEXT:      [](std::shared_ptr<
//CHECK-NEXT:             oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
//CHECK-NEXT:                                          oneapi::mkl::dft::domain::REAL>>
//CHECK-NEXT:             desc,
//CHECK-NEXT:         sycl::double2 *in_data, double *out_data) {
//CHECK-NEXT:        /*
//CHECK-NEXT:        DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require
//CHECK-NEXT:        review.
//CHECK-NEXT:        */
//CHECK-NEXT:        desc->commit(dpct::get_default_queue());
//CHECK-NEXT:        if ((void *)in_data == (void *)out_data) {
//CHECK-NEXT:          oneapi::mkl::dft::compute_backward(*desc, (double *)in_data);
//CHECK-NEXT:        } else {
//CHECK-NEXT:          oneapi::mkl::dft::compute_backward(*desc, (double *)in_data,
//CHECK-NEXT:                                             out_data);
//CHECK-NEXT:        }
//CHECK-NEXT:        return 0;
//CHECK-NEXT:      };
  static Func_t FuncPtr  = &cufftExecZ2D;

//CHECK:  std::shared_ptr<oneapi::mkl::dft::descriptor<
//CHECK-NEXT:      oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>
//CHECK-NEXT:      plan1;
//CHECK-NEXT:  /*
//CHECK-NEXT:  DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be
//CHECK-NEXT:  deduced, so it is assumed out-of-place. You may need to adjust the code.
//CHECK-NEXT:  */
//CHECK-NEXT:  plan1 = std::make_shared<oneapi::mkl::dft::descriptor<
//CHECK-NEXT:      oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(10);
//CHECK-NEXT:  plan1->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
//CHECK-NEXT:                   DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
//CHECK-NEXT:  std::int64_t input_stride_ct{{[0-9]+}}[2] = {0, 1};
//CHECK-NEXT:  plan1->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
//CHECK-NEXT:                   input_stride_ct{{[0-9]+}});
  cufftHandle plan1;
  cufftPlan1d(&plan1, 10, CUFFT_Z2D, 1);

  double* odata;
  double2* idata;

  FuncPtr(plan1, idata, odata);

  return 0;
}

int foo2() {
//CHECK:  using Func_t = int (*)(
//CHECK-NEXT:      std::shared_ptr<oneapi::mkl::dft::descriptor<
//CHECK-NEXT:          oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>,
//CHECK-NEXT:      sycl::double2 *, double *);
  using Func_t = cufftResult (*)(cufftHandle, cufftDoubleComplex *, double *);

//CHECK:  auto FuncPtr2 =
//CHECK-NEXT:      [](std::shared_ptr<
//CHECK-NEXT:             oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
//CHECK-NEXT:                                          oneapi::mkl::dft::domain::REAL>>
//CHECK-NEXT:             desc,
//CHECK-NEXT:         sycl::double2 *in_data, double *out_data) {
//CHECK-NEXT:        /*
//CHECK-NEXT:        DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require
//CHECK-NEXT:        review.
//CHECK-NEXT:        */
//CHECK-NEXT:        desc->commit(dpct::get_default_queue());
//CHECK-NEXT:        if ((void *)in_data == (void *)out_data) {
//CHECK-NEXT:          oneapi::mkl::dft::compute_backward(*desc, (double *)in_data);
//CHECK-NEXT:        } else {
//CHECK-NEXT:          oneapi::mkl::dft::compute_backward(*desc, (double *)in_data,
//CHECK-NEXT:                                             out_data);
//CHECK-NEXT:        }
//CHECK-NEXT:        return 0;
//CHECK-NEXT:      };
  Func_t FuncPtr2  = &cufftExecZ2D;

//CHECK:  std::shared_ptr<oneapi::mkl::dft::descriptor<
//CHECK-NEXT:      oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>
//CHECK-NEXT:      plan1;
//CHECK-NEXT:  /*
//CHECK-NEXT:  DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be
//CHECK-NEXT:  deduced, so it is assumed out-of-place. You may need to adjust the code.
//CHECK-NEXT:  */
//CHECK-NEXT:  plan1 = std::make_shared<oneapi::mkl::dft::descriptor<
//CHECK-NEXT:      oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(10);
//CHECK-NEXT:  plan1->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
//CHECK-NEXT:                   DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
//CHECK-NEXT:  std::int64_t input_stride_ct{{[0-9]+}}[2] = {0, 1};
//CHECK-NEXT:  plan1->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
//CHECK-NEXT:                   input_stride_ct{{[0-9]+}});
  cufftHandle plan1;
  cufftPlan1d(&plan1, 10, CUFFT_Z2D, 1);

  double* odata;
  double2* idata;

  FuncPtr2(plan1, idata, odata);

  return 0;
}

int foo3() {
//CHECK:  using Func_t = int (*)(
//CHECK-NEXT:      std::shared_ptr<oneapi::mkl::dft::descriptor<
//CHECK-NEXT:          oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>,
//CHECK-NEXT:      sycl::double2 *, double *);
  using Func_t = cufftResult (*)(cufftHandle, cufftDoubleComplex *, double *);

//CHECK:  Func_t FuncPtr3;
//CHECK-NEXT:  FuncPtr3 =
//CHECK-NEXT:      [](std::shared_ptr<
//CHECK-NEXT:             oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
//CHECK-NEXT:                                          oneapi::mkl::dft::domain::REAL>>
//CHECK-NEXT:             desc,
//CHECK-NEXT:         sycl::double2 *in_data, double *out_data) {
//CHECK-NEXT:        /*
//CHECK-NEXT:        DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require
//CHECK-NEXT:        review.
//CHECK-NEXT:        */
//CHECK-NEXT:        desc->commit(dpct::get_default_queue());
//CHECK-NEXT:        if ((void *)in_data == (void *)out_data) {
//CHECK-NEXT:          oneapi::mkl::dft::compute_backward(*desc, (double *)in_data);
//CHECK-NEXT:        } else {
//CHECK-NEXT:          oneapi::mkl::dft::compute_backward(*desc, (double *)in_data,
//CHECK-NEXT:                                             out_data);
//CHECK-NEXT:        }
//CHECK-NEXT:        return 0;
//CHECK-NEXT:      };
  Func_t FuncPtr3;
  FuncPtr3 = &cufftExecZ2D;

//CHECK:  std::shared_ptr<oneapi::mkl::dft::descriptor<
//CHECK-NEXT:      oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>
//CHECK-NEXT:      plan1;
//CHECK-NEXT:  /*
//CHECK-NEXT:  DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be
//CHECK-NEXT:  deduced, so it is assumed out-of-place. You may need to adjust the code.
//CHECK-NEXT:  */
//CHECK-NEXT:  plan1 = std::make_shared<oneapi::mkl::dft::descriptor<
//CHECK-NEXT:      oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(10);
//CHECK-NEXT:  plan1->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
//CHECK-NEXT:                   DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
//CHECK-NEXT:  std::int64_t input_stride_ct{{[0-9]+}}[2] = {0, 1};
//CHECK-NEXT:  plan1->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
//CHECK-NEXT:                   input_stride_ct{{[0-9]+}});
  cufftHandle plan1;
  cufftPlan1d(&plan1, 10, CUFFT_Z2D, 1);

  double* odata;
  double2* idata;

  FuncPtr3(plan1, idata, odata);

  return 0;
}

int foo4() {
//CHECK:  int (*FuncPtr4)(
//CHECK-NEXT:      std::shared_ptr<oneapi::mkl::dft::descriptor<
//CHECK-NEXT:          oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>,
//CHECK-NEXT:      sycl::double2 *, double *);
  cufftResult (*FuncPtr4)(cufftHandle, cufftDoubleComplex *, double *);

//CHECK:  FuncPtr4 =
//CHECK-NEXT:      [](std::shared_ptr<
//CHECK-NEXT:             oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
//CHECK-NEXT:                                          oneapi::mkl::dft::domain::REAL>>
//CHECK-NEXT:             desc,
//CHECK-NEXT:         sycl::double2 *in_data, double *out_data) {
//CHECK-NEXT:        /*
//CHECK-NEXT:        DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require
//CHECK-NEXT:        review.
//CHECK-NEXT:        */
//CHECK-NEXT:        desc->commit(dpct::get_default_queue());
//CHECK-NEXT:        if ((void *)in_data == (void *)out_data) {
//CHECK-NEXT:          oneapi::mkl::dft::compute_backward(*desc, (double *)in_data);
//CHECK-NEXT:        } else {
//CHECK-NEXT:          oneapi::mkl::dft::compute_backward(*desc, (double *)in_data,
//CHECK-NEXT:                                             out_data);
//CHECK-NEXT:        }
//CHECK-NEXT:        return 0;
//CHECK-NEXT:      };
  FuncPtr4 = &cufftExecZ2D;

//CHECK:  std::shared_ptr<oneapi::mkl::dft::descriptor<
//CHECK-NEXT:      oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>
//CHECK-NEXT:      plan1;
//CHECK-NEXT:  /*
//CHECK-NEXT:  DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be
//CHECK-NEXT:  deduced, so it is assumed out-of-place. You may need to adjust the code.
//CHECK-NEXT:  */
//CHECK-NEXT:  plan1 = std::make_shared<oneapi::mkl::dft::descriptor<
//CHECK-NEXT:      oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(10);
//CHECK-NEXT:  plan1->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
//CHECK-NEXT:                   DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
//CHECK-NEXT:  std::int64_t input_stride_ct{{[0-9]+}}[2] = {0, 1};
//CHECK-NEXT:  plan1->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
//CHECK-NEXT:                   input_stride_ct{{[0-9]+}});
  cufftHandle plan1;
  cufftPlan1d(&plan1, 10, CUFFT_Z2D, 1);

  double* odata;
  double2* idata;

  FuncPtr4(plan1, idata, odata);

  return 0;
}