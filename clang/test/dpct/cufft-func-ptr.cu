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
//CHECK-NEXT:      if ((void *)in_data == (void *)out_data) {
//CHECK-NEXT:        oneapi::mkl::dft::compute_backward(*desc, (double *)in_data);
//CHECK-NEXT:      } else {
//CHECK-NEXT:        oneapi::mkl::dft::compute_backward(*desc, (double *)in_data, out_data);
//CHECK-NEXT:      }
//CHECK-NEXT:    };
static cufftResult (*pt2CufftExec)(cufftHandle, cufftDoubleComplex *,
                                    double *) = &cufftExecZ2D;

int main() {
//CHECK:  std::shared_ptr<oneapi::mkl::dft::descriptor<
//CHECK-NEXT:      oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>
//CHECK-NEXT:      plan1;
//CHECK-NEXT:  /*
//CHECK-NEXT:  DPCT1071:0: The placement of the FFT computational function cannot be deduced.
//CHECK-NEXT:  It is migrated as out-of-place. You may need to adjust the code.
//CHECK-NEXT:  */
//CHECK-NEXT:  plan1 = std::make_shared<oneapi::mkl::dft::descriptor<
//CHECK-NEXT:      oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(10);
//CHECK-NEXT:  plan1->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
//CHECK-NEXT:                   DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
//CHECK-NEXT:  std::int64_t input_stride_ct0[2] = {0, 1};
//CHECK-NEXT:  plan1->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
//CHECK-NEXT:                   input_stride_ct0);
//CHECK-NEXT:  plan1->commit(dpct::get_default_queue());
  cufftHandle plan1;
  cufftPlan1d(&plan1, 10, CUFFT_Z2D, 1);

  double* odata;
  double2* idata;

  pt2CufftExec(plan1, idata, odata);

  return 0;
}