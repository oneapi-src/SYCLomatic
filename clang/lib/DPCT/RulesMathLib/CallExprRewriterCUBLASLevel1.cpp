//===--------------- CallExprRewriterCUBLASLevel1.cpp ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterCUBLAS.h"

using namespace clang::dpct;

RewriterMap dpct::createCUBLASLevel1RewriterMap() {
  return RewriterMap{
      // cublasIsamax_v2_64
      WARNING_FACTORY_ENTRY(
          "cublasIsamax_v2_64",
          LAMBDA_FACTORY_ENTRY(
              "cublasIsamax_v2_64",
              DECLARE(MapNames::getLibraryHelperNamespace() +
                          "blas::wrapper_int64_out",
                      "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"),
                      ARG(4)),
              CALL("oneapi::mkl::blas::column_major::iamax",
                   MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
                   BUFFER_OR_USM_PTR(ARG(2), "float"), ARG(3),
                   BUFFER_OR_USM_PTR(
                       MEMBER_CALL(ARG("res_wrapper_ct4"), false, "get_ptr"),
                       "std::int64_t"),
                   ARG("oneapi::mkl::index_base::one")),
              LITERAL("return 0")),
          Diagnostics::NOERROR_RETURN_LAMBDA)
      // cublasIdamax_v2_64
      WARNING_FACTORY_ENTRY(
          "cublasIdamax_v2_64",
          LAMBDA_FACTORY_ENTRY(
              "cublasIdamax_v2_64",
              DECLARE(MapNames::getLibraryHelperNamespace() +
                          "blas::wrapper_int64_out",
                      "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"),
                      ARG(4)),
              CALL("oneapi::mkl::blas::column_major::iamax",
                   MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
                   BUFFER_OR_USM_PTR(ARG(2), "double"), ARG(3),
                   BUFFER_OR_USM_PTR(
                       MEMBER_CALL(ARG("res_wrapper_ct4"), false, "get_ptr"),
                       "std::int64_t"),
                   ARG("oneapi::mkl::index_base::one")),
              LITERAL("return 0")),
          Diagnostics::NOERROR_RETURN_LAMBDA)
      // cublasIcamax_v2_64
      WARNING_FACTORY_ENTRY(
          "cublasIcamax_v2_64",
          LAMBDA_FACTORY_ENTRY(
              "cublasIcamax_v2_64",
              DECLARE(MapNames::getLibraryHelperNamespace() +
                          "blas::wrapper_int64_out",
                      "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"),
                      ARG(4)),
              CALL("oneapi::mkl::blas::column_major::iamax",
                   MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
                   BUFFER_OR_USM_PTR(ARG(2), "std::complex<float>"), ARG(3),
                   BUFFER_OR_USM_PTR(
                       MEMBER_CALL(ARG("res_wrapper_ct4"), false, "get_ptr"),
                       "std::int64_t"),
                   ARG("oneapi::mkl::index_base::one")),
              LITERAL("return 0")),
          Diagnostics::NOERROR_RETURN_LAMBDA)
      // cublasIzamax_v2_64
      WARNING_FACTORY_ENTRY(
          "cublasIzamax_v2_64",
          LAMBDA_FACTORY_ENTRY(
              "cublasIzamax_v2_64",
              DECLARE(MapNames::getLibraryHelperNamespace() +
                          "blas::wrapper_int64_out",
                      "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"),
                      ARG(4)),
              CALL("oneapi::mkl::blas::column_major::iamax",
                   MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
                   BUFFER_OR_USM_PTR(ARG(2), "std::complex<double>"), ARG(3),
                   BUFFER_OR_USM_PTR(
                       MEMBER_CALL(ARG("res_wrapper_ct4"), false, "get_ptr"),
                       "std::int64_t"),
                   ARG("oneapi::mkl::index_base::one")),
              LITERAL("return 0")),
          Diagnostics::NOERROR_RETURN_LAMBDA)
      // cublasIsamin_v2_64
      WARNING_FACTORY_ENTRY(
          "cublasIsamin_v2_64",
          LAMBDA_FACTORY_ENTRY(
              "cublasIsamin_v2_64",
              DECLARE(MapNames::getLibraryHelperNamespace() +
                          "blas::wrapper_int64_out",
                      "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"),
                      ARG(4)),
              CALL("oneapi::mkl::blas::column_major::iamin",
                   MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
                   BUFFER_OR_USM_PTR(ARG(2), "float"), ARG(3),
                   BUFFER_OR_USM_PTR(
                       MEMBER_CALL(ARG("res_wrapper_ct4"), false, "get_ptr"),
                       "std::int64_t"),
                   ARG("oneapi::mkl::index_base::one")),
              LITERAL("return 0")),
          Diagnostics::NOERROR_RETURN_LAMBDA)
      // cublasIdamin_v2_64
      WARNING_FACTORY_ENTRY(
          "cublasIdamin_v2_64",
          LAMBDA_FACTORY_ENTRY(
              "cublasIdamin_v2_64",
              DECLARE(MapNames::getLibraryHelperNamespace() +
                          "blas::wrapper_int64_out",
                      "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"),
                      ARG(4)),
              CALL("oneapi::mkl::blas::column_major::iamin",
                   MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
                   BUFFER_OR_USM_PTR(ARG(2), "double"), ARG(3),
                   BUFFER_OR_USM_PTR(
                       MEMBER_CALL(ARG("res_wrapper_ct4"), false, "get_ptr"),
                       "std::int64_t"),
                   ARG("oneapi::mkl::index_base::one")),
              LITERAL("return 0")),
          Diagnostics::NOERROR_RETURN_LAMBDA)
      // cublasIcamin_v2_64
      WARNING_FACTORY_ENTRY(
          "cublasIcamin_v2_64",
          LAMBDA_FACTORY_ENTRY(
              "cublasIcamin_v2_64",
              DECLARE(MapNames::getLibraryHelperNamespace() +
                          "blas::wrapper_int64_out",
                      "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"),
                      ARG(4)),
              CALL("oneapi::mkl::blas::column_major::iamin",
                   MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
                   BUFFER_OR_USM_PTR(ARG(2), "std::complex<float>"), ARG(3),
                   BUFFER_OR_USM_PTR(
                       MEMBER_CALL(ARG("res_wrapper_ct4"), false, "get_ptr"),
                       "std::int64_t"),
                   ARG("oneapi::mkl::index_base::one")),
              LITERAL("return 0")),
          Diagnostics::NOERROR_RETURN_LAMBDA)
      // cublasIzamin_v2_64
      WARNING_FACTORY_ENTRY(
          "cublasIzamin_v2_64",
          LAMBDA_FACTORY_ENTRY(
              "cublasIzamin_v2_64",
              DECLARE(MapNames::getLibraryHelperNamespace() +
                          "blas::wrapper_int64_out",
                      "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"),
                      ARG(4)),
              CALL("oneapi::mkl::blas::column_major::iamin",
                   MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
                   BUFFER_OR_USM_PTR(ARG(2), "std::complex<double>"), ARG(3),
                   BUFFER_OR_USM_PTR(
                       MEMBER_CALL(ARG("res_wrapper_ct4"), false, "get_ptr"),
                       "std::int64_t"),
                   ARG("oneapi::mkl::index_base::one")),
              LITERAL("return 0")),
          Diagnostics::NOERROR_RETURN_LAMBDA)
      // cublasIsamax_v2
      WARNING_FACTORY_ENTRY(
          "cublasIsamax_v2",
          LAMBDA_FACTORY_ENTRY(
              "cublasIsamax_v2",
              DECLARE(MapNames::getLibraryHelperNamespace() +
                          "blas::wrapper_int_to_int64_out",
                      "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"),
                      ARG(4)),
              CALL("oneapi::mkl::blas::column_major::iamax",
                   MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
                   BUFFER_OR_USM_PTR(ARG(2), "float"), ARG(3),
                   BUFFER_OR_USM_PTR(
                       MEMBER_CALL(ARG("res_wrapper_ct4"), false, "get_ptr"),
                       "std::int64_t"),
                   ARG("oneapi::mkl::index_base::one")),
              LITERAL("return 0")),
          Diagnostics::NOERROR_RETURN_LAMBDA)
      // cublasIdamax_v2
      WARNING_FACTORY_ENTRY(
          "cublasIdamax_v2",
          LAMBDA_FACTORY_ENTRY(
              "cublasIdamax_v2",
              DECLARE(MapNames::getLibraryHelperNamespace() +
                          "blas::wrapper_int_to_int64_out",
                      "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"),
                      ARG(4)),
              CALL("oneapi::mkl::blas::column_major::iamax",
                   MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
                   BUFFER_OR_USM_PTR(ARG(2), "double"), ARG(3),
                   BUFFER_OR_USM_PTR(
                       MEMBER_CALL(ARG("res_wrapper_ct4"), false, "get_ptr"),
                       "std::int64_t"),
                   ARG("oneapi::mkl::index_base::one")),
              LITERAL("return 0")),
          Diagnostics::NOERROR_RETURN_LAMBDA)
      // cublasIcamax_v2
      WARNING_FACTORY_ENTRY(
          "cublasIcamax_v2",
          LAMBDA_FACTORY_ENTRY(
              "cublasIcamax_v2",
              DECLARE(MapNames::getLibraryHelperNamespace() +
                          "blas::wrapper_int_to_int64_out",
                      "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"),
                      ARG(4)),
              CALL("oneapi::mkl::blas::column_major::iamax",
                   MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
                   BUFFER_OR_USM_PTR(ARG(2), "std::complex<float>"), ARG(3),
                   BUFFER_OR_USM_PTR(
                       MEMBER_CALL(ARG("res_wrapper_ct4"), false, "get_ptr"),
                       "std::int64_t"),
                   ARG("oneapi::mkl::index_base::one")),
              LITERAL("return 0")),
          Diagnostics::NOERROR_RETURN_LAMBDA)
      // cublasIzamax_v2
      WARNING_FACTORY_ENTRY(
          "cublasIzamax_v2",
          LAMBDA_FACTORY_ENTRY(
              "cublasIzamax_v2",
              DECLARE(MapNames::getLibraryHelperNamespace() +
                          "blas::wrapper_int_to_int64_out",
                      "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"),
                      ARG(4)),
              CALL("oneapi::mkl::blas::column_major::iamax",
                   MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
                   BUFFER_OR_USM_PTR(ARG(2), "std::complex<double>"), ARG(3),
                   BUFFER_OR_USM_PTR(
                       MEMBER_CALL(ARG("res_wrapper_ct4"), false, "get_ptr"),
                       "std::int64_t"),
                   ARG("oneapi::mkl::index_base::one")),
              LITERAL("return 0")),
          Diagnostics::NOERROR_RETURN_LAMBDA)
      // cublasIsamin_v2
      WARNING_FACTORY_ENTRY(
          "cublasIsamin_v2",
          LAMBDA_FACTORY_ENTRY(
              "cublasIsamin_v2",
              DECLARE(MapNames::getLibraryHelperNamespace() +
                          "blas::wrapper_int_to_int64_out",
                      "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"),
                      ARG(4)),
              CALL("oneapi::mkl::blas::column_major::iamin",
                   MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
                   BUFFER_OR_USM_PTR(ARG(2), "float"), ARG(3),
                   BUFFER_OR_USM_PTR(
                       MEMBER_CALL(ARG("res_wrapper_ct4"), false, "get_ptr"),
                       "std::int64_t"),
                   ARG("oneapi::mkl::index_base::one")),
              LITERAL("return 0")),
          Diagnostics::NOERROR_RETURN_LAMBDA)
      // cublasIdamin_v2
      WARNING_FACTORY_ENTRY(
          "cublasIdamin_v2",
          LAMBDA_FACTORY_ENTRY(
              "cublasIdamin_v2",
              DECLARE(MapNames::getLibraryHelperNamespace() +
                          "blas::wrapper_int_to_int64_out",
                      "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"),
                      ARG(4)),
              CALL("oneapi::mkl::blas::column_major::iamin",
                   MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
                   BUFFER_OR_USM_PTR(ARG(2), "double"), ARG(3),
                   BUFFER_OR_USM_PTR(
                       MEMBER_CALL(ARG("res_wrapper_ct4"), false, "get_ptr"),
                       "std::int64_t"),
                   ARG("oneapi::mkl::index_base::one")),
              LITERAL("return 0")),
          Diagnostics::NOERROR_RETURN_LAMBDA)
      // cublasIcamin_v2
      WARNING_FACTORY_ENTRY(
          "cublasIcamin_v2",
          LAMBDA_FACTORY_ENTRY(
              "cublasIcamin_v2",
              DECLARE(MapNames::getLibraryHelperNamespace() +
                          "blas::wrapper_int_to_int64_out",
                      "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"),
                      ARG(4)),
              CALL("oneapi::mkl::blas::column_major::iamin",
                   MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
                   BUFFER_OR_USM_PTR(ARG(2), "std::complex<float>"), ARG(3),
                   BUFFER_OR_USM_PTR(
                       MEMBER_CALL(ARG("res_wrapper_ct4"), false, "get_ptr"),
                       "std::int64_t"),
                   ARG("oneapi::mkl::index_base::one")),
              LITERAL("return 0")),
          Diagnostics::NOERROR_RETURN_LAMBDA)
      // cublasIzamin_v2
      WARNING_FACTORY_ENTRY(
          "cublasIzamin_v2",
          LAMBDA_FACTORY_ENTRY(
              "cublasIzamin_v2",
              DECLARE(MapNames::getLibraryHelperNamespace() +
                          "blas::wrapper_int_to_int64_out",
                      "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"),
                      ARG(4)),
              CALL("oneapi::mkl::blas::column_major::iamin",
                   MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
                   BUFFER_OR_USM_PTR(ARG(2), "std::complex<double>"), ARG(3),
                   BUFFER_OR_USM_PTR(
                       MEMBER_CALL(ARG("res_wrapper_ct4"), false, "get_ptr"),
                       "std::int64_t"),
                   ARG("oneapi::mkl::index_base::one")),
              LITERAL("return 0")),
          Diagnostics::NOERROR_RETURN_LAMBDA)

#define NRM2(FUNC, TYPE1, TYPE2, TYPE3)                                        \
  WARNING_FACTORY_ENTRY(                                                       \
      #FUNC,                                                                   \
      LAMBDA_FACTORY_ENTRY(                                                    \
          #FUNC,                                                               \
          DECLARE(MapNames::getLibraryHelperNamespace() + "blas::" #TYPE1,     \
                  "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"),   \
                  ARG(4)),                                                     \
          CALL("oneapi::mkl::blas::column_major::nrm2",                        \
               MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),                 \
               BUFFER_OR_USM_PTR(ARG(2), #TYPE2), ARG(3),                      \
               BUFFER_OR_USM_PTR(                                              \
                   MEMBER_CALL(ARG("res_wrapper_ct4"), false, "get_ptr"),      \
                   #TYPE3)),                                                   \
          LITERAL("return 0")),                                                \
      Diagnostics::NOERROR_RETURN_LAMBDA)
      // cublasSnrm2_v2
      NRM2(cublasSnrm2_v2, wrapper_float_out, float, float)
      // cublasDnrm2_v2
      NRM2(cublasDnrm2_v2, wrapper_double_out, double, double)
      // cublasScnrm2_v2
      NRM2(cublasScnrm2_v2, wrapper_float_out, std::complex<float>, float)
      // cublasDznrm2_v2
      NRM2(cublasDznrm2_v2, wrapper_double_out, std::complex<double>, double)
      // cublasSnrm2_v2_64
      NRM2(cublasSnrm2_v2_64, wrapper_float_out, float, float)
      // cublasDnrm2_v2_64
      NRM2(cublasDnrm2_v2_64, wrapper_double_out, double, double)
      // cublasScnrm2_v2_64
      NRM2(cublasScnrm2_v2_64, wrapper_float_out, std::complex<float>, float)
      // cublasDznrm2_v2_64
      NRM2(cublasDznrm2_v2_64, wrapper_double_out, std::complex<double>, double)
#undef NRM2

#define DOT(FUNC, NEW_FUNC, TYPE1, TYPE2, TYPE3)                               \
  WARNING_FACTORY_ENTRY(                                                       \
      #FUNC,                                                                   \
      LAMBDA_FACTORY_ENTRY(                                                    \
          #FUNC,                                                               \
          DECLARE(MapNames::getLibraryHelperNamespace() + "blas::" #TYPE1,     \
                  "res_wrapper_ct6", MEMBER_CALL(ARG(0), true, "get_queue"),   \
                  ARG(6)),                                                     \
          CALL("oneapi::mkl::blas::column_major::" #NEW_FUNC,                  \
               MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),                 \
               BUFFER_OR_USM_PTR(ARG(2), #TYPE2), ARG(3),                      \
               BUFFER_OR_USM_PTR(ARG(4), #TYPE2), ARG(5),                      \
               BUFFER_OR_USM_PTR(                                              \
                   MEMBER_CALL(ARG("res_wrapper_ct6"), false, "get_ptr"),      \
                   #TYPE3)),                                                   \
          LITERAL("return 0")),                                                \
      Diagnostics::NOERROR_RETURN_LAMBDA)
      // cublasSdot_v2
      DOT(cublasSdot_v2, dot, wrapper_float_out, float, float)
      // cublasDdot_v2
      DOT(cublasDdot_v2, dot, wrapper_double_out, double, double)
      // cublasCdotu_v2
      DOT(cublasCdotu_v2, dotu, wrapper_float2_out, std::complex<float>,
          std::complex<float>)
      // cublasCdotc_v2
      DOT(cublasCdotc_v2, dotc, wrapper_float2_out, std::complex<float>,
          std::complex<float>)
      // cublasZdotu_v2
      DOT(cublasZdotu_v2, dotu, wrapper_double2_out, std::complex<double>,
          std::complex<double>)
      // cublasZdotc_v2
      DOT(cublasZdotc_v2, dotc, wrapper_double2_out, std::complex<double>,
          std::complex<double>)
      // cublasSdot_v2_64
      DOT(cublasSdot_v2_64, dot, wrapper_float_out, float, float)
      // cublasDdot_v2_64
      DOT(cublasDdot_v2_64, dot, wrapper_double_out, double, double)
      // cublasCdotu_v2_64
      DOT(cublasCdotu_v2_64, dotu, wrapper_float2_out, std::complex<float>,
          std::complex<float>)
      // cublasCdotc_v2_64
      DOT(cublasCdotc_v2_64, dotc, wrapper_float2_out, std::complex<float>,
          std::complex<float>)
      // cublasZdotu_v2_64
      DOT(cublasZdotu_v2_64, dotu, wrapper_double2_out, std::complex<double>,
          std::complex<double>)
      // cublasZdotc_v2_64
      DOT(cublasZdotc_v2_64, dotc, wrapper_double2_out, std::complex<double>,
          std::complex<double>)
#undef DOT

#define SCAL(FUNC, TYPE1)                                                      \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #FUNC, CALL("oneapi::mkl::blas::column_major::scal",                     \
                  MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),              \
                  SCALAR_INPUT(ARG(2), #TYPE1),                                \
                  BUFFER_OR_USM_PTR(ARG(3), #TYPE1), ARG(4))))
      // cublasSscal_v2
      SCAL(cublasSscal_v2, float)
      // cublasDscal_v2
      SCAL(cublasDscal_v2, double)
      // cublasCscal_v2
      SCAL(cublasCscal_v2, std::complex<float>)
      // cublasCsscal_v2
      SCAL(cublasCsscal_v2, std::complex<float>)
      // cublasZscal_v2
      SCAL(cublasZscal_v2, std::complex<double>)
      // cublasZdscal_v2
      SCAL(cublasZdscal_v2, std::complex<double>)
      // cublasSscal_v2_64
      SCAL(cublasSscal_v2_64, float)
      // cublasDscal_v2_64
      SCAL(cublasDscal_v2_64, double)
      // cublasCscal_v2_64
      SCAL(cublasCscal_v2_64, std::complex<float>)
      // cublasCsscal_v2_64
      SCAL(cublasCsscal_v2_64, std::complex<float>)
      // cublasZscal_v2_64
      SCAL(cublasZscal_v2_64, std::complex<double>)
      // cublasZdscal_v2_64
      SCAL(cublasZdscal_v2_64, std::complex<double>)
#undef SCAL

#define AXPY(FUNC, TYPE1)                                                      \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #FUNC,                                                                   \
      CALL("oneapi::mkl::blas::column_major::axpy",                            \
           MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),                     \
           SCALAR_INPUT(ARG(2), #TYPE1), BUFFER_OR_USM_PTR(ARG(3), #TYPE1),    \
           ARG(4), BUFFER_OR_USM_PTR(ARG(5), #TYPE1), ARG(6))))
      // cublasSaxpy_v2
      AXPY(cublasSaxpy_v2, float)
      // cublasDaxpy_v2
      AXPY(cublasDaxpy_v2, double)
      // cublasCaxpy_v2
      AXPY(cublasCaxpy_v2, std::complex<float>)
      // cublasZaxpy_v2
      AXPY(cublasZaxpy_v2, std::complex<double>)
      // cublasSaxpy_v2_64
      AXPY(cublasSaxpy_v2_64, float)
      // cublasDaxpy_v2_64
      AXPY(cublasDaxpy_v2_64, double)
      // cublasCaxpy_v2_64
      AXPY(cublasCaxpy_v2_64, std::complex<float>)
      // cublasZaxpy_v2_64
      AXPY(cublasZaxpy_v2_64, std::complex<double>)
#undef AXPY

#define COPY(FUNC, TYPE1)                                                      \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #FUNC, CALL("oneapi::mkl::blas::column_major::copy",                     \
                  MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),              \
                  BUFFER_OR_USM_PTR(ARG(2), #TYPE1), ARG(3),                   \
                  BUFFER_OR_USM_PTR(ARG(4), #TYPE1), ARG(5))))
      // cublasScopy_v2
      COPY(cublasScopy_v2, float)
      // cublasDcopy_v2
      COPY(cublasDcopy_v2, double)
      // cublasCcopy_v2
      COPY(cublasCcopy_v2, std::complex<float>)
      // cublasZcopy_v2
      COPY(cublasZcopy_v2, std::complex<double>)
      // cublasScopy_v2_64
      COPY(cublasScopy_v2_64, float)
      // cublasDcopy_v2_64
      COPY(cublasDcopy_v2_64, double)
      // cublasCcopy_v2_64
      COPY(cublasCcopy_v2_64, std::complex<float>)
      // cublasZcopy_v2_64
      COPY(cublasZcopy_v2_64, std::complex<double>)
#undef COPY

#define SWAP(FUNC, TYPE1)                                                      \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #FUNC, CALL("oneapi::mkl::blas::column_major::swap",                     \
                  MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),              \
                  BUFFER_OR_USM_PTR(ARG(2), #TYPE1), ARG(3),                   \
                  BUFFER_OR_USM_PTR(ARG(4), #TYPE1), ARG(5))))
      // cublasSswap_v2
      SWAP(cublasSswap_v2, float)
      // cublasDswap_v2
      SWAP(cublasDswap_v2, double)
      // cublasCswap_v2
      SWAP(cublasCswap_v2, std::complex<float>)
      // cublasZswap_v2
      SWAP(cublasZswap_v2, std::complex<double>)
      // cublasSswap_v2_64
      SWAP(cublasSswap_v2_64, float)
      // cublasDswap_v2_64
      SWAP(cublasDswap_v2_64, double)
      // cublasCswap_v2_64
      SWAP(cublasCswap_v2_64, std::complex<float>)
      // cublasZswap_v2_64
      SWAP(cublasZswap_v2_64, std::complex<double>)
#undef SWAP

#define ASUM(FUNC, TYPE1, TYPE2, TYPE3)                                        \
  WARNING_FACTORY_ENTRY(                                                       \
      #FUNC,                                                                   \
      LAMBDA_FACTORY_ENTRY(                                                    \
          #FUNC,                                                               \
          DECLARE(MapNames::getLibraryHelperNamespace() + "blas::" #TYPE1,     \
                  "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"),   \
                  ARG(4)),                                                     \
          CALL("oneapi::mkl::blas::column_major::asum",                        \
               MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),                 \
               BUFFER_OR_USM_PTR(ARG(2), #TYPE2), ARG(3),                      \
               BUFFER_OR_USM_PTR(                                              \
                   MEMBER_CALL(ARG("res_wrapper_ct4"), false, "get_ptr"),      \
                   #TYPE3)),                                                   \
          LITERAL("return 0")),                                                \
      Diagnostics::NOERROR_RETURN_LAMBDA)
      // cublasSasum_v2
      ASUM(cublasSasum_v2, wrapper_float_out, float, float)
      // cublasDasum_v2
      ASUM(cublasDasum_v2, wrapper_double_out, double, double)
      // cublasScasum_v2
      ASUM(cublasScasum_v2, wrapper_float_out, std::complex<float>, float)
      // cublasDzasum_v2
      ASUM(cublasDzasum_v2, wrapper_double_out, std::complex<double>, double)
      // cublasSasum_v2_64
      ASUM(cublasSasum_v2_64, wrapper_float_out, float, float)
      // cublasDasum_v2_64
      ASUM(cublasDasum_v2_64, wrapper_double_out, double, double)
      // cublasScasum_v2_64
      ASUM(cublasScasum_v2_64, wrapper_float_out, std::complex<float>, float)
      // cublasDzasum_v2_64
      ASUM(cublasDzasum_v2_64, wrapper_double_out, std::complex<double>, double)
#undef ASUM

#define ROT(FUNC, TYPE1)                                                       \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #FUNC,                                                                   \
      CALL("oneapi::mkl::blas::column_major::rot",                             \
           MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),                     \
           BUFFER_OR_USM_PTR(ARG(2), #TYPE1), ARG(3),                          \
           BUFFER_OR_USM_PTR(ARG(4), #TYPE1), ARG(5),                          \
           SCALAR_INPUT(ARG(6), #TYPE1), SCALAR_INPUT(ARG(7), #TYPE1))))
      // cublasSrot_v2
      ROT(cublasSrot_v2, float)
      // cublasDrot_v2
      ROT(cublasDrot_v2, double)
      // cublasCrot_v2
      ROT(cublasCrot_v2, std::complex<float>)
      // cublasCsrot_v2
      ROT(cublasCsrot_v2, std::complex<float>)
      // cublasZrot_v2
      ROT(cublasZrot_v2, std::complex<double>)
      // cublasZdrot_v2
      ROT(cublasZdrot_v2, std::complex<double>)
      // cublasSrot_v2_64
      ROT(cublasSrot_v2_64, float)
      // cublasDrot_v2_64
      ROT(cublasDrot_v2_64, double)
      // cublasCrot_v2_64
      ROT(cublasCrot_v2_64, std::complex<float>)
      // cublasCsrot_v2_64
      ROT(cublasCsrot_v2_64, std::complex<float>)
      // cublasZrot_v2_64
      ROT(cublasZrot_v2_64, std::complex<double>)
      // cublasZdrot_v2_64
      ROT(cublasZdrot_v2_64, std::complex<double>)
#undef ROT

#define ROTG(FUNC, TYPE1, TYPE2, TYPE3, TYPE4, TYPE5)                          \
  WARNING_FACTORY_ENTRY(                                                       \
      #FUNC,                                                                   \
      LAMBDA_FACTORY_ENTRY(                                                    \
          #FUNC,                                                               \
          DECLARE(MapNames::getLibraryHelperNamespace() + "blas::" #TYPE1,     \
                  "res_wrapper_ct1", MEMBER_CALL(ARG(0), true, "get_queue"),   \
                  ARG(1)),                                                     \
          DECLARE(MapNames::getLibraryHelperNamespace() + "blas::" #TYPE1,     \
                  "res_wrapper_ct2", MEMBER_CALL(ARG(0), true, "get_queue"),   \
                  ARG(2)),                                                     \
          DECLARE(MapNames::getLibraryHelperNamespace() + "blas::" #TYPE2,     \
                  "res_wrapper_ct3", MEMBER_CALL(ARG(0), true, "get_queue"),   \
                  ARG(3)),                                                     \
          DECLARE(MapNames::getLibraryHelperNamespace() + "blas::" #TYPE3,     \
                  "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"),   \
                  ARG(4)),                                                     \
          CALL("oneapi::mkl::blas::column_major::rotg",                        \
               MEMBER_CALL(ARG(0), true, "get_queue"),                         \
               BUFFER_OR_USM_PTR(                                              \
                   MEMBER_CALL(ARG("res_wrapper_ct1"), false, "get_ptr"),      \
                   #TYPE4),                                                    \
               BUFFER_OR_USM_PTR(                                              \
                   MEMBER_CALL(ARG("res_wrapper_ct2"), false, "get_ptr"),      \
                   #TYPE4),                                                    \
               BUFFER_OR_USM_PTR(                                              \
                   MEMBER_CALL(ARG("res_wrapper_ct3"), false, "get_ptr"),      \
                   #TYPE5),                                                    \
               BUFFER_OR_USM_PTR(                                              \
                   MEMBER_CALL(ARG("res_wrapper_ct4"), false, "get_ptr"),      \
                   #TYPE4)),                                                   \
          LITERAL("return 0")),                                                \
      Diagnostics::NOERROR_RETURN_LAMBDA)
      // cublasSrotg_v2
      ROTG(cublasSrotg_v2, wrapper_float_inout, wrapper_float_out,
           wrapper_float_out, float, float)
      // cublasDrotg_v2
      ROTG(cublasDrotg_v2, wrapper_double_inout, wrapper_double_out,
           wrapper_double_out, double, double)
      // cublasCrotg_v2
      ROTG(cublasCrotg_v2, wrapper_float2_inout, wrapper_float_out,
           wrapper_float2_out, std::complex<float>, float)
      // cublasZrotg_v2
      ROTG(cublasZrotg_v2, wrapper_double2_inout, wrapper_double_out,
           wrapper_double2_out, std::complex<double>, double)
#undef ROTG

#define ROTM(FUNC, TYPE1, TYPE2)                                               \
  WARNING_FACTORY_ENTRY(                                                       \
      #FUNC,                                                                   \
      LAMBDA_FACTORY_ENTRY(                                                    \
          #FUNC,                                                               \
          DECLARE(MapNames::getLibraryHelperNamespace() + "blas::" #TYPE1,     \
                  "res_wrapper_ct6", MEMBER_CALL(ARG(0), true, "get_queue"),   \
                  ARG(6), ARG("5")),                                           \
          CALL("oneapi::mkl::blas::column_major::rotm",                        \
               MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),                 \
               BUFFER_OR_USM_PTR(ARG(2), #TYPE2), ARG(3),                      \
               BUFFER_OR_USM_PTR(ARG(4), #TYPE2), ARG(5),                      \
               BUFFER_OR_USM_PTR(                                              \
                   MEMBER_CALL(ARG("res_wrapper_ct6"), false, "get_ptr"),      \
                   #TYPE2)),                                                   \
          LITERAL("return 0")),                                                \
      Diagnostics::NOERROR_RETURN_LAMBDA)
      // cublasSrotm_v2
      ROTM(cublasSrotm_v2, wrapper_float_in, float)
      // cublasDrotm_v2
      ROTM(cublasDrotm_v2, wrapper_double_in, double)
      // cublasSrotm_v2_64
      ROTM(cublasSrotm_v2_64, wrapper_float_in, float)
      // cublasDrotm_v2_64
      ROTM(cublasDrotm_v2_64, wrapper_double_in, double)
#undef ROTM

#define ROTMG(FUNC, TYPE1, TYPE2, TYPE3)                                       \
  WARNING_FACTORY_ENTRY(                                                       \
      #FUNC,                                                                   \
      LAMBDA_FACTORY_ENTRY(                                                    \
          #FUNC,                                                               \
          DECLARE(MapNames::getLibraryHelperNamespace() + "blas::" #TYPE1,     \
                  "res_wrapper_ct1", MEMBER_CALL(ARG(0), true, "get_queue"),   \
                  ARG(1)),                                                     \
          DECLARE(MapNames::getLibraryHelperNamespace() + "blas::" #TYPE1,     \
                  "res_wrapper_ct2", MEMBER_CALL(ARG(0), true, "get_queue"),   \
                  ARG(2)),                                                     \
          DECLARE(MapNames::getLibraryHelperNamespace() + "blas::" #TYPE1,     \
                  "res_wrapper_ct3", MEMBER_CALL(ARG(0), true, "get_queue"),   \
                  ARG(3)),                                                     \
          DECLARE(MapNames::getLibraryHelperNamespace() + "blas::" #TYPE2,     \
                  "res_wrapper_ct5", MEMBER_CALL(ARG(0), true, "get_queue"),   \
                  ARG(5), ARG("5")),                                           \
          CALL("oneapi::mkl::blas::column_major::rotmg",                       \
               MEMBER_CALL(ARG(0), true, "get_queue"),                         \
               BUFFER_OR_USM_PTR(                                              \
                   MEMBER_CALL(ARG("res_wrapper_ct1"), false, "get_ptr"),      \
                   #TYPE3),                                                    \
               BUFFER_OR_USM_PTR(                                              \
                   MEMBER_CALL(ARG("res_wrapper_ct2"), false, "get_ptr"),      \
                   #TYPE3),                                                    \
               BUFFER_OR_USM_PTR(                                              \
                   MEMBER_CALL(ARG("res_wrapper_ct3"), false, "get_ptr"),      \
                   #TYPE3),                                                    \
               SCALAR_INPUT(ARG(4), #TYPE3),                                   \
               BUFFER_OR_USM_PTR(                                              \
                   MEMBER_CALL(ARG("res_wrapper_ct5"), false, "get_ptr"),      \
                   #TYPE3)),                                                   \
          LITERAL("return 0")),                                                \
      Diagnostics::NOERROR_RETURN_LAMBDA)
      // cublasSrotmg_v2
      ROTMG(cublasSrotmg_v2, wrapper_float_inout, wrapper_float_out, float)
      // cublasDrotmg_v2
      ROTMG(cublasDrotmg_v2, wrapper_double_inout, wrapper_double_out, double)
#undef ROTMG

  };
}
