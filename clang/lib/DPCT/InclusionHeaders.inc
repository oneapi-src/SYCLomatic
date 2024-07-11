//===--------------- InclusionHeaders.inc ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/****************************************************************************
*
* ENTRY(Filename, MatchMode, RuleGroupKind, ProcessFlag, MustAngled, Headers...)
*  Filename: File name in inclusion stmt
*  MatchMode: FullMatch|Startwith
*    FullMatch:  this entry will be available only when file name full matched.
*    Startwith:  this entry will be available when file name start with "Filename"
*  RuleGroupKind: Sparse|BLas|Solver|Rng|FFT|DNN|NCCL|Libcu|Thrust
*    Rule group to be enabled.
*  ProcessFlag:  MarkInserted|Replace|Remove|DoNothing
*    MarkInserted: Specified Headers will be marked as inserted.
*    Replace:  Remove current inclusion and insert specified Headers.
*    Remove:  Remove current inclusion.
*    DoNothing:  Do nothing, just a placeholder to avoid other logic.
*  MustAngled:  true|false
*    true:  only angled inclusion can enable this entry.
*    false:  all inclusion can enable this entry.
*  Headers: specified HeaderTypes.
*
****************************************************************************/

REGIST_INCLUSION("time.h", FullMatch, Common, MarkInserted, true,
                 HeaderType::HT_Time)

REGIST_INCLUSION("math.h", FullMatch, Common, MarkInserted, true,
                 HeaderType::HT_Math)
REGIST_INCLUSION("cmath", FullMatch, Common, MarkInserted, true,
                 HeaderType::HT_Math)

REGIST_INCLUSION("algorithm", FullMatch, Common, MarkInserted, true,
                 HeaderType::HT_Algorithm)

REGIST_INCLUSION("cublas_v2.h", FullMatch, BLas, Replace, false,
                 HeaderType::HT_DPCT_BLAS_Utils)
REGIST_INCLUSION("cublas.h", FullMatch, BLas, Replace, false,
                 HeaderType::HT_DPCT_BLAS_Utils)

REGIST_INCLUSION("curand.h", FullMatch, Rng, Replace, false,
                 HeaderType::HT_DPCT_RNG_Utils)
REGIST_INCLUSION("curand_kernel.h", FullMatch, Rng, Replace, false,
                 HeaderType::HT_DPCT_RNG_Utils)

REGIST_INCLUSION("cusparse.h", FullMatch, Sparse, Replace, false,
                 HeaderType::HT_DPCT_SPBLAS_Utils)
REGIST_INCLUSION("cusparse_v2.h", FullMatch, Sparse, Replace, false,
                 HeaderType::HT_DPCT_SPBLAS_Utils,
                 HeaderType::HT_DPCT_BLAS_Utils)

REGIST_INCLUSION("cufft.h", FullMatch, FFT, Replace, false,
                 HeaderType::HT_DPCT_FFT_Utils)

REGIST_INCLUSION("cusolverDn.h", FullMatch, Solver, Replace, false,
                 HeaderType::HT_DPCT_LAPACK_Utils)

REGIST_INCLUSION("nccl.h", FullMatch, NCCL, Replace, false,
                 HeaderType::HT_DPCT_CCL_Utils)

REGIST_INCLUSION("cudnn.h", FullMatch, DNN, Replace, false,
                 HeaderType::HT_DPCT_DNNL_Utils)

REGIST_INCLUSION("cuda/atomic", FullMatch, Libcu, Replace, false,
                 HeaderType::HT_DPCT_Atomic)
REGIST_INCLUSION("cuda/std/atomic", FullMatch, Libcu, Replace, false,
                 HeaderType::HT_DPCT_Atomic)

REGIST_INCLUSION("cuda/std/complex", FullMatch, Libcu, Replace, false,
                 HeaderType::HT_Complex)

REGIST_INCLUSION("cuda/std/array", FullMatch, Libcu, Replace, false,
                 HeaderType::HT_Array)

REGIST_INCLUSION("cuda/std/tuple", FullMatch, Libcu, Replace, false,
                 HeaderType::HT_Tuple)

REGIST_INCLUSION("thrust/complex.h", FullMatch, Thrust, Replace, false,
                 HeaderType::HT_Complex)

REGIST_INCLUSION("thrust/uninitialized_copy.h", FullMatch, Thrust, Replace,
                 false, HeaderType::HT_DPL_Memory)
REGIST_INCLUSION("thrust/uninitialized_fill.h", FullMatch, Thrust, Replace,
                 false, HeaderType::HT_DPL_Memory)

REGIST_INCLUSION("thrust/random.h", FullMatch, Thrust, Replace, false,
                 HeaderType::HT_DPL_Random)

REGIST_INCLUSION("thrust/functional.h", FullMatch, Thrust, Replace, false,
                 HeaderType::HT_Functional, HeaderType::HT_DPCT_DPL_Utils,
                 HeaderType::HT_DPL_Execution, HeaderType::HT_DPL_Algorithm)

REGIST_INCLUSION("thrust/", Startwith, Thrust, Replace, false,
                 HeaderType::HT_DPCT_DPL_Utils, HeaderType::HT_DPL_Execution,
                 HeaderType::HT_DPL_Algorithm)

REGIST_INCLUSION("cub/", Startwith, Thrust, Replace, false,
                 HeaderType::HT_DPCT_DPL_Utils, HeaderType::HT_DPL_Execution,
                 HeaderType::HT_DPL_Algorithm)

REGIST_INCLUSION("CL/", Startwith, Common, DoNothing, false)

REGIST_INCLUSION("cuda_rutime.h", FullMatch, Libcu, Remove, true)

REGIST_INCLUSION("cublasLt.h", FullMatch, BLas, Replace, false,
                 HeaderType::HT_DPCT_BLAS_GEMM_Utils)
