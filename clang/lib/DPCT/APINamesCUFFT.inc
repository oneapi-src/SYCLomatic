//===--------------- APINamesCUFFT.inc ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::LibCommonUtils_mkl_get_version,
    CALL_FACTORY_ENTRY("cufftGetVersion",
                       CALL(MapNames::getDpctNamespace() + "mkl_get_version",
                            ARG(MapNames::getDpctNamespace() + "version_field::major"), ARG(0)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::LibCommonUtils_mkl_get_version,
    CALL_FACTORY_ENTRY("cufftGetProperty",
                       CALL(MapNames::getDpctNamespace() + "mkl_get_version",
                            ARG(0), ARG(1)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_engine,
    ASSIGN_FACTORY_ENTRY("cufftPlan1d", DEREF(0),
                       CALL(MapNames::getDpctNamespace() + "fft::fft_engine::create",
                            QUEUEPTRSTR, ARG(1), ARG(2), ARG(3)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_engine,
    ASSIGN_FACTORY_ENTRY("cufftPlan2d", DEREF(0),
                       CALL(MapNames::getDpctNamespace() + "fft::fft_engine::create",
                            QUEUEPTRSTR, ARG(1), ARG(2), ARG(3)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_engine,
    ASSIGN_FACTORY_ENTRY("cufftPlan3d", DEREF(0),
                       CALL(MapNames::getDpctNamespace() + "fft::fft_engine::create",
                            QUEUEPTRSTR, ARG(1), ARG(2), ARG(3), ARG(4)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_engine,
    ASSIGN_FACTORY_ENTRY("cufftPlanMany", DEREF(0),
                       CALL(MapNames::getDpctNamespace() + "fft::fft_engine::create",
                            QUEUEPTRSTR, ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_engine,
    WARNING_FACTORY_ENTRY("cufftMakePlan1d",
    WARNING_FACTORY_ENTRY("cufftMakePlan1d",
    MEMBER_CALL_FACTORY_ENTRY("cufftMakePlan1d", ARG(0), true, "commit", QUEUEPTRSTR, ARG(1), ARG(2), ARG(3), ARG(4)), Diagnostics::FFT_DIRECTION_AND_PLACEMENT, ARG("commit")), Diagnostics::FFT_EXTERNAL_WORKSPACE)))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_engine,
    WARNING_FACTORY_ENTRY("cufftMakePlan2d",
    WARNING_FACTORY_ENTRY("cufftMakePlan2d",
    MEMBER_CALL_FACTORY_ENTRY("cufftMakePlan2d", ARG(0), true, "commit", QUEUEPTRSTR, ARG(1), ARG(2), ARG(3), ARG(4)), Diagnostics::FFT_DIRECTION_AND_PLACEMENT, ARG("commit")), Diagnostics::FFT_EXTERNAL_WORKSPACE)))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_engine,
    WARNING_FACTORY_ENTRY("cufftMakePlan3d",
    WARNING_FACTORY_ENTRY("cufftMakePlan3d",
    MEMBER_CALL_FACTORY_ENTRY("cufftMakePlan3d", ARG(0), true, "commit", QUEUEPTRSTR, ARG(1), ARG(2), ARG(3), ARG(4), ARG(5)), Diagnostics::FFT_DIRECTION_AND_PLACEMENT, ARG("commit")), Diagnostics::FFT_EXTERNAL_WORKSPACE)))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_engine,
    WARNING_FACTORY_ENTRY("cufftMakePlanMany",
    WARNING_FACTORY_ENTRY("cufftMakePlanMany",
    MEMBER_CALL_FACTORY_ENTRY("cufftMakePlanMany", ARG(0), true, "commit",
                              QUEUEPTRSTR, ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10), ARG(11)), Diagnostics::FFT_DIRECTION_AND_PLACEMENT, ARG("commit")), Diagnostics::FFT_EXTERNAL_WORKSPACE)))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_engine,
    WARNING_FACTORY_ENTRY("cufftMakePlanMany64",
    WARNING_FACTORY_ENTRY("cufftMakePlanMany64",
    MEMBER_CALL_FACTORY_ENTRY("cufftMakePlanMany64", ARG(0), true, "commit",
                              QUEUEPTRSTR, ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10), ARG(11)), Diagnostics::FFT_DIRECTION_AND_PLACEMENT, ARG("commit")), Diagnostics::FFT_EXTERNAL_WORKSPACE)))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_engine,
    WARNING_FACTORY_ENTRY("cufftXtMakePlanMany",
    WARNING_FACTORY_ENTRY("cufftXtMakePlanMany",
    MEMBER_CALL_FACTORY_ENTRY("cufftXtMakePlanMany", ARG(0), true, "commit",
                              QUEUEPTRSTR, ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10), ARG(11), ARG(12)), Diagnostics::FFT_DIRECTION_AND_PLACEMENT, ARG("commit")), Diagnostics::FFT_EXTERNAL_WORKSPACE)))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_engine,
    ASSIGN_FACTORY_ENTRY("cufftCreate", DEREF(0),
                       CALL(MapNames::getDpctNamespace() + "fft::fft_engine::create"))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_engine,
    CALL_FACTORY_ENTRY("cufftDestroy",
                       CALL(MapNames::getDpctNamespace() + "fft::fft_engine::destroy", ARG(0)))))

ASSIGNABLE_FACTORY_WITH_PAREN(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_engine,
    MEMBER_CALL_FACTORY_ENTRY("cufftExecR2C", ARG(0), true, "compute<float, " + MapNames::getClNamespace() + "float2>", ARG(1), ARG(2), ARG(MapNames::getDpctNamespace() + "fft::fft_direction::forward"))))
ASSIGNABLE_FACTORY_WITH_PAREN(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_engine,
    MEMBER_CALL_FACTORY_ENTRY("cufftExecC2R", ARG(0), true, "compute<" + MapNames::getClNamespace() + "float2, float>", ARG(1), ARG(2), ARG(MapNames::getDpctNamespace() + "fft::fft_direction::backward"))))
ASSIGNABLE_FACTORY_WITH_PAREN(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_engine,
    MEMBER_CALL_FACTORY_ENTRY("cufftExecD2Z", ARG(0), true, "compute<double, " + MapNames::getClNamespace() + "double2>", ARG(1), ARG(2), ARG(MapNames::getDpctNamespace() + "fft::fft_direction::forward"))))
ASSIGNABLE_FACTORY_WITH_PAREN(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_engine,
    MEMBER_CALL_FACTORY_ENTRY("cufftExecZ2D", ARG(0), true, "compute<" + MapNames::getClNamespace() + "double2, double>", ARG(1), ARG(2), ARG(MapNames::getDpctNamespace() + "fft::fft_direction::backward"))))
ASSIGNABLE_FACTORY_WITH_PAREN(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_engine,
    MEMBER_CALL_FACTORY_ENTRY("cufftExecC2C", ARG(0), true, "compute<" + MapNames::getClNamespace() + "float2, " + MapNames::getClNamespace() + "float2>", ARG(1), ARG(2), makeFFTDirExprCallArgCreator(3))))
ASSIGNABLE_FACTORY_WITH_PAREN(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_engine,
    MEMBER_CALL_FACTORY_ENTRY("cufftExecZ2Z", ARG(0), true, "compute<" + MapNames::getClNamespace() + "double2, " + MapNames::getClNamespace() + "double2>", ARG(1), ARG(2), makeFFTDirExprCallArgCreator(3))))
ASSIGNABLE_FACTORY_WITH_PAREN(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_engine,
    MEMBER_CALL_FACTORY_ENTRY("cufftXtExec", ARG(0), true, "compute<void, void>", ARG(1), ARG(2), makeFFTDirExprCallArgCreator(3))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_engine,
    MEMBER_CALL_FACTORY_ENTRY("cufftSetStream", ARG(0), true, "set_queue", ARG(1))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_engine,
    WARNING_FACTORY_ENTRY("cufftEstimate1d",
    WARNING_FACTORY_ENTRY("cufftEstimate1d",
    CALL_FACTORY_ENTRY("cufftEstimate1d",
                       CALL(MapNames::getDpctNamespace() + "fft::fft_engine::estimate_size",
                            ARG(0), ARG(1), ARG(2), ARG(3))), Diagnostics::FFT_DIRECTION_AND_PLACEMENT, ARG(MapNames::getDpctNamespace() + "fft::fft_engine::estimate_size")), Diagnostics::FFT_EXTERNAL_WORKSPACE)))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_engine,
    WARNING_FACTORY_ENTRY("cufftEstimate2d",
    WARNING_FACTORY_ENTRY("cufftEstimate2d",
    CALL_FACTORY_ENTRY("cufftEstimate2d",
                       CALL(MapNames::getDpctNamespace() + "fft::fft_engine::estimate_size",
                            ARG(0), ARG(1), ARG(2), ARG(3))), Diagnostics::FFT_DIRECTION_AND_PLACEMENT, ARG(MapNames::getDpctNamespace() + "fft::fft_engine::estimate_size")), Diagnostics::FFT_EXTERNAL_WORKSPACE)))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_engine,
    WARNING_FACTORY_ENTRY("cufftEstimate3d",
    WARNING_FACTORY_ENTRY("cufftEstimate3d",
    CALL_FACTORY_ENTRY("cufftEstimate3d",
                       CALL(MapNames::getDpctNamespace() + "fft::fft_engine::estimate_size",
                            ARG(0), ARG(1), ARG(2), ARG(3), ARG(4))), Diagnostics::FFT_DIRECTION_AND_PLACEMENT, ARG(MapNames::getDpctNamespace() + "fft::fft_engine::estimate_size")), Diagnostics::FFT_EXTERNAL_WORKSPACE)))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_engine,
    WARNING_FACTORY_ENTRY("cufftEstimateMany",
    WARNING_FACTORY_ENTRY("cufftEstimateMany",
    CALL_FACTORY_ENTRY("cufftEstimateMany",
                       CALL(MapNames::getDpctNamespace() + "fft::fft_engine::estimate_size",
                            ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10))), Diagnostics::FFT_DIRECTION_AND_PLACEMENT, ARG(MapNames::getDpctNamespace() + "fft::fft_engine::estimate_size")), Diagnostics::FFT_EXTERNAL_WORKSPACE)))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_engine,
    WARNING_FACTORY_ENTRY("cufftGetSize1d",
    WARNING_FACTORY_ENTRY("cufftGetSize1d",
    CALL_FACTORY_ENTRY("cufftGetSize1d",
                       CALL(MapNames::getDpctNamespace() + "fft::fft_engine::estimate_size",
                            ARG(1), ARG(2), ARG(3), ARG(4))), Diagnostics::FFT_DIRECTION_AND_PLACEMENT, ARG(MapNames::getDpctNamespace() + "fft::fft_engine::estimate_size")), Diagnostics::FFT_EXTERNAL_WORKSPACE)))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_engine,
    WARNING_FACTORY_ENTRY("cufftGetSize2d",
    WARNING_FACTORY_ENTRY("cufftGetSize2d",
    CALL_FACTORY_ENTRY("cufftGetSize2d",
                       CALL(MapNames::getDpctNamespace() + "fft::fft_engine::estimate_size",
                            ARG(1), ARG(2), ARG(3), ARG(4))), Diagnostics::FFT_DIRECTION_AND_PLACEMENT, ARG(MapNames::getDpctNamespace() + "fft::fft_engine::estimate_size")), Diagnostics::FFT_EXTERNAL_WORKSPACE)))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_engine,
    WARNING_FACTORY_ENTRY("cufftGetSize3d",
    WARNING_FACTORY_ENTRY("cufftGetSize3d",
    CALL_FACTORY_ENTRY("cufftGetSize3d",
                       CALL(MapNames::getDpctNamespace() + "fft::fft_engine::estimate_size",
                            ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))), Diagnostics::FFT_DIRECTION_AND_PLACEMENT, ARG(MapNames::getDpctNamespace() + "fft::fft_engine::estimate_size")), Diagnostics::FFT_EXTERNAL_WORKSPACE)))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_engine,
    WARNING_FACTORY_ENTRY("cufftGetSizeMany",
    WARNING_FACTORY_ENTRY("cufftGetSizeMany",
    CALL_FACTORY_ENTRY("cufftGetSizeMany",
                       CALL(MapNames::getDpctNamespace() + "fft::fft_engine::estimate_size",
                            ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10), ARG(11))), Diagnostics::FFT_DIRECTION_AND_PLACEMENT, ARG(MapNames::getDpctNamespace() + "fft::fft_engine::estimate_size")), Diagnostics::FFT_EXTERNAL_WORKSPACE)))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_engine,
    WARNING_FACTORY_ENTRY("cufftGetSizeMany64",
    WARNING_FACTORY_ENTRY("cufftGetSizeMany64",
    CALL_FACTORY_ENTRY("cufftGetSizeMany64",
                       CALL(MapNames::getDpctNamespace() + "fft::fft_engine::estimate_size",
                            ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10), ARG(11))), Diagnostics::FFT_DIRECTION_AND_PLACEMENT, ARG(MapNames::getDpctNamespace() + "fft::fft_engine::estimate_size")), Diagnostics::FFT_EXTERNAL_WORKSPACE)))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_engine,
    WARNING_FACTORY_ENTRY("cufftGetSize",
    MEMBER_CALL_FACTORY_ENTRY("cufftGetSize", ARG(0), true, "get_workspace_size", ARG(1)), Diagnostics::FFT_EXTERNAL_WORKSPACE)))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_engine,
    WARNING_FACTORY_ENTRY("cufftSetAutoAllocation",
    MEMBER_CALL_FACTORY_ENTRY("cufftSetAutoAllocation", ARG(0), true, "use_internal_workspace", ARG(1)), Diagnostics::FFT_EXTERNAL_WORKSPACE)))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_engine,
    WARNING_FACTORY_ENTRY("cufftSetWorkArea",
    MEMBER_CALL_FACTORY_ENTRY("cufftSetWorkArea", ARG(0), true, "set_workspace", ARG(1)), Diagnostics::FFT_EXTERNAL_WORKSPACE)))
