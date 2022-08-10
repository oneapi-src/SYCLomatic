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
    HelperFeatureEnum::FftUtils_fft_solver,
    ASSIGN_FACTORY_ENTRY("cufftPlan1d", DEREF(0),
                       CALL("std::make_shared<" + MapNames::getDpctNamespace() + "fft::fft_solver>",
                            ARG(1), ARG(2), ARG(3)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_solver,
    ASSIGN_FACTORY_ENTRY("cufftMakePlan1d", ARG(0),
                       CALL("std::make_shared<" + MapNames::getDpctNamespace() + "fft::fft_solver>",
                            ARG(1), ARG(2), ARG(3)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_solver,
    ASSIGN_FACTORY_ENTRY("cufftPlan2d", DEREF(0),
                       CALL("std::make_shared<" + MapNames::getDpctNamespace() + "fft::fft_solver>",
                            ARG(1), ARG(2), ARG(3)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_solver,
    ASSIGN_FACTORY_ENTRY("cufftMakePlan2d", ARG(0),
                       CALL("std::make_shared<" + MapNames::getDpctNamespace() + "fft::fft_solver>",
                            ARG(1), ARG(2), ARG(3)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_solver,
    ASSIGN_FACTORY_ENTRY("cufftPlan3d", DEREF(0),
                       CALL("std::make_shared<" + MapNames::getDpctNamespace() + "fft::fft_solver>",
                            ARG(1), ARG(2), ARG(3), ARG(4)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_solver,
    ASSIGN_FACTORY_ENTRY("cufftMakePlan3d", ARG(0),
                       CALL("std::make_shared<" + MapNames::getDpctNamespace() + "fft::fft_solver>",
                            ARG(1), ARG(2), ARG(3), ARG(4)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_solver,
    ASSIGN_FACTORY_ENTRY("cufftPlanMany", DEREF(0),
                       CALL("std::make_shared<" + MapNames::getDpctNamespace() + "fft::fft_solver>",
                            ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_solver,
    ASSIGN_FACTORY_ENTRY("cufftMakePlanMany", ARG(0),
                       CALL("std::make_shared<" + MapNames::getDpctNamespace() + "fft::fft_solver>",
                            ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_solver,
    ASSIGN_FACTORY_ENTRY("cufftMakePlanMany64", ARG(0),
                       CALL("std::make_shared<" + MapNames::getDpctNamespace() + "fft::fft_solver>",
                            ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_solver,
    ASSIGN_FACTORY_ENTRY("cufftXtMakePlanMany", ARG(0),
                       CALL("std::make_shared<" + MapNames::getDpctNamespace() + "fft::fft_solver>",
                            ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10), ARG(11)))))

REMOVE_API_FACTORY_ENTRY("cufftCreate")
REMOVE_API_FACTORY_ENTRY("cufftDestroy")

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_solver,
    MEMBER_CALL_FACTORY_ENTRY("cufftExecR2C", ARG(0), true, "compute", ARG(1), ARG(2), ARG("dpct::fft::fft_dir::forward"))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_solver,
    MEMBER_CALL_FACTORY_ENTRY("cufftExecC2R", ARG(0), true, "compute", ARG(1), ARG(2), ARG("dpct::fft::fft_dir::backward"))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_solver,
    MEMBER_CALL_FACTORY_ENTRY("cufftExecD2Z", ARG(0), true, "compute", ARG(1), ARG(2), ARG("dpct::fft::fft_dir::forward"))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_solver,
    MEMBER_CALL_FACTORY_ENTRY("cufftExecZ2D", ARG(0), true, "compute", ARG(1), ARG(2), ARG("dpct::fft::fft_dir::backward"))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_solver,
    MEMBER_CALL_FACTORY_ENTRY("cufftExecC2C", ARG(0), true, "compute", ARG(1), ARG(2), ARG(3))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_solver,
    MEMBER_CALL_FACTORY_ENTRY("cufftExecZ2Z", ARG(0), true, "compute", ARG(1), ARG(2), ARG(3))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_solver,
    MEMBER_CALL_FACTORY_ENTRY("cufftXtExec", ARG(0), true, "compute", ARG(1), ARG(2), ARG(3))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::FftUtils_fft_solver,
    MEMBER_CALL_FACTORY_ENTRY("cufftSetStream", ARG(0), true, "set_queue", ARG(1))))
