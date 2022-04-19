//===---APINamesCUBLAS.inc ---------------------------------------------===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::LibCommonUtils_mkl_get_version,
    CALL_FACTORY_ENTRY("cublasGetVersion_v2",
                       CALL(MapNames::getDpctNamespace() + "mkl_get_version",
                            ARG(MapNames::getDpctNamespace() + "version_field::major"), ARG(1))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::LibCommonUtils_mkl_get_version,
    CALL_FACTORY_ENTRY("cublasGetVersion",
                       CALL(MapNames::getDpctNamespace() + "mkl_get_version",
                            ARG(MapNames::getDpctNamespace() + "version_field::major"), ARG(0))))
