//===---APINamesNccl.inc--------------------------------------------===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::CclUtils_get_version,
                        ASSIGNABLE_FACTORY(
                                  ASSIGN_FACTORY_ENTRY("ncclGetVersion",DEREF(0),
                                                        CALL(MapNames::getDpctNamespace() + "ccl::get_version"))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::CclUtils_create_kvs_address,
                        ASSIGNABLE_FACTORY(
                                  ASSIGN_FACTORY_ENTRY("ncclGetUniqueId",DEREF(0),
                                                        CALL(MapNames::getDpctNamespace() + "ccl::create_kvs_address"))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::CclUtils_create_kvs,
                        ASSIGNABLE_FACTORY(
                                  ASSIGN_FACTORY_ENTRY("ncclCommInitRank",DEREF(0),
                                                        CALL("oneapi::ccl::create_communicator",
                                                        ARG(1), ARG(3),
                                                        CALL(MapNames::getDpctNamespace() + "ccl::create_kvs", ARG(2))))))