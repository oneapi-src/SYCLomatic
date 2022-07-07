//===--------------------APINamesLIBCU.inc----------------------------===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

// Base API

CALL_FACTORY_ENTRY("cuda::std::atomic_thread_fence", CALL(MapNames::getClNamespace() + "atomic_fence",
                                          ARG(0)))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::AtomicUtils_atomic_utils_store,
                        MEMBER_CALL_FACTORY_ENTRY(
                            "store",  DEREF(0), false, "store",
                            ARG(0),ARG(1)))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::AtomicUtils_atomic_utils_load,
                        MEMBER_CALL_FACTORY_ENTRY(
                            "load",  DEREF(0), false, "load",
                            ARG(0)))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::AtomicUtils_atomic_utils_exchange,
                        MEMBER_CALL_FACTORY_ENTRY(
                            "exchange",  DEREF(0), false, "exchange",
                            ARG(0),ARG(1)))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::AtomicUtils_atomic_utils_compare_exchange_weak,
                        MEMBER_CALL_FACTORY_ENTRY(
                            "compare_exchange_weak",  DEREF(0), false, "compare_exchange_weak",
                            ARG(0),ARG(1), ARG(2), ARG(3)))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::AtomicUtils_atomic_utils_compare_exchange_strong,
                        MEMBER_CALL_FACTORY_ENTRY(
                            "compare_exchange_strong",  DEREF(0), false, "compare_exchange_strong",
                            ARG(0),ARG(1), ARG(2), ARG(3)))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::AtomicUtils_atomic_utils_fetch_add,
                        MEMBER_CALL_FACTORY_ENTRY(
                            "fetch_add",  DEREF(0), false, "fetch_add",
                            ARG(0),ARG(1)))
                            
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::AtomicUtils_atomic_utils_fetch_sub,
                        MEMBER_CALL_FACTORY_ENTRY(
                            "fetch_sub",  DEREF(0), false, "fetch_sub",
                            ARG(0),ARG(1)))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::AtomicUtils_atomic_utils_fetch_sub,
                        MEMBER_CALL_FACTORY_ENTRY(
                            "fetch_and",  DEREF(0), false, "fetch_and",
                            ARG(0),ARG(1)))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::AtomicUtils_atomic_utils_fetch_or,
                        MEMBER_CALL_FACTORY_ENTRY(
                            "fetch_or",  DEREF(0), false, "fetch_or",
                            ARG(0),ARG(1))) 

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::AtomicUtils_atomic_utils_fetch_xor,
                        MEMBER_CALL_FACTORY_ENTRY(
                            "fetch_xor",  DEREF(0), false, "fetch_xor",
                            ARG(0),ARG(1)))           