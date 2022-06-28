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

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::AtomicHelper_atomic_helper_store,
ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                            "store",  false, "store",
                            ARG(0),ARG(1))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::AtomicHelper_atomic_helper_load,
ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                            "load",  false, "load",
                            ARG(0))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::AtomicHelper_atomic_helper_exchange,
ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                            "exchange",  false, "exchange",
                            ARG(0),ARG(1))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::AtomicHelper_atomic_helper_compare_exchange_weak,
ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                            "compare_exchange_weak",  false, "compare_exchange_weak",
                            ARG(0),ARG(1), ARG(2), ARG(3))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::AtomicHelper_atomic_helper_compare_exchange_strong,
ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                            "compare_exchange_strong",  false, "compare_exchange_strong",
                            ARG(0),ARG(1), ARG(2), ARG(3))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::AtomicHelper_atomic_helper_fetch_add,
ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                            "fetch_add",  false, "fetch_add",
                            ARG(0),ARG(1))))
                            
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::AtomicHelper_atomic_helper_fetch_sub,
ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                            "fetch_sub",  false, "fetch_sub",
                            ARG(0),ARG(1))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::AtomicHelper_atomic_helper_fetch_sub,
ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                            "fetch_and",  false, "fetch_and",
                            ARG(0),ARG(1))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::AtomicHelper_atomic_helper_fetch_or,
ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                            "fetch_or",  false, "fetch_or",
                            ARG(0),ARG(1))))  

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::AtomicHelper_atomic_helper_fetch_xor,
ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                            "fetch_xor",  false, "fetch_xor",
                            ARG(0),ARG(1))))             