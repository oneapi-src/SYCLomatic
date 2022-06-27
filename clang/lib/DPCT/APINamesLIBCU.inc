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
                            ARG(0),ARG(1))))