//===---APINamesComplex.inc--------------------------------------------===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

CALL_FACTORY_ENTRY("make_cuComplex",
                   CALL(MapNames::getClNamespace() + "float2", ARG(0), ARG(1)))
CALL_FACTORY_ENTRY("make_cuDoubleComplex",
                   CALL(MapNames::getClNamespace() + "double2", ARG(0), ARG(1)))