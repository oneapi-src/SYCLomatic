//===--- uuid.h-------------------------*- C++ -*---===//
//
// Copyright (C) 2018-1019 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//
#pragma once
#include "GaNamespace.h"
#include <string>

//TODO:  remove the macro to enable windows support.
#if defined(__linux__)

GAHELPER_NS_BEGIN

std::string generate_uuid4();

GAHELPER_NS_END
#endif