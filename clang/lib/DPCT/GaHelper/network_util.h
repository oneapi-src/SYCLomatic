//===--- network_util.h-------------------------*- C++ -*---===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//
#pragma once
#include "GaNamespace.h"

GAHELPER_NS_BEGIN

enum class network_status_t
{
    inside_intel_network,
    outside_intel_network,
    network_error
};

network_status_t getNetworkStatus();

GAHELPER_NS_END
