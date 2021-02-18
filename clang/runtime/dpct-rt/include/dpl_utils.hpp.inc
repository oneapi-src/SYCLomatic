//==---- dpl_utils.hpp ----------------------------*- C++ -*----------------==//
//
// Copyright (C) 2018 - 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPL_UTILS_HPP
#define __DPL_UTILS_HPP

// DPCT_LABEL_BEGIN|non_local_include_dependency|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
#define ONEDPL_USE_DPCPP_BACKEND 1
#define __USE_DPCT 1

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/numeric>
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|local_include_dependency|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
// DPCT_LABEL_END
#include "dpl_extras/memory.h"
#include "dpl_extras/algorithm.h"
#include "dpl_extras/numeric.h"
#include "dpl_extras/iterators.h"
#include "dpl_extras/vector.h"

// DPCT_LABEL_BEGIN|dummy_function|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
// DPCT_LABEL_END
#endif // __DPL_UTILS_HPP
