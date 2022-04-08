// C2S_LABEL_BEGIN|License|
// C2S_DEPENDENCY_EMPTY
// C2S_CODE
//==---- dpl_utils.hpp ----------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
// C2S_LABEL_END

#ifndef __C2S_DPL_UTILS_HPP__
#define __C2S_DPL_UTILS_HPP__
// C2S_COMMENT
// C2S_COMMENT Example1:
// C2S_COMMENT // C2S_LABEL_BEGIN|FeatureNameDef|[Namespace]
// C2S_COMMENT // C2S_DEPENDENCY_EMPTY
// C2S_COMMENT // C2S_CODE
// C2S_COMMENT some code
// C2S_COMMENT // C2S_LABEL_END
// C2S_COMMENT
// C2S_COMMENT Example2:
// C2S_COMMENT // C2S_LABEL_BEGIN|FeatureNameDef|[Namespace]
// C2S_COMMENT // C2S_DEPENDENCY_BEGIN
// C2S_COMMENT // FileID|FeatureNameRef
// C2S_COMMENT [// FileID|FeatureNameRef]
// C2S_COMMENT ...
// C2S_COMMENT // C2S_DEPENDENCY_END
// C2S_COMMENT // C2S_CODE
// C2S_COMMENT some code
// C2S_COMMENT // C2S_LABEL_END
// C2S_COMMENT
// C2S_COMMENT For header file including dependency, please use predefined feature name:
// C2S_COMMENT   local_include_dependency: c2s helper files
// C2S_COMMENT   non_local_include_dependency: other headler files

// C2S_LABEL_BEGIN|non_local_include_dependency|
// C2S_DEPENDENCY_EMPTY
// C2S_CODE
#define ONEDPL_USE_DPCPP_BACKEND 1
#define __USE_C2S 1

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/numeric>
// C2S_LABEL_END

// C2S_LABEL_BEGIN|local_include_dependency|
// C2S_DEPENDENCY_EMPTY
// C2S_CODE
// C2S_LABEL_END
#include "dpl_extras/memory.h"
#include "dpl_extras/algorithm.h"
#include "dpl_extras/numeric.h"
#include "dpl_extras/iterators.h"
#include "dpl_extras/vector.h"
#include "dpl_extras/dpcpp_extensions.h"

#endif // __C2S_DPL_UTILS_HPP__
