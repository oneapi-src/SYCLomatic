// C2S_LABEL_BEGIN|License|
// C2S_DEPENDENCY_EMPTY
// C2S_CODE
//==---- numeric.h --------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
// C2S_LABEL_END

#ifndef __C2S_DPL_EXTRAS_NUMERIC_H__
#define __C2S_DPL_EXTRAS_NUMERIC_H__
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
// C2S_LABEL_END
// C2S_LABEL_BEGIN|local_include_dependency|
// C2S_DEPENDENCY_EMPTY
// C2S_CODE
// C2S_LABEL_END
namespace c2s {

// C2S_LABEL_BEGIN|inner_product|c2s
// C2S_DEPENDENCY_EMPTY
// C2S_CODE
template <typename Policy, typename InputIt1, typename InputIt2, typename T>
T inner_product(Policy &&policy, InputIt1 first1, InputIt1 last1,
                InputIt2 first2, T init) {
  return std::transform_reduce(std::forward<Policy>(policy), first1, last1,
                               first2, init);
}

template <typename Policy, typename InputIt1, typename InputIt2, typename T,
          typename BinaryOperation1, typename BinaryOperation2>
T inner_product(Policy &&policy, InputIt1 first1, InputIt1 last1,
                InputIt2 first2, T init, BinaryOperation1 op1,
                BinaryOperation2 op2) {
  return std::transform_reduce(std::forward<Policy>(policy), first1, last1,
                               first2, init, op1, op2);
}
// C2S_LABEL_END

} // end namespace c2s

#endif
