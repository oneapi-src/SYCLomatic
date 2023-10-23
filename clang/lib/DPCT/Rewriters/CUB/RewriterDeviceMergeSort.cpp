//===--------------- RewriterDeviceMergeSort.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterCUB.h"

using namespace clang::dpct;

RewriterMap dpct::createDeviceMergeSortRewriterMap() {
  return RewriterMap{
    // cub::DeviceMergeSort::SortPairs
    CONDITIONAL_FACTORY_ENTRY(
      CheckCubRedundantFunctionCall(), 
      REMOVE_API_FACTORY_ENTRY("cub::DeviceMergeSort::SortPairs"),
        HEADER_INSERT_FACTORY(HeaderType::HT_DPCT_DPL_Utils, 
          REMOVE_CUB_TEMP_STORAGE_FACTORY(
            CONDITIONAL_FACTORY_ENTRY(makeCheckAnd(CheckArgCount(7, std::greater_equal<>(), /*IncludeDefaultArg=*/false), makeCheckNot(CheckArgIsDefaultCudaStream(6))), 
              CALL_FACTORY_ENTRY("cub::DeviceMergeSort::SortPairs", 
                CALL(MapNames::getDpctNamespace() + "stable_sort",
                  CALL("oneapi::dpl::execution::device_policy",
                                            STREAM(6)),
                  ARG(2), BO(BinaryOperatorKind::BO_Add, ARG(2),
                                          ARG(4)), ARG(3), ARG(5)                    
                )
              ),
              CALL_FACTORY_ENTRY("cub::DeviceMergeSort::SortPairs", 
                CALL(MapNames::getDpctNamespace() + "stable_sort",
                  CALL("oneapi::dpl::execution::device_policy",
                                            QUEUESTR),
                  ARG(2), BO(BinaryOperatorKind::BO_Add, ARG(2),
                                          ARG(4)), ARG(3), ARG(5)                    
                )
              )
            )
          )
        )
      )
  };
}

#if 0



#endif

