//===--------------- CallExprRewriterCUB.h --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DPCT_CALL_EXPR_REWRITER_CUB_H
#define CLANG_DPCT_CALL_EXPR_REWRITER_CUB_H

#include "CUBAPIMigration.h"
#include "CallExprRewriter.h"
#include "CallExprRewriterCommon.h"

namespace clang::dpct {

class CheckCubRedundantFunctionCall {
public:
  bool operator()(const CallExpr *C) {
    return CubDeviceLevelRule::isRedundantCallExpr(C);
  }
};

inline std::shared_ptr<CallExprRewriter>
RemoveCubTempStorageFactory::create(const CallExpr *C) const {
  CubDeviceLevelRule::removeRedundantTempVar(C);
  return Inner->create(C);
}

inline std::function<bool(const CallExpr *)> checkEnableUserDefineReductions() {
  return [=](const CallExpr *) -> bool {
    return DpctGlobalInfo::useUserDefineReductions();
  };
}

inline std::function<bool(const CallExpr *)>
checkArgCanMappingToSyclNativeBinaryOp(size_t ArgIdx) {
  return [=](const CallExpr *C) -> bool {
    const Expr *Arg = C->getArg(ArgIdx);
    std::string TypeName = DpctGlobalInfo::getUnqualifiedTypeName(
        Arg->getType().getCanonicalType());
    return CubTypeRule::CanMappingToSyclNativeBinaryOp(TypeName);
  };
}

inline std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createRemoveCubTempStorageFactory(
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        &&Input) {
  return std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>(
      std::move(Input.first),
      std::make_shared<RemoveCubTempStorageFactory>(Input.second));
}

template <class T>
inline std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createRemoveCubTempStorageFactory(
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        &&Input,
    T) {
  return createRemoveCubTempStorageFactory(std::move(Input));
}

#define REMOVE_CUB_TEMP_STORAGE_FACTORY(INNER)                                 \
  createRemoveCubTempStorageFactory(INNER 0),

typedef std::unordered_map<std::string,
                           std::shared_ptr<CallExprRewriterFactoryBase>>
    RewriterMap;

RewriterMap createDeviceReduceRewriterMap();
RewriterMap createDeviceScanRewriterMap();
RewriterMap createDeviceSelectRewriterMap();
RewriterMap createDeviceRunLengthEncodeRewriterMap();
RewriterMap createDeviceSegmentedReduceRewriterMap();
RewriterMap createDeviceRadixSortRewriterMap();
RewriterMap createDeviceSegmentedRadixSortRewriterMap();
RewriterMap createDeviceSegmentedSortRewriterMap();
RewriterMap createDeviceHistgramRewriterMap();
RewriterMap createDeviceMergeSortRewriterMap();
RewriterMap createClassMethodsRewriterMap();
RewriterMap createUtilityFunctionsRewriterMap();
} // namespace clang::dpct

#endif // CLANG_DPCT_CALL_EXPR_REWRITER_CUB_H
