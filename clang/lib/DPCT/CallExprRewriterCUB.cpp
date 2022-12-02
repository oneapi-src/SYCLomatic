//===--------------- CallExprRewriterCUB.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriter.h"
#include "CallExprRewriterCommon.h"
#include "CUBAPIMigration.h"

namespace clang {
namespace dpct {

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

std::function<bool(const CallExpr *)>
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

void CallExprRewriterFactoryBase::initRewriterMapCUB() {
  RewriterMap->merge(
      std::unordered_map<std::string,
                         std::shared_ptr<CallExprRewriterFactoryBase>>({
#include "APINamesCUB.inc"
      }));
}

} // namespace dpct
} // namespace clang
