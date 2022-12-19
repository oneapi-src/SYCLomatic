//===--------------- CallExprRewriterTexture.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriter.h"
#include "CallExprRewriterCommon.h"

namespace clang {
namespace dpct {

#define TEX_FUNCTION_FACTORY_ENTRY(FuncName, TexType, ...)                     \
  {FuncName,                                                                   \
   createTextureReaderRewriterFactory<__VA_ARGS__>(FuncName, TexType)},
#define BIND_TEXTURE_FACTORY_ENTRY(FuncName, ...)                              \
  {FuncName, createBindTextureRewriterFactory<__VA_ARGS__>(FuncName)},

#define REWRITER_FACTORY_ENTRY(FuncName, RewriterFactory, ...)                 \
  {FuncName, std::make_shared<RewriterFactory>(FuncName, __VA_ARGS__)},
#define FUNC_NAME_FACTORY_ENTRY(FuncName, RewriterName)                        \
  REWRITER_FACTORY_ENTRY(FuncName, FuncCallExprRewriterFactory, RewriterName)
#define UNSUPPORTED_FACTORY_ENTRY(FuncName, MsgID)                             \
  REWRITER_FACTORY_ENTRY(FuncName, UnsupportFunctionRewriterFactory<>, MsgID)

void CallExprRewriterFactoryBase::initRewriterMapTexture() {
  RewriterMap->merge(
      std::unordered_map<std::string,
                         std::shared_ptr<CallExprRewriterFactoryBase>>({
#define ENTRY_RENAMED(SOURCEAPINAME, TARGETAPINAME)                            \
  FUNC_NAME_FACTORY_ENTRY(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_TEXTURE(SOURCEAPINAME, TEXTYPE, ...)                             \
  TEX_FUNCTION_FACTORY_ENTRY(SOURCEAPINAME, TEXTYPE, __VA_ARGS__)
#define ENTRY_UNSUPPORTED(SOURCEAPINAME, MSGID)                                \
  UNSUPPORTED_FACTORY_ENTRY(SOURCEAPINAME, MSGID)
#define ENTRY_BIND(SOURCEAPINAME, ...)                                         \
  BIND_TEXTURE_FACTORY_ENTRY(SOURCEAPINAME, __VA_ARGS__)
#define ENTRY_TEMPLATED(SOURCEAPINAME, ...)                                    \
  TEMPLATED_CALL_FACTORY_ENTRY(SOURCEAPINAME, __VA_ARGS__)
#include "APINamesTexture.inc"
#undef ENTRY_RENAMED
#undef ENTRY_TEXTURE
#undef ENTRY_UNSUPPORTED
#undef ENTRY_TEMPLATED
#undef ENTRY_BIND
      }));
}

} // namespace dpct
} // namespace clang
