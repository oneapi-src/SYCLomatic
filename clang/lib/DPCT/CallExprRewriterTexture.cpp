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
template <size_t... Idx>
class TextureReadRewriterFactory : public CallExprRewriterFactoryBase {
  std::string Source;
  int TexType;

  inline int getDim() const { return TexType & 0x0f; }

  template <class BaseT>
  std::shared_ptr<CallExprRewriter>
  createRewriter(const CallExpr *C, bool RetAssign, BaseT Base) const {
    const static std::string MemberName = "read";
    using ReaderPrinter = decltype(makeMemberCallCreator<false>(
        std::declval<std::function<BaseT(const CallExpr *)>>(), false,
        MemberName, makeCallArgCreatorWithCall(Idx)...)(C));
    if (RetAssign) {
      return std::make_shared<PrinterRewriter<
          BinaryOperatorPrinter<BO_Assign, DerefExpr, ReaderPrinter>>>(
          C, Source, DerefExpr(C->getArg(0), C),
          ReaderPrinter(std::move(Base), false, MemberName,
                        std::make_pair(C, C->getArg(Idx + 1))...));
    }
    return std::make_shared<PrinterRewriter<ReaderPrinter>>(
        C, Source, Base, false, MemberName,
        std::make_pair(C, C->getArg(Idx))...);
  }

  template <typename VecType>
  std::shared_ptr<CallExprRewriter>
  createbindlessRewriterNormal(const CallExpr *C, bool RetAssign,
                               const TemplateArgumentInfo &TAI,
                               const std::string &VecTypeName) const {
    const static std::string FuncName =
        MapNames::getClNamespace() + "ext::oneapi::experimental::sample_image";
    using FuncNamePrinter =
        TemplatedNamePrinter<StringRef, std::vector<TemplateArgumentInfo>>;
    using ReaderPrinter =
        CallExprPrinter<FuncNamePrinter,
                        std::pair<const CallExpr *, const Expr *>, VecType>;
    if (RetAssign) {
      return std::make_shared<PrinterRewriter<
          BinaryOperatorPrinter<BO_Assign, DerefExpr, ReaderPrinter>>>(
          C, Source, DerefExpr(C->getArg(0), C),
          ReaderPrinter(
              FuncNamePrinter(FuncName, {TAI}), std::make_pair(C, C->getArg(1)),
              VecType(VecTypeName, std::make_pair(C, C->getArg(Idx + 1))...)));
    }
    return std::make_shared<PrinterRewriter<ReaderPrinter>>(
        C, Source, FuncNamePrinter(FuncName, {TAI}),
        std::make_pair(C, C->getArg(0)),
        VecType(VecTypeName, std::make_pair(C, C->getArg(Idx))...));
  }

  template <typename VecType>
  std::shared_ptr<CallExprRewriter>
  createbindlessRewriterLod(const CallExpr *C, bool RetAssign,
                            const TemplateArgumentInfo &TAI,
                            const std::string &VecTypeName) const {
    const static std::string FuncName =
        MapNames::getClNamespace() + "ext::oneapi::experimental::sample_mipmap";
    using FuncNamePrinter =
        TemplatedNamePrinter<StringRef, std::vector<TemplateArgumentInfo>>;
    using ReaderPrinter =
        CallExprPrinter<FuncNamePrinter,
                        std::pair<const CallExpr *, const Expr *>, VecType,
                        std::pair<const CallExpr *, const Expr *>>;
    if (RetAssign) {
      return std::make_shared<PrinterRewriter<
          BinaryOperatorPrinter<BO_Assign, DerefExpr, ReaderPrinter>>>(
          C, Source, DerefExpr(C->getArg(0), C),
          ReaderPrinter(
              FuncNamePrinter(FuncName, {TAI}), std::make_pair(C, C->getArg(1)),
              VecType(VecTypeName, std::make_pair(C, C->getArg(Idx + 1))...),
              std::make_pair(C, C->getArg(C->getNumArgs() - 1))));
    }
    return std::make_shared<PrinterRewriter<ReaderPrinter>>(
        C, Source, FuncNamePrinter(FuncName, {TAI}),
        std::make_pair(C, C->getArg(0)),
        VecType(VecTypeName, std::make_pair(C, C->getArg(Idx))...),
        std::make_pair(C, C->getArg(C->getNumArgs() - 1)));
  }

  template <typename VecType>
  std::shared_ptr<CallExprRewriter>
  createbindlessRewriterLayered(const CallExpr *C, bool RetAssign,
                                const TemplateArgumentInfo &TAI,
                                const std::string &VecTypeName) const {
    const static std::string FuncName =
        MapNames::getClNamespace() +
        "ext::oneapi::experimental::sample_image_array";
    using FuncNamePrinter =
        TemplatedNamePrinter<StringRef, std::vector<TemplateArgumentInfo>>;
    using ReaderPrinter =
        CallExprPrinter<FuncNamePrinter,
                        std::pair<const CallExpr *, const Expr *>, VecType,
                        std::pair<const CallExpr *, const Expr *>>;
    if (RetAssign) {
      return std::make_shared<PrinterRewriter<
          BinaryOperatorPrinter<BO_Assign, DerefExpr, ReaderPrinter>>>(
          C, Source, DerefExpr(C->getArg(0), C),
          ReaderPrinter(
              FuncNamePrinter(FuncName, {TAI}), std::make_pair(C, C->getArg(1)),
              VecType(VecTypeName, std::make_pair(C, C->getArg(Idx + 1))...),
              std::make_pair(C, C->getArg(C->getNumArgs() - 1))));
    }
    return std::make_shared<PrinterRewriter<ReaderPrinter>>(
        C, Source, FuncNamePrinter(FuncName, {TAI}),
        std::make_pair(C, C->getArg(0)),
        VecType(VecTypeName, std::make_pair(C, C->getArg(Idx))...),
        std::make_pair(C, C->getArg(C->getNumArgs() - 1)));
  }

  std::shared_ptr<CallExprRewriter>
  createbindlessRewriter(const CallExpr *C, bool RetAssign,
                         QualType TargetType) const {
    TemplateArgumentInfo TAI;
    auto TAL = getTemplateArgsList(C);
    if (TAL.empty()) {
      if (const auto *ET = dyn_cast<ElaboratedType>(TargetType)) {
        // For texture referece API, need desugar the __nv_tex_rmet_ret type.
        TargetType =
            ET->desugar().getDesugaredType(DpctGlobalInfo::getContext());
      }
      TAI.setAsType(TargetType);
      if (TargetType->isDependentType()) {
        // Texture read APIs without template arg must have the first argument
        // as the return value.
        RetAssign = true;
      }
    } else {
      TAI = TAL[0];
    }
    std::string VecTypeName = "float";
    if (getDim() != 1)
      VecTypeName =
          MapNames::getClNamespace() + VecTypeName + std::to_string(getDim());
    using VecType =
        CallExprPrinter<std::string,
                        decltype(std::make_pair(C, C->getArg(Idx)))...>;
    if ((TexType & 0xf0) == 0x10)
      return createbindlessRewriterLod<VecType>(C, RetAssign, TAI, VecTypeName);
    if ((TexType & 0xf0) == 0xf0)
      return createbindlessRewriterLayered<VecType>(C, RetAssign, TAI,
                                                    VecTypeName);
    return createbindlessRewriterNormal<VecType>(C, RetAssign, TAI,
                                                 VecTypeName);
  }

public:
  TextureReadRewriterFactory(std::string Name, int Tex)
      : Source(std::move(Name)), TexType(Tex) {}
  std::shared_ptr<CallExprRewriter>
  create(const CallExpr *Call) const override {
    const Expr *SourceExpr = Call->getArg(0);
    unsigned SourceIdx = 0;
    QualType TargetType = Call->getType();
    StringRef SourceName;
    bool RetAssign = false;
    if (SourceExpr->getType()->isPointerType()) {
      TargetType = SourceExpr->getType()->getPointeeType();
      SourceExpr = Call->getArg(1);
      SourceIdx = 1;
      RetAssign = true;
      if (auto UO = dyn_cast<UnaryOperator>(SourceExpr)) {
        if (UO->getOpcode() == UnaryOperator::Opcode::UO_AddrOf) {
          SourceExpr = UO->getSubExpr();
        }
      }
    }
    if (DpctGlobalInfo::useExtBindlessImages()) {
      return createbindlessRewriter(Call, RetAssign, TargetType);
    }
    SourceExpr = SourceExpr->IgnoreImpCasts();
    if (auto FD = DpctGlobalInfo::getParentFunction(Call)) {
      auto FuncInfo = DeviceFunctionDecl::LinkRedecls(FD);
      if (FuncInfo) {
        auto CallInfo = FuncInfo->addCallee(Call);
        if (auto ME = dyn_cast<MemberExpr>(SourceExpr)) {
          auto MemberInfo =
              CallInfo->addStructureTextureObjectArg(SourceIdx, ME, false);
          if (MemberInfo) {
            FuncInfo->addTexture(MemberInfo);
            MemberInfo->setType(
                DpctGlobalInfo::getUnqualifiedTypeName(TargetType), TexType);
            SourceName = MemberInfo->getName();
            return createRewriter(Call, RetAssign, SourceName);
          }
        } else if (auto DRE = dyn_cast<DeclRefExpr>(SourceExpr)) {
          auto TexInfo = CallInfo->addTextureObjectArg(SourceIdx, DRE, false);
          if (TexInfo) {
            TexInfo->setType(DpctGlobalInfo::getUnqualifiedTypeName(TargetType),
                             TexType);
          }
        }
      }
    }

    return createRewriter(Call, RetAssign,
                          std::make_pair(Call, Call->getArg(RetAssign & 0x01)));
  }
};

/// Create rewriter factory for texture reader APIs.
/// Predicate: check the first arg if is pointer and set texture info with
/// corresponding data. Migrate the call expr to an assign expr if Pred result
/// is true; e.g.: tex1D(&u, tex, 1.0f) -> u = tex.read(1.0f) Migrate the call
/// expr to an assign expr if Pred result is false; e.g.: tex1D(tex, 1.0f) ->
/// tex.read(1.0f) The template arguments is the member call arguments' index in
/// original call expr.
template <size_t... Idx>
inline std::shared_ptr<CallExprRewriterFactoryBase>
createTextureReaderRewriterFactory(const std::string &Source, int TextureType) {
  return std::make_shared<TextureReadRewriterFactory<Idx...>>(Source,
                                                              TextureType);
}

#define TEX_FUNCTION_FACTORY_ENTRY(FuncName, TexType, ...)                     \
  {FuncName,                                                                   \
   createTextureReaderRewriterFactory<__VA_ARGS__>(FuncName, TexType)},
#define BIND_TEXTURE_FACTORY_ENTRY(FuncName, ...)                              \
  {FuncName, createBindTextureRewriterFactory<__VA_ARGS__>(FuncName)},

#define FUNC_NAME_FACTORY_ENTRY(FuncName, RewriterName)                        \
  REWRITER_FACTORY_ENTRY(FuncName, FuncCallExprRewriterFactory, RewriterName)
#define UNSUPPORTED_FACTORY_ENTRY(FuncName, MsgID)                             \
  REWRITER_FACTORY_ENTRY(FuncName,                                             \
      UnsupportFunctionRewriterFactory<std::string>, MsgID, FuncName)

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
