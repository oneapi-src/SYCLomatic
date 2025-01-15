//===--------------- TypeLocRewriters.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RuleInfra/TypeLocRewriters.h"
#include "UserDefinedRules/UserDefinedRules.h"
#include "clang/AST/TypeLoc.h"

namespace clang {
namespace dpct {
inline auto UseSYCLCompat() {
  return [](const TypeLoc) -> bool { return DpctGlobalInfo::useSYCLCompat(); };
}

TemplateArgumentInfo getTemplateArg(const TypeLoc &TL, unsigned Idx) {
  if (auto TSTL = TL.getAs<TemplateSpecializationTypeLoc>()) {
    if (TSTL.getNumArgs() > Idx) {
      auto TAL = TSTL.getArgLoc(Idx);
      return TemplateArgumentInfo(TAL, TL.getSourceRange());
    }
  }
  return TemplateArgumentInfo("");
}

std::function<std::string(const TypeLoc)>
makeStringCreator(std::string TypeName,
                  clang::dpct::HelperFeatureEnum RequestFeature,
                  const std::vector<std::string> &Headers) {
  return [=](const TypeLoc TL) -> std::string {
    requestFeature(RequestFeature);
    return TypeName;
  };
}

template <typename T>
std::function<std::string(const TypeLoc)>
makeAddPointerCreator(std::function<T(const TypeLoc)> f) {
  return [=](const TypeLoc TL) {
    std::string s;
    llvm::raw_string_ostream OS(s);
    dpct::print(OS, f(TL));
    OS << " *";
    return s;
  };
}

std::function<std::string(const TypeLoc)>
makeTypeStrCreator() {
  return [=](const TypeLoc TL) {
    if(!TL)
      return std::string();
    auto PP = DpctGlobalInfo::getContext().getPrintingPolicy();
    PP.SuppressTagKeyword = true;
    PP.FullyQualifiedName = true;
    return TL.getType().getAsString(PP);
  };
}

std::function<TemplateArgumentInfo(const TypeLoc)>
makeTemplateArgCreator(unsigned Idx) {
  return [=](const TypeLoc TL) -> TemplateArgumentInfo {
    return getTemplateArg(TL, Idx);
  };
}

std::function<std::string(const TypeLoc)>
makeUserDefinedTypeStrCreator(MetaRuleObject &R,
                              std::shared_ptr<TypeOutputBuilder> TOB) {
  return [&R, Builder = std::move(TOB)](const TypeLoc TL) {
    if (!TL)
      return std::string();
    auto Range = getDefinitionRange(TL.getBeginLoc(), TL.getEndLoc());
    for (const auto &ItHeader : R.Includes) {
      DpctGlobalInfo::getInstance().insertHeader(Range.getBegin(), ItHeader);
    }

    std::string ResultStr;
    llvm::raw_string_ostream OS(ResultStr);
    for (auto &tob : Builder->SubBuilders) {
      switch (tob->Kind) {
      case (OutputBuilder::Kind::String):
        OS << tob->Str;
        break;
      case (OutputBuilder::Kind::TemplateArg): {
        OS << getTemplateArg(TL, tob->ArgIndex).getString();
        break;
      }
      default:
        DpctDebugs() << "[OutputBuilder::Kind] Unexpected value: " << tob->Kind
                     << "\n";
        assert(0);
      }
    }
    OS.flush();
    return ResultStr;
  };
}

class CheckTemplateArgCount {
  unsigned Count;

public:
  CheckTemplateArgCount(unsigned I) : Count(I) {}
  bool operator()(const TypeLoc TL) {
    if(auto TSTL = TL.getAs<TemplateSpecializationTypeLoc>()){
      return TSTL.getNumArgs() == Count;
    }
    return false;
  }
};

inline auto CheckForPostfixDeclaratorType(unsigned Idx) {
  return [=](const TypeLoc TL){
    if (const auto TSTL = TL.getAs<TemplateSpecializationTypeLoc>()) {
      const auto TAT = TSTL.getArgLoc(Idx).getArgument().getAsType();
      const auto CT = TAT.getCanonicalType();
      return isTypePostfix(CT);
    }
    return false;
  };
}

class CheckTypeNameAndInVarDecl {
  std::string Name;
public:
  CheckTypeNameAndInVarDecl(const std::string &N) : Name(N) {}
  bool operator()(const TypeLoc TL) {
    if (const auto *VD = DpctGlobalInfo::findAncestor<VarDecl>(&TL)) {
      return DpctGlobalInfo::getTypeName(VD->getType()) == Name;
    }
    return false;
  }
};

inline std::function<bool(const TypeLoc TL)>
checkTemplateArgSpelling(size_t Index, std::string Str) {
  auto getQualtifiedNameStr = [=](const NamedDecl *NL) -> std::string {
    if (NL == nullptr)
      return "";
    if (const auto *NSD = dyn_cast<NamespaceDecl>(NL->getDeclContext())) {
      std::string TypeQualifiedString =
          getNameSpace(NSD) + "::" + NL->getNameAsString();
      return TypeQualifiedString;
    }
    return NL->getNameAsString();
  };

  return [=](const TypeLoc TL) -> bool {
    if (const auto &TSTL = TL.getAs<TemplateSpecializationTypeLoc>()) {
      if (Index > TSTL.getNumArgs())
        return false;
      const auto TA = TSTL.getArgLoc(Index).getArgument();
      if (TA.getKind() == TemplateArgument::ArgKind::Type) {
        std::string ResStr =
            getQualtifiedNameStr(TA.getAsType()->getAsTagDecl());
        if (ResStr.empty())
          return TA.getAsType().getAsString() == Str;
        return ResStr == Str;
      } else if (TA.getKind() == TemplateArgument::ArgKind::Declaration) {
        return getQualtifiedNameStr(TA.getAsDecl()) == Str;
      } else if (TA.getKind() == TemplateArgument::ArgKind::Integral) {
        return std::to_string(TA.getAsIntegral().getExtValue()) == Str;
      } else if (TA.getKind() == TemplateArgument::ArgKind::Expression) {
        return getStmtSpelling(TA.getAsExpr()) == Str;
      }
    }
    return false;
  };
}

std::function<bool(const TypeLoc)> checkEnableJointMatrixForType() {
  return [=](const TypeLoc) -> bool {
    return DpctGlobalInfo::useExtJointMatrix();
  };
}

std::function<bool(const TypeLoc)> checkEnableGraphForType() {
  return [=](const TypeLoc) -> bool { return DpctGlobalInfo::useExtGraph(); };
}

std::function<bool(const TypeLoc)> checkEnableBindlessImagesForType() {
  return [=](const TypeLoc) -> bool { return DpctGlobalInfo::useExtBindlessImages(); };
}

std::function<bool(const TypeLoc)> isUseNonUniformGroupsForType() {
  return [=](const TypeLoc) -> bool {
    return DpctGlobalInfo::useExpNonUniformGroups();
  };
}

std::function<bool(const TypeLoc)> isUseLogicalGroupsForType() {
  return
      [=](const TypeLoc) -> bool { return DpctGlobalInfo::useLogicalGroup(); };
}
std::function<bool(const TypeLoc)> isUseRootGroupForType() {
  return [=](const TypeLoc) -> bool { return DpctGlobalInfo::useRootGroup(); };
}
std::function<bool(const TypeLoc)> isUseNdRangeBarrier() {
  return [=](const TypeLoc) -> bool {
    return DpctGlobalInfo::useNdRangeBarrier();
  };
}

// Print a templated type. Pass a STR("") as a template argument for types with
// empty template argument e.g. MyType<>, If --enable-ctad is set, the template
// arguments which could be deduced with class template argument deduction(CTAD)
// will be omitted in the generated code.
template <class TypeNameT, class... TemplateArgsT>
std::shared_ptr<TypeLocRewriterFactoryBase> createCtadTypeLocRewriterFactory(
    std::function<TypeNameT(const TypeLoc)> TypeNameCreator,
    std::function<TemplateArgsT(const TypeLoc)>... TAsCreator) {
  return std::make_shared<
      TypeLocRewriterFactory<CtadTemplateTypeLocRewriter<TypeNameT, TemplateArgsT...>,
                             std::function<TypeNameT(const TypeLoc)>,
                             std::function<TemplateArgsT(const TypeLoc)>...>>(
      std::forward<std::function<TypeNameT(const TypeLoc)>>(TypeNameCreator),
      std::forward<std::function<TemplateArgsT(const TypeLoc)>>(TAsCreator)...);
}

// Print a type without template.
template <class TypeNameT>
std::shared_ptr<TypeLocRewriterFactoryBase> createCtadTypeLocRewriterFactory(
    std::function<TypeNameT(const TypeLoc)> TypeNameCreator) {
  return std::make_shared<
      TypeLocRewriterFactory<TypeNameTypeLocRewriter<TypeNameT>,
                             std::function<TypeNameT(const TypeLoc)>>>(
      std::forward<std::function<TypeNameT(const TypeLoc)>>(TypeNameCreator));
}

std::shared_ptr<TypeLocRewriterFactoryBase> createTypeLocConditionalFactory(
    std::function<bool(const TypeLoc)> Pred,
    std::shared_ptr<TypeLocRewriterFactoryBase> &&First,
    std::shared_ptr<TypeLocRewriterFactoryBase> &&Second) {
  return std::make_shared<TypeLocConditionalRewriterFactory>(Pred, First,
                                                             Second);
}

template <typename... ArgsT>
std::shared_ptr<TypeLocRewriterFactoryBase> createTypeLocEmitWarningFactory(
    std::shared_ptr<TypeLocRewriterFactoryBase> &&Inner, Diagnostics MsgID,
    ArgsT... Args) {
  return std::make_shared<TypeLocEmitWarningRewriterFactory<ArgsT...>>(
      Inner, MsgID, Args...);
}

template <typename... Args> 
std::shared_ptr<TypeLocRewriterFactoryBase>
createReportWarningTypeLocRewriterFactory(Diagnostics MsgId,
                                          Args&&... args) {
  return std::make_shared<
    TypeLocRewriterFactory<ReportWarningTypeLocRewriter, Diagnostics, Args...>>
    (MsgId, std::forward<Args>(args)...);
}

std::pair<TypeMatchingDesc, std::shared_ptr<TypeLocRewriterFactoryBase>>
createFeatureRequestFactory(
    HelperFeatureEnum Feature,
    std::pair<TypeMatchingDesc, std::shared_ptr<TypeLocRewriterFactoryBase>>
        &&Input) {
  return std::pair<TypeMatchingDesc, std::shared_ptr<TypeLocRewriterFactoryBase>>(
      std::move(Input.first),
      std::make_shared<TypeLocRewriterFactoryWithFeatureRequest>(Feature,
                                                          Input.second));
}
template <class T>
std::pair<TypeMatchingDesc, std::shared_ptr<TypeLocRewriterFactoryBase>>
createFeatureRequestFactory(
    HelperFeatureEnum Feature,
    std::pair<TypeMatchingDesc, std::shared_ptr<TypeLocRewriterFactoryBase>>
        &&Input,
    T) {
  return createFeatureRequestFactory(Feature, std::move(Input));
}

std::shared_ptr<TypeLocRewriterFactoryBase> createHeaderInsertionFactory(
    HeaderType Header,
    std::shared_ptr<TypeLocRewriterFactoryBase> &&SubRewriterFactory) {
  return std::make_shared<HeaderInsertionRewriterFactory>(Header,
                                                          SubRewriterFactory);
}
std::unique_ptr<std::unordered_map<
      TypeMatchingDesc, std::shared_ptr<TypeLocRewriterFactoryBase>,
      TypeMatchingDesc::hash>> TypeLocRewriterFactoryBase::TypeLocRewriterMap;

void initTypeLocSYCLCompatRewriterMap(
    std::unordered_map<TypeMatchingDesc,
                       std::shared_ptr<TypeLocRewriterFactoryBase>,
                       TypeMatchingDesc::hash> &TypeLocRewriterMap);

void TypeLocRewriterFactoryBase::initTypeLocRewriterMap() {
  TypeLocRewriterMap = std::make_unique<std::unordered_map<
      TypeMatchingDesc, std::shared_ptr<TypeLocRewriterFactoryBase>,
      TypeMatchingDesc::hash>>(
      std::unordered_map<TypeMatchingDesc,
                         std::shared_ptr<TypeLocRewriterFactoryBase>,
                         TypeMatchingDesc::hash>({
#define STR(Str) makeStringCreator(Str)
#define TEMPLATE_ARG(Idx) makeTemplateArgCreator(Idx)
#define TYPE_REWRITE_ENTRY(Name, Factory) {{Name}, Factory},
#define TYPE_CONDITIONAL_FACTORY(Pred, First, Second)                          \
  createTypeLocConditionalFactory(Pred, First, Second)
#define TYPE_FACTORY(...) createTypeLocRewriterFactory(__VA_ARGS__)
#define CTAD_TYPE_FACTORY(...) createCtadTypeLocRewriterFactory(__VA_ARGS__)
#define FEATURE_REQUEST_FACTORY(FEATURE, x)                                    \
  createFeatureRequestFactory(FEATURE, x 0),
#define HEADER_INSERTION_FACTORY(HEADER, SUB)                                  \
  createHeaderInsertionFactory(HEADER, SUB)
#define TYPESTR makeTypeStrCreator()
#define WARNING_FACTORY(MSGID, ...)                                            \
  createReportWarningTypeLocRewriterFactory(MSGID, __VA_ARGS__)
#define EMIT_WARNING_FACTORY(...) createTypeLocEmitWarningFactory(__VA_ARGS__)
#define ADD_POINTER(CREATOR) makeAddPointerCreator(CREATOR)
#include "APINamesTemplateType.inc"
#undef WARNING_FACTORY
#undef ADD_POINTER
#undef HEADER_INSERTION_FACTORY
#undef FEATURE_REQUEST_FACTORY
#undef TYPE_FACTORY
#undef TYPE_CONDITIONAL_FACTORY
#undef TYPE_REWRITE_ENTRY
#undef TEMPLATE_ARG
#undef STR
      }));
  if (DpctGlobalInfo::useSYCLCompat())
    initTypeLocSYCLCompatRewriterMap(*TypeLocRewriterMap);
}

void initTypeLocSYCLCompatRewriterMap(
    std::unordered_map<TypeMatchingDesc,
                       std::shared_ptr<TypeLocRewriterFactoryBase>,
                       TypeMatchingDesc::hash> &TypeLocRewriterMap) {
#define SYCLCOMPAT_UNSUPPORT(NAME)                                             \
  TypeLocRewriterMap[TypeMatchingDesc(NAME)] =                                 \
      createReportWarningTypeLocRewriterFactory(                               \
          Diagnostics::UNSUPPORT_SYCLCOMPAT, makeStringCreator(NAME));

  SYCLCOMPAT_UNSUPPORT("cudaGraph_t")
  SYCLCOMPAT_UNSUPPORT("cudaGraphExec_t")
  SYCLCOMPAT_UNSUPPORT("cudaGraphNode_t")
  SYCLCOMPAT_UNSUPPORT("cudaGraphicsResource")
  SYCLCOMPAT_UNSUPPORT("cudaGraphicsResource_t")
  SYCLCOMPAT_UNSUPPORT("CUgraphicsResource")
  SYCLCOMPAT_UNSUPPORT("cudaExternalMemory_t")
  SYCLCOMPAT_UNSUPPORT("cudaExternalMemoryHandleDesc")
  SYCLCOMPAT_UNSUPPORT("cudaExternalMemoryMipmappedArrayDesc")
  SYCLCOMPAT_UNSUPPORT("cudaExternalMemoryBufferDesc")
  SYCLCOMPAT_UNSUPPORT("thrust::system::cuda::experimental::pinned_allocator")
  SYCLCOMPAT_UNSUPPORT("thrust::cuda::experimental::pinned_allocator")
  SYCLCOMPAT_UNSUPPORT("thrust::device_allocator")

#undef SYCLCOMPAT_UNSUPPORT
}
} // namespace dpct
} // namespace clang
