//===--------------- TypeLocRewriters.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypeLocRewriters.h"
#include "clang/AST/TypeLoc.h"

namespace clang {
namespace dpct {

std::function<std::string(const TypeLoc)>
makeStringCreator(std::string TypeName) {
  return [=](const TypeLoc TL) -> std::string {
    return TypeName;
  };
}

std::function<TemplateArgumentInfo(const TypeLoc)>
makeTemplateArgCreator(unsigned Idx) {
  return [=](const TypeLoc TL) -> TemplateArgumentInfo {
    if (auto TSTL = TL.getAs<TemplateSpecializationTypeLoc>()) {
      if (TSTL.getNumArgs() > Idx) {
        auto TAL = TSTL.getArgLoc(Idx);
        return TemplateArgumentInfo(TAL, TL.getSourceRange());
      }
    }
    return TemplateArgumentInfo("");
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
    auto PP = DpctGlobalInfo::getContext().getPrintingPolicy();
    PP.SuppressTagKeyword = true;
    PP.FullyQualifiedName = true;
    return TL.getType().getAsString(PP);
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
      return CT->isPointerType() || CT->isFunctionType() || CT->isArrayType();
    }
    return false;
  };
}

// Print a templated type. Pass a STR("") as a template argument for types with
// no template argument e.g. MyType<>
template <class TypeNameT, class... TemplateArgsT>
std::shared_ptr<TypeLocRewriterFactoryBase> createTypeLocRewriterFactory(
    std::function<TypeNameT(const TypeLoc)> TypeNameCreator,
    std::function<TemplateArgsT(const TypeLoc)>... TAsCreator) {
  return std::make_shared<
      TypeLocRewriterFactory<TemplateTypeLocRewriter<TypeNameT, TemplateArgsT...>,
                             std::function<TypeNameT(const TypeLoc)>,
                             std::function<TemplateArgsT(const TypeLoc)>...>>(
      std::forward<std::function<TypeNameT(const TypeLoc)>>(TypeNameCreator),
      std::forward<std::function<TemplateArgsT(const TypeLoc)>>(TAsCreator)...);
}

// Print a type with no template.
template <class TypeNameT>
std::shared_ptr<TypeLocRewriterFactoryBase> createTypeLocRewriterFactory(
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

template <typename... Args> 
std::shared_ptr<TypeLocRewriterFactoryBase>
createReportWarningTypeLocRewriterFactory(Diagnostics MsgId,
                                          Args&&... args) {
  return std::make_shared<
    TypeLocRewriterFactory<ReportWarningTypeLocRewriter, Diagnostics, Args...>>
    (MsgId, std::forward<Args>(args)...);
}

std::pair<std::string, std::shared_ptr<TypeLocRewriterFactoryBase>>
createFeatureRequestFactory(
    HelperFeatureEnum Feature,
    std::pair<std::string, std::shared_ptr<TypeLocRewriterFactoryBase>>
        &&Input) {
  return std::pair<std::string, std::shared_ptr<TypeLocRewriterFactoryBase>>(
      std::move(Input.first),
      std::make_shared<TypeLocRewriterFactoryWithFeatureRequest>(Feature,
                                                          Input.second));
}
template <class T>
std::pair<std::string, std::shared_ptr<TypeLocRewriterFactoryBase>>
createFeatureRequestFactory(
    HelperFeatureEnum Feature,
    std::pair<std::string, std::shared_ptr<TypeLocRewriterFactoryBase>>
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
    std::string, std::shared_ptr<TypeLocRewriterFactoryBase>>>
    TypeLocRewriterFactoryBase::TypeLocRewriterMap;

void TypeLocRewriterFactoryBase::initTypeLocRewriterMap() {
  TypeLocRewriterMap = std::make_unique<std::unordered_map<
      std::string, std::shared_ptr<TypeLocRewriterFactoryBase>>>(
      std::unordered_map<std::string,
                         std::shared_ptr<TypeLocRewriterFactoryBase>>({
#define STR(Str) makeStringCreator(Str)
#define TEMPLATE_ARG(Idx) makeTemplateArgCreator(Idx)
#define TYPE_REWRITE_ENTRY(Name, Factory) {Name, Factory},
#define TYPE_CONDITIONAL_FACTORY(Pred, First, Second)                          \
  createTypeLocConditionalFactory(Pred, First, Second)
#define TYPE_FACTORY(...) createTypeLocRewriterFactory(__VA_ARGS__)
#define FEATURE_REQUEST_FACTORY(FEATURE, x)                                    \
  createFeatureRequestFactory(FEATURE, x 0),
#define HEADER_INSERTION_FACTORY(HEADER, SUB)                                  \
  createHeaderInsertionFactory(HEADER, SUB)
#define TYPESTR makeTypeStrCreator()
#define WARNING_FACTORY(MSGID, ARGS) \
  createReportWarningTypeLocRewriterFactory(MSGID, ARGS)
#define ADD_POINTER(CREATOR) \
  makeAddPointerCreator(CREATOR)
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
}
} // namespace dpct
} // namespace clang
