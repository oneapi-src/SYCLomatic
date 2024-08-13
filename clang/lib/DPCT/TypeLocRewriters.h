//===--------------- TypeLocRewriters.h -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AnalysisInfo.h"
#include "CallExprRewriter.h"
#include "Rules.h"
#include "Utility.h"
#include "clang/AST/ASTFwd.h"
#include "clang/AST/DeclCXX.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/Support/Casting.h"
#include <iostream>
#include <utility>
#include <vector>

namespace clang {
namespace dpct {

class TypeLocRewriter {
protected:
  const TypeLoc TL;

protected:
  TypeLocRewriter(const TypeLoc TL, bool ReplTemplate = false)
      : TL(TL), ReplaceTemplate(ReplTemplate) {}

public:
  bool ReplaceTemplate = false;
  virtual ~TypeLocRewriter() {}
  virtual std::optional<std::string> rewrite() = 0;
};

template <class Printer>
class TypePrinterRewriter : Printer, public TypeLocRewriter {
public:
  template <class... ArgsT>
  TypePrinterRewriter(const TypeLoc TL, ArgsT &&...Args)
      : Printer(std::forward<ArgsT>(Args)...), TypeLocRewriter(TL) {}
  template <class... ArgsT>
  TypePrinterRewriter(
      const TypeLoc TL,
      const std::function<ArgsT(const CallExpr *)> &...ArgCreators)
      : TypePrinterRewriter(TL, ArgCreators(TL)...) {}
  std::optional<std::string> rewrite() override {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    Printer::print(OS);
    return OS.str();
  }
};

template <class TypeNameT, class... TemplateArgsT>
class TemplateTypeLocRewriter
    : public TypePrinterRewriter<TemplatedNamePrinter<TypeNameT, TemplateArgsT...>> {
public:
  TemplateTypeLocRewriter(
      const TypeLoc TL,
      const std::function<TypeNameT(const TypeLoc)> &TypeNameCreator,
      const std::function<TemplateArgsT(const TypeLoc)> &...TAsCreator)
      : TypePrinterRewriter<TemplatedNamePrinter<TypeNameT, TemplateArgsT...>>(
            TL, TypeNameCreator(TL), TAsCreator(TL)...) {}
};

template <class TypeNameT, class... TemplateArgsT>
class CtadTemplateTypeLocRewriter
    : public TypePrinterRewriter<CtadTemplatedNamePrinter<TypeNameT, TemplateArgsT...>> {
public:
  CtadTemplateTypeLocRewriter(
      const TypeLoc TL,
      const std::function<TypeNameT(const TypeLoc)> &TypeNameCreator,
      const std::function<TemplateArgsT(const TypeLoc)> &...TAsCreator)
      : TypePrinterRewriter<CtadTemplatedNamePrinter<TypeNameT, TemplateArgsT...>>(
            TL, TypeNameCreator(TL), TAsCreator(TL)...) {}
};

template <class TypeNameT>
struct TypeNameTypeLocRewriter
    : public TypePrinterRewriter<TypeNamePrinter<TypeNameT>> {
  TypeNameTypeLocRewriter(
      const TypeLoc TL,
      const std::function<TypeNameT(const TypeLoc)> &TypeNameCreator)
      : TypePrinterRewriter<TypeNamePrinter<TypeNameT>>(TL,
                                                        TypeNameCreator(TL)) {}
};

class ReportWarningTypeLocRewriter : public TypeLocRewriter {
public:
  template <class F>
  static std::string getMsgArg(F&& f, const TypeLoc TL) {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    print(OS, f(TL));
    return OS.str();
  }

  template <typename IDTy, typename... Ts>
  inline void report(IDTy MsgID, bool UseTextBegin, Ts &&...Vals) {
    TransformSetTy TS;
    auto SL = TL.getBeginLoc();
    DiagnosticsUtils::report(
        SL, MsgID, &TS, UseTextBegin, std::forward<Ts>(Vals)...);
    for (auto &T : TS)
      DpctGlobalInfo::getInstance().addReplacement(
          T->getReplacement(DpctGlobalInfo::getContext()));
  }

  template <class... MsgArgs>
  ReportWarningTypeLocRewriter(const TypeLoc TL,
                               Diagnostics MsgID, MsgArgs&&...Args)
    : TypeLocRewriter(TL) {
    report(MsgID, false, getMsgArg(std::forward<MsgArgs>(Args), TL)...);
  }

  std::optional<std::string> rewrite() override {
    return {};
  }
};

class TypeMatchingDesc {
private:
  std::string Name;

public:
  TypeMatchingDesc(const std::string &Name, const int TAC = -1)
      : Name(Name), TemplateArgCount(TAC) {}
  TypeMatchingDesc(TypeLoc TL) {
    auto &Context = dpct::DpctGlobalInfo::getContext();
    // ignore template args
    if (auto ETL = TL.getAs<ElaboratedTypeLoc>()) {
      // A::B::C::typename<int>
      //          ^           ^ <-- getNamedTypeLoc
      TL = ETL.getNamedTypeLoc();
    }
    if (auto TSTL = TL.getAs<TemplateSpecializationTypeLoc>()) {
      llvm::raw_string_ostream OS(Name);
      auto PP = Context.getPrintingPolicy();
      TSTL.getTypePtr()->getTemplateName().print(
          OS, PP, TemplateName::Qualified::Fully);
      if (auto DeclPtr =
              TSTL.getTypePtr()->getTemplateName().getAsTemplateDecl()) {
        TemplateArgCount = DeclPtr->getTemplateParameters()->size();
      }
    } else {
      Name = dpct::DpctGlobalInfo::getTypeName(TL.getType(), Context);
    }
  }
  bool operator==(const TypeMatchingDesc &RHS) const {
    if (!Name.compare(RHS.Name) &&
        (RHS.TemplateArgCount == TemplateArgCount ||
         RHS.TemplateArgCount == -1 || TemplateArgCount == -1)) {
      return true;
    }
    return false;
  }
  const std::string &getName() const { return Name; }

  struct hash {
    std::size_t operator()(const TypeMatchingDesc &TM) const noexcept {
      return std::hash<std::string>{}(TM.getName());
    }
  };
  // -1 means ignore template args in matching and replacing.
  // 0, 1, 2, 3... means the explicit template arg count in matching and
  // replacing.
  int TemplateArgCount = -1;
};

class TypeLocRewriterFactoryBase {
public:
  virtual std::shared_ptr<TypeLocRewriter> create(const TypeLoc) const = 0;
  virtual ~TypeLocRewriterFactoryBase() {}
  // Caller need to handle header insertion since no location is available
  inline static std::string findReplacedName(const std::string &Name) {
    static const std::string EmptyString;
    auto Iter = TypeLocRewriterMap->find({Name});
    if (Iter != TypeLocRewriterMap->end()) {
      // No need for real TypeLoc since the factory should be created with
      // makeStringCreator. In makeStringCreator TL is not dereferenced.
      auto Rewriter = Iter->second->create(TypeLoc());
      auto Result = Rewriter->rewrite();
      if (Result.has_value()) {
        return Result.value();
      }
    }
    return EmptyString;
  }

  inline static std::pair<std::string, clang::SourceRange>
  findReplacement(TypeLoc TL) {
    static const std::string EmptyString;
    TypeMatchingDesc TMD(TL);
    std::cout<<"findReplacement TMD: "<<TMD.getName()<<" "<<TMD.TemplateArgCount<<std::endl;
    auto ReplRangeBegin = TL.getBeginLoc();
    auto ReplRangeEnd = TL.getEndLoc();

    // Process the beginloc of an elaborated type
    if (auto ETL = TL.getAs<ElaboratedTypeLoc>()) {
      // A::B::C::typename
      // ^ get the location of the qualifier
      auto QualifierLoc = ETL.getQualifierLoc();
      // A::B::C::typename<int>
      //          ^           ^ <-- getNamedTypeLoc
      TL = ETL.getNamedTypeLoc();
      if (ETL.getTypePtr()->getKeyword() == ElaboratedTypeKeyword::Typename) {
        ReplRangeBegin = QualifierLoc.getBeginLoc();
      }
    }

    auto Iter = TypeLocRewriterMap->find(TMD);
    if (Iter != TypeLocRewriterMap->end()) {
      auto Rewriter = Iter->second->create(TL);
      auto Result = Rewriter->rewrite();
      if (Result.has_value()) {
        // Exclude template arg part in the replace range if the
        // TemplateArgCount is -1
        if (auto TSTL = TL.getAs<TemplateSpecializationTypeLoc>()) {
          if (Iter->first.TemplateArgCount == -1) {
            // A::B::C::typename<int>
            //          ^
            // getTemplateNameLoc should be here after ETL is processed.
            // addReplacement will extend the end loc for 1 token.
            auto &SM = dpct::DpctGlobalInfo::getSourceManager();
            ReplRangeEnd = TSTL.getTemplateNameLoc();
            std::cout << "-1: " << ReplRangeBegin.printToString(SM) << " "
                      << ReplRangeEnd.printToString(SM) << " " << Result.value()
                      << std::endl;
          }
        }

        return std::make_pair(Result.value(),
                              clang::SourceRange(ReplRangeBegin, ReplRangeEnd));
      }
    }
    return std::make_pair(EmptyString, clang::SourceRange());
  }

  inline static std::string findReplacedName(TypeLoc TL) {
    static const std::string EmptyString;
    TypeMatchingDesc TMD(TL);

    auto Iter = TypeLocRewriterMap->find(TMD);
    if (Iter != TypeLocRewriterMap->end()) {
      auto Rewriter = Iter->second->create(TL);
      auto Result = Rewriter->rewrite();
      if (Result.has_value()) {
        return Result.value();
      }
    }
    return EmptyString;
  }

  static std::unique_ptr<std::unordered_map<
      TypeMatchingDesc, std::shared_ptr<TypeLocRewriterFactoryBase>,
      TypeMatchingDesc::hash>>
      TypeLocRewriterMap;
  static void initTypeLocRewriterMap();
  RulePriority Priority = RulePriority::Fallback;
};

class TypeLocRewriterFactoryWithFeatureRequest
    : public TypeLocRewriterFactoryBase {
  std::shared_ptr<TypeLocRewriterFactoryBase> Inner;
  HelperFeatureEnum Feature;

public:
  TypeLocRewriterFactoryWithFeatureRequest(
      HelperFeatureEnum Feature,
      std::shared_ptr<TypeLocRewriterFactoryBase> InnerFactory)
      : Inner(InnerFactory), Feature(Feature) {}
  std::shared_ptr<TypeLocRewriter> create(const TypeLoc TL) const override {
    requestFeature(Feature);
    return Inner->create(TL);
  }
};

template <class RewriterTy, class... TAs>
class TypeLocRewriterFactory : public TypeLocRewriterFactoryBase {
  std::tuple<TAs...> Initializer;

private:
  template <size_t... Idx>
  inline std::shared_ptr<TypeLocRewriter>
  createRewriter(const TypeLoc TL, std::index_sequence<Idx...>) const {
    return std::shared_ptr<RewriterTy>(
        new RewriterTy(TL, std::get<Idx>(Initializer)...));
  }

public:
  TypeLocRewriterFactory(TAs... TemplateArgs)
      : Initializer(std::forward<TAs>(TemplateArgs)...) {}
  std::shared_ptr<TypeLocRewriter> create(const TypeLoc TL) const override {
    // if (!TL)
    //   return std::shared_ptr<TypeLocRewriter>();
    return createRewriter(
        TL, std::index_sequence_for<TAs...>());
  }
};

class TypeLocConditionalRewriterFactory : public TypeLocRewriterFactoryBase {
  std::function<bool(const TypeLoc)> Pred;
  std::shared_ptr<TypeLocRewriterFactoryBase> First, Second;

public:
  template <class InputPred>
  TypeLocConditionalRewriterFactory(
      InputPred &&P, std::shared_ptr<TypeLocRewriterFactoryBase> FirstFactory,
      std::shared_ptr<TypeLocRewriterFactoryBase> SecondFactory)
      : Pred(std::forward<InputPred>(P)), First(FirstFactory),
        Second(SecondFactory) {}
  std::shared_ptr<TypeLocRewriter> create(const TypeLoc TL) const override {
    if (Pred(TL))
      return First->create(TL);
    else
      return Second->create(TL);
  }
};

class HeaderInsertionRewriterFactory : public TypeLocRewriterFactoryBase {
  HeaderType Header;
  std::shared_ptr<TypeLocRewriterFactoryBase> SubRewriterFactory;

public:
  HeaderInsertionRewriterFactory(
      HeaderType Header,
      std::shared_ptr<TypeLocRewriterFactoryBase> SubRewriterFactory)
      : Header(Header), SubRewriterFactory(SubRewriterFactory) {}

  std::shared_ptr<TypeLocRewriter> create(const TypeLoc TL) const override {
    DpctGlobalInfo::getInstance().insertHeader(TL.getBeginLoc(), Header);
    return SubRewriterFactory->create(TL);
  }
};

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

std::function<std::string(const TypeLoc)>
makeUserDefinedTypeStrCreator(MetaRuleObject &R, TypeOutputBuilder &TOB);

std::function<std::string(const TypeLoc)> makeStringCreator(
    std::string TypeName,
    clang::dpct::HelperFeatureEnum RequestFeature =
        clang::dpct::HelperFeatureEnum::none,
    const std::vector<std::string> &Headers = std::vector<std::string>());

} // namespace dpct
} // namespace clang
