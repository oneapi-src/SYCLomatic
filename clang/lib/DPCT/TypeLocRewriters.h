//===--------------- TypeLocRewriters.h -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriter.h"

namespace clang {
namespace dpct {

class TypeLocRewriter {
protected:
  const TypeLoc TL;

protected:
  TypeLocRewriter(const TypeLoc TL) : TL(TL) {}

public:
  virtual ~TypeLocRewriter() {}
  virtual Optional<std::string> rewrite() = 0;
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
  Optional<std::string> rewrite() override {
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

template <class TypeNameT>
struct TypeNameTypeLocRewriter
    : public TypePrinterRewriter<TypeNamePrinter<TypeNameT>> {
  TypeNameTypeLocRewriter(
      const TypeLoc TL,
      const std::function<TypeNameT(const TypeLoc)> &TypeNameCreator)
      : TypePrinterRewriter<TypeNamePrinter<TypeNameT>>(TL,
                                                        TypeNameCreator(TL)) {}
};

class TypeLocRewriterFactoryBase {
public:
  virtual std::shared_ptr<TypeLocRewriter> create(const TypeLoc) const = 0;
  virtual ~TypeLocRewriterFactoryBase() {}

  static std::unique_ptr<std::unordered_map<
    std::string, std::shared_ptr<TypeLocRewriterFactoryBase>>>
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
    requestFeature(Feature, TL.getBeginLoc());
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
    if (!TL)
      return std::shared_ptr<TypeLocRewriter>();
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

} // namespace dpct
} // namespace clang