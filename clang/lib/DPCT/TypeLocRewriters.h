#include "CallExprRewriter.h"

namespace clang {
namespace dpct {

template <class TypeNameT, class... TemplateArgsT> class TypeLocPrinter {
  TypeNameT TypeName;
  ArgsPrinter<false, TemplateArgsT...> TAs;

public:
  TypeLocPrinter(TypeNameT TypeName, TemplateArgsT &&...TAs)
      : TypeName(TypeName), TAs(std::forward<TemplateArgsT>(TAs)...) {}
  template <class StreamT> void print(StreamT &Stream) const {
    dpct::print(Stream, TypeName);
    Stream << "<";
    TAs.print(Stream);
    Stream << ">";
  }
};

class TypeLocRewriter {
protected:
  const TypeLoc *TL;
  StringRef OriginalTypeName;

protected:
  TypeLocRewriter(const TypeLoc *TL, StringRef OriginalTypeName)
      : TL(TL), OriginalTypeName(OriginalTypeName) {}

public:
  ArgumentAnalysis Analyzer;
  virtual ~TypeLocRewriter() {}
  virtual Optional<std::string> rewrite() = 0;

protected:
  StringRef getOriginalTypeName() { return OriginalTypeName; }
};

template <class Printer>
class TypePrinterRewriter : Printer, public TypeLocRewriter {
public:
  template <class... ArgsT>
  TypePrinterRewriter(const TypeLoc *TL, StringRef Source, ArgsT &&...Args)
      : Printer(std::forward<ArgsT>(Args)...), TypeLocRewriter(TL, Source) {}
  template <class... ArgsT>
  TypePrinterRewriter(const TypeLoc *TL, StringRef Source,
                  const std::function<ArgsT(const CallExpr *)> &...ArgCreators)
      : TypePrinterRewriter(TL, Source, ArgCreators(TL)...) {}
  Optional<std::string> rewrite() override {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    Printer::print(OS);
    return OS.str();
  }
};

template <class TypeNameT, class... TemplateArgsT>
class TemplateTypeLocRewriter
    : public TypePrinterRewriter<TypeLocPrinter<TypeNameT, TemplateArgsT...>> {
public:
  TemplateTypeLocRewriter(
      const TypeLoc *TL, StringRef OriginalName,
      const std::function<TypeNameT(const TypeLoc *)> &TypeNameCreator,
      const std::function<TemplateArgsT(const TypeLoc *)> &...TAsCreator)
      : TypePrinterRewriter<TypeLocPrinter<TypeNameT, TemplateArgsT...>>(
            TL, OriginalName, TypeNameCreator(TL), TAsCreator(TL)...) {}
};

class TypeLocRewriterFactoryBase {
public:
  virtual std::shared_ptr<TypeLocRewriter> create(const TypeLoc *) const = 0;
  virtual ~TypeLocRewriterFactoryBase() {}

  static std::unique_ptr<std::unordered_map<
    std::string, std::shared_ptr<TypeLocRewriterFactoryBase>>>
    TypeLocRewriterMap;
  static void initTypeLocRewriterMap();
  RulePriority Priority = RulePriority::Fallback;
};

template <class RewriterTy, class... TAs>
class TypeLocRewriterFactory : public TypeLocRewriterFactoryBase {
  std::tuple<std::string, TAs...> Initializer;

private:
  template <size_t... Idx>
  inline std::shared_ptr<TypeLocRewriter>
  createRewriter(const TypeLoc *TL, std::index_sequence<Idx...>) const {
    return std::shared_ptr<RewriterTy>(
        new RewriterTy(TL, std::get<Idx>(Initializer)...));
  }

public:
  TypeLocRewriterFactory(StringRef OriginalTypeName, TAs... TemplateArgs)
      : Initializer(OriginalTypeName.str(),
                    std::forward<TAs>(TemplateArgs)...) {}
  std::shared_ptr<TypeLocRewriter> create(const TypeLoc *TL) const override {
    if (!TL)
      return std::shared_ptr<TypeLocRewriter>();
    return createRewriter(
        TL, std::index_sequence_for<std::string, TAs...>());
  }
};

class TypeLocConditionalRewriterFactory : public TypeLocRewriterFactoryBase {
  std::function<bool(const TypeLoc *)> Pred;
  std::shared_ptr<TypeLocRewriterFactoryBase> First, Second;

public:
  template <class InputPred>
  TypeLocConditionalRewriterFactory(
      InputPred &&P, std::shared_ptr<TypeLocRewriterFactoryBase> FirstFactory,
      std::shared_ptr<TypeLocRewriterFactoryBase> SecondFactory)
      : Pred(std::forward<InputPred>(P)), First(FirstFactory),
        Second(SecondFactory) {}
  std::shared_ptr<TypeLocRewriter> create(const TypeLoc *TL) const override {
    if (Pred(TL))
      return First->create(TL);
    else
      return Second->create(TL);
  }
};


} // namespace dpct
} // namespace clang