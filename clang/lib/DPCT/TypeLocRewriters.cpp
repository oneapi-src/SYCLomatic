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
        return TemplateArgumentInfo(TAL);
      }
    }
    return TemplateArgumentInfo("");
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

template <class TypeNameT, class... TemplateArgsT>
std::shared_ptr<TypeLocRewriterFactoryBase> creatTypeLocRewriterFactory(
    std::function<TypeNameT(const TypeLoc)> TypeNameCreator,
    std::function<TemplateArgsT(const TypeLoc)>... TAsCreator) {
  return std::make_shared<
      TypeLocRewriterFactory<TemplateTypeLocRewriter<TypeNameT, TemplateArgsT...>,
                             std::function<TypeNameT(const TypeLoc)>,
                             std::function<TemplateArgsT(const TypeLoc)>...>>(
      std::forward<std::function<TypeNameT(const TypeLoc)>>(TypeNameCreator),
      std::forward<std::function<TemplateArgsT(const TypeLoc)>>(TAsCreator)...);
}

std::shared_ptr<TypeLocRewriterFactoryBase> createTypeLocConditionalFactory(
    std::function<bool(const TypeLoc)> Pred,
    std::shared_ptr<TypeLocRewriterFactoryBase> &&First,
    std::shared_ptr<TypeLocRewriterFactoryBase> &&Second) {
  return std::make_shared<TypeLocConditionalRewriterFactory>(Pred, First,
                                                             Second);
}

std::unique_ptr<std::unordered_map<
    std::string, std::shared_ptr<TypeLocRewriterFactoryBase>>>
    TypeLocRewriterFactoryBase::TypeLocRewriterMap;

void TypeLocRewriterFactoryBase::initTypeLocRewriterMap() {
  TypeLocRewriterMap = std::make_unique<std::unordered_map<
      std::string, std::shared_ptr<TypeLocRewriterFactoryBase>>>(
      std::unordered_map<std::string,
                         std::shared_ptr<TypeLocRewriterFactoryBase>>(
          {{"cuda::atomic", createTypeLocConditionalFactory(
                         CheckTemplateArgCount(2),
                         creatTypeLocRewriterFactory(
                             makeStringCreator("atomic_ext"),
                             makeTemplateArgCreator(0),
                             makeStringCreator(MapNames::getClNamespace() + "memory_order::relaxed"),
                             makeTemplateArgCreator(1)),
                         creatTypeLocRewriterFactory(
                             makeStringCreator("atomic_ext"),
                             makeTemplateArgCreator(0)))}}));
}
} // namespace dpct
} // namespace clang