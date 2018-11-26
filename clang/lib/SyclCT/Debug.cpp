#include "Debug.h"
#include "ASTTraversal.h"

namespace clang {
namespace syclct {

void DebugInfo::printTranslationRules(
    const std::vector<std::unique_ptr<ASTTraversal>> &TRs) {
  if (!IsDebugBuild) {
    return;
  }

  auto print = [&]() {
    llvm::dbgs() << "Translation Rules:\n";

    constexpr char Indent[] = "  ";
    if (TRs.empty()) {
      llvm::dbgs() << Indent << "None\n";
      return;
    }

    for (auto &TR : TRs) {
      if (auto I = dyn_cast<TranslationRule>(&*TR)) {
        llvm::dbgs() << Indent << I->getName() << "\n";
      }
    }
  };

  DEBUG_WITH_TYPE("TranslationRules", print());
}

void DebugInfo::printMatchedRules(
    const std::vector<std::unique_ptr<ASTTraversal>> &MatchedRules) {
  if (!IsDebugBuild) {
    return;
  }

  for (auto &MR : MatchedRules) {
    if (auto TR = dyn_cast<TranslationRule>(&*MR)) {
#define CHECK_RULE(TYPE)                                                       \
  if (TR->getName() == #TYPE) {                                                \
    DEBUG_WITH_TYPE(#TYPE, TR->print(llvm::dbgs()));                           \
    continue;                                                                  \
  }
      CHECK_RULE(IterationSpaceBuiltinRule)
      CHECK_RULE(ErrorHandlingIfStmtRule)
      CHECK_RULE(AlignAttrsRule)
      CHECK_RULE(FunctionAttrsRule)
      CHECK_RULE(TypeInVarDeclRule)
      CHECK_RULE(SyclStyleVectorRule)
      CHECK_RULE(SyclStyleVectorCtorRule)
      CHECK_RULE(ReplaceDim3CtorRule)
      CHECK_RULE(Dim3MemberFieldsRule)
      CHECK_RULE(ReturnTypeRule)
      CHECK_RULE(DevicePropVarRule)
      CHECK_RULE(EnumConstantRule)
      CHECK_RULE(ErrorConstantsRule)
      CHECK_RULE(FunctionCallRule)
      CHECK_RULE(KernelCallRule)
      CHECK_RULE(SharedMemVarRule)
      CHECK_RULE(ConstantMemVarRule)
      CHECK_RULE(DeviceMemVarRule)
      CHECK_RULE(MemoryTranslationRule)
      CHECK_RULE(ErrorTryCatchRule)
      CHECK_RULE(KernelIterationSpaceRule)
      CHECK_RULE(UnnamedTypesRule)
      CHECK_RULE(MathFunctionsRule)
      CHECK_RULE(SyncThreadsRule)
      CHECK_RULE(KernelFunctionInfoRule)
#undef CHECK_RULE
    }
  }
}

void DebugInfo::printReplacements(ReplacementFilter &ReplFilter,
                                  clang::ASTContext &Context) {
  if (!IsDebugBuild) {
    return;
  }

  for (const ExtReplacement &Repl : ReplFilter) {
    const TextModification *TM = nullptr;
#define TRANSFORMATION(TYPE)                                                   \
  TM = Repl.getParentTM();                                                     \
  if (TM && TMID::TYPE == TM->getID()) {                                       \
    DEBUG_WITH_TYPE(#TYPE, TM->print(llvm::dbgs(), Context));                  \
    continue;                                                                  \
  }
#include "Transformations.inc"
#undef TRANSFORMATION
  }
}

} // namespace syclct
} // namespace clang
