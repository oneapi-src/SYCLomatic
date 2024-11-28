//===--------------- ASTTraversal.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASTTraversal.h"
#include "RulesLang/RulesLang.h"
#include "AnalysisInfo.h"
#include "RulesAsm/AsmMigration.h"
#include "RulesDNN/DNNAPIMigration.h"
#include "RulesMathLib/FFTAPIMigration.h"
#include "RulesMathLib/RandomAPIMigration.h"
#include "RulesMathLib/SolverAPIMigration.h"
#include "RulesMathLib/BLASAPIMigration.h"
#include "CodePin/GenCodePinHeader.h"
#include "RulesSecurity/Homoglyph.h"
#include "RulesLangLib/LIBCUAPIMigration.h"
#include "MigrationRuleManager.h"
#include "RulesSecurity/MisleadingBidirectional.h"
#include "RulesCCL/NCCLAPIMigration.h"
#include "RulesLang/OptimizeMigration.h"
#include "RulesMathLib/SpBLASAPIMigration.h"
#include "TextModification.h"
#include "RulesLangLib/ThrustAPIMigration.h"
#include "Utility.h"
#include "RulesLang/WMMAAPIMigration.h"

#include <string>
#include <unordered_map>
#include <utility>

using namespace clang;
using namespace clang::dpct;

namespace clang {
namespace dpct {

unsigned MigrationRule::PairID = 0;

void MigrationRule::print(llvm::raw_ostream &OS) {
  const auto &EmittedTransformations = getEmittedTransformations();
  if (EmittedTransformations.empty()) {
    return;
  }

  OS << "[" << getName() << "]" << getNL();
  constexpr char Indent[] = "  ";
  for (const auto &TM : EmittedTransformations) {
    OS << Indent;
    TM->print(OS, DpctGlobalInfo::getContext(),
              /* Print parent */ false);
  }
}

void MigrationRule::printStatistics(llvm::raw_ostream &OS) {
  const auto &EmittedTransformations = getEmittedTransformations();
  if (EmittedTransformations.empty()) {
    return;
  }

  OS << "<Statistics of " << getName() << ">" << getNL();
  std::unordered_map<std::string, size_t> TMNameCountMap;
  for (const auto &TM : EmittedTransformations) {
    const std::string Name = TM->getName();
    if (TMNameCountMap.count(Name) == 0) {
      TMNameCountMap.emplace(std::make_pair(Name, 1));
    } else {
      ++TMNameCountMap[Name];
    }
  }

  constexpr char Indent[] = "  ";
  for (const auto &Pair : TMNameCountMap) {
    const std::string &Name = Pair.first;
    const size_t &Numbers = Pair.second;
    OS << Indent << "Emitted # of replacement <" << Name << ">: " << Numbers
       << getNL();
  }
}

void MigrationRule::emplaceTransformation(TextModification *TM) {
  auto T = std::shared_ptr<TextModification>(TM);
  Transformations.emplace_back(T);
  TransformSet->emplace_back(T);
}

// RuleLang
REGISTER_RULE(IterationSpaceBuiltinRule, PassKind::PK_Analysis)
REGISTER_RULE(ErrorHandlingIfStmtRule, PassKind::PK_Migration)
REGISTER_RULE(ErrorHandlingHostAPIRule, PassKind::PK_Migration)
REGISTER_RULE(AtomicFunctionRule, PassKind::PK_Migration)
REGISTER_RULE(ZeroLengthArrayRule, PassKind::PK_Migration)
REGISTER_RULE(MiscAPIRule, PassKind::PK_Migration)
REGISTER_RULE(TypeInDeclRule, PassKind::PK_Migration)
REGISTER_RULE(VectorTypeNamespaceRule, PassKind::PK_Migration)
REGISTER_RULE(VectorTypeMemberAccessRule, PassKind::PK_Migration)
REGISTER_RULE(VectorTypeOperatorRule, PassKind::PK_Migration)
REGISTER_RULE(DeviceInfoVarRule, PassKind::PK_Migration)
REGISTER_RULE(EnumConstantRule, PassKind::PK_Migration)
REGISTER_RULE(ErrorConstantsRule, PassKind::PK_Migration)
REGISTER_RULE(LinkageSpecDeclRule, PassKind::PK_Migration)
REGISTER_RULE(CU_JITEnumsRule, PassKind::PK_Migration)
REGISTER_RULE(FunctionCallRule, PassKind::PK_Migration)
REGISTER_RULE(EventAPICallRule, PassKind::PK_Migration)
REGISTER_RULE(ProfilingEnableOnDemandRule, PassKind::PK_Analysis)
REGISTER_RULE(StreamAPICallRule, PassKind::PK_Migration)
REGISTER_RULE(KernelCallRule, PassKind::PK_Analysis)
REGISTER_RULE(DeviceFunctionDeclRule, PassKind::PK_Analysis)
REGISTER_RULE(MemVarRefMigrationRule, PassKind::PK_Migration)
REGISTER_RULE(ConstantMemVarMigrationRule, PassKind::PK_Migration)
REGISTER_RULE(MemVarMigrationRule, PassKind::PK_Migration)
REGISTER_RULE(MemVarAnalysisRule, PassKind::PK_Analysis)
REGISTER_RULE(MemoryMigrationRule, PassKind::PK_Migration)
REGISTER_RULE(MemoryDataTypeRule, PassKind::PK_Migration)
REGISTER_RULE(UnnamedTypesRule, PassKind::PK_Migration)
REGISTER_RULE(TypeMmberRule, PassKind::PK_Migration)
REGISTER_RULE(CMemoryAPIRule, PassKind::PK_Migration)
REGISTER_RULE(GuessIndentWidthRule, PassKind::PK_Migration)
REGISTER_RULE(MathFunctionsRule, PassKind::PK_Migration)
REGISTER_RULE(WarpFunctionsRule, PassKind::PK_Analysis)
REGISTER_RULE(CooperativeGroupsFunctionRule, PassKind::PK_Analysis)
REGISTER_RULE(SyncThreadsRule, PassKind::PK_Analysis)
REGISTER_RULE(SyncThreadsMigrationRule, PassKind::PK_Migration)
REGISTER_RULE(KernelFunctionInfoRule, PassKind::PK_Migration)
REGISTER_RULE(RecognizeAPINameRule, PassKind::PK_Migration)
REGISTER_RULE(RecognizeTypeRule, PassKind::PK_Migration)
REGISTER_RULE(TextureMemberSetRule, PassKind::PK_Migration)
REGISTER_RULE(TextureRule, PassKind::PK_Analysis)
REGISTER_RULE(CXXNewExprRule, PassKind::PK_Migration)
REGISTER_RULE(NamespaceRule, PassKind::PK_Migration)
REGISTER_RULE(RemoveBaseClassRule, PassKind::PK_Migration)
REGISTER_RULE(AsmRule, PassKind::PK_Analysis)
REGISTER_RULE(VirtualMemRule, PassKind::PK_Migration)
REGISTER_RULE(DriverModuleAPIRule, PassKind::PK_Migration)
REGISTER_RULE(DriverDeviceAPIRule, PassKind::PK_Migration)
REGISTER_RULE(DriverContextAPIRule, PassKind::PK_Migration)
REGISTER_RULE(CudaArchMacroRule, PassKind::PK_Migration)
REGISTER_RULE(ConfusableIdentifierDetectionRule, PassKind::PK_Migration)
REGISTER_RULE(MisleadingBidirectionalRule, PassKind::PK_Migration)
REGISTER_RULE(WMMARule, PassKind::PK_Analysis)
REGISTER_RULE(ForLoopUnrollRule, PassKind::PK_Migration)
REGISTER_RULE(SpBLASTypeLocRule, PassKind::PK_Migration)
REGISTER_RULE(DeviceConstantVarOptimizeAnalysisRule, PassKind::PK_Analysis)
REGISTER_RULE(GenCodePinHeaderRule, PassKind::PK_Migration)
REGISTER_RULE(ComplexAPIRule, PassKind::PK_Migration)
REGISTER_RULE(TemplateSpecializationTypeLocRule, PassKind::PK_Migration)
REGISTER_RULE(CudaStreamCastRule, PassKind::PK_Migration)
REGISTER_RULE(CudaExtentRule, PassKind::PK_Analysis)
REGISTER_RULE(CudaUuidRule, PassKind::PK_Analysis)
REGISTER_RULE(TypeRemoveRule, PassKind::PK_Analysis)
REGISTER_RULE(CompatWithClangRule, PassKind::PK_Migration)
REGISTER_RULE(AssertRule, PassKind::PK_Migration)
REGISTER_RULE(GraphRule, PassKind::PK_Migration)
REGISTER_RULE(GraphicsInteropRule, PassKind::PK_Migration)
REGISTER_RULE(RulesLangAddrSpaceConvRule, PassKind::PK_Migration)

REGISTER_RULE(BLASEnumsRule, PassKind::PK_Migration, RuleGroupKind::RK_BLas)
REGISTER_RULE(BLASFunctionCallRule, PassKind::PK_Migration,RuleGroupKind::RK_BLas)

REGISTER_RULE(SPBLASEnumsRule, PassKind::PK_Migration, RuleGroupKind::RK_Sparse)
REGISTER_RULE(SPBLASFunctionCallRule, PassKind::PK_Migration,RuleGroupKind::RK_Sparse)

REGISTER_RULE(RandomEnumsRule, PassKind::PK_Migration, RuleGroupKind::RK_Rng)
REGISTER_RULE(RandomFunctionCallRule, PassKind::PK_Migration,RuleGroupKind::RK_Rng)
REGISTER_RULE(DeviceRandomFunctionCallRule, PassKind::PK_Migration,RuleGroupKind::RK_Rng)

REGISTER_RULE(SOLVEREnumsRule, PassKind::PK_Migration, RuleGroupKind::RK_Solver)
REGISTER_RULE(SOLVERFunctionCallRule, PassKind::PK_Migration,RuleGroupKind::RK_Solver)

REGISTER_RULE(LIBCURule, PassKind::PK_Migration, RuleGroupKind::RK_Libcu)

REGISTER_RULE(ThrustAPIRule, PassKind::PK_Migration, RuleGroupKind::RK_Thrust)
REGISTER_RULE(ThrustTypeRule, PassKind::PK_Migration, RuleGroupKind::RK_Thrust)

REGISTER_RULE(ManualMigrateEnumsRule, PassKind::PK_Migration, RuleGroupKind::RK_NCCL)
REGISTER_RULE(NCCLRule, PassKind::PK_Migration, RuleGroupKind::RK_NCCL)

REGISTER_RULE(FFTEnumsRule, PassKind::PK_Migration, RuleGroupKind::RK_FFT)
REGISTER_RULE(FFTFunctionCallRule, PassKind::PK_Migration,RuleGroupKind::RK_FFT)

REGISTER_RULE(CuDNNTypeRule, PassKind::PK_Migration, RuleGroupKind::RK_DNN)
REGISTER_RULE(CuDNNAPIRule, PassKind::PK_Migration, RuleGroupKind::RK_DNN)

} // namespace dpct
} // namespace clang