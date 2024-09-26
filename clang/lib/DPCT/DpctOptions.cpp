//===--------------- DpctOptions.cpp--------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/DPCT/DpctOptions.h"

#include "Error.h"
#include "Utility.h"

namespace clang {
namespace dpct {

namespace {

DpctOptionBase::BitsType ExplicitOptions = 0;
DpctOptionBase::BitsType DependedOptions = 0;
DpctOptionBase::BitsType ActionOptions = 0;
DpctOptionBase::BitsType
    ActionGroup[static_cast<unsigned>(DpctActionKind::DAK_NUM)] = {0};

std::array<DpctOptionBase *, static_cast<unsigned>(DpctOptionNameKind::OPT_NUM)>
    DpctOptionList;

inline llvm::raw_ostream &getErrorStream() { return llvm::errs(); }
inline DpctOptionBase *getOption(DpctOptionNameKind Kind) {
  return DpctOptionList[static_cast<unsigned>(Kind)];
}

inline constexpr DpctOptionBase::BitsType getFlagBit(DpctOptionNameKind Kind) {
  return DpctOptionBase::BitsType(1) << static_cast<unsigned>(Kind);
}

inline void fatalError() {
  ShowStatus(MigrationErrorConflictOptions);
  dpctExit(MigrationErrorConflictOptions);
}

#ifdef DPCT_DEBUG_BUILD
StringRef getOptionName(DpctOptionNameKind Kind) {
#define DPCT_OPTIONS_IN_CLANG_TOOLING 0
#define DPCT_OPTION(TEMPLATE, TYPE, NAME, ...)                                 \
  case DpctOptionNameKind::OPT_##NAME:                                         \
    return #NAME;
  switch (Kind) {
#include "clang/DPCT/DPCTOptions.inc"
  default:
    return "";
  }
}
#endif

} // namespace

DpctOptionBase::DpctOptionBase(DpctOptionNameKind Name, DpctOptionClass OC,
                               std::initializer_list<DpctActionKind> Actions)
    : Name(Name), Class(OC), FlagBit(0), Exclusive(0), Dependencies(0) {
  FlagBit = getFlagBit(Name);
  Exclusive = ~FlagBit;
  DpctOptionList[static_cast<unsigned>(Name)] = this;
  for (auto Action : Actions)
    addActionGroup(Action);
  if (OC == DpctOptionClass::OC_Action)
    ActionOptions |= FlagBit;
}

inline void DpctOptionBase::addActionGroup(DpctActionKind Action) {
  ActionGroup[static_cast<unsigned>(Action)] |= FlagBit;
}

std::vector<DpctOptionBase *> DpctOptionBase::getOptions(BitsType Flag) {
  static constexpr BitsType Mask = getFlagBit(DpctOptionNameKind::OPT_NUM) - 1;
  Flag &= Mask;
  std::vector<DpctOptionBase *> Ret;
  auto Begin = DpctOptionList.rbegin();
  while (Flag) {
    const auto OptionIter = std::lower_bound(
        Begin, DpctOptionList.rend(), Flag,
        [](const DpctOptionBase *Option, BitsType Flag) -> bool {
          return Option->FlagBit > Flag;
        });
    assert(OptionIter != DpctOptionList.rend());
    Ret.push_back(*OptionIter);
    Flag ^= (*OptionIter)->FlagBit;
    Begin = OptionIter;
  }
  return Ret;
}

inline void DpctOptionBase::printName(llvm::raw_ostream &OS) {
  OS << "\"-" << getName() << "\"";
}

inline void DpctOptionBase::setExclusive(DpctOptionNameKind OptionA,
                                         DpctOptionNameKind OptionB) {
  DpctOptionBase *OptA = getOption(OptionA), *OptB = getOption(OptionB);
  OptA->Exclusive |= OptB->FlagBit;
  OptB->Exclusive |= OptA->FlagBit;
}

inline void DpctOptionBase::setDependency(DpctOptionNameKind DependentOption,
                                          DpctOptionNameKind DependedOption) {
  getOption(DependentOption)->Dependencies |= getFlagBit(DependedOption);
}

void DpctOptionBase::setExclusiveByAction() {
  DpctOptionBase::BitsType Mask = 0;
  if (Class == DpctOptionClass::OC_Action)
    // Even in the same action group, actions are still exclusive with each
    // other.
    Mask = ActionOptions;
  else
    // By default, attribute options wouldn't be exclusive with another
    // attribute.
    Exclusive &= ActionOptions;

  for (auto Action : ActionGroup) {
    if (FlagBit & Action) {
      // Exclude other options in the same action group from 'Exclusive'.
      Exclusive &= ~Action | Mask;
    }
  }
}

void DpctOptionBase::reportAsIgnored(DpctOptionBase *ConflictBy,
                                     llvm::raw_ostream &OutStream) {
  reset();
  OutStream << "Warning: Option ";
  printName(OutStream);
  OutStream << " is ignored because it conflicts with option ";
  ConflictBy->printName(OutStream);
  OutStream << ".\n";
}

void DpctOptionBase::setOccurrenced() {
  static auto &ErrorStream = getErrorStream();
  if (auto Conflicts = ExplicitOptions & Exclusive) {
    if (Class == DpctOptionClass::OC_Attribute) {
      auto ConflictOptions = getOptions(Conflicts);
      for (auto Option : ConflictOptions) {
        reportAsIgnored(Option, ErrorStream);
      }
      return;
    } else if (auto ConflictActions = Conflicts & ActionOptions) {
      auto Actions = getOptions(ConflictActions);
      for (auto Action : Actions) {
        ErrorStream << "Error: Option ";
        printName(ErrorStream);
        ErrorStream << " and option ";
        Action->printName(ErrorStream);
        ErrorStream << " can not be used together.\n";
      }
      fatalError();
    } else {
      auto Options = getOptions(Conflicts);
      for (auto Option : Options) {
        Option->reportAsIgnored(this, ErrorStream);
      }
    }
  }
  ExplicitOptions |= FlagBit;
  DependedOptions |= Dependencies;
}

void DpctOptionBase::init() {
  for (auto Option : DpctOptionList) {
    if (!Option)
      continue;
    Option->setExclusiveByAction();
  }
  setDependency(DpctOptionNameKind::OPT_ProcessAll,
                DpctOptionNameKind::OPT_InRoot);
  setDependency(DpctOptionNameKind::OPT_BuildScriptFile,
                DpctOptionNameKind::OPT_GenBuildScript);
}

void DpctOptionBase::check() {
  auto &ErrorStream = getErrorStream();
  auto NoSepcifiedOptions = ~ExplicitOptions;
  if (auto Lack = DependedOptions & NoSepcifiedOptions) {
    auto ReportErrMsg = [&](DpctOptionBase *Required) {
      bool IsFirst = true;
      ErrorStream << "Error: Option ";
      for (auto Option : DpctOptionList) {
        if (Option->Dependencies & Required->FlagBit) {
          if (IsFirst) {
            IsFirst = false;
          } else {
            ErrorStream << ", option ";
          }
          Option->printName(ErrorStream);
        }
      }
      ErrorStream << " require(s) that option ";
      Required->printName(ErrorStream);
      ErrorStream << " be specified explicitly.\n";
    };
    auto LackOptions = getOptions(Lack);
    for (auto Option : LackOptions)
      ReportErrMsg(Option);
    fatalError();
  }

  if ((ExplicitOptions & getFlagBit(DpctOptionNameKind::OPT_UseSYCLCompat)) &&
      (ExplicitOptions &
       getFlagBit(DpctOptionNameKind::OPT_GenHelperFunction))) {
    getOption(DpctOptionNameKind::OPT_GenHelperFunction)
        ->reportAsIgnored(getOption(DpctOptionNameKind::OPT_UseSYCLCompat),
                          ErrorStream);
  }
}

} // namespace dpct
} // namespace clang