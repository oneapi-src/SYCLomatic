//===--------------- DpctOptions.h-----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_OPTIONS_H__
#define __DPCT_OPTIONS_H__

#include "llvm/Support/CommandLine.h"

#ifdef NDEBUG
#undef DPCT_DEBUG_BUILD
#else
#undef DPCT_DEBUG_BUILD
#define DPCT_DEBUG_BUILD 1
#endif

namespace clang {
namespace dpct {

#define DPCT_OPTION(TEMPLATE, TYPE, NAME, ...) OPT_##NAME,
#define DPCT_OPTIONS_IN_CLANG_TOOLING 0
enum class DpctOptionNameKind {
#include "DPCTOptions.inc"
  OPT_NUM
};
enum class DpctOptionClass { OC_Action, OC_Attribute };
enum class DpctActionKind {
  DAK_Migration = 0,
  DAK_Analysis,
  DAK_Query,
  DAK_BuildScript,
  DAK_Help,
  DAK_Independent,
  DAK_NUM
};

class DpctOptionBase {
public:
  using BitsType = int64_t;

  static_assert(static_cast<unsigned>(DpctOptionNameKind::OPT_NUM) <
                    sizeof(BitsType) * CHAR_BIT,
                "Option number is larger than BitsType width");

private:
  DpctOptionNameKind Name;
  DpctOptionClass Class;
  BitsType FlagBit;
  BitsType Exclusive;
  BitsType Dependencies;

  static std::vector<DpctOptionBase *> getOptions(BitsType Flag);

  /// The usage of option /p DependerOption requires option /p DependeeOption to
  /// be specified.
  /// Requires to be called in DpctOptionBase::init().
  static void setDependency(DpctOptionNameKind DependentOption,
                            DpctOptionNameKind DependeeOption);

  /// Sepcify that /p OptionA and /p OptionB cannot be used together.
  /// Requires to be called in DpctOptionBase::init().
  static void setExclusive(DpctOptionNameKind OptionA,
                           DpctOptionNameKind OptionB);

  void setExclusiveByAction();
  void printName(llvm::raw_ostream &OS);

  void reportAsIgnored(DpctOptionBase *ConflictedBy,
                       llvm::raw_ostream &OutStream);

  virtual void reset() = 0;
  virtual llvm::StringRef getName() const noexcept = 0;

  friend struct OptionActions;

protected:
  void setOccurrenced();
  void addActionGroup(DpctActionKind);

  DpctOptionBase(DpctOptionNameKind, DpctOptionClass,
                 std::initializer_list<DpctActionKind>);

public:
  virtual ~DpctOptionBase() = default;
  static void init();
  static void check();
};

// Using alias to unify template parameters' difference.
// Always using default storage.
template <class DataType, class ParserClass = llvm::cl::parser<DataType>>
using list = llvm::cl::list<DataType, bool, ParserClass>;
template <class DataType, class ParserClass = llvm::cl::parser<DataType>>
using bits = llvm::cl::bits<DataType, bool, ParserClass>;
template <class DataType, class ParserClass = llvm::cl::parser<DataType>>
using opt = llvm::cl::opt<DataType, false, ParserClass>;

template <template <class, class> class OptionTemplate, class DataType,
          class ParserClass>
class DpctOptionConstructImpl : public DpctOptionBase,
                                public OptionTemplate<DataType, ParserClass> {
public:
  using ClOptionType = OptionTemplate<DataType, ParserClass>;

  template <class... Args>
  DpctOptionConstructImpl(DpctOptionNameKind Name, DpctOptionClass Class,
                          std::initializer_list<DpctActionKind> Actions,
                          Args &&...ClOptionArgs)
      : DpctOptionBase(Name, Class, Actions),
        ClOptionType(std::forward<Args>(ClOptionArgs)...) {
    ClOptionType::setCallback(
        [this](const typename ParserClass::parser_data_type &) {
          this->setOccurrenced();
        });
  }

private:
  llvm::StringRef getName() const noexcept override { return this->ArgStr; }
};

template <template <class, class> class OptionTemplate, class DataType,
          class ParserClass = llvm::cl::parser<DataType>>
class DpctOption
    : public DpctOptionConstructImpl<OptionTemplate, DataType, ParserClass> {
  using BaseType =
      DpctOptionConstructImpl<OptionTemplate, DataType, ParserClass>;

public:
  using BaseType::BaseType;

private:
  void reset() override { BaseType::clear(); }
};

template <class DataType, class ParserClass>
class DpctOption<clang::dpct::opt, DataType, ParserClass>
    : public DpctOptionConstructImpl<clang::dpct::opt, DataType, ParserClass> {
  using BaseType =
      DpctOptionConstructImpl<clang::dpct::opt, DataType, ParserClass>;

public:
  using BaseType::BaseType;

private:
  void reset() override {
    const llvm::cl::OptionValue<DataType> &V = BaseType::getDefault();
    if (V.hasValue())
      BaseType::setValue(V.getValue());
    else
      BaseType::setValue(DataType());
  }
};

} // namespace dpct
} // namespace clang

#endif //!__DPCT_OPTIONS_H__

