//===----------------------- AsmMigration.cpp -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AsmMigration.h"
#include "AnalysisInfo.h"
#include "Asm/AsmParser.h"
#include "CallExprRewriter.h"
#include "CrashRecovery.h"
#include "MigrationRuleManager.h"
#include "TextModification.h"
#include "Utility.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Parse/RAIIObjectsForParser.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>

using namespace clang;
using namespace clang::dpct;

namespace {

inline bool SYCLGenError() { return true; }
inline bool SYCLGenSuccess() { return false; }

/// This is used to handle all the AST nodes (except specific instructions, Eg.
/// mov/setp), and generate functionally equivalent SYCL code.
class SYCLGenBase {
  bool IsInMacroDef = false;
  bool EmitNewLine = true;
  bool EmitSemi = true;
  unsigned NumIndent = 0;
  llvm::SmallString<4> IndentUnit{"  "};
  llvm::SmallString<16> Indent;
  llvm::SmallString<4> NewLine;
  llvm::raw_ostream *Stream;
  const GCCAsmStmt *GAS;

  class BlockDelimiterGuard {
    SYCLGenBase &CodeGen;

  public:
    BlockDelimiterGuard(SYCLGenBase &CG) : CodeGen(CG) {
      CodeGen.OS() << "{";
      CodeGen.endl();
      CodeGen.incIndent();
    }

    ~BlockDelimiterGuard() {
      CodeGen.decIndent();
      CodeGen.indent();
      CodeGen.OS() << "}";
      CodeGen.endl();
    }
  };

public:
  SYCLGenBase(llvm::raw_ostream &OS, const GCCAsmStmt *G)
      : Stream(&OS), GAS(G) {}

  virtual ~SYCLGenBase() = default;

  unsigned getNumIndent() const { return NumIndent; }

  void setIndentUnit(StringRef Unit) {
    if (!Unit.empty())
      IndentUnit = Unit;
  }

  void setInMacroDefine() { IsInMacroDef = true; }
  bool isInMacroDefine() const { return IsInMacroDef; }

  void decIndent(unsigned Num = 1) {
    if (NumIndent >= Num) {
      NumIndent -= Num;
      for (unsigned I = 0; I < Num; ++I)
        Indent.pop_back_n(IndentUnit.size());
    } else {
      NumIndent = 0;
      Indent.clear();
    }
  }

  void incIndent(unsigned Num = 1) {
    NumIndent += Num;
    for (unsigned I = 0; I < Num; ++I)
      Indent.append(IndentUnit);
  }

protected:
  void indent() { OS() << Indent; }

  void endl() {
    if (EmitNewLine) {
      if (NewLine.empty()) {
        if (isInMacroDefine())
          NewLine.append("\\");
        NewLine.append(getNL());
      }
      OS() << NewLine;
    }
  }

  void semi() {
    if (EmitSemi)
      OS() << ";";
  }

  void endstmt() {
    if (EmitSemi)
      OS() << ";";
    endl();
  }

  llvm::raw_ostream &OS() { return *Stream; }

  void switchOutStream(llvm::raw_ostream &NewOS) { Stream = &NewOS; }

  bool tryEmitStmt(llvm::raw_ostream &TmpOS, const InlineAsmStmt *S) {
    llvm::SaveAndRestore<llvm::raw_ostream *> OutStream(Stream);
    switchOutStream(TmpOS);
    if (emitStmt(S))
      return SYCLGenError();
    TmpOS.flush();
    return SYCLGenSuccess();
  }

  bool tryEmitStmt(std::string &Buffer, const InlineAsmStmt *S) {
    llvm::raw_string_ostream TmpOS(Buffer);
    return tryEmitStmt(TmpOS, S);
  }

  // Types
  bool emitType(const InlineAsmType *T);
  bool emitBuiltinType(const InlineAsmBuiltinType *T);
  bool emitVectorType(const InlineAsmVectorType *T);

  // Declarations
  bool emitDeclaration(const InlineAsmDecl *D);
  bool emitVariableDeclaration(const InlineAsmVariableDecl *D);

  // Statements && Expressions
  bool emitStmt(const InlineAsmStmt *S);
  bool emitCompoundStmt(const InlineAsmCompoundStmt *S);
  bool emitDeclStmt(const InlineAsmDeclStmt *S);
  bool emitInstruction(const InlineAsmInstruction *I);
  bool emitConditionalInstruction(const InlineAsmConditionalInstruction *I);
  bool emitUnaryOperator(const InlineAsmUnaryOperator *Op);
  bool emitBinaryOperator(const InlineAsmBinaryOperator *Op);
  bool emitConditionalOperator(const InlineAsmConditionalOperator *Op);
  bool emitVectorExpr(const InlineAsmVectorExpr *E) { return SYCLGenError(); }
  bool emitDiscardExpr(const InlineAsmDiscardExpr *E) { return SYCLGenError(); }
  bool emitAddressExpr(const InlineAsmAddressExpr *E) { return SYCLGenError(); }
  bool emitCastExpr(const InlineAsmCastExpr *E);
  bool emitParenExpr(const InlineAsmParenExpr *E);
  bool emitDeclRefExpr(const InlineAsmDeclRefExpr *E);
  bool emitIntegerLiteral(const InlineAsmIntegerLiteral *I);
  bool emitFloatingLiteral(const InlineAsmFloatingLiteral *Fp);

  // Instructions
#define INSTRUCTION(X)                                                         \
  virtual bool handle_##X(const InlineAsmInstruction *I) {                     \
    return SYCLGenError();                                                     \
  }
#include "Asm/AsmTokenKinds.def"
};

bool SYCLGenBase::emitStmt(const InlineAsmStmt *S) {
  switch (S->getStmtClass()) {
#define STMT(CLASS, PARENT)                                                    \
  case InlineAsmStmt::CLASS##Class:                                            \
    return emit##CLASS(dyn_cast<InlineAsm##CLASS>(S));
#define ABSTRACT_STMT(STMT)
#include "Asm/AsmNodes.def"
  }
  return SYCLGenError();
}

bool SYCLGenBase::emitDeclStmt(const InlineAsmDeclStmt *S) {
  if (S->getNumDecl() == 0)
    return SYCLGenError();
  if (S->getDeclSpec().Alignment) {
    OS() << "alignas(";
    OS() << S->getDeclSpec().Alignment->getValue().getZExtValue();
    OS() << ") ";
  }
  if (emitType(S->getDeclSpec().Type))
    return SYCLGenError();
  OS() << " ";
  int NumCommas = S->getNumDecl() - 1;
  for (const auto *D : S->decls()) {
    if (emitDeclaration(D))
      return SYCLGenError();
    if (NumCommas-- > 0)
      OS() << ", ";
  }
  endstmt();
  return SYCLGenSuccess();
}

bool SYCLGenBase::emitCompoundStmt(const InlineAsmCompoundStmt *S) {
  llvm::SaveAndRestore<bool> StoreEndl(EmitNewLine);
  llvm::SaveAndRestore<bool> StoreSemi(EmitSemi);
  EmitNewLine = true;
  EmitSemi = true;
  BlockDelimiterGuard Guard(*this);
  for (const auto *SubStmt : S->stmts()) {
    indent();
    if (emitStmt(SubStmt))
      return SYCLGenError();
  }
  return SYCLGenSuccess();
}

bool SYCLGenBase::emitInstruction(const InlineAsmInstruction *I) {
  switch (I->getOpcode()->getTokenID()) {
#define INSTRUCTION(X)                                                         \
  case asmtok::op_##X:                                                         \
    return handle_##X(I);
#include "Asm/AsmTokenKinds.def"
  default:
    break;
  }
  return SYCLGenError();
}

bool SYCLGenBase::emitConditionalInstruction(
    const InlineAsmConditionalInstruction *I) {
  OS() << "if (";
  if (I->hasNot())
    OS() << "!";
  if (emitStmt(I->getPred()))
    return SYCLGenError();
  OS() << ") ";
  BlockDelimiterGuard Guard(*this);
  indent();
  return emitInstruction(I->getInstruction());
}

bool SYCLGenBase::emitUnaryOperator(const InlineAsmUnaryOperator *Op) {
  switch (Op->getOpcode()) {
    // clang-format off
  case InlineAsmUnaryOperator::Plus:  OS() << "+"; break;
  case InlineAsmUnaryOperator::Minus: OS() << "-"; break;
  case InlineAsmUnaryOperator::Not:   OS() << "~"; break;
  case InlineAsmUnaryOperator::LNot:  OS() << "!"; break;
    // clang-format on
  }
  return emitStmt(Op->getSubExpr());
}

bool SYCLGenBase::emitBinaryOperator(const InlineAsmBinaryOperator *Op) {
  if (emitStmt(Op->getLHS()))
    return SYCLGenError();
  OS() << " ";
  // clang-format off
  switch (Op->getOpcode()) {
  case InlineAsmBinaryOperator::Mul:    OS() << "*";  break;
  case InlineAsmBinaryOperator::Div:    OS() << "/";  break;
  case InlineAsmBinaryOperator::Rem:    OS() << "%";  break;
  case InlineAsmBinaryOperator::Add:    OS() << "+";  break;
  case InlineAsmBinaryOperator::Sub:    OS() << "-";  break;
  case InlineAsmBinaryOperator::Shl:    OS() << "<<"; break;
  case InlineAsmBinaryOperator::Shr:    OS() << ">>"; break;
  case InlineAsmBinaryOperator::LT:     OS() << "<";  break;
  case InlineAsmBinaryOperator::GT:     OS() << ">";  break;
  case InlineAsmBinaryOperator::LE:     OS() << "<="; break;
  case InlineAsmBinaryOperator::GE:     OS() << ">="; break;
  case InlineAsmBinaryOperator::EQ:     OS() << "=="; break;
  case InlineAsmBinaryOperator::NE:     OS() << "!="; break;
  case InlineAsmBinaryOperator::And:    OS() << "&";  break;
  case InlineAsmBinaryOperator::Xor:    OS() << "^";  break;
  case InlineAsmBinaryOperator::Or:     OS() << "|";  break;
  case InlineAsmBinaryOperator::LAnd:   OS() << "&&"; break;
  case InlineAsmBinaryOperator::LOr:    OS() << "||"; break;
  case InlineAsmBinaryOperator::Assign: OS() << "=";  break;
  }
  // clang-format on
  OS() << " ";
  return emitStmt(Op->getRHS());
}

bool SYCLGenBase::emitConditionalOperator(
    const InlineAsmConditionalOperator *Op) {
  if (emitStmt(Op->getCond()))
    return SYCLGenError();
  OS() << " ? ";
  if (emitStmt(Op->getLHS()))
    return SYCLGenError();
  OS() << " : ";
  return emitStmt(Op->getRHS());
}

bool SYCLGenBase::emitCastExpr(const InlineAsmCastExpr *E) {
  OS() << "static_cast<";
  if (emitType(E->getType()))
    return SYCLGenError();
  OS() << ">(";
  if (emitStmt(E->getSubExpr()))
    return SYCLGenError();
  OS() << ")";
  return SYCLGenSuccess();
}

bool SYCLGenBase::emitParenExpr(const InlineAsmParenExpr *E) {
  OS() << "(";
  if (emitStmt(E->getSubExpr()))
    return SYCLGenError();
  OS() << ")";
  return SYCLGenSuccess();
}

bool SYCLGenBase::emitDeclRefExpr(const InlineAsmDeclRefExpr *E) {
  if (E->getDecl().getDeclName()->isBuiltinID()) {
    switch (E->getDecl().getDeclName()->getTokenID()) {
    case asmtok::bi_laneid:
      OS() << DpctGlobalInfo::getItem(GAS)
           << ".get_sub_group().get_local_linear_id()";
      break;
    case asmtok::bi_warpid:
      OS() << DpctGlobalInfo::getItem(GAS)
           << ".get_sub_group().get_group_linear_id()";
      break;
    case asmtok::bi_WARP_SZ:
      OS() << DpctGlobalInfo::getItem(GAS)
           << ".get_sub_group().get_local_range().get(0)";
      break;
    default:
      return SYCLGenError();
    }
    return SYCLGenSuccess();
  }
  OS() << E->getDecl().getDeclName()->getName();
  if (E->hasParameterizedName()) {
    OS() << '[' << E->getParameterizedNameIndex() << ']';
  }
  return SYCLGenSuccess();
}

bool SYCLGenBase::emitIntegerLiteral(const InlineAsmIntegerLiteral *I) {
  OS() << I->getLiteral();
  return SYCLGenSuccess();
}

bool SYCLGenBase::emitFloatingLiteral(const InlineAsmFloatingLiteral *Fp) {
  if (!Fp->isExactMachineFloatingLiteral()) {
    OS() << Fp->getLiteral();
  } else {
    const char *Template = "sycl::bit_cast<{0}>({1}(0x{2}{3}))";
    if (const auto *T = dyn_cast<InlineAsmBuiltinType>(Fp->getType())) {
      switch (T->getKind()) {
      case InlineAsmBuiltinType::TK_f32:
        OS() << llvm::formatv(Template, "float", "uint32_t", Fp->getLiteral(),
                              "U");
        break;
      case InlineAsmBuiltinType::TK_f64:
        OS() << llvm::formatv(Template, "double", "uint64_t", Fp->getLiteral(),
                              "ULL");
        break;
      default:
        return SYCLGenError();
      }
      return SYCLGenSuccess();
    }
  }
  return SYCLGenSuccess();
}

bool SYCLGenBase::emitType(const InlineAsmType *T) {
  switch (T->getTypeClass()) {
  case InlineAsmType::BuiltinClass:
    return emitBuiltinType(dyn_cast<InlineAsmBuiltinType>(T));
  case InlineAsmType::VectorClass:
    return emitVectorType(dyn_cast<InlineAsmVectorType>(T));
  default:
    break;
  }
  return SYCLGenError();
}

bool SYCLGenBase::emitBuiltinType(const InlineAsmBuiltinType *T) {
  switch (T->getKind()) {
    // clang-format off
  case InlineAsmBuiltinType::TK_b8:     OS() << "uint8_t"; break;
  case InlineAsmBuiltinType::TK_b16:    OS() << "uint16_t"; break;
  case InlineAsmBuiltinType::TK_b32:    OS() << "uint32_t"; break;
  case InlineAsmBuiltinType::TK_b64:    OS() << "uint64_t"; break;
  case InlineAsmBuiltinType::TK_u8:     OS() << "uint8_t"; break;
  case InlineAsmBuiltinType::TK_u16:    OS() << "uint16_t"; break;
  case InlineAsmBuiltinType::TK_u32:    OS() << "uint32_t"; break;
  case InlineAsmBuiltinType::TK_u64:    OS() << "uint64_t"; break;
  case InlineAsmBuiltinType::TK_s8:     OS() << "int8_t"; break;
  case InlineAsmBuiltinType::TK_s16:    OS() << "int16_t"; break;
  case InlineAsmBuiltinType::TK_s32:    OS() << "int32_t"; break;
  case InlineAsmBuiltinType::TK_s64:    OS() << "int64_t"; break;
  case InlineAsmBuiltinType::TK_f16:    OS() << "sycl::half"; break;
  case InlineAsmBuiltinType::TK_f32:    OS() << "float"; break;
  case InlineAsmBuiltinType::TK_f64:    OS() << "double"; break;
  case InlineAsmBuiltinType::TK_byte:   OS() << "uint8_t"; break;
  case InlineAsmBuiltinType::TK_4byte:  OS() << "uint32_t"; break;
  case InlineAsmBuiltinType::TK_pred:   OS() << "bool"; break;
  case InlineAsmBuiltinType::TK_bf16:
  case InlineAsmBuiltinType::TK_e4m3:
  case InlineAsmBuiltinType::TK_e5m2:
  case InlineAsmBuiltinType::TK_tf32:
  case InlineAsmBuiltinType::TK_f16x2:
  case InlineAsmBuiltinType::TK_bf16x2:
  case InlineAsmBuiltinType::TK_e4m3x2:
  case InlineAsmBuiltinType::TK_e5m2x2:
  case InlineAsmBuiltinType::TK_s16x2:
  case InlineAsmBuiltinType::TK_u16x2:
    [[fallthrough]];
  default:
    return SYCLGenError();
  }
  // clang-format on
  return SYCLGenSuccess();
}

bool SYCLGenBase::emitVectorType(const InlineAsmVectorType *T) {
  OS() << "sycl::vec<";
  if (emitType(T->getElementType()))
    return SYCLGenError();
  OS() << ", ";
  switch (T->getKind()) {
  case InlineAsmVectorType::TK_v2:
    OS() << 2;
    break;
  case InlineAsmVectorType::TK_v4:
    OS() << 4;
    break;
  case InlineAsmVectorType::TK_v8:
    OS() << 8;
    break;
  }
  OS() << ">";
  return SYCLGenSuccess();
}

bool SYCLGenBase::emitDeclaration(const InlineAsmDecl *D) {
  switch (D->getDeclClass()) {
  case InlineAsmDecl::VariableDeclClass:
    return emitVariableDeclaration(dyn_cast<InlineAsmVariableDecl>(D));
  default:
    break;
  }
  return SYCLGenError();
}

bool SYCLGenBase::emitVariableDeclaration(const InlineAsmVariableDecl *D) {
  OS() << D->getDeclName()->getName();
  if (D->isParameterizedNameDecl())
    OS() << '[' << D->getNumParameterizedNames() << ']';
  return SYCLGenSuccess();
}

/// This used to handle the specific instruction.
class SYCLGen : public SYCLGenBase {
public:
  SYCLGen(llvm::raw_ostream &OS, const GCCAsmStmt *G) : SYCLGenBase(OS, G) {}

  bool handleStatement(const InlineAsmStmt *S) { return emitStmt(S); }

protected:
  bool handle_mov(const InlineAsmInstruction *I) override {
    if (I->getNumInputOperands() != 1)
      return SYCLGenError();
    if (emitStmt(I->getOutputOperand()))
      return SYCLGenError();
    OS() << " = ";
    if (emitStmt(I->getInputOperand(0)))
      return SYCLGenError();
    endstmt();
    return SYCLGenSuccess();
  }

  bool handle_setp(const InlineAsmInstruction *I) override {
    if (I->getNumInputOperands() != 2 && I->getNumInputOperands() == 1)
      return SYCLGenError();

    auto *T = dyn_cast<InlineAsmBuiltinType>(I->getType(0));
    if (!T)
      return SYCLGenError();

    if (emitStmt(I->getOutputOperand()))
      return SYCLGenError();

    OS() << " = ";

    std::string Op1, Op2;

    if (tryEmitStmt(Op1, I->getInputOperand(0)))
      return SYCLGenError();

    if (tryEmitStmt(Op2, I->getInputOperand(1)))
      return SYCLGenError();

    const char *Template = nullptr;

    for (const auto A : I->attrs()) {
      switch (A) {
      case InstAttr::eq:
        if (T->isSignedInt() || T->isUnsignedInt() || T->isBitSize())
          Template = "{0} == {1}";
        else if (T->getKind() == InlineAsmBuiltinType::TK_f32)
          Template = "!sycl::isnan({0}) && !sycl::isnan({1}) && "
                     "sycl::isequal<float>({0}, {1})";
        else if (T->getKind() == InlineAsmBuiltinType::TK_f64)
          Template = "!sycl::isnan({0}) && !sycl::isnan({1}) && "
                     "sycl::isequal<double>({0}, {1})";
        else
          return SYCLGenError();
        break;
      case InstAttr::ne:
        if (T->isSignedInt() || T->isUnsignedInt() || T->isBitSize())
          Template = "{0} != {1}";
        else if (T->getKind() == InlineAsmBuiltinType::TK_f32)
          Template = "!sycl::isnan({0}) && !sycl::isnan({1}) && "
                     "sycl::isnotequal<float>({0}, {1})";
        else if (T->getKind() == InlineAsmBuiltinType::TK_f64)
          Template = "!sycl::isnan({0}) && !sycl::isnan({1}) && "
                     "sycl::isnotequal<double>({0}, {1})";
        else
          return SYCLGenError();
        break;
      case InstAttr::lt:
        if (T->isSignedInt())
          Template = "{0} < {1}";
        else if (T->getKind() == InlineAsmBuiltinType::TK_f32)
          Template = "!sycl::isnan({0}) && !sycl::isnan({1}) && "
                     "sycl::isless<float>({0}, {1})";
        else if (T->getKind() == InlineAsmBuiltinType::TK_f64)
          Template = "!sycl::isnan({0}) && !sycl::isnan({1}) && "
                     "sycl::isless<double>({0}, {1})";
        else
          return SYCLGenError();
        break;
      case InstAttr::le:
        if (T->isSignedInt())
          Template = "{0} <= {1}";
        else if (T->getKind() == InlineAsmBuiltinType::TK_f32)
          Template = "!sycl::isnan({0}) && !sycl::isnan({1}) && "
                     "sycl::islessequal<float>({0}, {1})";
        else if (T->getKind() == InlineAsmBuiltinType::TK_f64)
          Template = "!sycl::isnan({0}) && !sycl::isnan({1}) && "
                     "sycl::islessequal<double>({0}, {1})";
        else
          return SYCLGenError();
        break;
      case InstAttr::gt:
        if (T->isSignedInt())
          Template = "{0} > {1}";
        else if (T->getKind() == InlineAsmBuiltinType::TK_f32)
          Template = "!sycl::isnan({0}) && !sycl::isnan({1}) && "
                     "sycl::isgreater<float>({0}, {1})";
        else if (T->getKind() == InlineAsmBuiltinType::TK_f64)
          Template = "!sycl::isnan({0}) && !sycl::isnan({1}) && "
                     "sycl::isgreater<double>({0}, {1})";
        else
          return SYCLGenError();
        break;
      case InstAttr::ge:
        if (T->isSignedInt())
          Template = "{0} >= {1}";
        else if (T->getKind() == InlineAsmBuiltinType::TK_f32)
          Template = "!sycl::isnan({0}) && !sycl::isnan({1}) && "
                     "sycl::isgreaterequal<float>({0}, {1})";
        else if (T->getKind() == InlineAsmBuiltinType::TK_f64)
          Template = "!sycl::isnan({0}) && !sycl::isnan({1}) && "
                     "sycl::isgreaterequal<double>({0}, {1})";
        else
          return SYCLGenError();
        break;
      case InstAttr::lo:
        if (T->isUnsignedInt())
          Template = "{0} < {1}";
        else
          return SYCLGenError();
        break;
      case InstAttr::ls:
        if (T->isUnsignedInt())
          Template = "{0} <= {1}";
        else
          return SYCLGenError();
        break;
      case InstAttr::hi:
        if (T->isUnsignedInt())
          Template = "{0} > {1}";
        else
          return SYCLGenError();
        break;
      case InstAttr::hs:
        if (T->isUnsignedInt())
          Template = "{0} >= {1}";
        else
          return SYCLGenError();
        break;
      case InstAttr::equ:
        if (T->getKind() == InlineAsmBuiltinType::TK_f32)
          Template = "sycl::isnan({0}) || sycl::isnan({1}) || "
                     "sycl::isequal<float>({0}, {1})";
        else if (T->getKind() == InlineAsmBuiltinType::TK_f64)
          Template = "sycl::isnan({0}) || sycl::isnan({1}) || "
                     "sycl::isequal<double>({0}, {1})";
        else
          return SYCLGenError();
        break;
      case InstAttr::neu:
        if (T->getKind() == InlineAsmBuiltinType::TK_f32)
          Template = "sycl::isnan({0}) || sycl::isnan({1}) || "
                     "sycl::isnotequal<float>({0}, {1})";
        else if (T->getKind() == InlineAsmBuiltinType::TK_f64)
          Template = "sycl::isnan({0}) || sycl::isnan({1}) || "
                     "sycl::isnotequal<double>({0}, {1})";
        else
          return SYCLGenError();
        break;
      case InstAttr::ltu:
        if (T->getKind() == InlineAsmBuiltinType::TK_f32)
          Template = "sycl::isnan({0}) || sycl::isnan({1}) || "
                     "sycl::isless<float>({0}, {1})";
        else if (T->getKind() == InlineAsmBuiltinType::TK_f64)
          Template = "sycl::isnan({0}) || sycl::isnan({1}) || "
                     "sycl::isless<double>({0}, {1})";
        else
          return SYCLGenError();
        break;
      case InstAttr::leu:
        if (T->getKind() == InlineAsmBuiltinType::TK_f32)
          Template = "sycl::isnan({0}) || sycl::isnan({1}) || "
                     "sycl::islessequal<float>({0}, {1})";
        else if (T->getKind() == InlineAsmBuiltinType::TK_f64)
          Template = "sycl::isnan({0}) || sycl::isnan({1}) || "
                     "sycl::islessequal<double>({0}, {1})";
        else
          return SYCLGenError();
        break;
      case InstAttr::gtu:
        if (T->getKind() == InlineAsmBuiltinType::TK_f32)
          Template = "sycl::isnan({0}) || sycl::isnan({1}) || "
                     "sycl::isgreater<float>({0}, {1})";
        else if (T->getKind() == InlineAsmBuiltinType::TK_f64)
          Template = "sycl::isnan({0}) || sycl::isnan({1}) || "
                     "sycl::isgreater<double>({0}, {1})";
        else
          return SYCLGenError();
        break;
      case InstAttr::geu:
        if (T->getKind() == InlineAsmBuiltinType::TK_f32)
          Template = "sycl::isnan({0}) || sycl::isnan({1}) || "
                     "sycl::isgreaterequal<float>({0}, {1})";
        else if (T->getKind() == InlineAsmBuiltinType::TK_f64)
          Template = "sycl::isnan({0}) || sycl::isnan({1}) || "
                     "sycl::isgreaterequal<double>({0}, {1})";
        else
          return SYCLGenError();
        break;
      case InstAttr::num:
        if (T->isFloating())
          Template = "!sycl::isnan({0}) && !sycl::isnan({1})";
        else
          return SYCLGenError();
        break;
      case InstAttr::nan:
        if (T->isFloating())
          Template = "sycl::isnan({0}) || sycl::isnan({1})";
        else
          return SYCLGenError();
        break;
      default:
        break;
      }
    }

    if (!Template)
      return SYCLGenError();

    OS() << llvm::formatv(Template, Op1, Op2);
    endstmt();
    return SYCLGenSuccess();
  }

  bool handle_lop3(const InlineAsmInstruction *I) override {
    if (I->getNumInputOperands() != 4 || I->getNumTypes() != 1 ||
        !isa<InlineAsmBuiltinType>(I->getType(0)) ||
        dyn_cast<InlineAsmBuiltinType>(I->getType(0))->getKind() !=
            InlineAsmBuiltinType::TK_b32)
      return SYCLGenError();
    if (emitStmt(I->getOutputOperand()))
      return SYCLGenError();
    OS() << " = ";

    std::string Op[3];
    for (auto Idx : llvm::seq(0, 3)) {
      if (tryEmitStmt(Op[Idx], I->getInputOperand(Idx)))
        return SYCLGenError();
    }

    if (!isa<InlineAsmIntegerLiteral>(I->getInputOperand(3)))
      return SYCLGenError();
    unsigned Imm = dyn_cast<InlineAsmIntegerLiteral>(I->getInputOperand(3))
                       ->getValue()
                       .getZExtValue();

#define EMPTY nullptr
#define EMPTY4 EMPTY, EMPTY, EMPTY, EMPTY
#define EMPTY16 EMPTY4, EMPTY4, EMPTY4, EMPTY4
    constexpr const char *FastMap[256] = {
        /*0x00*/ "0",
        // clang-format off
      EMPTY16, EMPTY4, EMPTY4, EMPTY,
      /*0x1a*/ "({0} & {1} | {2}) ^ {0}",
      EMPTY, EMPTY, EMPTY,
      /*0x1e*/ "{0} ^ ({1} | {2})",
      EMPTY4, EMPTY4, EMPTY4, EMPTY, EMPTY,
      /*0x2d*/ "~{0} ^ (~{1} & {2})",
      EMPTY16, EMPTY, EMPTY,
      /*0x40*/ "{0} & {1} & ~{2}",
      EMPTY16, EMPTY16, EMPTY16, EMPTY4, EMPTY, EMPTY, EMPTY,
      /*0x78*/ "{0} ^ ({1} & {2})",
      EMPTY4, EMPTY, EMPTY, EMPTY,
      /*0x80*/ "{0} & {1} & {2}",
      EMPTY16, EMPTY4, EMPTY,
      /*0x96*/ "{0} ^ {1} ^ {2}",
      EMPTY16, EMPTY4, EMPTY4, EMPTY4, EMPTY,
      /*0xb4*/ "{0} ^ ({1} & ~{2})",
      EMPTY, EMPTY, EMPTY,
      /*0xb8*/ "({0} ^ ({1} & ({2} ^ {0})))",
      EMPTY16, EMPTY4, EMPTY4, EMPTY,
      /*0xd2*/ "{0} ^ (~{1} & {2})",
      EMPTY16, EMPTY4, EMPTY,
      /*0xe8*/ "(({0} & ({1} | {2})) | ({1} & {2}))",
      EMPTY,
      /*0xea*/ "({0} & {1}) | {2}",
      EMPTY16, EMPTY, EMPTY, EMPTY,
      /*0xfe*/ "{0} | {1} | {2}",
      /*0xff*/ "uint32_t(-1)"};
    // clang-format on

#undef EMPTY16
#undef EMPTY4
#undef EMPTY
    // clang-format off
    constexpr const char *SlowMap[8] = {
      /* 0x01*/ "(~{0} & ~{1} & ~{2})",
      /* 0x02*/ "(~{0} & ~{1} & {2})",
      /* 0x04*/ "(~{0} & {1} & ~{2})",
      /* 0x08*/ "(~{0} & {1} & {2})",
      /* 0x10*/ "({0} & ~{1} & ~{2})",
      /* 0x20*/ "({0} & ~{1} & {2})",
      /* 0x40*/ "({0} & {1} & ~{2})",
      /* 0x80*/ "({0} & {1} & {2})"
    };
    // clang-format on

    if (FastMap[Imm]) {
      OS() << llvm::formatv(FastMap[Imm], Op[0], Op[1], Op[2]);
    } else {
      SmallVector<std::string, 8> Templates;
      for (auto Bit : llvm::seq(0, 8)) {
        if (Imm & (1U << Bit)) {
          Templates.push_back(
              llvm::formatv(SlowMap[Bit], Op[0], Op[1], Op[2]).str());
        }
      }

      OS() << llvm::join(Templates, " | ");
    }

    endstmt();
    return SYCLGenSuccess();
  }
};
} // namespace

void AsmRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  using namespace clang::ast_matchers;
  MF.addMatcher(
      asmStmt(hasAncestor(functionDecl(
                  anyOf(hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal)))))
          .bind("asm"),
      this);
}

void AsmRule::doMigrateInternel(const GCCAsmStmt *GAS) {
  const auto &C = DpctGlobalInfo::getContext();
  auto &SM = DpctGlobalInfo::getSourceManager();
  std::string S = GAS->generateAsmString(C);
  InlineAsmContext Context;
  llvm::SourceMgr Mgr;
  std::string Buffer = GAS->getAsmString()->getString().str();
  Mgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(Buffer),
                         llvm::SMLoc());
  InlineAsmParser Parser(Context, Mgr);
  std::string ReplaceString;
  llvm::raw_string_ostream OS(ReplaceString);
  SYCLGen CodeGen(OS, GAS);
  StringRef Indent =
      getIndent(GAS->getBeginLoc(), DpctGlobalInfo::getSourceManager());

  CodeGen.setIndentUnit(Indent);
  CodeGen.incIndent();
  if (isInMacroDefinition(GAS->getBeginLoc(), GAS->getEndLoc()))
    CodeGen.setInMacroDefine();

  auto getReplaceString = [&](const Expr *E) {
    ArgumentAnalysis AA(CodeGen.isInMacroDefine());
    AA.setCallSpelling(SM.getSpellingLoc(GAS->getBeginLoc()),
                       SM.getSpellingLoc(GAS->getEndLoc()));
    AA.analyze(E);
    if (needExtraParens(E))
      return "(" + AA.getRewriteString() + ")";
    return AA.getRewriteString();
  };
  Parser.addBuiltinIdentifier();
  for (unsigned I = 0, E = GAS->getNumOutputs(); I != E; ++I)
    Parser.addInlineAsmOperands(getReplaceString(GAS->getOutputExpr(I)),
                                GAS->getOutputConstraint(I));

  for (unsigned I = 0, E = GAS->getNumInputs(); I != E; ++I)
    Parser.addInlineAsmOperands(getReplaceString(GAS->getInputExpr(I)),
                                GAS->getInputConstraint(I));

  do {
    auto Inst = Parser.ParseStatement();
    if (Inst.isInvalid()) {
      report(GAS->getAsmLoc(), Diagnostics::DEVICE_ASM, true);
      return;
    }

    if (CodeGen.handleStatement(Inst.get())) {
      report(GAS->getAsmLoc(), Diagnostics::DEVICE_ASM, true);
      return;
    }
  } while (!Parser.getCurToken().is(asmtok::eof));

  StringRef Ref = ReplaceString;
  if (CodeGen.isInMacroDefine()) {
    if (Ref.ends_with("\\\n")) {
      ReplaceString.erase(ReplaceString.end() - 2);
    } else if (Ref.ends_with("\\\r\n")) {
      ReplaceString.erase(ReplaceString.end() - 3);
    }
  }

  auto *Repl = new ReplaceStmt(GAS, std::move(ReplaceString));
  Repl->setBlockLevelFormatFlag();
  emplaceTransformation(Repl);

  auto Tok = Lexer::findNextToken(GAS->getEndLoc(), SM, C.getLangOpts());

  if (Tok.has_value() && Tok->is(tok::semi) &&
      (!CodeGen.isInMacroDefine() ||
       isInMacroDefinition(Tok->getLocation(), Tok->getEndLoc()))) {
    emplaceTransformation(new ReplaceToken(Tok->getLocation(), ""));
  }
  return;
}

void AsmRule::runRule(const ast_matchers::MatchFinder::MatchResult &Result) {
  if (auto *AS = getNodeAsType<AsmStmt>(Result, "asm")) {
    if (const auto *GAS = dyn_cast<GCCAsmStmt>(AS)) {
      if (!runWithCrashGuard([&]() { doMigrateInternel(GAS); }, ""))
        report(AS->getAsmLoc(), Diagnostics::DEVICE_ASM, true);

      return;
    }
    report(AS->getAsmLoc(), Diagnostics::DEVICE_ASM, true);
  }
  return;
}
