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
#include "MigrationRuleManager.h"
#include "TextModification.h"
#include "Utility.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/TokenKinds.h"
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

using namespace clang;
using namespace clang::dpct;

namespace {

/// This is used to handle all the AST nodes (except specific instructions).
class SYCLGenBase {
  llvm::raw_ostream *Stream;
  bool EmitNewLine = true;
  bool EmitSemi = true;
  bool IndentTopLevelCompoundStmt = true;
  unsigned NumIndent = 0;
  llvm::SmallString<4> IndentUnit{"  "};
  llvm::SmallString<16> Indent;

public:
  SYCLGenBase(llvm::raw_ostream &OS) : Stream(&OS) {}
  virtual ~SYCLGenBase() = default;

  unsigned getNumIndent() const { return NumIndent; }

  void setNumIndent(unsigned Indent) { NumIndent = Indent; }

  void setIndentUnit(StringRef Unit) {
    if (!Unit.empty())
      IndentUnit = Unit;
  }

protected:
  void decIndent(unsigned Num = 1) {
    if (NumIndent >= Num)
      NumIndent -= Num;
    else
      NumIndent = 0;
  }

  void incIndent(unsigned Num = 1) { NumIndent += Num; }

  StringRef indent() {
    size_t Len = IndentUnit.size() * NumIndent;
    if (Len == Indent.size())
      return Indent;

    Indent.reserve(Len);
    Indent.clear();
    for (unsigned I = 0; I < NumIndent; ++I)
      Indent += IndentUnit;

    return Indent;
  }

  StringRef endl() const {
    if (EmitNewLine)
      return getNL();
    return "";
  }

  StringRef semi() const {
    if (EmitSemi)
      return ";";
    return "";
  }

  llvm::raw_ostream &OS() { return *Stream; }

  void switchOutStream(llvm::raw_ostream &NewOS) { Stream = &NewOS; }

  bool tryEmitStatement(llvm::raw_ostream &TmpOS, const InlineAsmStmt *S) {
    llvm::SaveAndRestore<llvm::raw_ostream *> OutStream(Stream);
    switchOutStream(TmpOS);
    if (emitStatement(S))
      return true;
    TmpOS.flush();
    return false;
  }

  bool tryEmitStatement(std::string &Buffer, const InlineAsmStmt *S) {
    llvm::raw_string_ostream TmpOS(Buffer);
    return tryEmitStatement(TmpOS, S);
  }

  // Types
  bool emitType(const InlineAsmType *T);
  bool emitBuiltinType(const InlineAsmBuiltinType *T);
  bool emitVectorType(const InlineAsmVectorType *T);

  // Declarations
  bool emitDeclaration(const InlineAsmDecl *D);
  bool emitVariableDeclaration(const InlineAsmVariableDecl *D);

  // Statements && Expressions
  bool emitStatement(const InlineAsmStmt *S);
  bool emitCompoundStatement(const InlineAsmCompoundStmt *S);
  bool emitDeclarationStatement(const InlineAsmDeclStmt *S);
  bool emitInstruction(const InlineAsmInstruction *I);
  bool emitConditionalInstruction(const InlineAsmConditionalInstruction *I);
  bool emitUnaryOperator(const InlineAsmUnaryOperator *Op);
  bool emitBinaryOperator(const InlineAsmBinaryOperator *Op);
  bool emitConditionalOperator(const InlineAsmConditionalOperator *Op);
  bool emitCastExpression(const InlineAsmCastExpr *E);
  bool emitParenExpression(const InlineAsmParenExpr *E);
  bool emitDeclRefExpression(const InlineAsmDeclRefExpr *E);
  bool emitIntegerLiteral(const InlineAsmIntegerLiteral *I);
  bool emitFloatingLiteral(const InlineAsmFloatingLiteral *Fp);

  // Instructions
#define INSTRUCTION(X)                                                         \
  bool handle_##X(const InlineAsmInstruction *I) { return true; }
#include "Asm/AsmTokenKinds.def"
};

bool SYCLGenBase::emitStatement(const InlineAsmStmt *S) {
  switch (S->getStmtClass()) {
  case InlineAsmStmt::CompoundStmtClass:
    return emitCompoundStatement(dyn_cast<InlineAsmCompoundStmt>(S));
  case InlineAsmStmt::DeclStmtClass:
    return emitDeclarationStatement(dyn_cast<InlineAsmDeclStmt>(S));
  case InlineAsmStmt::InstructionClass:
    return emitInstruction(dyn_cast<InlineAsmInstruction>(S));
  case InlineAsmStmt::ConditionalInstructionClass:
    return emitConditionalInstruction(dyn_cast<InlineAsmConditionalInstruction>(S));
  case InlineAsmStmt::UnaryOperatorClass:
    return emitUnaryOperator(dyn_cast<InlineAsmUnaryOperator>(S));
  case InlineAsmStmt::BinaryOperatorClass:
    return emitBinaryOperator(dyn_cast<InlineAsmBinaryOperator>(S));
  case InlineAsmStmt::ConditionalOperatorClass:
    return emitConditionalOperator(dyn_cast<InlineAsmConditionalOperator>(S));
  case InlineAsmStmt::CastExprClass:
    return emitCastExpression(dyn_cast<InlineAsmCastExpr>(S));
  case InlineAsmStmt::ParenExprClass:
    return emitParenExpression(dyn_cast<InlineAsmParenExpr>(S));
  case InlineAsmStmt::DeclRefExprClass:
    return emitDeclRefExpression(dyn_cast<InlineAsmDeclRefExpr>(S));
  case InlineAsmStmt::IntegerLiteralClass:
    return emitIntegerLiteral(dyn_cast<InlineAsmIntegerLiteral>(S));
  case InlineAsmStmt::FloatingLiteralClass:
    return emitFloatingLiteral(dyn_cast<InlineAsmFloatingLiteral>(S));
  default:
    return true;
  }
  return false;
}

bool SYCLGenBase::emitDeclarationStatement(const InlineAsmDeclStmt *S) {
  if (S->getNumDecl() == 0)
    return true;
  if (emitType(S->getBaseType()))
    return true;
  OS() << " ";
  int NumCommas = S->getNumDecl() - 1;
  for (const auto *D : S->decls()) {
    if (emitDeclaration(D))
      return true;
    if (NumCommas-- > 0)
      OS() << ", ";
  }
  OS() << semi() << endl();
  return false;
}

bool SYCLGenBase::emitCompoundStatement(const InlineAsmCompoundStmt *S) {
  llvm::SaveAndRestore<bool> StoreEndl(EmitNewLine);
  llvm::SaveAndRestore<bool> StoreSemi(EmitSemi);
  EmitNewLine = true;
  EmitSemi = true;
  OS() << "{" << endl();
  incIndent();
  for (const auto *SubStmt : S->stmts()) {
    OS() << indent();
    if (emitStatement(SubStmt))
      return true;
  }
  decIndent();
  if (IndentTopLevelCompoundStmt) {
    OS() << indent();
    IndentTopLevelCompoundStmt = false;
  }
  OS() << "}" << endl();
  return false;
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
  return true;
}

bool SYCLGenBase::emitConditionalInstruction(
    const InlineAsmConditionalInstruction *I) {
  OS() << "if (";
  if (I->hasNot())
    OS() << "!";
  if (emitStatement(I->getPred()))
    return true;
  OS() << ") {" << endl();
  incIndent();
  OS() << indent();
  if (emitInstruction(I->getInstruction()))
    return true;
  decIndent();
  OS() << indent() << "}";
  OS() << endl();
  return false;
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
  if (emitStatement(Op->getSubExpr()))
    return true;
  return false;
}

bool SYCLGenBase::emitBinaryOperator(const InlineAsmBinaryOperator *Op) {
  if (emitStatement(Op->getLHS()))
    return true;
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
    // clang-format on
  }
  OS() << " ";
  if (emitStatement(Op->getRHS()))
    return true;
  return false;
}

bool SYCLGenBase::emitConditionalOperator(
    const InlineAsmConditionalOperator *Op) {
  if (emitStatement(Op->getCond()))
    return true;
  OS() << " ? ";
  if (emitStatement(Op->getLHS()))
    return true;
  OS() << " : ";
  if (emitStatement(Op->getRHS()))
    return true;
  return false;
}

bool SYCLGenBase::emitCastExpression(const InlineAsmCastExpr *E) {
  OS() << "static_cast<";
  if (emitType(E->getType()))
    return true;
  OS() << ">(";
  if (emitStatement(E->getSubExpr()))
    return true;
  OS() << ")";
  return false;
}

bool SYCLGenBase::emitParenExpression(const InlineAsmParenExpr *E) {
  OS() << "(";
  if (emitStatement(E->getSubExpr()))
    return true;
  OS() << ")";
  return false;
}

bool SYCLGenBase::emitDeclRefExpression(const InlineAsmDeclRefExpr *E) {
  OS() << E->getDecl().getDeclName()->getName();
  if (E->hasParameterizedName()) {
    OS() << '[' << E->getParameterizedNameIndex() << ']';
  }
  return false;
}

bool SYCLGenBase::emitIntegerLiteral(const InlineAsmIntegerLiteral *I) {
  OS() << I->getLiteral();
  return false;
}

bool SYCLGenBase::emitFloatingLiteral(const InlineAsmFloatingLiteral *Fp) {
  if (!Fp->isExactMachineFloatingLiteral()) {
    OS() << Fp->getLiteral();
  } else {
    // [](){union {unsigned I; float F;}; I = 0x3f800000u; return F;}()
    constexpr char *Template =
        "[](){{union {{{0} I; {1} F;}; I = 0x{2}u; return F;}()";
    if (const auto *T = dyn_cast<InlineAsmBuiltinType>(Fp->getType())) {
      switch (T->getKind()) {
      case InlineAsmBuiltinType::TK_f32:
        OS() << llvm::formatv(Template, "uint32_t", "float", Fp->getLiteral());
        break;
      case InlineAsmBuiltinType::TK_f64:
        OS() << llvm::formatv(Template, "uint64_t", "double", Fp->getLiteral());
        break;
      default:
        return true;
      }
      return false;
    }
  }
  return false;
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
  return true;
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
    // clang-format on
    return true;
  }
  return false;
}

bool SYCLGenBase::emitVectorType(const InlineAsmVectorType *T) {
  OS() << "sycl::vec<";
  if (emitType(T->getElementType()))
    return true;
  OS() << ">";
  return false;
}

bool SYCLGenBase::emitDeclaration(const InlineAsmDecl *D) {
  switch (D->getDeclClass()) {
  case InlineAsmDecl::VariableDeclClass:
    return emitVariableDeclaration(dyn_cast<InlineAsmVariableDecl>(D));
  default:
    break;
  }
  return true;
}

bool SYCLGenBase::emitVariableDeclaration(const InlineAsmVariableDecl *D) {
  OS() << D->getDeclName()->getName();
  if (D->isParameterizedNameDecl())
    OS() << '[' << D->getNumParameterizedNames() << ']';
  return false;
}

/// This used to handle the specific instruction.
class SYCLGen : public SYCLGenBase {
public:
  SYCLGen(llvm::raw_ostream &OS) : SYCLGenBase(OS) {}

  bool handleStatement(const InlineAsmStmt *S) { return emitStatement(S); }

protected:
  bool handle_mov(const InlineAsmInstruction *I) {
    if (I->getNumInputOperands() != 1)
      return true;
    if (emitStatement(I->getOutputOperand()))
      return true;
    OS() << " = ";
    if (emitStatement(I->getInputOperand(0)))
      return true;
    OS() << semi() << endl();
    return false;
  }

  bool handle_setp(const InlineAsmInstruction *I) {
    if (I->getNumInputOperands() != 2 && I->getNumTypes() == 1)
      return true;

    auto *T = dyn_cast<InlineAsmBuiltinType>(I->getType(0));
    if (!T)
      return true;

    if (emitStatement(I->getOutputOperand()))
      return true;

    OS() << " = ";

    std::string Op1, Op2;

    if (tryEmitStatement(Op1, I->getInputOperand(0)))
      return true;

    if (tryEmitStatement(Op2, I->getInputOperand(1)))
      return true;

    const char *Template = nullptr;

    for (const auto *A : I->attrs()) {
      switch (A->getTokenID()) {
      case asmtok::kw_eq:
        if (T->isSignedInt() || T->isUnsignedInt() || T->isBitSize())
          Template = "{0} == {1}";
        else if (T->isFloating())
          Template = "!sycl::isnan({0}) && !sycl::isnan({1}) && {0} == {1}";
        else
          return true;
        break;
      case asmtok::kw_ne:
        if (T->isSignedInt() || T->isUnsignedInt() || T->isBitSize())
          Template = "{0} != {1}";
        else if (T->isFloating())
          Template = "!sycl::isnan({0}) && !sycl::isnan({1}) && {0} != {1}";
        else
          return true;
        break;
      case asmtok::kw_lt:
        if (T->isSignedInt())
          Template = "{0} < {1}";
        else if (T->isFloating())
          Template = "!sycl::isnan({0}) && !sycl::isnan({1}) && {0} < {1}";
        else
          return true;
        break;
      case asmtok::kw_le:
        if (T->isSignedInt())
          Template = "{0} <= {1}";
        else if (T->isFloating())
          Template = "!sycl::isnan({0}) && !sycl::isnan({1}) && {0} <= {1}";
        else
          return true;
        break;
      case asmtok::kw_gt:
        if (T->isSignedInt())
          Template = "{0} > {1}";
        else if (T->isFloating())
          Template = "!sycl::isnan({0}) && !sycl::isnan({1}) && {0} > {1}";
        else
          return true;
        break;
      case asmtok::kw_ge:
        if (T->isSignedInt())
          Template = "{0} >= {1}";
        else if (T->isFloating())
          Template = "!sycl::isnan({0}) && !sycl::isnan({1}) && {0} >= {1}";
        else
          return true;
        break;
      case asmtok::kw_lo:
        if (T->isUnsignedInt())
          Template = "{0} < {1}";
        else
          return true;
        break;
      case asmtok::kw_ls:
        if (T->isUnsignedInt())
          Template = "{0} <= {1}";
        else
          return true;
        break;
      case asmtok::kw_hi:
        if (T->isUnsignedInt())
          Template = "{0} > {1}";
        else
          return true;
        break;
      case asmtok::kw_hs:
        if (T->isUnsignedInt())
          Template = "{0} >= {1}";
        else
          return true;
        break;
      case asmtok::kw_equ:
        if (T->isFloating())
          Template = "sycl::isnan({0}) || sycl::isnan({1}) || {0} == {1}";
        else
          return true;
        break;
      case asmtok::kw_neu:
        if (T->isFloating())
          Template = " sycl::isnan({0}) || sycl::isnan({1}) || {0} != {1}";
        else
          return true;
        break;
      case asmtok::kw_ltu:
        if (T->isFloating())
          Template = "sycl::isnan({0}) || sycl::isnan({1}) || {0} < {1}";
        else
          return true;
        break;
      case asmtok::kw_leu:
        if (T->isFloating())
          Template = "sycl::isnan({0}) || sycl::isnan({1}) || {0} <= {1}";
        else
          return true;
        break;
      case asmtok::kw_gtu:
        if (T->isFloating())
          Template = "sycl::isnan({0}) || sycl::isnan({1}) || {0} > {1}";
        else
          return true;
        break;
      case asmtok::kw_geu:
        if (T->isFloating())
          Template = "sycl::isnan({0}) || sycl::isnan({1}) || {0} >= {1}";
        else
          return true;
        break;
      case asmtok::kw_num:
        if (T->isFloating())
          Template = "!sycl::isnan({0}) && !sycl::isnan({1})";
        else
          return true;
        break;
      case asmtok::kw_nan:
        if (T->isFloating())
          Template = "sycl::isnan({0}) || sycl::isnan({1})";
        else
          return true;
        break;
      default:
        break;
      }
    }

    if (!Template)
      return true;

    OS() << llvm::formatv(Template, Op1, Op2);
    OS() << semi() << endl();
    return false;
  }

  bool handle_lop3(const InlineAsmInstruction *I) {
    if (I->getNumInputOperands() != 4 || I->getNumTypes() != 1 ||
        !isa<InlineAsmBuiltinType>(I->getType(0)) ||
        dyn_cast<InlineAsmBuiltinType>(I->getType(0))->getKind() !=
            InlineAsmBuiltinType::TK_b32)
      return true;

    if (emitStatement(I->getOutputOperand()))
      return true;

    OS() << " = ";

    std::string Op[3];
    for (auto Idx : llvm::seq(0, 3)) {
      if (tryEmitStatement(Op[Idx], I->getInputOperand(Idx)))
        return true;
    }

    if (!isa<InlineAsmIntegerLiteral>(I->getInputOperand(3)))
      return true;

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
        // clang-format on
        /*0xfe*/ "{0} | {1} | {2}",
        /*0xff*/ "1"};
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

    OS() << semi() << endl();
    return false;
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

void AsmRule::runRule(const ast_matchers::MatchFinder::MatchResult &Result) {
  if (auto *AS = getNodeAsType<AsmStmt>(Result, "asm")) {
    if (const auto *Asm = dyn_cast<GCCAsmStmt>(AS)) {
      const auto &C = DpctGlobalInfo::getContext();
      std::string S = Asm->generateAsmString(C);
      InlineAsmContext Context;
      std::string Replacement;
      llvm::raw_string_ostream OS(Replacement);
      llvm::SourceMgr Mgr;
      std::string Buffer = Asm->getAsmString()->getString().str();
      Mgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(Buffer),
                             llvm::SMLoc());
      InlineAsmParser Parser(Context, Mgr);
      SYCLGen CodeGen(OS);
      std::string AsmString;

      auto getReplaceString = [&](const Expr *E) {
        ExprAnalysis EA;
        EA.analyze(E);
        if (needExtraParens(E))
          return "(" + EA.getReplacedString() + ")";
        return EA.getReplacedString();
      };

      for (unsigned I = 0, E = AS->getNumOutputs(); I != E; ++I)
        Parser.addInlineAsmOperands(getReplaceString(AS->getOutputExpr(I)),
                                    AS->getOutputConstraint(I));

      for (unsigned I = 0, E = AS->getNumInputs(); I != E; ++I)
        Parser.addInlineAsmOperands(getReplaceString(AS->getInputExpr(I)),
                                    AS->getInputConstraint(I));

      CodeGen.setNumIndent(1);
      CodeGen.setIndentUnit(
          getIndent(AS->getBeginLoc(), DpctGlobalInfo::getSourceManager()));

      do {
        auto Inst = Parser.ParseStatement();
        if (Inst.isInvalid()) {
          report(AS->getAsmLoc(), Diagnostics::DEVICE_ASM, true);
          return;
        }

        if (CodeGen.handleStatement(Inst.get())) {
          report(AS->getAsmLoc(), Diagnostics::DEVICE_ASM, true);
          return;
        }

      } while (!Parser.getCurToken().is(asmtok::eof));

      OS.flush();
      auto *Repl = new ReplaceStmt(AS, Replacement);
      Repl->setBlockLevelFormatFlag();
      emplaceTransformation(Repl);

      auto Tok = Lexer::findNextToken(
          Asm->getEndLoc(), DpctGlobalInfo::getSourceManager(),
          DpctGlobalInfo::getContext().getLangOpts());
      if (Tok.has_value() && Tok->is(tok::semi))
        emplaceTransformation(new ReplaceToken(Tok->getLocation(), ""));
      return;
    }
    report(AS->getAsmLoc(), Diagnostics::DEVICE_ASM, true);
  }
  return;
}
