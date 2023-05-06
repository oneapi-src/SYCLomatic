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
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <limits>
#include <sstream>

using namespace clang;
using namespace clang::dpct;

namespace {

class SYCLGenBase {
  llvm::StringMap<std::string> NameAliasTable;
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

  unsigned getNumIndent() const {
    return NumIndent;
  }

  void setNumIndent(unsigned Indent) {
    NumIndent = Indent;
  }

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

  void incIndent(unsigned Num = 1) {
    NumIndent += Num;
  }

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

  bool tryEmitStatement(llvm::raw_ostream &TmpOS, const DpctAsmStmt *S) {
    llvm::SaveAndRestore<llvm::raw_ostream *> OutStream(Stream);
    switchOutStream(TmpOS);
    if (emitStatement(S))
      return true;
    TmpOS.flush();
    return false;
  }

  bool tryEmitStatement(std::string &Buffer, const DpctAsmStmt *S) {
    llvm::raw_string_ostream TmpOS(Buffer);
    return tryEmitStatement(TmpOS, S);
  } 

  // Types
  bool emitType(const DpctAsmType *T);
  bool emitBuiltinType(const DpctAsmBuiltinType *T);
  bool emitVectorType(const DpctAsmVectorType *T);

  // Declarations
  bool emitDeclaration(const DpctAsmDecl *D);
  bool emitVariableDeclaration(const DpctAsmVariableDecl *D);

  // Statements && Expressions
  bool emitStatement(const DpctAsmStmt *S);
  bool emitCompoundStatement(const DpctAsmCompoundStmt *S);
  bool emitDeclarationStatement(const DpctAsmDeclStmt *S);
  bool emitInstruction(const DpctAsmInstruction *I);
  bool emitGuardInstruction(const DpctAsmGuardInstruction *I);
  bool emitUnaryOperator(const DpctAsmUnaryOperator *Op);
  bool emitBinaryOperator(const DpctAsmBinaryOperator *Op);
  bool emitConditionalOperator(const DpctAsmConditionalOperator *Op);
  bool emitCastExpression(const DpctAsmCastExpr *E);
  bool emitParenExpression(const DpctAsmParenExpr *E);
  bool emitDeclRefExpression(const DpctAsmDeclRefExpr *E);
  bool emitIntegerLiteral(const DpctAsmIntegerLiteral *I);
  bool emitFloatingLiteral(const DpctAsmFloatingLiteral *Fp);
  bool emitExactMachineFloatingLiteral(const DpctAsmExactMachineFloatingLiteral *Fp);

  // Instructions
#define INSTRUCTION(X) virtual bool handle_ ## X(const DpctAsmInstruction *I) { return true; }
#include "Asm/AsmTokenKinds.def"
};

bool SYCLGenBase::emitStatement(const DpctAsmStmt *S) {
   switch (S->getStmtClass()) {
    case DpctAsmStmt::CompoundStmtClass:
      return emitCompoundStatement(dyn_cast<DpctAsmCompoundStmt>(S));
    case DpctAsmStmt::DeclStmtClass:
      return emitDeclarationStatement(dyn_cast<DpctAsmDeclStmt>(S));
    case DpctAsmStmt::InstructionClass:
      return emitInstruction(dyn_cast<DpctAsmInstruction>(S));
    case DpctAsmStmt::GuardInstructionClass:
      return emitGuardInstruction(dyn_cast<DpctAsmGuardInstruction>(S));
    case DpctAsmStmt::UnaryOperatorClass:
      return emitUnaryOperator(dyn_cast<DpctAsmUnaryOperator>(S));
    case DpctAsmStmt::BinaryOperatorClass:
      return emitBinaryOperator(dyn_cast<DpctAsmBinaryOperator>(S));
    case DpctAsmStmt::ConditionalOperatorClass:
      return emitConditionalOperator(dyn_cast<DpctAsmConditionalOperator>(S));
    case DpctAsmStmt::CastExprClass:
      return emitCastExpression(dyn_cast<DpctAsmCastExpr>(S));
    case DpctAsmStmt::ParenExprClass:
      return emitParenExpression(dyn_cast<DpctAsmParenExpr>(S));
    case DpctAsmStmt::DeclRefExprClass:
      return emitDeclRefExpression(dyn_cast<DpctAsmDeclRefExpr>(S));
    case DpctAsmStmt::IntegerLiteralClass:
      return emitIntegerLiteral(dyn_cast<DpctAsmIntegerLiteral>(S));
    case DpctAsmStmt::FloatingLiteralClass:
      return emitFloatingLiteral(dyn_cast<DpctAsmFloatingLiteral>(S));
    case DpctAsmStmt::ExactMachineFloatingClass:
      return emitExactMachineFloatingLiteral(dyn_cast<DpctAsmExactMachineFloatingLiteral>(S));
    default:
      return true;
    }
    return false;
}

bool SYCLGenBase::emitDeclarationStatement(const DpctAsmDeclStmt *S) {
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

bool SYCLGenBase::emitCompoundStatement(const DpctAsmCompoundStmt *S) {
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

bool SYCLGenBase::emitInstruction(const DpctAsmInstruction *I) {
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

bool SYCLGenBase::emitGuardInstruction(const DpctAsmGuardInstruction *I) {
  OS() << "if (";
  if (I->isNeg())
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

bool SYCLGenBase::emitUnaryOperator(const DpctAsmUnaryOperator *Op) {
  switch (Op->getOpcode()) {
// clang-format off
  case DpctAsmUnaryOperator::Plus:  OS() << "+"; break;
  case DpctAsmUnaryOperator::Minus: OS() << "-"; break;
  case DpctAsmUnaryOperator::Not:   OS() << "~"; break;
  case DpctAsmUnaryOperator::LNot:  OS() << "!"; break;
// clang-format on
  }
  if (emitStatement(Op->getSubExpr()))
    return true;
  return false;
}

bool SYCLGenBase::emitBinaryOperator(const DpctAsmBinaryOperator *Op) {
  if (emitStatement(Op->getLHS()))
      return true;
    OS() << " ";
// clang-format off
    switch (Op->getOpcode()) {
    case DpctAsmBinaryOperator::Mul:    OS() << "*";  break;
    case DpctAsmBinaryOperator::Div:    OS() << "/";  break;
    case DpctAsmBinaryOperator::Rem:    OS() << "%";  break;
    case DpctAsmBinaryOperator::Add:    OS() << "+";  break;
    case DpctAsmBinaryOperator::Sub:    OS() << "-";  break;
    case DpctAsmBinaryOperator::Shl:    OS() << "<<"; break;
    case DpctAsmBinaryOperator::Shr:    OS() << ">>"; break;
    case DpctAsmBinaryOperator::LT:     OS() << "<";  break;
    case DpctAsmBinaryOperator::GT:     OS() << ">";  break;
    case DpctAsmBinaryOperator::LE:     OS() << "<="; break;
    case DpctAsmBinaryOperator::GE:     OS() << ">="; break;
    case DpctAsmBinaryOperator::EQ:     OS() << "=="; break;
    case DpctAsmBinaryOperator::NE:     OS() << "!="; break;
    case DpctAsmBinaryOperator::And:    OS() << "&";  break;
    case DpctAsmBinaryOperator::Xor:    OS() << "^";  break;
    case DpctAsmBinaryOperator::Or:     OS() << "|";  break;
    case DpctAsmBinaryOperator::LAnd:   OS() << "&&"; break;
    case DpctAsmBinaryOperator::LOr:    OS() << "||"; break;
    case DpctAsmBinaryOperator::Assign: OS() << "=";  break;
// clang-format on
    }
    OS() << " ";
    if (emitStatement(Op->getRHS()))
      return true;
    return false;
}

bool SYCLGenBase::emitConditionalOperator(const DpctAsmConditionalOperator *Op) {
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

bool SYCLGenBase::emitCastExpression(const DpctAsmCastExpr *E) {
  OS() << "static_cast<";
  if (emitType(E->getType()))
    return true;
  OS() << ">(";
  if (emitStatement(E->getSubExpr()))
    return true;
  OS() << ")";
  return false;
}

bool SYCLGenBase::emitParenExpression(const DpctAsmParenExpr *E) {
  OS() << "(";
  if (emitStatement(E->getSubExpr()))
    return true;
  OS() << ")";
  return false;
}

bool SYCLGenBase::emitDeclRefExpression(const DpctAsmDeclRefExpr *E) {
  OS() << E->getDecl().getDeclName()->getName();
  return false;
}

bool SYCLGenBase::emitIntegerLiteral(const DpctAsmIntegerLiteral *I) {
  OS() << I->getValue();
  return false;
}

bool SYCLGenBase::emitFloatingLiteral(const DpctAsmFloatingLiteral *Fp) {
  Fp->getValue().print(OS());
  return false;
}

bool SYCLGenBase::emitExactMachineFloatingLiteral(const DpctAsmExactMachineFloatingLiteral *Fp) {
  // [](){union {unsigned I; float F;}; I = 0x3f800000u; return F;}()
  constexpr char *Template = "[](){{union {{{0} I; {1} F;}; I = 0x{2}u; return F;}()";
  if (const auto *T = dyn_cast<DpctAsmBuiltinType>(Fp->getType())) {
    switch (T->getKind()) {
    case DpctAsmBuiltinType::TK_f32:
      OS() << llvm::formatv(Template, "uint32_t", "float", Fp->getHexLiteral());
      break;
    case DpctAsmBuiltinType::TK_f64:
      OS() << llvm::formatv(Template, "uint64_t", "double", Fp->getHexLiteral());
      break;
    default:
      return true;
    }
    return false;
  }
  return true;
}

bool SYCLGenBase::emitType(const DpctAsmType *T) {
  switch (T->getTypeClass()) {
  case DpctAsmType::BuiltinClass:
    return emitBuiltinType(dyn_cast<DpctAsmBuiltinType>(T));
  case DpctAsmType::VectorClass:
    return emitVectorType(dyn_cast<DpctAsmVectorType>(T));
  default:
    break;
  }
  return true;
}

bool SYCLGenBase::emitBuiltinType(const DpctAsmBuiltinType *T) {
  switch (T->getKind()) {
// clang-format off
  case DpctAsmBuiltinType::TK_b8:     OS() << "uint8_t"; break;
  case DpctAsmBuiltinType::TK_b16:    OS() << "uint16_t"; break;
  case DpctAsmBuiltinType::TK_b32:    OS() << "uint32_t"; break;
  case DpctAsmBuiltinType::TK_b64:    OS() << "uint64_t"; break;
  case DpctAsmBuiltinType::TK_u8:     OS() << "uint8_t"; break;
  case DpctAsmBuiltinType::TK_u16:    OS() << "uint16_t"; break;
  case DpctAsmBuiltinType::TK_u32:    OS() << "uint32_t"; break;
  case DpctAsmBuiltinType::TK_u64:    OS() << "uint64_t"; break;
  case DpctAsmBuiltinType::TK_s8:     OS() << "int8_t"; break;
  case DpctAsmBuiltinType::TK_s16:    OS() << "int16_t"; break;
  case DpctAsmBuiltinType::TK_s32:    OS() << "int32_t"; break;
  case DpctAsmBuiltinType::TK_s64:    OS() << "int64_t"; break;
  case DpctAsmBuiltinType::TK_f16:    OS() << "sycl::half"; break;
  case DpctAsmBuiltinType::TK_f32:    OS() << "float"; break;
  case DpctAsmBuiltinType::TK_f64:    OS() << "double"; break;
  case DpctAsmBuiltinType::TK_byte:   OS() << "uint8_t"; break;
  case DpctAsmBuiltinType::TK_4byte:  OS() << "uint32_t"; break;
  case DpctAsmBuiltinType::TK_pred:   OS() << "bool"; break;
  case DpctAsmBuiltinType::TK_bf16:
  case DpctAsmBuiltinType::TK_e4m3:
  case DpctAsmBuiltinType::TK_e5m2:
  case DpctAsmBuiltinType::TK_tf32:
  case DpctAsmBuiltinType::TK_f16x2:
  case DpctAsmBuiltinType::TK_bf16x2:
  case DpctAsmBuiltinType::TK_e4m3x2:
  case DpctAsmBuiltinType::TK_e5m2x2:
  case DpctAsmBuiltinType::TK_s16x2:
  case DpctAsmBuiltinType::TK_u16x2:
// clang-format on
    return true;
  }
  return false;
}

bool SYCLGenBase::emitVectorType(const DpctAsmVectorType *T) {
  OS() << "sycl::vec<";
  if (emitType(T->getElementType()))
    return true;
  OS() << ">";
  return false;
}

bool SYCLGenBase::emitDeclaration(const DpctAsmDecl *D) {
  switch (D->getDeclClass()) {
  case DpctAsmDecl::VariableDeclClass:
    return emitVariableDeclaration(dyn_cast<DpctAsmVariableDecl>(D));
  default:
    break;
  }
  return true;
}

bool SYCLGenBase::emitVariableDeclaration(const DpctAsmVariableDecl *D) {
  OS() << D->getDeclName()->getName();
  return false;
}

class SYCLGen : public SYCLGenBase {
public:
  SYCLGen(llvm::raw_ostream &OS) : SYCLGenBase(OS) {}

  bool handleStatement(const DpctAsmStmt *S) {
    return emitStatement(S);
  }

protected:
  bool handle_mov(const DpctAsmInstruction *I) override {
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

  bool handle_setp(const DpctAsmInstruction *I) override {
    if (I->getNumInputOperands() != 2 &&
        I->getNumTypes() == 1)
      return true;
    
    auto *T = dyn_cast<DpctAsmBuiltinType>(I->getType(0));
    if (!T) return true;
    
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
          Template = "{0} == {1} && !sycl::isnan({0}) && !sycl::isnan({1})";
        else
          return true;
        break;
      case asmtok::kw_ne:
        if (T->isSignedInt() || T->isUnsignedInt() || T->isBitSize())
          Template = "{0} != {1}";
        else if (T->isFloating())
          Template = "{0} != {1} && !sycl::isnan({0}) && !sycl::isnan({1})";
        else
          return true;
        break;
      case asmtok::kw_lt:
        if (T->isSignedInt())
          Template = "{0} < {1}";
        else if (T->isFloating())
          Template = "{0} < {1} && !sycl::isnan({0}) && !sycl::isnan({1})";
        else
          return true;
        break;
      case asmtok::kw_le:
        if (T->isSignedInt())
          Template = "{0} <= {1}";
        else if (T->isFloating())
          Template = "{0} <= {1} && !sycl::isnan({0}) && !sycl::isnan({1})";
        else
          return true;
        break;
      case asmtok::kw_gt:
        if (T->isSignedInt())
          Template = "{0} > {1}";
        else if (T->isFloating())
          Template = "{0} > {1} && !sycl::isnan({0}) && !sycl::isnan({1})";
        else
          return true;
        break;
      case asmtok::kw_ge:
        if (T->isSignedInt())
          Template = "{0} >= {1}";
        else if (T->isFloating())
          Template = "{0} >= {1} && !sycl::isnan({0}) && !sycl::isnan({1})";
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
          Template = "{0} == {1} || sycl::isnan({0}) || sycl::isnan({1})";
        else
          return true;
        break;
      case asmtok::kw_neu:
        if (T->isFloating())
          Template = "{0} != {1} || sycl::isnan({0}) || sycl::isnan({1})";
        else
          return true;
        break;
      case asmtok::kw_ltu:
        if (T->isFloating())
          Template = "{0} < {1} || sycl::isnan({0}) || sycl::isnan({1})";
        else
          return true;
        break;
      case asmtok::kw_leu:
        if (T->isFloating())
          Template = "{0} <= {1} || sycl::isnan({0}) || sycl::isnan({1})";
        else
          return true;
        break;
      case asmtok::kw_gtu:
        if (T->isFloating())
          Template = "{0} > {1} || sycl::isnan({0}) || sycl::isnan({1})";
        else
          return true;
        break;
      case asmtok::kw_geu:
        if (T->isFloating())
          Template = "{0} >= {1} || sycl::isnan({0}) || sycl::isnan({1})";
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

  bool handle_lop3(const DpctAsmInstruction *I) override {
    if (I->getNumInputOperands() != 4 || I->getNumTypes() != 1 || 
        !isa<DpctAsmBuiltinType>(I->getType(0)) ||
        dyn_cast<DpctAsmBuiltinType>(I->getType(0))->getKind() != DpctAsmBuiltinType::TK_b32)
      return true;

    if (emitStatement(I->getOutputOperand()))
      return true;
    
    OS() << " = ";

    std::string Op[3];
    for (auto Idx : llvm::seq(0, 3)) {
      if (tryEmitStatement(Op[Idx], I->getInputOperand(Idx)))
        return true;
    }

    if (!isa<DpctAsmIntegerLiteral>(I->getInputOperand(3)))
      return true;

    unsigned Imm = dyn_cast<DpctAsmIntegerLiteral>(I->getInputOperand(3))
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
          Templates.push_back(llvm::formatv(SlowMap[Bit], Op[0], Op[1], Op[2]).str());
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
      DpctAsmContext Context;
      std::string Replacement;
      llvm::raw_string_ostream OS(Replacement);
      llvm::SourceMgr Mgr;
      std::string Buffer = Asm->getAsmString()->getString().str();
      Mgr.AddNewSourceBuffer(
          llvm::MemoryBuffer::getMemBuffer(Buffer),
          llvm::SMLoc());
      DpctAsmParser Parser(Context, Mgr);
      SYCLGen CodeGen(OS);
      std::string AsmString;

      auto getReplaceString = [&](const Expr *E) {
        ExprAnalysis EA;
        EA.analyze(E);
        if (needExtraParens(E))
          return "(" + EA.getReplacedString() + ")";
        return EA.getReplacedString();
      };

      for (unsigned I = 0, E = AS->getNumOutputs(); I != E; ++I) {
        Parser.AddInlineAsmOperands(getReplaceString(AS->getOutputExpr(I)),
                                    AS->getOutputConstraint(I));
      }

      for (unsigned I = 0, E = AS->getNumInputs(); I != E; ++I) {
        Parser.AddInlineAsmOperands(getReplaceString(AS->getInputExpr(I)),
                                    AS->getInputConstraint(I));
      }

      CodeGen.setNumIndent(1);
      CodeGen.setIndentUnit(getIndent(AS->getBeginLoc(), DpctGlobalInfo::getSourceManager()));

      do {
        auto Inst = Parser.ParseStatement();
        if (Inst.isInvalid()) {
          report(AS->getAsmLoc(), Diagnostics::DEVICE_ASM, true);
          return;
        }

        if (CodeGen.handleStatement(Inst.get())) {
          report(AS->getAsmLoc(), Diagnostics::DEVICE_ASM, true);;
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
