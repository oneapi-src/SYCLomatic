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
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/raw_ostream.h"
#include <iterator>
#include <limits>
#include <sstream>

using namespace clang;
using namespace clang::dpct;

namespace {

class SYCLGen {
  llvm::StringMap<std::string> NameAliasTable;
  llvm::raw_ostream *Stream;
  bool EmitNewLine = false;
  bool EmitSemi = false;
public:
  SYCLGen(llvm::raw_ostream &OS) : Stream(&OS) {}
  ~SYCLGen() = default;

  void AddAlias(StringRef Origin, StringRef Alias) {
    NameAliasTable[Origin] = Alias;
  }

  bool HasAlias(StringRef Name) const { return NameAliasTable.contains(Name); }

  StringRef FindAlias(StringRef Name) {
    if (!HasAlias(Name))
      return "";
    return NameAliasTable[Name];
  }

  bool Emit(const PtxStmt *S) {
    switch (S->getStmtClass()) {
    case PtxStmt::CompoundStmtClass:
      return EmitCompoundStmt(dyn_cast<PtxCompoundStmt>(S));
    case PtxStmt::DeclStmtClass:
      return EmitDeclStmt(dyn_cast<PtxDeclStmt>(S));
    case PtxStmt::DeclRefExprClass:
      return EmitDeclRefExpr(dyn_cast<PtxDeclRefExpr>(S));
    case PtxStmt::IntegerLiteralClass:
      return EmitIntegerLiteral(dyn_cast<PtxIntegerLiteral>(S));
    case PtxStmt::FloatingLiteralClass:
      return EmitFloatingLiteral(dyn_cast<PtxFloatingLiteral>(S));
    case PtxStmt::CastExprClass:
      return EmitCastExpr(dyn_cast<PtxCastExpr>(S));
    case PtxStmt::ParenExprClass:
      return EmitParenExpr(dyn_cast<PtxParenExpr>(S));
    case PtxStmt::UnaryOperatorClass:
      return EmitUnaryOperator(dyn_cast<PtxUnaryOperator>(S));
    case PtxStmt::BinaryOperatorClass:
      return EmitBinaryOperator(dyn_cast<PtxBinaryOperator>(S));
    case PtxStmt::ConditionalOperatorClass:
      return EmitConditionalOperator(dyn_cast<PtxConditionalOperator>(S));
    case PtxStmt::GuardInstructionClass:
      return EmitGuardInstruction(dyn_cast<PtxGuardInstruction>(S));
    case PtxStmt::InstructionClass:
      return EmitInstruction(dyn_cast<PtxInstruction>(S));
    default:
      break;
    }
    return false;
  }

private:
  StringRef Endl() const {
    if (EmitNewLine)
      return getNL();
    return "";
  }

  StringRef Semi() const {
    if (EmitSemi)
      return ";";
    return "";
  }

  llvm::raw_ostream &OS() { return *Stream; }

  void SwitchOutStream(llvm::raw_ostream &NewOS) { Stream = &NewOS; }

  std::string ReNameVariable(StringRef Origin) const {
    std::string NewName;
    for (char C : Origin) {
      if (C == '%' || C == '$')
        continue;
      NewName += C;
    }

    return NewName + "_ct";
  }

  bool EmitMov(const PtxInstruction *I);
  bool EmitLop3(const PtxInstruction *I);
  bool EmitSetp(const PtxInstruction *I);

  bool EmitType(const PtxType *Type) {
    if (isa<PtxFundamentalType>(Type))
      return EmitFundamentalTyp(dyn_cast<PtxFundamentalType>(Type));
    return false;
  }

  bool EmitFundamentalTyp(const PtxFundamentalType *Type) {
    switch (Type->getKind()) {
    case PtxFundamentalType::TK_B8:
      OS() << "sycl::buffer<uint8_t, 1>";
      break;
    case PtxFundamentalType::TK_B16:
      OS() << "sycl::buffer<uint8_t, 2>";
      break;
    case PtxFundamentalType::TK_B32:
      OS() << "sycl::buffer<uint8_t, 4>";
      break;
    case PtxFundamentalType::TK_B64:
      OS() << "sycl::buffer<uint8_t, 8>";
      break;
    case PtxFundamentalType::TK_B128:
      OS() << "sycl::buffer<uint8_t, 16>";
      break;
    case PtxFundamentalType::TK_S2:
    case PtxFundamentalType::TK_S4:
      return true;
    case PtxFundamentalType::TK_S8:
      OS() << "int8_t";
      break;
    case PtxFundamentalType::TK_S16:
      OS() << "int16_t";
      break;
    case PtxFundamentalType::TK_S32:
      OS() << "int32_t";
      break;
    case PtxFundamentalType::TK_S64:
      OS() << "int64_t";
      break;
    case PtxFundamentalType::TK_U2:
    case PtxFundamentalType::TK_U4:
      return true;
    case PtxFundamentalType::TK_U8:
      OS() << "uint8_t";
      break;
    case PtxFundamentalType::TK_U16:
      OS() << "uint16_t";
      break;
    case PtxFundamentalType::TK_U32:
      OS() << "uint32_t";
      break;
    case PtxFundamentalType::TK_U64:
      OS() << "uint64_t";
      break;
    case PtxFundamentalType::TK_F16:
      OS() << "sycl::half";
      break;
    case PtxFundamentalType::TK_F16x2:
      OS() << "sycl::half2";
      break;
    case PtxFundamentalType::TK_F32:
      OS() << "float";
      break;
    case PtxFundamentalType::TK_F64:
      OS() << "double";
      break;
    case PtxFundamentalType::TK_E4m3:
    case PtxFundamentalType::TK_E5m2:
    case PtxFundamentalType::TK_E4m3x2:
    case PtxFundamentalType::TK_E5m2x2:
    case PtxFundamentalType::TK_Byte:
    case PtxFundamentalType::TK_4Byte:
      return true;
    case PtxFundamentalType::TK_Pred:
      OS() << "bool";
      break;
    }
    return false;
  }

  bool EmitDecl(const PtxDecl *D) {
    switch (D->getDeclClass()) {
    case PtxDecl::VariableDeclClass:
      return EmitVariableDecl(dyn_cast<PtxVariableDecl>(D));
    case PtxDecl::LabelDeclClass:
      return true;
    }
    return true;
  }

  bool EmitVariableDecl(const PtxVariableDecl *Variable) {
    if (Variable->getDeclName().contains('%') ||
        Variable->getDeclName().contains('$')) {
      std::string NewName = ReNameVariable(Variable->getDeclName());
      const_cast<PtxVariableDecl *>(Variable)->setDeclName(NewName);
    }
    OS() << Variable->getDeclName();
    return false;
  }

  bool EmitDeclRefExpr(const PtxDeclRefExpr *DRE) {
    StringRef Name = DRE->getSymbol().getDeclName();
    if (HasAlias(Name))
      OS() << FindAlias(Name);
    else
      OS() << Name;
    return false;
  }

  bool EmitIntegerLiteral(const PtxIntegerLiteral *Int) {
    OS() << Int->getValue();
    return false;
  }

  bool EmitFloatingLiteral(const PtxFloatingLiteral *FP) {
    FP->getValue().print(OS());
    return false;
  }

  bool EmitCastExpr(const PtxCastExpr *Cast) {
    OS() << "(";
    if (EmitType(Cast->getType()))
      return true;
    OS() << ")";
    if (Emit(Cast->getSubExpr()))
      return true;
    return false;
  }

  bool EmitParenExpr(const PtxParenExpr *Paren) {
    OS() << "(";
    if (Emit(Paren->getSubExpr()))
      return true;
    OS() << ")";
    return false;
  }

  bool EmitBinaryOperator(const PtxBinaryOperator *BinOp) {
    if (Emit(BinOp->getLHS()))
      return true;
    OS() << " ";
    switch (BinOp->getOpcode()) {
    case PtxBinaryOperator::Mul:
      OS() << "*";
      break;
    case PtxBinaryOperator::Div:
      OS() << "/";
      break;
    case PtxBinaryOperator::Rem:
      OS() << "%";
      break;
    case PtxBinaryOperator::Add:
      OS() << "+";
      break;
    case PtxBinaryOperator::Sub:
      OS() << "-";
      break;
    case PtxBinaryOperator::Shl:
      OS() << "<<";
      break;
    case PtxBinaryOperator::Shr:
      OS() << ">>";
      break;
    case PtxBinaryOperator::LT:
      OS() << "<";
      break;
    case PtxBinaryOperator::GT:
      OS() << ">";
      break;
    case PtxBinaryOperator::LE:
      OS() << "<=";
      break;
    case PtxBinaryOperator::GE:
      OS() << ">=";
      break;
    case PtxBinaryOperator::EQ:
      OS() << "==";
      break;
    case PtxBinaryOperator::NE:
      OS() << "!=";
      break;
    case PtxBinaryOperator::And:
      OS() << "&";
      break;
    case PtxBinaryOperator::Xor:
      OS() << "^";
      break;
    case PtxBinaryOperator::Or:
      OS() << "|";
      break;
    case PtxBinaryOperator::LAnd:
      OS() << "&&";
      break;
    case PtxBinaryOperator::LOr:
      OS() << "||";
      break;
      break;
    }
    OS() << " ";
    if (Emit(BinOp->getRHS()))
      return true;
    return false;
  }

  bool EmitUnaryOperator(const PtxUnaryOperator *UnaryOp) {
    switch (UnaryOp->getOpcode()) {
    case PtxUnaryOperator::Plus:
      OS() << "+";
      break;
    case PtxUnaryOperator::Minus:
      OS() << "-";
      break;
    case PtxUnaryOperator::Not:
      OS() << "~";
      break;
    case PtxUnaryOperator::LNot:
      OS() << "!";
      break;
      break;
    }
    if (Emit(UnaryOp->getSubExpr()))
      return true;
    return false;
  }

  bool EmitConditionalOperator(const PtxConditionalOperator *CondOp) {
    if (Emit(CondOp->getCond()))
      return true;
    OS() << " ? ";
    if (Emit(CondOp->getLHS()))
      return true;
    OS() << " : ";
    if (Emit(CondOp->getRHS()))
      return true;
    return false;
  }

  bool EmitCompoundStmt(const PtxCompoundStmt *C) {
    llvm::SaveAndRestore<bool> StoreEndl(EmitNewLine);
    llvm::SaveAndRestore<bool> StoreSemi(EmitSemi);
    EmitNewLine = true;
    EmitSemi = true;
    OS() << "{" << Endl();
    for (const auto *SubStmt : C->getStmts()) {
      if (Emit(SubStmt))
        return true;
    }
    OS() << "}" << Endl();
    return false;
  }

  bool EmitDeclStmt(const PtxDeclStmt *D) {
    if (D->decls().empty())
      return false;

    if (EmitType(D->getBaseType()))
      return true;
    OS() << " ";
    int NumComma = std::distance(D->decls().begin(), D->decls().end()) - 1;
    for (const auto &V : D->decls()) {
      if (EmitDecl(V))
        return true;
      if (NumComma-- > 0)
        OS() << ", ";
    }
    OS() << Semi() << Endl();
    return false;
  }

  bool EmitGuardInstruction(const PtxGuardInstruction *I) {
    {
      llvm::SaveAndRestore<bool> StoreEndl(EmitNewLine);
      llvm::SaveAndRestore<bool> StoreSemi(EmitSemi);
      EmitNewLine = false;
      EmitSemi = false;
      OS() << "(";
      if (I->isNeg())
        OS() << "!";
      if (Emit(I->getPred()))
        return true;
      OS() << " && (";
      if (EmitInstruction(I->getInstruction()))
        return true;
      OS() << "))";
    }
    OS() << Semi() << Endl();
    return false;
  }

  bool EmitInstruction(const PtxInstruction *I) {
    switch (I->getOpcode()) {
    case ptx::Mov:
      return EmitMov(I);
    case ptx::Lop3:
      return EmitLop3(I);
    case ptx::Setp:
      return EmitSetp(I);
    default:
      break;
    }
    return true;
  }
};

} // namespace

bool SYCLGen::EmitMov(const PtxInstruction *I) {
  if (I->getOperands().size() != 2)
    return true;
  if (Emit(I->getOperand(0)))
    return true;
  OS() << " = ";
  if (Emit(I->getOperand(1)))
    return true;
  OS() << Semi() << Endl();
  return false;
}

bool SYCLGen::EmitSetp(const PtxInstruction *I) {
  if (I->getNumOperands() != 3)
    return true;

  if (Emit(I->getOperand(0)))
    return true;

  OS() << " = ";

  if (Emit(I->getOperand(1)))
    return true;
  OS() << " ";
  ptx::ComparisonOp Op =
      static_cast<ptx::ComparisonOp>(I->getAttributes().ComparisonOp);
  switch (Op) {
  case ptx::CO_Eq:
    OS() << "==";
    break;
  case ptx::CO_Ne:
    OS() << "!=";
    break;
  case ptx::CO_Lt:
    OS() << "<";
    break;
  case ptx::CO_Le:
    OS() << "<=";
    break;
  case ptx::CO_Gt:
    OS() << ">";
    break;
  case ptx::CO_Ge:
    OS() << ">=";
    break;
  case ptx::CO_Lo:
  case ptx::CO_Ls:
  case ptx::CO_Hi:
  case ptx::CO_Hs:
  case ptx::CO_Equ:
  case ptx::CO_Neu:
  case ptx::CO_Ltu:
  case ptx::CO_Leu:
  case ptx::CO_Gtu:
  case ptx::CO_Geu:
  case ptx::CO_Num:
  case ptx::CO_Nan:
    return true;
  }
  OS() << " ";
  if (Emit(I->getOperand(2)))
    return true;
  OS() << Semi() << Endl();
  return false;
}

static bool canAsmLop3ExprFast(llvm::raw_ostream &OS, const std::string &a,
                               const std::string &b, const std::string &c,
                               const std::uint8_t imm) {
#define EMPTY4 "", "", "", ""
#define EMPTY16 EMPTY4, EMPTY4, EMPTY4, EMPTY4
  static const std::string FastMap[256] = {
      /*0x00*/ "0",
      // clang-format off
      EMPTY16, EMPTY4, EMPTY4, "",
      /*0x1a*/ "(@a & @b | @c) ^ @a",
      "", "", "",
      /*0x1e*/ "@a ^ (@b | @c)",
      EMPTY4, EMPTY4, EMPTY4, "", "",
      /*0x2d*/ "~@a ^ (~@b & @c)",
      EMPTY16, "", "",
      /*0x40*/ "@a & @b & ~@c",
      EMPTY16, EMPTY16, EMPTY16, EMPTY4, "", "", "",
      /*0x78*/ "@a ^ (@b & @c)",
      EMPTY4, "", "", "",
      /*0x80*/ "@a & @b & @c",
      EMPTY16, EMPTY4, "",
      /*0x96*/ "@a ^ @b ^ @c",
      EMPTY16, EMPTY4, EMPTY4, EMPTY4, "",
      /*0xb4*/ "@a ^ (@b & ~@c)",
      "", "", "",
      /*0xb8*/ "(@a ^ (@b & (@c ^ @a)))",
      EMPTY16, EMPTY4, EMPTY4, "",
      /*0xd2*/ "@a ^ (~@b & @c)",
      EMPTY16, EMPTY4, "",
      /*0xe8*/ "((@a & (@b | @c)) | (@b & @c))",
      "",
      /*0xea*/ "(@a & @b) | @c",
      EMPTY16, "", "", "",
      // clang-format on
      /*0xfe*/ "@a | @b | @c",
      /*0xff*/ "1"};
#undef EMPTY16
#undef EMPTY4
  const StringRef Expr = FastMap[imm];
  if (Expr.empty()) {
    return false;
  }
  OS << " ";
  const StringRef ReplaceMap[3] = {a, b, c};
  std::string::size_type Pre = 0;
  auto Pos = Expr.find('@');
  while (Pos != std::string::npos) {
    OS << Expr.substr(Pre, Pos - Pre).str();
    OS << ReplaceMap[Expr[Pos + 1] - 'a'].str();
    Pre = Pos + 2;
    Pos = Expr.find('@', Pre);
  }
  OS << Expr.substr(Pre).str();
  return true;
}

bool SYCLGen::EmitLop3(const PtxInstruction *Inst) {
  if (Inst->getNumOperands() != 5 || Inst->getAttributes().Types.size() != 1 ||
      !llvm::isa<PtxFundamentalType>(Inst->getAttributes().Types.front()) ||
      llvm::dyn_cast<PtxFundamentalType>(Inst->getAttributes().Types.front())
              ->getKind() != PtxFundamentalType::TK_B32) {
    return true;
  }

  std::string Op[4];

  // Switch to temporary ostream, and emit each operand.
  {
    llvm::SaveAndRestore<llvm::raw_ostream *> OutStream(Stream);

    std::string TmpBuf;
    llvm::raw_string_ostream TmpOS(TmpBuf);
    SwitchOutStream(TmpOS);

    for (unsigned I = 0; I < 4; ++I) {
      if (Emit(Inst->getOperand(I)))
        return true;
      TmpOS.flush();
      Op[I] = TmpBuf;
      TmpBuf.clear();
    }
  }

  unsigned imm = 0;

  if (const auto *Int = dyn_cast<PtxIntegerLiteral>(Inst->getOperand(4))) {
    imm = Int->getValue().getZExtValue();
  } else {
    return true;
  }

  if (imm > std::numeric_limits<uint8_t>::max())
    return true;

  OS() << Op[0] << " =";

  if (canAsmLop3ExprFast(OS(), Op[1], Op[2], Op[3], imm))
    return false;

  std::string Src;
  llvm::raw_string_ostream Tmp(Src);
  if (imm & 0x01)
    Tmp << " (~" << Op[1] << " & ~" << Op[2] << " & ~" << Op[3] << ") |";
  if (imm & 0x02)
    Tmp << " (~" << Op[1] << " & ~" << Op[2] << " & " << Op[3] << ") |";
  if (imm & 0x04)
    Tmp << " (~" << Op[1] << " & " << Op[2] << " & ~" << Op[3] << ") |";
  if (imm & 0x08)
    Tmp << " (~" << Op[1] << " & " << Op[2] << " & " << Op[3] << ") |";
  if (imm & 0x10)
    Tmp << " (" << Op[1] << " & ~" << Op[2] << " & ~" << Op[3] << ") |";
  if (imm & 0x20)
    Tmp << " (" << Op[1] << " & ~" << Op[2] << " & " << Op[3] << ") |";
  if (imm & 0x40)
    Tmp << " (" << Op[1] << " & " << Op[2] << " & ~" << Op[3] << ") |";
  if (imm & 0x80)
    Tmp << " (" << Op[1] << " & " << Op[2] << " & " << Op[3] << ") |";

  OS() << StringRef(Src).drop_back(2);
  return false;
}

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
      PtxContext Context;
      std::string Replacement;
      llvm::raw_string_ostream OS(Replacement);
      llvm::SourceMgr Mgr;
      Mgr.AddNewSourceBuffer(
          llvm::MemoryBuffer::getMemBuffer(Asm->getAsmString()->getString()),
          llvm::SMLoc());
      PtxParser Parser(Context, Mgr);
      SYCLGen CodeGen(OS);
      unsigned OperandIdx = 0;
      std::string AsmString;

      auto getReplaceString = [&](const Expr *E) {
        ExprAnalysis EA;
        EA.analyze(E);
        if (isa<IntegerLiteral>(E) || isa<DeclRefExpr>(E) ||
            isa<ImplicitCastExpr>(E)) {
          return EA.getReplacedString();
        }
        return "(" + EA.getReplacedString() + ")";
      };

      for (const auto *E : Asm->outputs()) {
        std::string Placeholder = "%" + std::to_string(OperandIdx++);
        Parser.AddBuiltinSymbol(Placeholder, nullptr);
        CodeGen.AddAlias(Placeholder, getReplaceString(E));
      }

      for (const auto *E : Asm->inputs()) {
        std::string Placeholder = "%" + std::to_string(OperandIdx++);
        Parser.AddBuiltinSymbol(Placeholder, nullptr);
        CodeGen.AddAlias(Placeholder, getReplaceString(E));
      }

      auto Inst = Parser.ParseStatement();
      if (Inst.isInvalid())
        goto MigrateAsmFail;

      if (CodeGen.Emit(Inst.get()))
        goto MigrateAsmFail;

      OS.flush();
                   
      auto *Repl = new ReplaceStmt(AS, Replacement);
      Repl->setBlockLevelFormatFlag();
      emplaceTransformation(Repl);

      if (isa<PtxCompoundStmt>(Inst.get())) {
        auto Tok = Lexer::findNextToken(
            Asm->getEndLoc(), DpctGlobalInfo::getSourceManager(),
            DpctGlobalInfo::getContext().getLangOpts());
        if (Tok.has_value() && Tok->is(tok::semi))
          emplaceTransformation(new ReplaceToken(Tok->getLocation(), ""));
      }
      return;
    }
  MigrateAsmFail:
    report(AS->getAsmLoc(), Diagnostics::DEVICE_ASM, true);
  }
  return;
}
