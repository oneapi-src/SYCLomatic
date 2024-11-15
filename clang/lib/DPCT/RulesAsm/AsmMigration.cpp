//===----------------------- AsmMigration.cpp -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AsmMigration.h"
#include "AnalysisInfo.h"
#include "RulesAsm/Parser/AsmNodes.h"
#include "RulesAsm/Parser/AsmParser.h"
#include "RulesAsm/Parser/AsmTokenKinds.h"
#include "ErrorHandle/CrashRecovery.h"
#include "Diagnostics/Diagnostics.h"
#include "RuleInfra/MapNames.h"
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
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdio>

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
  SmallString<4> IndentUnit{"  "};
  SmallString<16> Indent;
  SmallString<4> NewLine;
  SmallVector<SmallString<10>, 4> VecExprTypeRecord;
  raw_ostream *Stream;
  InlineAsmContext &Context;
  bool MigrationStopped = false;

protected:
  const InlineAsmInstruction *CurrInst = nullptr;
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

  struct EnterEmitVectorExpressionContext {
    SYCLGenBase &CodeGen;
    bool ShouldEnter;
    EnterEmitVectorExpressionContext(SYCLGenBase &CodeGen, StringRef Type,
                                     bool ShouldEnter = true)
        : CodeGen(CodeGen), ShouldEnter(ShouldEnter) {
      if (ShouldEnter)
        CodeGen.VecExprTypeRecord.push_back(Type);
    }

    ~EnterEmitVectorExpressionContext() {
      if (ShouldEnter)
        CodeGen.VecExprTypeRecord.pop_back();
    }
  };

  template <typename IDTy, typename... Ts>
  inline void report(IDTy MsgID, bool UseTextBegin, Ts &&...Vals) {
    TransformSetTy TS;
    auto SL = GAS->getBeginLoc();
    DiagnosticsUtils::report<IDTy, Ts...>(SL, MsgID, &TS, UseTextBegin,
                                          std::forward<Ts>(Vals)...);
    for (auto &T : TS)
      DpctGlobalInfo::getInstance().addReplacement(
          T->getReplacement(DpctGlobalInfo::getContext()));
  }

public:
  SYCLGenBase(llvm::raw_ostream &OS, InlineAsmContext &Ctx, const GCCAsmStmt *G)
      : Stream(&OS), Context(Ctx), GAS(G) {}

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

  void cutOffMigration() { MigrationStopped = true; }

  bool isMigrationStopped() const { return MigrationStopped; }

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

  void insertHeader(HeaderType HT) const {
    DpctGlobalInfo::getInstance().insertHeader(GAS->getBeginLoc(), HT);
  }

  llvm::raw_ostream &OS() { return *Stream; }
  void switchOutStream(llvm::raw_ostream &NewOS) { Stream = &NewOS; }

  bool tryEmitStmt(std::string &Buffer, const InlineAsmStmt *S) {
    llvm::raw_string_ostream TmpOS(Buffer);
    llvm::SaveAndRestore<llvm::raw_ostream *> OutStream(Stream);
    switchOutStream(TmpOS);
    if (emitStmt(S))
      return SYCLGenError();
    TmpOS.flush();
    return SYCLGenSuccess();
  }

  bool tryEmitType(std::string &Buffer, const InlineAsmType *T) {
    llvm::raw_string_ostream TmpOS(Buffer);
    llvm::SaveAndRestore<llvm::raw_ostream *> OutStream(Stream);
    switchOutStream(TmpOS);
    if (emitType(T))
      return SYCLGenError();
    TmpOS.flush();
    return SYCLGenSuccess();
  }

  bool tryEmitAllInputOperands(MutableArrayRef<std::string> Ops,
                               const InlineAsmInstruction *Inst) {
    for (unsigned I = 0, E = Inst->getNumInputOperands(); I != E; ++I)
      if (tryEmitStmt(Ops[I], Inst->getInputOperand(I)))
        return SYCLGenError();
    return SYCLGenSuccess();
  }

  bool needBitCast(const InlineAsmType *From, const InlineAsmType *To) {
    if (From == To)
      return false;
    if (const auto *BIFrom = dyn_cast<InlineAsmBuiltinType>(From),
        *BITo = dyn_cast<InlineAsmBuiltinType>(To);
        BIFrom->isScalar() && BITo->isScalar())
      return false;
    return true;
  }

  bool emitBitCast(const InlineAsmType *From, const InlineAsmType *To,
                   std::string &Val) {
    assert(needBitCast(From, To) && "Bit cast is unnecessary");
    std::string Buffer;
    llvm::raw_string_ostream TmpOS(Buffer);
    llvm::SaveAndRestore<llvm::raw_ostream *> OutStream(Stream);
    switchOutStream(TmpOS);
    std::string FromT, ToT;
    if (tryEmitType(FromT, From))
      return SYCLGenError();
    if (tryEmitType(ToT, To))
      return SYCLGenError();
    auto isVecTy = [&](const InlineAsmType *Ty) {
      if (isa<InlineAsmVectorType>(Ty))
        return true;
      const auto *BI = dyn_cast<InlineAsmBuiltinType>(Ty);
      return BI && BI->isVector();
    };
    if (isVecTy(From))
      OS() << Val;
    else
      OS() << MapNames::getClNamespace() << "vec<" << FromT << ", 1>(" << Val
           << ')';
    OS() << ".template as<";
    if (isVecTy(To))
      OS() << ToT << ">()";
    else
      OS() << MapNames::getClNamespace() << "vec<" << ToT << ", 1>>().x()";
    Val = std::move(Buffer);
    return SYCLGenSuccess();
  }

  // Types
  bool emitType(const InlineAsmType *T);
  bool emitBuiltinType(const InlineAsmBuiltinType *T);
  bool emitVectorType(const InlineAsmVectorType *T);

  // Declarations
  bool emitDeclaration(const InlineAsmDecl *D);
  bool emitVariableDeclaration(const InlineAsmVarDecl *D);

  // Statements && Expressions
  bool emitStmt(const InlineAsmStmt *S);
  bool emitCompoundStmt(const InlineAsmCompoundStmt *S);
  bool emitDeclStmt(const InlineAsmDeclStmt *S);
  bool emitInstruction(const InlineAsmInstruction *I);
  bool emitConditionalInstruction(const InlineAsmConditionalInstruction *I);
  bool emitUnaryOperator(const InlineAsmUnaryOperator *Op);
  bool emitBinaryOperator(const InlineAsmBinaryOperator *Op);
  bool emitConditionalOperator(const InlineAsmConditionalOperator *Op);
  bool emitVectorExpr(const InlineAsmVectorExpr *E);
  bool emitDiscardExpr(const InlineAsmDiscardExpr *E) { return SYCLGenError(); }
  bool emitAddressExpr(const InlineAsmAddressExpr *E);
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
#include "RulesAsm/Parser/AsmTokenKinds.def"
};

bool SYCLGenBase::emitStmt(const InlineAsmStmt *S) {
  switch (S->getStmtClass()) {
#define STMT(CLASS, PARENT)                                                    \
  case InlineAsmStmt::CLASS##Class:                                            \
    return emit##CLASS(dyn_cast<InlineAsm##CLASS>(S));
#define ABSTRACT_STMT(STMT)
#include "RulesAsm/Parser/AsmNodes.def"
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
  switch (I->getOpcodeID()->getTokenID()) {
#define INSTRUCTION(X)                                                         \
  case asmtok::op_##X:                                                         \
    return handle_##X(I);
#include "RulesAsm/Parser/AsmTokenKinds.def"
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
  auto *VD = dyn_cast_or_null<InlineAsmVarDecl>(&E->getDecl());
  if (!VD)
    return SYCLGenError();
  if (VD->getDeclName()->isSpecialReg()) {
    switch (VD->getDeclName()->getTokenID()) {
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
  OS() << VD->getDeclName()->getName();
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
    std::string Fmt =
        MapNames::getClNamespace() + "bit_cast<{0}>({1}(0x{2}{3}))";
    if (const auto *T = dyn_cast<InlineAsmBuiltinType>(Fp->getType())) {
      switch (T->getKind()) {
      case InlineAsmBuiltinType::f32:
        OS() << llvm::formatv(Fmt.c_str(), "float", "uint32_t",
                              Fp->getLiteral(), "U");
        break;
      case InlineAsmBuiltinType::f64:
        OS() << llvm::formatv(Fmt.c_str(), "double", "uint64_t",
                              Fp->getLiteral(), "ULL");
        break;
      default:
        return SYCLGenError();
      }
      return SYCLGenSuccess();
    }
  }
  return SYCLGenSuccess();
}

bool SYCLGenBase::emitVectorExpr(const InlineAsmVectorExpr *E) {
  if (!VecExprTypeRecord.empty())
    OS() << VecExprTypeRecord.back();
  OS() << "{";
  unsigned NumCommas = E->getNumElements() - 1;
  for (const auto *Element : E->elements()) {
    if (emitStmt(Element))
      return SYCLGenError();
    if (NumCommas) {
      OS() << ", ";
      NumCommas--;
    }
  }
  OS() << "}";
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
  case InlineAsmBuiltinType::b8:     OS() << "uint8_t"; break;
  case InlineAsmBuiltinType::b16:    OS() << "uint16_t"; break;
  case InlineAsmBuiltinType::b32:    OS() << "uint32_t"; break;
  case InlineAsmBuiltinType::b64:    OS() << "uint64_t"; break;
  case InlineAsmBuiltinType::u8:     OS() << "uint8_t"; break;
  case InlineAsmBuiltinType::u16:    OS() << "uint16_t"; break;
  case InlineAsmBuiltinType::u32:    OS() << "uint32_t"; break;
  case InlineAsmBuiltinType::u64:    OS() << "uint64_t"; break;
  case InlineAsmBuiltinType::s8:     OS() << "int8_t"; break;
  case InlineAsmBuiltinType::s16:    OS() << "int16_t"; break;
  case InlineAsmBuiltinType::s32:    OS() << "int32_t"; break;
  case InlineAsmBuiltinType::s64:    OS() << "int64_t"; break;
  case InlineAsmBuiltinType::f16:    OS() << MapNames::getClNamespace() + "half"; break;
  case InlineAsmBuiltinType::f32:    OS() << "float"; break;
  case InlineAsmBuiltinType::f64:    OS() << "double"; break;
  case InlineAsmBuiltinType::byte:   OS() << "uint8_t"; break;
  // case InlineAsmBuiltinType::4byte:  OS() << "uint32_t"; break;
  case InlineAsmBuiltinType::pred:   OS() << "bool"; break;
  case InlineAsmBuiltinType::s16x2:  OS() << MapNames::getClNamespace() + "short2"; break;
  case InlineAsmBuiltinType::u16x2:  OS() << MapNames::getClNamespace() + "ushort2"; break;
  case InlineAsmBuiltinType::bf16:   OS() << MapNames::getClNamespace() + "ext::oneapi::bfloat16"; break;
  case InlineAsmBuiltinType::f16x2:  OS() << MapNames::getClNamespace() + "half2"; break;
  case InlineAsmBuiltinType::e4m3:
  case InlineAsmBuiltinType::e5m2:
  case InlineAsmBuiltinType::tf32:
  case InlineAsmBuiltinType::bf16x2:
  case InlineAsmBuiltinType::e4m3x2:
  case InlineAsmBuiltinType::e5m2x2:
    [[fallthrough]];
  default:
    return SYCLGenError();
  }
  // clang-format on
  return SYCLGenSuccess();
}

bool SYCLGenBase::emitVectorType(const InlineAsmVectorType *T) {
  OS() << MapNames::getClNamespace() << "vec<";
  if (emitType(T->getElementType()))
    return SYCLGenError();
  OS() << ", ";
  switch (T->getKind()) {
  case InlineAsmVectorType::v2:
    OS() << 2;
    break;
  case InlineAsmVectorType::v4:
    OS() << 4;
    break;
  case InlineAsmVectorType::v8:
    OS() << 8;
    break;
  }
  OS() << ">";
  return SYCLGenSuccess();
}

bool SYCLGenBase::emitDeclaration(const InlineAsmDecl *D) {
  switch (D->getDeclClass()) {
  case InlineAsmDecl::VariableDeclClass:
    return emitVariableDeclaration(dyn_cast<InlineAsmVarDecl>(D));
  default:
    break;
  }
  return SYCLGenError();
}

bool SYCLGenBase::emitVariableDeclaration(const InlineAsmVarDecl *D) {
  OS() << D->getDeclName()->getName();
  if (D->isParameterizedNameDecl())
    OS() << '[' << D->getNumParameterizedNames() << ']';
  return SYCLGenSuccess();
}

bool SYCLGenBase::emitAddressExpr(const InlineAsmAddressExpr *Dst) {
  // Address expression only support ld/st instructions.
  if (!CurrInst || !CurrInst->is(asmtok::op_st, asmtok::op_ld, asmtok::op_atom))
    return SYCLGenError();
  std::string Type;
  if (tryEmitType(Type, CurrInst->getType(0)))
    return SYCLGenError();
  auto CanSuppressCast = [&](InlineAsmDeclRefExpr *DRE) {
    auto *VD = dyn_cast<InlineAsmVarDecl>(&Dst->getSymbol()->getDecl());
    if (VD->getInlineAsmOp()) {
      if (const auto *Ptr = dyn_cast<PointerType>(
              VD->getInlineAsmOp()->getType().getTypePtr())) {
        return Context.getTypeFromClangType(
                   Ptr->getPointeeType().getTypePtr()) == CurrInst->getType(0);
      }
    }
    return false;
  };

  if (CurrInst->is(asmtok::op_st, asmtok::op_ld))
    OS() << "*";
  switch (Dst->getMemoryOpKind()) {
  case InlineAsmAddressExpr::Imm:
    OS() << llvm::formatv("(({0} *)(uintptr_t){1})", Type,
                          Dst->getImmAddr()->getValue().getZExtValue());
    break;
  case InlineAsmAddressExpr::Reg: {
    std::string Reg;
    if (tryEmitStmt(Reg, Dst->getSymbol()))
      return SYCLGenSuccess();
    if (CanSuppressCast(Dst->getSymbol()))
      OS() << llvm::formatv("{0}", Reg);
    else
      OS() << llvm::formatv("(({0} *)(uintptr_t){1})", Type, Reg);
    break;
  }
  case InlineAsmAddressExpr::RegImm: {
    std::string Reg;
    if (tryEmitStmt(Reg, Dst->getSymbol()))
      return SYCLGenSuccess();
    OS() << llvm::formatv("(({0} *)((uintptr_t){1} + {2}))", Type, Reg,
                          Dst->getImmAddr()->getValue().getZExtValue());
    break;
  }
  case InlineAsmAddressExpr::Var: {
    std::string Reg;
    if (tryEmitStmt(Reg, Dst->getSymbol()))
      return SYCLGenSuccess();
    if (CanSuppressCast(Dst->getSymbol()))
      OS() << llvm::formatv("{0}", Reg);
    else
      OS() << llvm::formatv("(({0} *)&{1})", Type, Reg);
    break;
  }
  case InlineAsmAddressExpr::VarImm: {
    std::string Reg;
    if (tryEmitStmt(Reg, Dst->getSymbol()))
      return SYCLGenSuccess();
    OS() << llvm::formatv("(({0} *)((uintptr_t)&{1} + {2}))", Type, Reg,
                          Dst->getImmAddr()->getValue().getZExtValue());
    break;
  }
  }
  return SYCLGenSuccess();
}

/// This used to handle the specific instruction.
class SYCLGen : public SYCLGenBase {
public:
  SYCLGen(llvm::raw_ostream &OS, InlineAsmContext &Ctx, const GCCAsmStmt *G)
      : SYCLGenBase(OS, Ctx, G) {}

  bool handleStatement(const InlineAsmStmt *S) { return emitStmt(S); }

protected:
  unsigned getBitSizeTypeWidth(const InlineAsmBuiltinType *T) const {
    switch (T->getKind()) {
    case InlineAsmBuiltinType::b8:
      return 8;
    case InlineAsmBuiltinType::b16:
      return 16;
    case InlineAsmBuiltinType::b32:
      return 32;
    case InlineAsmBuiltinType::b64:
      return 64;
    default:
      return 0;
    }
  }

  const char *unpackBitSizeType(const InlineAsmBuiltinType *T,
                                unsigned N) const {
    assert(T->isBit());
    unsigned OriginTypeWidth = getBitSizeTypeWidth(T);
    unsigned UnpackTypeWidth = OriginTypeWidth / N;
    switch (UnpackTypeWidth) {
    case 8:
      return "uint8_t";
    case 16:
      return "uint16_t";
    case 32:
      return "uint32_t";
    default:
      return nullptr;
    }
  }

  bool handle_mov(const InlineAsmInstruction *I) override {
    if (I->getNumInputOperands() != 1)
      return SYCLGenError();
    // Handle data unpack mov.
    // mov.b32 {%0, %1}, %2;
    // %0 = sycl::vec<uint32_t, 1>(%2).template as<sycl::vec<uint16_t, 2>>()[0];
    // %1 = sycl::vec<uint32_t, 1>(%2).template as<sycl::vec<uint16_t, 2>>()[1];
    if (const auto *VE = dyn_cast<InlineAsmVectorExpr>(I->getOutputOperand())) {
      const auto *Type =
          llvm::dyn_cast_or_null<InlineAsmBuiltinType>(I->getType(0));
      if (!Type || !Type->isBit())
        return SYCLGenError();
      std::string OriginType, InputOp;
      std::string UnpackType = unpackBitSizeType(Type, VE->getNumElements());
      if (tryEmitType(OriginType, Type) ||
          tryEmitStmt(InputOp, I->getInputOperand(0)))
        return SYCLGenError();
      std::string SYCLVec;
      {
        llvm::raw_string_ostream TmpOS(SYCLVec);
        TmpOS << MapNames::getClNamespace() << "vec<" << OriginType << ", 1>("
              << InputOp << ").template as<" << MapNames::getClNamespace()
              << "vec<" << UnpackType << ", " << VE->getNumElements() << ">>()";
      }
      for (unsigned I = 0, E = VE->getNumElements(); I != E; ++I) {
        if (isa<InlineAsmDiscardExpr>(VE->getElement(I)))
          continue;
        if (I > 0)
          indent();
        if (emitStmt(VE->getElement(I)))
          return SYCLGenError();
        OS() << " = " << SYCLVec << '[' << I << ']';
        endstmt();
      }
    } else if (const auto *VE =
                   dyn_cast<InlineAsmVectorExpr>(I->getInputOperand(0))) {
      // Handle data pack ov.
      // mov.b32 %0, {%1, %2};
      // %0 = sycl::vec<uint16_t, 2>{%1, %2}.template as<sycl::vec<uint32_t,
      // 1>>()[0];
      const auto *Type =
          llvm::dyn_cast_or_null<InlineAsmBuiltinType>(I->getType(0));
      if (!Type || !Type->isBit())
        return SYCLGenError();
      std::string PackType, OutputOp;
      std::string OriginType = unpackBitSizeType(Type, VE->getNumElements());
      if (tryEmitType(PackType, Type))
        return SYCLGenError();
      if (emitStmt(I->getOutputOperand()))
        return SYCLGenError();
      OS() << " = " << MapNames::getClNamespace() << "vec<" << OriginType
           << ", " << VE->getNumElements() << ">(";
      if (emitStmt(VE))
        return SYCLGenError();
      OS() << ").template as<" << MapNames::getClNamespace() << "vec<"
           << PackType << ", 1>>()[0]";
      endstmt();
    } else {
      if (emitStmt(I->getOutputOperand()))
        return SYCLGenError();
      OS() << " = ";
      if (emitStmt(I->getInputOperand(0)))
        return SYCLGenError();
      endstmt();
    }
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
    std::string Op[2];
    for (int i = 0; i < 2; ++i)
      if (tryEmitStmt(Op[i], I->getInputOperand(i)))
        return SYCLGenError();

    std::string TypeStr;
    if (tryEmitType(TypeStr, I->getType(0)))
      return SYCLGenError();

    auto ExprCmp = [&](StringRef BinOp) {
      OS() << Op[0] << ' ' << BinOp << ' ' << Op[1];
    };
    auto DpctCmp = [&](const char *Fmt) {
      OS() << MapNames::getDpctNamespace()
           << llvm::formatv(Fmt, TypeStr, Op[0], Op[1]);
    };

    bool IsInteger = T->isInt() || T->isBit();

    for (const auto A : I->attrs()) {
      switch (A) {
      case InstAttr::eq:
        if (IsInteger)
          ExprCmp("==");
        else if (T->isFloat())
          DpctCmp("compare<{0}>({1}, {2}, std::equal_to<>())");
        else
          return SYCLGenError();
        break;
      case InstAttr::ne:
        if (IsInteger)
          ExprCmp("!=");
        else if (T->isFloat())
          DpctCmp("compare<{0}>({1}, {2}, std::not_equal_to<>())");
        else
          return SYCLGenError();
        break;
      case InstAttr::lt:
        if (IsInteger)
          ExprCmp("<");
        else if (T->isFloat())
          DpctCmp("compare<{0}>({1}, {2}, std::less<>())");
        else
          return SYCLGenError();
        break;
      case InstAttr::le:
        if (IsInteger)
          ExprCmp("<=");
        else if (T->isFloat())
          DpctCmp("compare<{0}>({1}, {2}, std::less_equal<>())");
        else
          return SYCLGenError();
        break;
      case InstAttr::gt:
        if (IsInteger)
          ExprCmp(">");
        else if (T->isFloat())
          DpctCmp("compare<{0}>({1}, {2}, std::greater<>())");
        else
          return SYCLGenError();
        break;
      case InstAttr::ge:
        if (IsInteger)
          ExprCmp(">=");
        else if (T->isFloat())
          DpctCmp("compare<{0}>({1}, {2}, std::greater_equal<>())");
        else
          return SYCLGenError();
        break;
      case InstAttr::lo:
        if (T->isUnsigned())
          ExprCmp("<");
        else
          return SYCLGenError();
        break;
      case InstAttr::ls:
        if (T->isUnsigned())
          ExprCmp("<=");
        else
          return SYCLGenError();
        break;
      case InstAttr::hi:
        if (T->isUnsigned())
          ExprCmp(">");
        else
          return SYCLGenError();
        break;
      case InstAttr::hs:
        if (T->isUnsigned())
          ExprCmp(">=");
        else
          return SYCLGenError();
        break;
      case InstAttr::equ:
        if (T->isFloat())
          DpctCmp("unordered_compare<{0}>({1}, {2}, std::equal_to<>())");
        else
          return SYCLGenError();
        break;
      case InstAttr::neu:
        if (T->isFloat())
          DpctCmp("unordered_compare<{0}>({1}, {2}, std::not_equal_to<>())");
        else
          return SYCLGenError();
        break;
      case InstAttr::ltu:
        if (T->isFloat())
          DpctCmp("unordered_compare<{0}>({1}, {2}, std::less<>())");
        else
          return SYCLGenError();
        break;
      case InstAttr::leu:
        if (T->isFloat())
          DpctCmp("unordered_compare<{0}>({1}, {2}, std::less_equal<>())");
        else
          return SYCLGenError();
        break;
      case InstAttr::gtu:
        if (T->isFloat())
          DpctCmp("unordered_compare<{0}>({1}, {2}, std::greater<>())");
        else
          return SYCLGenError();
        break;
      case InstAttr::geu:
        if (T->isFloat())
          DpctCmp("unordered_compare<{0}>({1}, {2}, std::greater_equal<>())");
        else
          return SYCLGenError();
        break;
      case InstAttr::num:
        if (T->isFloat()) {
          OS() << '!' << MapNames::getClNamespace();
          OS() << "isnan(" << Op[0] << ")";
          OS() << " && ";
          OS() << '!' << MapNames::getClNamespace();
          OS() << "isnan(" << Op[1] << ")";
        } else
          return SYCLGenError();
        break;
      case InstAttr::nan:
        if (T->isFloat()) {
          OS() << MapNames::getClNamespace();
          OS() << "isnan(" << Op[0] << ")";
          OS() << " || ";
          OS() << MapNames::getClNamespace();
          OS() << "isnan(" << Op[1] << ")";
        } else
          return SYCLGenError();
        break;
      default:
        break;
      }
    }

    endstmt();
    return SYCLGenSuccess();
  }

  bool handle_lop3(const InlineAsmInstruction *I) override {
    if (I->getNumInputOperands() != 4 || I->getNumTypes() != 1 ||
        !isa<InlineAsmBuiltinType>(I->getType(0)) ||
        dyn_cast<InlineAsmBuiltinType>(I->getType(0))->getKind() !=
            InlineAsmBuiltinType::b32)
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

  bool CheckAddSubMinMaxType(const InlineAsmInstruction *Inst, bool &isVec) {
    if (const auto *BI = dyn_cast<InlineAsmBuiltinType>(Inst->getType(0))) {
      if (Inst->hasAttr(InstAttr::sat) &&
          BI->getKind() != InlineAsmBuiltinType::s32)
        return false;
      if (BI->getKind() != InlineAsmBuiltinType::s16 &&
          BI->getKind() != InlineAsmBuiltinType::u16 &&
          BI->getKind() != InlineAsmBuiltinType::s32 &&
          BI->getKind() != InlineAsmBuiltinType::u32 &&
          BI->getKind() != InlineAsmBuiltinType::s64 &&
          BI->getKind() != InlineAsmBuiltinType::u64 &&
          BI->getKind() != InlineAsmBuiltinType::s16x2 &&
          BI->getKind() != InlineAsmBuiltinType::u16x2)
        return false;
      isVec = BI->getKind() == InlineAsmBuiltinType::s16x2 ||
              BI->getKind() == InlineAsmBuiltinType::u16x2;
    } else {
      return false;
    }
    return true;
  }

  bool HandleAddSub(const InlineAsmInstruction *Inst) {
    if (Inst->getNumInputOperands() != 2 || Inst->getNumTypes() != 1)
      return SYCLGenError();

    bool isVec = false;
    if (!CheckAddSubMinMaxType(Inst, isVec))
      return SYCLGenError();

    if (emitStmt(Inst->getOutputOperand()))
      return SYCLGenError();
    OS() << " = ";

    std::string TypeRepl;
    if (tryEmitType(TypeRepl, Inst->getType(0)))
      return SYCLGenError();
    EnterEmitVectorExpressionContext Ctx(*this, TypeRepl, isVec);

    std::string Op[2];
    for (unsigned I = 0; I < Inst->getNumInputOperands(); ++I) {
      if (tryEmitStmt(Op[I], Inst->getInputOperand(I)))
        return SYCLGenError();
      if (Inst->hasAttr(InstAttr::sat))
        Op[I] =
            Cast(Inst->getType(0), Inst->getInputOperand(I)->getType(), Op[I]);
    }

    if (Inst->hasAttr(InstAttr::sat)) {
      if (Inst->is(asmtok::op_add))
        OS() << MapNames::getClNamespace()
             << llvm::formatv("add_sat({0}, {1})", Op[0], Op[1]);
      else
        OS() << MapNames::getClNamespace()
             << llvm::formatv("sub_sat({0}, {1})", Op[0], Op[1]);
    } else {
      if (Inst->is(asmtok::op_add))
        OS() << llvm::formatv("{0} + {1}", Op[0], Op[1]);
      else
        OS() << llvm::formatv("{0} - {1}", Op[0], Op[1]);
    }

    endstmt();
    return SYCLGenSuccess();
  }

  bool handle_add(const InlineAsmInstruction *Inst) override {
    return HandleAddSub(Inst);
  }

  bool HandleShflSync(const InlineAsmInstruction *Inst) {

    if (emitStmt(Inst->getOutputOperand()))
      return SYCLGenError();
    OS() << " = ";

    std::string Op[4];
    if (tryEmitAllInputOperands(Op, Inst))
      return SYCLGenError();

    OS() << MapNames::getDpctNamespace();

    if (DpctGlobalInfo::useMaskedSubGroupFunction()) {
      OS() << "experimental::";
      if (Inst->hasAttr(InstAttr::up)) {
        // to handle "shfl.up.b32 %0, %1, %2, %3;"
        OS() << "shift_sub_group_right(" << Op[3] << ", ";
      } else if (Inst->hasAttr(InstAttr::down)) {
        // to handle "shfl.down.b32 %0, %1, %2, %3;"
        OS() << "shift_sub_group_left(" << Op[3] << ", ";
      } else if (Inst->hasAttr(InstAttr::idx)) {
        // to handle "shfl.sync.idx.b32 %0, %1, %2, %3, %4;"
        OS() << "select_from_sub_group(" << Op[3] << ", ";
      } else if (Inst->hasAttr(InstAttr::bfly)) {
        // to handle "shfl.bfly.b32 %0, %1, %2, %3;"
        OS() << "permute_sub_group_by_xor(" << Op[3] << ", ";
      }

      OS() << DpctGlobalInfo::getItem(GAS) << ".get_sub_group(), " << Op[0]
           << ", " << Op[1] << ")";
    } else {
      llvm::StringRef Str =
          ". You can specify "
          "\"--use-experimental-features=masked-sub-group-operation\" to "
          "use the experimental helper function to migrate inline asm "
          "instruction ";

      auto CommonStr = llvm::Twine(Str)
                           .concat("\"")
                           .concat(GAS->getAsmString()->getString())
                           .concat("\"")
                           .str();

      if (Inst->hasAttr(InstAttr::up)) {
        // to handle "shfl.sync.up.b32 %0, %1, %2, %3, %4;"
        report(Diagnostics::MASK_UNSUPPORTED, true,
               llvm::Twine("shift_sub_group_right").concat(CommonStr).str());
        OS() << "shift_sub_group_right(";
      } else if (Inst->hasAttr(InstAttr::down)) {
        // to handle "shfl.sync.down.b32 %0, %1, %2, %3, %4;"
        report(Diagnostics::MASK_UNSUPPORTED, true,
               llvm::Twine("shift_sub_group_left").concat(CommonStr).str());
        OS() << "shift_sub_group_left(";
      } else if (Inst->hasAttr(InstAttr::idx)) {
        // to handle "shfl.sync.idx.b32 %0, %1, %2, %3, %4;"
        report(Diagnostics::MASK_UNSUPPORTED, true,
               llvm::Twine("select_from_sub_group").concat(CommonStr).str());
        OS() << "select_from_sub_group(";
      } else if (Inst->hasAttr(InstAttr::bfly)) {
        // to handle "shfl.sync.bfly.b32 %0, %1, %2, %3, %4;"
        report(Diagnostics::MASK_UNSUPPORTED, true,
               llvm::Twine("permute_sub_group_by_xor").concat(CommonStr).str());
        OS() << "permute_sub_group_by_xor(";
      }

      OS() << DpctGlobalInfo::getItem(GAS) << ".get_sub_group(), " << Op[0]
           << ", " << Op[1] << ")";
    }

    endstmt();
    return SYCLGenSuccess();
  }

  bool HandleShfl(const InlineAsmInstruction *Inst) {
    if (emitStmt(Inst->getOutputOperand()))
      return SYCLGenError();
    OS() << " = ";

    std::string Op[3];
    if (tryEmitAllInputOperands(Op, Inst))
      return SYCLGenError();

    OS() << MapNames::getDpctNamespace();
    if (Inst->hasAttr(InstAttr::up)) {
      // to handle "shfl.up.b32 %0, %1, %2, %3;"
      OS() << "shift_sub_group_right(";
    } else if (Inst->hasAttr(InstAttr::down)) {
      // to handle "shfl.down.b32 %0, %1, %2, %3;"
      OS() << "shift_sub_group_left(";
    } else if (Inst->hasAttr(InstAttr::idx)) {
      // to handle "shfl.idx.b32 %0, %1, %2, %3;"
      OS() << "select_from_sub_group(";
    } else if (Inst->hasAttr(InstAttr::bfly)) {
      // to handle "shfl.bfly.b32 %0, %1, %2, %3;"
      OS() << "permute_sub_group_by_xor(";
    }

    OS() << DpctGlobalInfo::getItem(GAS) << ".get_sub_group(), " << Op[0]
         << ", " << Op[1] << ")";

    endstmt();
    return SYCLGenSuccess();
  }

  bool handle_shfl(const InlineAsmInstruction *Inst) override {

    if (DpctGlobalInfo::useSYCLCompat()) {
      report(Diagnostics::UNSUPPORT_SYCLCOMPAT, /*UseTextBegin=*/true,
             GAS->getAsmString()->getString());
      cutOffMigration();
      return SYCLGenSuccess();
    }

    if (Inst->getNumInputOperands() == 4 && Inst->getNumTypes() == 1 &&
        Inst->hasAttr(InstAttr::sync)) {
      return HandleShflSync(Inst);
    }

    if (Inst->getNumInputOperands() == 3 && Inst->getNumTypes() == 1 &&
        !Inst->hasAttr(InstAttr::sync)) {
      return HandleShfl(Inst);
    }

    return SYCLGenError();
  }

  bool handle_fence(const InlineAsmInstruction *Inst) override {
    if (Inst->getNumInputOperands() != 0)
      return SYCLGenError();

    OS() << MapNames::getClNamespace() << "atomic_fence(";
    if (Inst->hasAttr(InstAttr::sc) && Inst->hasAttr(InstAttr::cta)) {
      OS() << MapNames::getClNamespace() << "memory_order::seq_cst,"
           << MapNames::getClNamespace() << "memory_scope::work_group";
    } else if (Inst->hasAttr(InstAttr::sc) && Inst->hasAttr(InstAttr::gpu)) {
      OS() << MapNames::getClNamespace() << "memory_order::seq_cst,"
           << MapNames::getClNamespace() << "memory_scope::device";
    } else if (Inst->hasAttr(InstAttr::sc) && Inst->hasAttr(InstAttr::sys)) {
      OS() << MapNames::getClNamespace() << "memory_order::seq_cst,"
           << MapNames::getClNamespace() << "memory_scope::system";
    } else if (Inst->hasAttr(InstAttr::acq_rel) &&
               Inst->hasAttr(InstAttr::cta)) {
      OS() << MapNames::getClNamespace() << "memory_order::acq_rel,"
           << MapNames::getClNamespace() << "memory_scope::work_group";

    } else if (Inst->hasAttr(InstAttr::acq_rel) &&
               Inst->hasAttr(InstAttr::gpu)) {
      OS() << MapNames::getClNamespace() << "memory_order::acq_rel,"
           << MapNames::getClNamespace() << "memory_scope::device";
    } else if (Inst->hasAttr(InstAttr::acq_rel) &&
               Inst->hasAttr(InstAttr::sys)) {
      OS() << MapNames::getClNamespace() << "memory_order::acq_rel,"
           << MapNames::getClNamespace() << "memory_scope::system";
    } else {
      return SYCLGenError();
    }

    OS() << ')';
    endstmt();
    return SYCLGenSuccess();
  }

  bool handle_sub(const InlineAsmInstruction *Inst) override {
    return HandleAddSub(Inst);
  }

  bool handle_membar(const InlineAsmInstruction *Inst) override {
    if (Inst->getNumInputOperands() != 0)
      return SYCLGenError();

    OS() << MapNames::getClNamespace() << "atomic_fence("
         << MapNames::getClNamespace() << "memory_order::seq_cst,";

    if (Inst->hasAttr(InstAttr::cta)) {
      OS() << MapNames::getClNamespace() << "memory_scope::work_group";
    } else if (Inst->hasAttr(InstAttr::gl)) {
      OS() << MapNames::getClNamespace() << "memory_scope::device";
    } else if (Inst->hasAttr(InstAttr::sys)) {
      OS() << MapNames::getClNamespace() << "memory_scope::system";
    } else {
      return SYCLGenError();
    }

    OS() << ')';
    endstmt();
    return SYCLGenSuccess();
  }

  StringRef GetWiderTypeAsString(const InlineAsmBuiltinType *Type) const {
    switch (Type->getKind()) {
    case InlineAsmBuiltinType::s16:
      return "int32_t";
    case InlineAsmBuiltinType::u16:
      return "uint32_t";
    case InlineAsmBuiltinType::s32:
      return "int64_t";
    case InlineAsmBuiltinType::u32:
      return "uint64_t";
    default:
      assert(false && "Can not find wide type");
    }
    return "";
  }

  bool CheckMulMadType(const InlineAsmBuiltinType *Type) const {
    return Type->getKind() == InlineAsmBuiltinType::s16 ||
           Type->getKind() == InlineAsmBuiltinType::u16 ||
           Type->getKind() == InlineAsmBuiltinType::s32 ||
           Type->getKind() == InlineAsmBuiltinType::u32 ||
           Type->getKind() == InlineAsmBuiltinType::s64 ||
           Type->getKind() == InlineAsmBuiltinType::u64;
  }

  std::string Cast(StringRef Type, StringRef Op) {
    return llvm::Twine("(").concat(Type).concat(")").concat(Op).str();
  }

  std::string Cast(const InlineAsmType *DestType, const InlineAsmType *OpType,
                   const std::string &Op) {
    const auto *DT = llvm::dyn_cast_if_present<InlineAsmBuiltinType>(DestType);
    const auto *OT = llvm::dyn_cast_if_present<InlineAsmBuiltinType>(OpType);
    if (!DT || !OT)
      return Op;
    std::string DestTypeStr;
    if (tryEmitType(DestTypeStr, DestType))
      return Op;
    return Cast(DestTypeStr, Op);
  }

  bool handle_mul(const InlineAsmInstruction *Inst) override {
    if (Inst->getNumInputOperands() != 2 || Inst->getNumTypes() != 1)
      return SYCLGenError();
    const auto *Type = dyn_cast<InlineAsmBuiltinType>(Inst->getType(0));

    if (!Type || !CheckMulMadType(Type))
      return SYCLGenError();

    // Can not use .wide attr on 64-bit integer.
    if (Inst->hasAttr(InstAttr::wide) &&
        (Type->getKind() == InlineAsmBuiltinType::s64 ||
         Type->getKind() == InlineAsmBuiltinType::u64))
      return SYCLGenError();

    if (emitStmt(Inst->getOutputOperand()))
      return SYCLGenError();

    OS() << " = ";

    std::string Op[2];
    for (unsigned I = 0; I < Inst->getNumInputOperands(); ++I) {
      if (tryEmitStmt(Op[I], Inst->getInputOperand(I)))
        return SYCLGenError();
      if (Inst->hasAttr(InstAttr::hi))
        Op[I] =
            Cast(Inst->getType(0), Inst->getInputOperand(I)->getType(), Op[I]);
    }

    // mul.hi
    if (Inst->hasAttr(InstAttr::hi)) {
      OS() << MapNames::getClNamespace() << "mul_hi(" << Op[0] << ", " << Op[1]
           << ")";
      // mul.wide
    } else if (Inst->hasAttr(InstAttr::wide)) {
      OS() << Cast(GetWiderTypeAsString(Type), Op[0]) << " * "
           << Cast(GetWiderTypeAsString(Type), Op[1]);
      // mul.lo
    } else {
      // Need to add a new help function.
      // OS() << Op[0] << " * " << Op[1];
      return SYCLGenError();
    }

    endstmt();
    return SYCLGenSuccess();
  }

  bool handle_mad(const InlineAsmInstruction *Inst) override {
    if (Inst->getNumInputOperands() != 3 || Inst->getNumTypes() != 1)
      return SYCLGenError();
    const auto *Type = dyn_cast<InlineAsmBuiltinType>(Inst->getType(0));

    if (!Type || !CheckMulMadType(Type))
      return SYCLGenError();

    // Can not use .wide attr on 64-bit integer.
    if (Inst->hasAttr(InstAttr::wide) &&
        (Type->getKind() == InlineAsmBuiltinType::s64 ||
         Type->getKind() == InlineAsmBuiltinType::u64))
      return SYCLGenError();

    // The attribute .sat only work with .hi mode.
    if (Inst->hasAttr(InstAttr::sat) &&
        (!Inst->hasAttr(InstAttr::hi) ||
         Type->getKind() != InlineAsmBuiltinType::s32))
      return SYCLGenError();

    if (emitStmt(Inst->getOutputOperand()))
      return SYCLGenError();

    OS() << " = ";

    std::string Op[3];
    for (unsigned I = 0; I < Inst->getNumInputOperands(); ++I) {
      if (tryEmitStmt(Op[I], Inst->getInputOperand(I)))
        return SYCLGenError();
      if (Inst->hasAttr(InstAttr::sat, InstAttr::hi))
        Op[I] =
            Cast(Inst->getType(0), Inst->getInputOperand(I)->getType(), Op[I]);
    }

    // mad.hi.sat
    if (Inst->hasAttr(InstAttr::sat))
      OS() << MapNames::getClNamespace() << "add_sat("
           << MapNames::getClNamespace() << "mul_hi(" << Op[0] << ", " << Op[1]
           << "), " << Op[2] << ")";
    // mad.hi
    else if (Inst->hasAttr(InstAttr::hi))
      OS() << MapNames::getClNamespace() << "mad_hi(" << Op[0] << ", " << Op[1]
           << ", " << Op[2] << ")";
    // mad.wide
    else if (Inst->hasAttr(InstAttr::wide))
      OS() << llvm::formatv("({3}){0} * ({3}){1} + ({3}){2}", Op[0], Op[1],
                            Op[2], GetWiderTypeAsString(Type));
    // mad.lo
    else {
      // Need to add a new help function.
      // OS() << Op[0] << " * " << Op[1] << " + " << Op[2];
      return SYCLGenError();
    }

    endstmt();
    return SYCLGenSuccess();
  }

  bool HandleDivRem(const InlineAsmInstruction *Inst, char Operator) {
    if (Inst->getNumInputOperands() != 2 || Inst->getNumTypes() != 1)
      return SYCLGenError();
    if (emitStmt(Inst->getOutputOperand()))
      return SYCLGenError();
    OS() << " = ";

    std::string Operand[2];
    for (unsigned I = 0; I < Inst->getNumInputOperands(); ++I) {
      if (tryEmitStmt(Operand[I], Inst->getInputOperand(I)))
        return SYCLGenError();
    }
    OS() << Operand[0] << ' ' << Operator << ' ' << Operand[1];
    endstmt();
    return SYCLGenSuccess();
  }

  bool handle_div(const InlineAsmInstruction *Inst) override {
    return HandleDivRem(Inst, '/');
  }

  bool handle_rem(const InlineAsmInstruction *Inst) override {
    return HandleDivRem(Inst, '%');
  }

  bool HandleMul24Mad24(const InlineAsmInstruction *Inst, bool isMad) {
    if (Inst->getNumInputOperands() != 2U + isMad || Inst->getNumTypes() != 1)
      return SYCLGenError();

    const auto *Type = dyn_cast<InlineAsmBuiltinType>(Inst->getType(0));

    if (!Type || !CheckMulMadType(Type))
      return SYCLGenError();

    // hi unsupport
    if (Inst->hasAttr(InstAttr::hi))
      return SYCLGenError();

    if (emitStmt(Inst->getOutputOperand()))
      return SYCLGenError();

    OS() << " = ";

    std::string Op[3];
    for (unsigned I = 0; I < Inst->getNumInputOperands(); ++I) {
      if (tryEmitStmt(Op[I], Inst->getInputOperand(I)))
        return SYCLGenError();
      Op[I] =
          Cast(Inst->getType(0), Inst->getInputOperand(I)->getType(), Op[I]);
    }

    if (isMad)
      OS() << MapNames::getClNamespace() << "mad24(" << Op[0] << ", " << Op[1]
           << ", " << Op[2] << ")";
    else
      OS() << MapNames::getClNamespace() << "mul24(" << Op[0] << ", " << Op[1]
           << ")";

    endstmt();
    return SYCLGenSuccess();
  }

  bool handle_mul24(const InlineAsmInstruction *Inst) override {
    return HandleMul24Mad24(Inst, /*isMad=*/false);
  }

  bool handle_mad24(const InlineAsmInstruction *Inst) override {
    return HandleMul24Mad24(Inst, /*isMad=*/true);
  }

  bool HandleAbsNeg(const InlineAsmInstruction *Inst) {
    if (Inst->getNumInputOperands() != 1 || Inst->getNumTypes() != 1)
      return SYCLGenError();

    const auto *Type = dyn_cast<InlineAsmBuiltinType>(Inst->getType(0));
    if (!Type)
      return SYCLGenError();

    if (Type->getKind() != InlineAsmBuiltinType::s16 &&
        Type->getKind() != InlineAsmBuiltinType::s32 &&
        Type->getKind() != InlineAsmBuiltinType::s64) {
      return SYCLGenError();
    }

    if (emitStmt(Inst->getOutputOperand()))
      return SYCLGenError();

    OS() << " = ";

    std::string Op;
    if (tryEmitStmt(Op, Inst->getInputOperand(0)))
      return SYCLGenError();

    std::string TypeStr;
    if (tryEmitType(TypeStr, Type))
      return SYCLGenError();

    if (Inst->is(asmtok::op_abs))
      OS() << MapNames::getClNamespace() << "abs(" << Op << ")";
    else
      OS() << "-" << Op;

    endstmt();
    return SYCLGenSuccess();
  }

  bool handle_abs(const InlineAsmInstruction *Inst) override {
    return HandleAbsNeg(Inst);
  }

  bool handle_neg(const InlineAsmInstruction *Inst) override {
    return HandleAbsNeg(Inst);
  }

  bool HandlePopcClz(const InlineAsmInstruction *Inst) {
    if (Inst->getNumInputOperands() != 1 || Inst->getNumTypes() != 1)
      return SYCLGenError();

    const auto *Type = dyn_cast<InlineAsmBuiltinType>(Inst->getType(0));
    if (!Type || (Type->getKind() != InlineAsmBuiltinType::b32 &&
                  Type->getKind() != InlineAsmBuiltinType::b64))
      return SYCLGenError();

    if (emitStmt(Inst->getOutputOperand()))
      return SYCLGenError();
    OS() << " = ";

    std::string TypeRepl, OpRepl;
    if (tryEmitStmt(OpRepl, Inst->getInputOperand(0)) ||
        tryEmitType(TypeRepl, Inst->getType(0)))
      return SYCLGenError();

    if (Inst->is(asmtok::op_popc))
      OS() << MapNames::getClNamespace() << "popcount<" << TypeRepl << ">("
           << OpRepl << ")";
    else
      OS() << MapNames::getClNamespace() << "clz<" << TypeRepl << ">(" << OpRepl
           << ")";

    endstmt();
    return SYCLGenSuccess();
  }

  bool handle_popc(const InlineAsmInstruction *Inst) override {
    return HandlePopcClz(Inst);
  }

  bool handle_clz(const InlineAsmInstruction *Inst) override {
    return HandlePopcClz(Inst);
  }

  bool HandleMinMax(const InlineAsmInstruction *Inst, StringRef Fn) {
    if (Inst->getNumInputOperands() != 2 || Inst->getNumTypes() != 1)
      return SYCLGenError();

    bool isVec = false;
    if (!CheckAddSubMinMaxType(Inst, isVec))
      return SYCLGenError();

    if (emitStmt(Inst->getOutputOperand()))
      return SYCLGenError();
    OS() << " = ";

    std::string Op[2];
    for (unsigned I = 0; I < Inst->getNumInputOperands(); ++I) {
      if (tryEmitStmt(Op[I], Inst->getInputOperand(I)))
        return SYCLGenError();
      Op[I] =
          Cast(Inst->getType(0), Inst->getInputOperand(I)->getType(), Op[I]);
    }

    std::string Base = llvm::Twine(Fn)
                           .concat("(")
                           .concat(Op[0])
                           .concat(", ")
                           .concat(Op[1])
                           .concat(")")
                           .str();

    if (Inst->hasAttr(InstAttr::relu))
      OS() << MapNames::getDpctNamespace() << "relu(" << Base << ")";
    else
      OS() << Base;

    endstmt();
    return SYCLGenSuccess();
  }

  bool handle_min(const InlineAsmInstruction *Inst) override {
    return HandleMinMax(Inst, MapNames::getClNamespace() + "min");
  }

  bool handle_max(const InlineAsmInstruction *Inst) override {
    return HandleMinMax(Inst, MapNames::getClNamespace() + "max");
  }

  bool HandleBitwiseBinaryOp(const InlineAsmInstruction *Inst,
                             StringRef Operator) {
    if (Inst->getNumInputOperands() != 2 || Inst->getNumTypes() != 1)
      return SYCLGenError();
    bool IsShift = Inst->is(asmtok::op_shl, asmtok::op_shr);
    const auto *BI = dyn_cast<InlineAsmBuiltinType>(Inst->getType(0));
    if (!BI || (BI->getKind() != InlineAsmBuiltinType::b16 &&
                BI->getKind() != InlineAsmBuiltinType::b32 &&
                BI->getKind() != InlineAsmBuiltinType::b64 &&
                (IsShift && BI->getKind() == InlineAsmBuiltinType::pred)))
      return SYCLGenError();

    if (emitStmt(Inst->getOutputOperand()))
      return SYCLGenError();
    OS() << " = ";

    std::string Operand[2];
    for (unsigned I = 0; I < Inst->getNumInputOperands(); ++I)
      if (tryEmitStmt(Operand[I], Inst->getInputOperand(I)))
        return SYCLGenError();

    OS() << Operand[0] << ' ' << Operator << ' ' << Operand[1];

    endstmt();
    return SYCLGenSuccess();
  }

  bool handle_and(const InlineAsmInstruction *Inst) override {
    return HandleBitwiseBinaryOp(Inst, "&");
  }

  bool handle_or(const InlineAsmInstruction *Inst) override {
    return HandleBitwiseBinaryOp(Inst, "|");
  }

  bool handle_xor(const InlineAsmInstruction *Inst) override {
    return HandleBitwiseBinaryOp(Inst, "^");
  }

  bool handle_shl(const InlineAsmInstruction *Inst) override {
    return HandleBitwiseBinaryOp(Inst, "<<");
  }

  bool handle_shr(const InlineAsmInstruction *Inst) override {
    return HandleBitwiseBinaryOp(Inst, ">>");
  }

  bool HandleNot(const InlineAsmInstruction *Inst) {
    std::string TypeRepl;
    if (tryEmitType(TypeRepl, Inst->getType(0)))
      return SYCLGenError();

    if (Inst->getNumInputOperands() != 1 || Inst->getNumTypes() != 1)
      return SYCLGenError();

    const auto *BI = dyn_cast<InlineAsmBuiltinType>(Inst->getType(0));
    if (!BI || (BI->getKind() != InlineAsmBuiltinType::b16 &&
                BI->getKind() != InlineAsmBuiltinType::b32 &&
                BI->getKind() != InlineAsmBuiltinType::b64 &&
                (Inst->is(asmtok::op_cnot) &&
                 BI->getKind() == InlineAsmBuiltinType::pred)))
      return SYCLGenError();

    if (emitStmt(Inst->getOutputOperand()))
      return SYCLGenError();
    OS() << " = ";

    std::string Op;
    if (tryEmitStmt(Op, Inst->getInputOperand(0)))
      return SYCLGenError();

    if (Inst->is(asmtok::op_not)) {
      OS() << llvm::formatv("~{0}", Op);
    } else {
      OS() << llvm::formatv("{0} == 0", Op);
    }

    endstmt();
    return SYCLGenSuccess();
  }

  bool handle_not(const InlineAsmInstruction *Inst) override {
    return HandleNot(Inst);
  }

  bool handle_cnot(const InlineAsmInstruction *Inst) override {
    return HandleNot(Inst);
  }

  bool HandleSinCosTanhSqrtLg2Ex2(const InlineAsmInstruction *Inst,
                                  StringRef MathFn) {
    if (Inst->getNumInputOperands() != 1)
      return SYCLGenError();
    if (emitStmt(Inst->getOutputOperand()))
      return SYCLGenError();
    OS() << " = ";
    std::string Op;
    if (tryEmitStmt(Op, Inst->getInputOperand(0)))
      return SYCLGenError();
    if (needBitCast(Inst->getInputOperand(0)->getType(), Inst->getType(0)) &&
        emitBitCast(Inst->getInputOperand(0)->getType(), Inst->getType(0), Op))
      return SYCLGenError();
    std::string ReplaceString = MapNames::getClNamespace() + MathFn.str() + '(';
    if (Inst->getOpcode() == asmtok::op_ex2)
      ReplaceString += "2, ";
    ReplaceString += Op + ")";
    if (Inst->hasAttr(InstAttr::rn, InstAttr::rz, InstAttr::rm, InstAttr::rp))
      report(Diagnostics::ROUNDING_MODE_UNSUPPORTED, true);
    if (needBitCast(Inst->getType(0), Inst->getOutputOperand()->getType()) &&
        emitBitCast(Inst->getType(0), Inst->getOutputOperand()->getType(),
                    ReplaceString))
      return SYCLGenError();
    OS() << ReplaceString;
    endstmt();
    return SYCLGenSuccess();
  }

  bool handle_cos(const InlineAsmInstruction *Inst) override {
    return HandleSinCosTanhSqrtLg2Ex2(Inst, "cos");
  }

  bool handle_sin(const InlineAsmInstruction *Inst) override {
    return HandleSinCosTanhSqrtLg2Ex2(Inst, "sin");
  }

  bool handle_tanh(const InlineAsmInstruction *Inst) override {
    return HandleSinCosTanhSqrtLg2Ex2(Inst, "tanh");
  }

  bool handle_sqrt(const InlineAsmInstruction *Inst) override {
    return HandleSinCosTanhSqrtLg2Ex2(Inst, "sqrt");
  }

  bool handle_rsqrt(const InlineAsmInstruction *Inst) override {
    return HandleSinCosTanhSqrtLg2Ex2(Inst, "rsqrt");
  }

  bool handle_lg2(const InlineAsmInstruction *Inst) override {
    return HandleSinCosTanhSqrtLg2Ex2(Inst, "log2");
  }

  bool handle_ex2(const InlineAsmInstruction *Inst) override {
    return HandleSinCosTanhSqrtLg2Ex2(Inst, "pow");
  }

  bool handle_sad(const InlineAsmInstruction *Inst) override {
    if (Inst->getNumInputOperands() != 3 && Inst->getNumTypes() != 0)
      return SYCLGenError();
    if (emitStmt(Inst->getOutputOperand()))
      return SYCLGenError();
    OS() << " = ";
    std::string Op[3];
    for (int i = 0; i < 3; ++i)
      if (tryEmitStmt(Op[i], Inst->getInputOperand(i)))
        return SYCLGenError();

    OS() << MapNames::getClNamespace() << "abs_diff(" << Op[0] << ", " << Op[1]
         << ") + " << Op[2];
    endstmt();
    return SYCLGenSuccess();
  }

  bool handle_testp(const InlineAsmInstruction *Inst) override {
    if (Inst->getNumInputOperands() != 1)
      return SYCLGenError();
    if (emitStmt(Inst->getOutputOperand()))
      return SYCLGenError();
    OS() << " = ";

    if (Inst->hasAttr(InstAttr::finite))
      OS() << MapNames::getClNamespace() << "isfinite(";
    else if (Inst->hasAttr(InstAttr::infinite))
      OS() << MapNames::getClNamespace() << "isinf(";
    else if (Inst->hasAttr(InstAttr::number))
      OS() << "!" << MapNames::getClNamespace() << "isnan(";
    else if (Inst->hasAttr(InstAttr::notanumber))
      OS() << MapNames::getClNamespace() << "isnan(";
    else if (Inst->hasAttr(InstAttr::normal))
      OS() << MapNames::getClNamespace() << "isnormal(";
    else if (Inst->hasAttr(InstAttr::subnormal))
      OS() << "!" << MapNames::getClNamespace() << "isnormal(";
    else
      return SYCLGenError();

    if (emitStmt(Inst->getInputOperand(0)))
      return SYCLGenError();
    OS() << ')';
    endstmt();
    return SYCLGenSuccess();
  }

  bool handle_selp(const InlineAsmInstruction *Inst) override {
    if (Inst->getNumInputOperands() != 3)
      return SYCLGenError();
    if (emitStmt(Inst->getOutputOperand()))
      return SYCLGenError();
    OS() << " = ";
    std::string Op[3];
    for (int i = 0; i < 3; ++i)
      if (tryEmitStmt(Op[i], Inst->getInputOperand(i)))
        return SYCLGenError();
    OS() << Op[2] << " == 1 ? " << Op[0] << " : " << Op[1];
    endstmt();
    return SYCLGenSuccess();
  }

  bool handle_copysign(const InlineAsmInstruction *Inst) override {
    if (Inst->getNumInputOperands() != 2)
      return SYCLGenError();
    if (emitStmt(Inst->getOutputOperand()))
      return SYCLGenError();
    OS() << " = ";
    std::string Op[2];
    for (int i = 0; i < 2; ++i)
      if (tryEmitStmt(Op[i], Inst->getInputOperand(i)))
        return SYCLGenError();
    OS() << MapNames::getClNamespace() << "copysign(" << Op[1] << ", " << Op[0]
         << ')';
    endstmt();
    insertHeader(HeaderType::HT_Math);
    return SYCLGenSuccess();
  }

  // The type of operands must be one of s32/u32.
  bool CheckSIMDInstructionType(const InlineAsmInstruction *Inst) {
    for (const auto *T : Inst->types()) {
      if (const auto *BI = dyn_cast<InlineAsmBuiltinType>(T)) {
        if (BI->getKind() == InlineAsmBuiltinType::s32 ||
            BI->getKind() == InlineAsmBuiltinType::u32)
          continue;
      }
      return SYCLGenError();
    }
    return SYCLGenSuccess();
  }

  bool HandleComparsionOp(const InlineAsmInstruction *Inst) {
    if (!Inst)
      return SYCLGenError();
    if (Inst->hasAttr(InstAttr::eq))
      OS() << ", "
           << "std::equal_to<>()";
    else if (Inst->hasAttr(InstAttr::ne))
      OS() << ", "
           << "std::not_equal_to<>()";
    else if (Inst->hasAttr(InstAttr::lt))
      OS() << ", "
           << "std::less<>()";
    else if (Inst->hasAttr(InstAttr::le))
      OS() << ", "
           << "std::less_equal<>()";
    else if (Inst->hasAttr(InstAttr::gt))
      OS() << ", "
           << "std::greater<>()";
    else if (Inst->hasAttr(InstAttr::ge))
      OS() << ", "
           << "std::greater_equal<>()";
    else
      return SYCLGenError();
    return SYCLGenSuccess();
  }

  // Handle the 1 element vadd/vsub/vmin/vmax/vabsdiff video instructions.
  bool HandleOneElementAddSubMinMax(const InlineAsmInstruction *Inst,
                                    StringRef Fn) {
    if (Inst->getNumInputOperands() < 2 || Inst->getNumTypes() != 3 ||
        CheckSIMDInstructionType(Inst))
      return SYCLGenError();

    // Arguments mismatch for instruction, which has a secondary arithmetic
    // operation.
    if (Inst->hasAttr(InstAttr::add, InstAttr::min, InstAttr::max) &&
        Inst->getNumInputOperands() < 3)
      return SYCLGenError();

    if (emitStmt(Inst->getOutputOperand()))
      return SYCLGenError();

    OS() << " = " << Fn;
    if (Inst->hasAttr(InstAttr::sat))
      OS() << "_sat";
    if (Inst->is(asmtok::op_vshl, asmtok::op_vshr))
      OS() << (Inst->hasAttr(InstAttr::clamp) ? "_clamp" : "_wrap");
    OS() << "<";
    if (emitType(Inst->getType(0)))
      return SYCLGenError();
    OS() << ">(";

    std::string Op[3];
    for (unsigned I = 0; I < Inst->getNumInputOperands(); ++I)
      if (tryEmitStmt(Op[I], Inst->getInputOperand(I)))
        return SYCLGenError();

    OS() << llvm::join(ArrayRef(Op, Inst->getNumInputOperands()), ", ");
    // The secondary arithmetic operation.
    if (Inst->hasAttr(InstAttr::add))
      OS() << ", " << MapNames::getClNamespace() << "plus<>()";
    else if (Inst->hasAttr(InstAttr::min))
      OS() << ", " << MapNames::getClNamespace() << "minimum<>()";
    else if (Inst->hasAttr(InstAttr::max))
      OS() << ", " << MapNames::getClNamespace() << "maximum<>()";

    OS() << ")";
    endstmt();
    insertHeader(HeaderType::HT_DPCT_Math);
    return SYCLGenSuccess();
  }

  bool handle_vadd(const InlineAsmInstruction *I) override {
    return HandleOneElementAddSubMinMax(I, MapNames::getDpctNamespace() +
                                               "extend_add");
  }

  bool handle_vsub(const InlineAsmInstruction *I) override {
    return HandleOneElementAddSubMinMax(I, MapNames::getDpctNamespace() +
                                               "extend_sub");
  }

  bool handle_vabsdiff(const InlineAsmInstruction *I) override {
    return HandleOneElementAddSubMinMax(I, MapNames::getDpctNamespace() +
                                               "extend_absdiff");
  }

  bool handle_vmax(const InlineAsmInstruction *I) override {
    return HandleOneElementAddSubMinMax(I, MapNames::getDpctNamespace() +
                                               "extend_max");
  }

  bool handle_vmin(const InlineAsmInstruction *I) override {
    return HandleOneElementAddSubMinMax(I, MapNames::getDpctNamespace() +
                                               "extend_min");
  }

  bool handle_vshl(const InlineAsmInstruction *I) override {
    return HandleOneElementAddSubMinMax(I, MapNames::getDpctNamespace() +
                                               "extend_shl");
  }

  bool handle_vshr(const InlineAsmInstruction *I) override {
    return HandleOneElementAddSubMinMax(I, MapNames::getDpctNamespace() +
                                               "extend_shr");
  }

  bool HandleVset(const InlineAsmInstruction *I, StringRef Fn) {
    if (DpctGlobalInfo::useSYCLCompat()) {
      report(Diagnostics::UNSUPPORT_SYCLCOMPAT, /*UseTextBegin=*/true,
             GAS->getAsmString()->getString());
      cutOffMigration();
      return SYCLGenSuccess();
    }
    if (I->getNumInputOperands() < 2 || I->getNumTypes() != 2 ||
        CheckSIMDInstructionType(I))
      return SYCLGenError();
    bool hasSecOp = I->hasAttr(InstAttr::add, InstAttr::min, InstAttr::max);
    if (hasSecOp && I->getNumInputOperands() < 3)
      return SYCLGenError();
    if (emitStmt(I->getOutputOperand()))
      return SYCLGenError();
    OS() << " = " << MapNames::getDpctNamespace() << Fn;
    if (I->is(asmtok::op_vset2, asmtok::op_vset4) && I->hasAttr(InstAttr::add))
      OS() << "_add";
    OS() << '<';
    for (int i = 0, e = I->getNumTypes(); i != e; ++i) {
      if (emitType(I->getType(i)))
        return SYCLGenError();
      if (i < e - 1)
        OS() << ", ";
    }
    OS() << '>' << '(';
    unsigned NumInputOp = I->getNumInputOperands();
    if (NumInputOp >= 3 && !hasSecOp)
      NumInputOp--;
    // If no second op, ignore third operand until we support operand mask.
    for (unsigned i = 0; i != NumInputOp; ++i) {
      if (emitStmt(I->getInputOperand(i)))
        return SYCLGenError();
      if (i < NumInputOp - 1)
        OS() << ", ";
    }
    if (HandleComparsionOp(I))
      return SYCLGenError();
    if (I->is(asmtok::op_vset)) {
      if (I->hasAttr(InstAttr::add))
        OS() << ", " << MapNames::getClNamespace() << "plus<>()";
      else if (I->hasAttr(InstAttr::min))
        OS() << ", " << MapNames::getClNamespace() << "minimum<>()";
      else if (I->hasAttr(InstAttr::max))
        OS() << ", " << MapNames::getClNamespace() << "maximum<>()";
    }
    OS() << ')';
    endstmt();
    return SYCLGenSuccess();
  }

  bool handle_vset(const InlineAsmInstruction *I) override {
    return HandleVset(I, "extend_compare");
  }

  // Handle the 2/4 element video instructions.
  bool handleMultiElementAddSubMinMax(const InlineAsmInstruction *Inst,
                                      StringRef Fn) {
    if (Inst->getNumInputOperands() < 3 || Inst->getNumTypes() != 3 ||
        CheckSIMDInstructionType(Inst))
      return SYCLGenError();
    if (emitStmt(Inst->getOutputOperand()))
      return SYCLGenError();

    OS() << " = " << Fn;
    if (Inst->hasAttr(InstAttr::sat))
      OS() << "_sat";
    if (Inst->hasAttr(InstAttr::add))
      OS() << "_add";
    OS() << "<";
    if (emitType(Inst->getType(0)))
      return SYCLGenError();
    OS() << ", ";
    if (emitType(Inst->getType(1)))
      return SYCLGenError();
    OS() << ", ";
    if (emitType(Inst->getType(2)))
      return SYCLGenError();
    OS() << ">(";

    std::string Op[3];
    for (unsigned I = 0; I < Inst->getNumInputOperands(); ++I)
      if (tryEmitStmt(Op[I], Inst->getInputOperand(I)))
        return SYCLGenError();

    OS() << llvm::join(ArrayRef(Op, Inst->getNumInputOperands()), ", ");

    OS() << ")";
    endstmt();
    insertHeader(HeaderType::HT_DPCT_Math);
    return SYCLGenSuccess();
  }

  bool handle_vadd2(const InlineAsmInstruction *I) override {
    return handleMultiElementAddSubMinMax(I, MapNames::getDpctNamespace() +
                                                 "extend_vadd2");
  }
  bool handle_vsub2(const InlineAsmInstruction *I) override {
    return handleMultiElementAddSubMinMax(I, MapNames::getDpctNamespace() +
                                                 "extend_vsub2");
  }
  bool handle_vabsdiff2(const InlineAsmInstruction *I) override {
    return handleMultiElementAddSubMinMax(I, MapNames::getDpctNamespace() +
                                                 "extend_vabsdiff2");
  }
  bool handle_vmin2(const InlineAsmInstruction *I) override {
    return handleMultiElementAddSubMinMax(I, MapNames::getDpctNamespace() +
                                                 "extend_vmin2");
  }
  bool handle_vmax2(const InlineAsmInstruction *I) override {
    return handleMultiElementAddSubMinMax(I, MapNames::getDpctNamespace() +
                                                 "extend_vmax2");
  }
  bool handle_vavrg2(const InlineAsmInstruction *I) override {
    return handleMultiElementAddSubMinMax(I, MapNames::getDpctNamespace() +
                                                 "extend_vavrg2");
  }
  bool handle_vset2(const InlineAsmInstruction *I) override {
    return HandleVset(I, "extend_vcompare2");
  }
  bool handle_vadd4(const InlineAsmInstruction *I) override {
    return handleMultiElementAddSubMinMax(I, MapNames::getDpctNamespace() +
                                                 "extend_vadd4");
  }
  bool handle_vsub4(const InlineAsmInstruction *I) override {
    return handleMultiElementAddSubMinMax(I, MapNames::getDpctNamespace() +
                                                 "extend_vsub4");
  }
  bool handle_vabsdiff4(const InlineAsmInstruction *I) override {
    return handleMultiElementAddSubMinMax(I, MapNames::getDpctNamespace() +
                                                 "extend_vabsdiff4");
  }
  bool handle_vmin4(const InlineAsmInstruction *I) override {
    return handleMultiElementAddSubMinMax(I, MapNames::getDpctNamespace() +
                                                 "extend_vmin4");
  }
  bool handle_vmax4(const InlineAsmInstruction *I) override {
    return handleMultiElementAddSubMinMax(I, MapNames::getDpctNamespace() +
                                                 "extend_vmax4");
  }
  bool handle_vavrg4(const InlineAsmInstruction *I) override {
    return handleMultiElementAddSubMinMax(I, MapNames::getDpctNamespace() +
                                                 "extend_vavrg4");
  }
  bool handle_vset4(const InlineAsmInstruction *I) override {
    return HandleVset(I, "extend_vcompare4");
  }

  bool handle_bfe(const InlineAsmInstruction *Inst) override {
    if (Inst->getNumInputOperands() != 3)
      return SYCLGenError();
    const auto *Type = dyn_cast<InlineAsmBuiltinType>(Inst->getType(0));
    if (!Type || (Type->getKind() != InlineAsmBuiltinType::s32 &&
                  Type->getKind() != InlineAsmBuiltinType::s64 &&
                  Type->getKind() != InlineAsmBuiltinType::u32 &&
                  Type->getKind() != InlineAsmBuiltinType::u64))
      return SYCLGenError();
    std::string TypeStr, Op[3];
    if (tryEmitType(TypeStr, Type))
      return SYCLGenError();
    for (int i = 0; i < 3; ++i)
      if (tryEmitStmt(Op[i], Inst->getInputOperand(i)))
        return SYCLGenError();
    if (emitStmt(Inst->getOutputOperand()))
      return SYCLGenError();
    OS() << " = ";
    OS() << MapNames::getDpctNamespace() << "bfe_safe<" << TypeStr << ">(" << Op[0]
         << ", " << Op[1] << ", " << Op[2] << ')';
    endstmt();
    insertHeader(HeaderType::HT_DPCT_Math);
    return SYCLGenSuccess();
  }

  bool handle_bfi(const InlineAsmInstruction *Inst) override {
    if (Inst->getNumInputOperands() != 4)
      return SYCLGenError();
    const auto *Type = dyn_cast<InlineAsmBuiltinType>(Inst->getType(0));
    if (!Type || (Type->getKind() != InlineAsmBuiltinType::b32 &&
                  Type->getKind() != InlineAsmBuiltinType::b64))
      return SYCLGenError();
    std::string TypeStr, Op[4];
    if (tryEmitType(TypeStr, Type))
      return SYCLGenError();
    for (int i = 0; i < 4; ++i)
      if (tryEmitStmt(Op[i], Inst->getInputOperand(i)))
        return SYCLGenError();
    if (emitStmt(Inst->getOutputOperand()))
      return SYCLGenError();
    OS() << " = ";
    OS() << MapNames::getDpctNamespace() << "bfi_safe<" << TypeStr << ">("
         << Op[0] << ", " << Op[1] << ", " << Op[2] << ", " << Op[3] << ')';
    endstmt();
    insertHeader(HeaderType::HT_DPCT_Math);
    return SYCLGenSuccess();
  }

  bool handle_brev(const InlineAsmInstruction *Inst) override {
    if (Inst->getNumInputOperands() != 1)
      return SYCLGenError();
    const auto *Type = dyn_cast<InlineAsmBuiltinType>(Inst->getType(0));
    if (!Type || (Type->getKind() != InlineAsmBuiltinType::b32 &&
                  Type->getKind() != InlineAsmBuiltinType::b64))
      return SYCLGenError();

    std::string TypeStr;
    if (tryEmitType(TypeStr, Type))
      return SYCLGenError();

    if (emitStmt(Inst->getOutputOperand()))
      return SYCLGenError();
    OS() << " = ";
    OS() << MapNames::getDpctNamespace() << "reverse_bits<" << TypeStr << ">(";
    if (emitStmt(Inst->getInputOperand(0)))
      return SYCLGenError();
    OS() << ")";
    endstmt();
    insertHeader(HeaderType::HT_DPCT_Dpct);
    return SYCLGenSuccess();
  }

  bool CheckDotProductAccType(const InlineAsmType *Type) {
    const auto *BIType = dyn_cast<const InlineAsmBuiltinType>(Type);
    if (!BIType || (BIType->getKind() != InlineAsmBuiltinType::s32 &&
                    BIType->getKind() != InlineAsmBuiltinType::u32))
      return SYCLGenError();
    return SYCLGenSuccess();
  }

  bool handle_dp4a(const InlineAsmInstruction *Inst) override {
    if (Inst->getNumInputOperands() != 3 || Inst->getNumTypes() != 2)
      return SYCLGenError();

    if (CheckDotProductAccType(Inst->getType(0)) ||
        CheckDotProductAccType(Inst->getType(1)))
      return SYCLGenError();

    std::string TypeStr[2];
    for (int i = 0; i < 2; ++i)
      if (tryEmitType(TypeStr[i], Inst->getType(i)))
        return SYCLGenError();
    if (emitStmt(Inst->getOutputOperand()))
      return SYCLGenError();
    OS() << " = ";
    OS() << MapNames::getDpctNamespace() << "dp4a<" << TypeStr[0] << ", "
         << TypeStr[1] << ">(";
    std::string Op[3];
    for (int i = 0; i < 3; ++i)
      if (tryEmitStmt(Op[i], Inst->getInputOperand(i)))
        return SYCLGenError();
    OS() << Op[0] << ", " << Op[1] << ", " << Op[2] << ")";
    endstmt();
    insertHeader(HeaderType::HT_DPCT_Math);
    return SYCLGenSuccess();
  }

  bool handle_dp2a(const InlineAsmInstruction *Inst) override {
    if (Inst->getNumInputOperands() != 3 || Inst->getNumTypes() != 2)
      return SYCLGenError();

    if (CheckDotProductAccType(Inst->getType(0)) ||
        CheckDotProductAccType(Inst->getType(1)))
      return SYCLGenError();

    bool lo = Inst->hasAttr(InstAttr::lo);
    bool hi = Inst->hasAttr(InstAttr::hi);
    if (!(lo ^ hi))
      return SYCLGenError();

    std::string TypeStr[2];
    for (int i = 0; i < 2; ++i)
      if (tryEmitType(TypeStr[i], Inst->getType(i)))
        return SYCLGenError();
    if (emitStmt(Inst->getOutputOperand()))
      return SYCLGenError();
    OS() << " = ";
    OS() << MapNames::getDpctNamespace() << "dp2a_";
    if (lo)
      OS() << "lo";
    else
      OS() << "hi";
    OS() << "<" << TypeStr[0] << ", " << TypeStr[1] << ">(";
    std::string Op[3];
    for (int i = 0; i < 3; ++i)
      if (tryEmitStmt(Op[i], Inst->getInputOperand(i)))
        return SYCLGenError();
    OS() << Op[0] << ", " << Op[1] << ", " << Op[2] << ")";
    endstmt();
    insertHeader(HeaderType::HT_DPCT_Math);
    return SYCLGenSuccess();
  }

  bool handle_bar(const InlineAsmInstruction *Inst) override {
    // Only support bar.warp.sync membermask
    if (Inst->getNumInputOperands() != 1 || !Inst->hasAttr(InstAttr::warp) ||
        !Inst->hasAttr(InstAttr::sync) ||
        !DpctGlobalInfo::useExpNonUniformGroups())
      return SYCLGenError();

    std::string MemberMask;
    if (tryEmitStmt(MemberMask, Inst->getInputOperand(0)))
      return SYCLGenError();

    OS() << MapNames::getClNamespace() << "group_barrier("
         << MapNames::getClNamespace()
         << "ext::oneapi::experimental::get_ballot_group("
         << DpctGlobalInfo::getItem(GAS) << ".get_sub_group(), " << MemberMask
         << " & (1 << " << DpctGlobalInfo::getItem(GAS)
         << ".get_local_linear_id())))";
    const auto *KernelDecl = getImmediateOuterFuncDecl(GAS);
    if (KernelDecl) {
      auto FuncInfo = DeviceFunctionDecl::LinkRedecls(KernelDecl);
      if (FuncInfo)
        FuncInfo->addSubGroupSizeRequest(32, GAS->getBeginLoc(),
                                         DpctGlobalInfo::getSubGroup(GAS));
    }
    endstmt();
    return SYCLGenSuccess();
  }

  bool handle_ret(const InlineAsmInstruction *) override {
    OS() << "return";
    endstmt();
    return SYCLGenSuccess();
  }

  StringRef
  GetIntelDevcieMathRoundingModifier(const InlineAsmInstruction *Inst) {
    if (Inst->hasAttr(InstAttr::rn, InstAttr::rni))
      return "rn";
    if (Inst->hasAttr(InstAttr::rz, InstAttr::rzi))
      return "rz";
    if (Inst->hasAttr(InstAttr::rm, InstAttr::rmi))
      return "rd";
    if (Inst->hasAttr(InstAttr::rp, InstAttr::rpi))
      return "ru";
    return "";
  }

  StringRef GetSyclVectorRoundingModifier(const InlineAsmInstruction *Inst) {
    if (Inst->hasAttr(InstAttr::rn, InstAttr::rni))
      return "rounding_mode::rte";
    if (Inst->hasAttr(InstAttr::rz, InstAttr::rzi))
      return "rounding_mode::rtz";
    if (Inst->hasAttr(InstAttr::rm, InstAttr::rmi))
      return "rounding_mode::rtn";
    if (Inst->hasAttr(InstAttr::rp, InstAttr::rpi))
      return "rounding_mode::rtp";
    return "";
  }

  bool handle_rcp(const InlineAsmInstruction *Inst) override {
    if (Inst->getNumInputOperands() != 1 ||
        !isa<InlineAsmBuiltinType>(Inst->getType(0)))
      return SYCLGenError();
    if (emitStmt(Inst->getOutputOperand()))
      return SYCLGenError();
    std::string Op[1];
    const auto *T = dyn_cast<InlineAsmBuiltinType>(Inst->getType(0));
    if (tryEmitAllInputOperands(Op, Inst))
      return SYCLGenError();
    OS() << " = ";

    StringRef RD = GetIntelDevcieMathRoundingModifier(Inst);
    // If intel-device-math extension enabled, we migrate to rcp
    // instruction to sycl::ext::intel::math::{f|d}rcp_{rd|rn|ru|rz} apis
    // for better performance.
    if (DpctGlobalInfo::useIntelDeviceMath() && !RD.empty()) {
      insertHeader(HeaderType::HT_SYCL_Math);
      OS() << MapNames::getClNamespace() << "ext::intel::math::"
           << (T->getKind() == InlineAsmBuiltinType::f32 ? 'f' : 'd')
           << "rcp_" << RD << '(' << Op[0] << ')';
    } else {
      OS() << "1 / " << Op[0];
    }
    endstmt();
    return SYCLGenSuccess();
  }

  StringRef
  ConvertTypeToIntelDeviceMathFuncNameSuffix(const InlineAsmBuiltinType *T) {
    switch (T->getKind()) {
    case InlineAsmBuiltinType::s8:
      return "short";
    case InlineAsmBuiltinType::u8:
      return "ushort";
    case InlineAsmBuiltinType::s16:
      return "short";
    case InlineAsmBuiltinType::u16:
      return "ushort";
    case InlineAsmBuiltinType::s32:
      return "int";
    case InlineAsmBuiltinType::u32:
      return "uint";
    case InlineAsmBuiltinType::s64:
      return "ll";
    case InlineAsmBuiltinType::u64:
      return "ull";
    case InlineAsmBuiltinType::f32:
      return "float";
    case InlineAsmBuiltinType::f64:
      return "double";
    default:
      return "";
    }
  }

  bool handle_cvt(const InlineAsmInstruction *Inst) override {
    if (Inst->getNumInputOperands() != 1 || Inst->getNumTypes() != 2 ||
        !isa<InlineAsmBuiltinType>(Inst->getType(0)) ||
        !isa<InlineAsmBuiltinType>(Inst->getType(1)))
      return SYCLGenError();
    if (emitStmt(Inst->getOutputOperand()))
      return SYCLGenError();
    std::string Op;
    if (tryEmitStmt(Op, Inst->getInputOperand(0)))
      return SYCLGenError();
    OS() << " = ";
    const auto *DesType = dyn_cast<InlineAsmBuiltinType>(Inst->getType(0));
    const auto *SrcType = dyn_cast<InlineAsmBuiltinType>(Inst->getType(1));
    const auto *RealDesType =
        dyn_cast<InlineAsmBuiltinType>(Inst->getOutputOperand()->getType());
    const auto *RealSrcType =
        dyn_cast<InlineAsmBuiltinType>(Inst->getInputOperand(0)->getType());
    std::string DesTypeStr, SrcTypeStr, RealDesTypeStr, RealSrcTypeStr;
    if (tryEmitType(DesTypeStr, DesType))
      return SYCLGenError();
    if (tryEmitType(SrcTypeStr, SrcType))
      return SYCLGenError();
    if (tryEmitType(RealDesTypeStr, RealDesType))
      return SYCLGenError();
    if (tryEmitType(RealSrcTypeStr, RealSrcType))
      return SYCLGenError();

    bool SrcNeedBitCast = SrcType != RealSrcType &&
                          (!SrcType->isScalar() || !RealSrcType->isScalar() ||
                           SrcType->is(InlineAsmBuiltinType::f16) ||
                           RealSrcType->is(InlineAsmBuiltinType::f16));
    bool DesNeedBitCast = DesType != RealDesType &&
                          (!DesType->isScalar() || !RealDesType->isScalar() ||
                           DesType->is(InlineAsmBuiltinType::f16) ||
                           RealDesType->is(InlineAsmBuiltinType::f16));

    if (SrcNeedBitCast) {
      std::string NewOp;
      llvm::raw_string_ostream O(NewOp);
      O << MapNames::getClNamespace() << "vec<" << RealSrcTypeStr << ", 1>("
        << Op << ").template as<" << MapNames::getClNamespace() << "vec<"
        << SrcTypeStr << ", 1>>()";
      Op = std::move(NewOp);
    }

    bool HasHalfOrBfloat16 =
        SrcType->getKind() == InlineAsmBuiltinType::f16 ||
        DesType->getKind() == InlineAsmBuiltinType::f16 ||
        SrcType->getKind() == InlineAsmBuiltinType::bf16 ||
        DesType->getKind() == InlineAsmBuiltinType::bf16;
    if (DpctGlobalInfo::useIntelDeviceMath() && HasHalfOrBfloat16) {
      insertHeader(HeaderType::HT_SYCL_Math);
      if (SrcNeedBitCast)
        Op.append(".x()");
      if (DesNeedBitCast)
        OS() << MapNames::getClNamespace() << "vec<" << RealDesTypeStr
             << ", 1>(";
      // sycl::ext::intel::math::half2{short|ushort|int|uint|ll|ull|float|double}
      if (SrcType->getKind() == InlineAsmBuiltinType::f16) {
        OS() << MapNames::getClNamespace() << "ext::intel::math::half2"
             << ConvertTypeToIntelDeviceMathFuncNameSuffix(DesType) << '_'
             << GetIntelDevcieMathRoundingModifier(Inst) << '(' << Op << ')';
      }
      // sycl::ext::intel::math::{short|ushort|int|uint|ll|ull|float|double}2half
      else if (DesType->getKind() == InlineAsmBuiltinType::f16) {
        OS() << MapNames::getClNamespace() << "ext::intel::math::"
             << ConvertTypeToIntelDeviceMathFuncNameSuffix(SrcType) << "2half_"
             << GetIntelDevcieMathRoundingModifier(Inst) << '(' << Op << ')';
      }
      // sycl::ext::intel::math::bfloat162{short|ushort|int|uint|ll|ull|float|double}
      else if (SrcType->getKind() == InlineAsmBuiltinType::bf16) {
        OS() << MapNames::getClNamespace() << "ext::intel::math::bfloat162"
             << ConvertTypeToIntelDeviceMathFuncNameSuffix(DesType) << '_'
             << GetIntelDevcieMathRoundingModifier(Inst) << '(' << Op << ')';
      }
      // sycl::ext::intel::math::{short|ushort|int|uint|ll|ull|float|double}2bfloat16
      else if (DesType->getKind() == InlineAsmBuiltinType::bf16) {
        OS() << MapNames::getClNamespace() << "ext::intel::math::"
             << ConvertTypeToIntelDeviceMathFuncNameSuffix(SrcType)
             << "2bfloat16_" << GetIntelDevcieMathRoundingModifier(Inst) << '('
             << Op << ')';
      }
      if (DesNeedBitCast)
        OS() << ").template as<" << MapNames::getClNamespace() << "vec<"
             << RealDesTypeStr << ", 1>>().x()";
    } else {
      // Destination type and source type is integer or float/double
      // point.
      if (
          // Dest type is integer type, float or double
          ((DesType->isInt() || DesType->isFloat()) &&
           DesType->getKind() != InlineAsmBuiltinType::f16) &&

          // Src type is integer type, float or double
          ((SrcType->isInt() || SrcType->isFloat()) &&
           SrcType->getKind() != InlineAsmBuiltinType::f16) &&

          // Instruction has no rounding modifier
          !Inst->hasAttr(InstAttr::rni, InstAttr::rn, InstAttr::rzi,
                         InstAttr::rz, InstAttr::rmi, InstAttr::rm,
                         InstAttr::rpi, InstAttr::rp)) {
        OS() << "static_cast<" << DesTypeStr << ">(" << Op << ")";
      } else {
        if (SrcNeedBitCast)
          OS() << Op;
        else
          OS() << MapNames::getClNamespace() << "vec<" << SrcTypeStr << ", 1>("
               << Op << ")";
        OS() << ".template convert<" << DesTypeStr << ", "
             << MapNames::getClNamespace()
             << GetSyclVectorRoundingModifier(Inst) << ">()";
        if (DesNeedBitCast)
          OS() << ".template as<" << MapNames::getClNamespace() << "vec<"
               << RealDesTypeStr << ", 1>>()";
        OS() << ".x()";
      }
    }

    endstmt();
    return SYCLGenSuccess();
  }

  // Handle fma instruction.
  // .sat/.ftz/.oob/.relu attributes was ignored.
  bool handle_fma(const InlineAsmInstruction *Inst) override {
    if (Inst->getNumInputOperands() != 3 || Inst->getNumTypes() != 1)
      return SYCLGenError();
    if (!isa<InlineAsmBuiltinType>(Inst->getType(0)) ||
        !isa<InlineAsmBuiltinType>(Inst->getOutputOperand()->getType()) ||
        !isa<InlineAsmBuiltinType>(Inst->getInputOperand(0)->getType()) ||
        !isa<InlineAsmBuiltinType>(Inst->getInputOperand(1)->getType()) ||
        !isa<InlineAsmBuiltinType>(Inst->getInputOperand(2)->getType()))
      return SYCLGenError();
    const InlineAsmBuiltinType *T =
        dyn_cast<InlineAsmBuiltinType>(Inst->getType(0));
    const InlineAsmBuiltinType *OpTy[4] = {
        dyn_cast<InlineAsmBuiltinType>(Inst->getInputOperand(0)->getType()),
        dyn_cast<InlineAsmBuiltinType>(Inst->getInputOperand(1)->getType()),
        dyn_cast<InlineAsmBuiltinType>(Inst->getInputOperand(2)->getType()),
        dyn_cast<InlineAsmBuiltinType>(Inst->getOutputOperand()->getType())};
    if (emitStmt(Inst->getOutputOperand()))
      return SYCLGenError();
    OS() << " = ";
    std::string Op[3];
    std::string OpTyStr[4];
    std::string InstTypeStr;
    if (tryEmitAllInputOperands(Op, Inst))
      return SYCLGenError();
    if (tryEmitType(InstTypeStr, T))
      return SYCLGenError();
    for (int i = 0; i < 4; ++i)
      if (tryEmitType(OpTyStr[i], OpTy[i]))
        return SYCLGenError();

    if (T->getKind() == InlineAsmBuiltinType::f32 ||
        T->getKind() == InlineAsmBuiltinType::f64) {
      StringRef RD = GetIntelDevcieMathRoundingModifier(Inst);
      // If intel-device-math extension enabled, we migrate to fma
      // instruction to sycl::ext::intel::math::fma{f}_{rd|rn|ru|rz} apis
      // for better performance.
      if (DpctGlobalInfo::useIntelDeviceMath() && !RD.empty()) {
        insertHeader(HeaderType::HT_SYCL_Math);
        OS() << MapNames::getClNamespace() << "ext::intel::math::fma"
             << (T->getKind() == InlineAsmBuiltinType::f32 ? "f" : "") << '_'
             << RD << '(' << llvm::join(Op, Op + 3, ", ") << ')';
      } else
        OS() << MapNames::getClNamespace() << "fma" << '('
             << llvm::join(Op, Op + 3, ", ") << ')';
    } else {
      if (T->getKind() == InlineAsmBuiltinType::f16 ||
          T->getKind() == InlineAsmBuiltinType::f16x2 ||
          T->getKind() == InlineAsmBuiltinType::bf16x2) {
        OS() << MapNames::getClNamespace() << "fma(";
        if (T->getKind() == InlineAsmBuiltinType::f16)
          InstTypeStr = MapNames::getClNamespace() + "vec<" +
                        MapNames::getClNamespace() + "half, 1>";
        for (int I = 0; I < 3; ++I) {
          if (T == OpTy[I])
            OS() << Op[I];
          else
            OS() << MapNames::getClNamespace() << "vec<" << OpTyStr[I]
                 << ", 1>(" << Op[I] << ").template as<" << InstTypeStr
                 << ">()";
          if (I < 2)
            OS() << ", ";
        }
        OS() << ')';
        if (Inst->getOutputOperand()->getType() != T) {
          OS() << ".template as<" << MapNames::getClNamespace() << "vec<"
               << OpTyStr[3] << ", 1>"
               << ">().x()";
        }
      } else
        // fma.bf16 is not supported now.
        return SYCLGenError();
    }
    endstmt();
    return SYCLGenSuccess();
  }

  bool handle_slct(const InlineAsmInstruction *Inst) override {
    if (Inst->getNumInputOperands() != 3 || Inst->getNumTypes() != 2 ||
        !isa<InlineAsmBuiltinType>(Inst->getType(0)) ||
        !isa<InlineAsmBuiltinType>(Inst->getType(1)))
      return SYCLGenError();
    const auto *T0 = dyn_cast<InlineAsmBuiltinType>(Inst->getType(0));
    const auto *T1 = dyn_cast<InlineAsmBuiltinType>(Inst->getType(1));
    if (!T0->isInt() && !T0->isBit() && !T0->isFloat())
      return SYCLGenError();
    if (T1->getKind() != InlineAsmBuiltinType::s32 &&
        T1->getKind() != InlineAsmBuiltinType::f32)
      return SYCLGenError();

    if (emitStmt(Inst->getOutputOperand()))
      return SYCLGenError();
    std::string Op[3];
    if (tryEmitAllInputOperands(Op, Inst))
      return SYCLGenError();
    OS() << " = (" << Op[2] << " >= "
         << (T1->getKind() == InlineAsmBuiltinType::f32 ? "0.0f" : "0")
         << ") ? " << Op[0] << " : " << Op[1];
    endstmt();
    return SYCLGenSuccess();
  }

  bool handle_st(const InlineAsmInstruction *Inst) override {
    if (Inst->getNumInputOperands() != 1)
      return SYCLGenError();
    llvm::SaveAndRestore<const InlineAsmInstruction *> Store(CurrInst);
    CurrInst = Inst;
    const auto *Src = Inst->getInputOperand(0);
    const auto *Dst =
        dyn_cast_or_null<InlineAsmAddressExpr>(Inst->getOutputOperand());
    if (!Dst)
      return false;
    std::string Type;
    if (tryEmitType(Type, Inst->getType(0)))
      return SYCLGenError();
    if (emitStmt(Dst))
      return SYCLGenError();
    OS() << " = ";
    if (emitStmt(Src))
      return SYCLGenError();
    endstmt();
    return SYCLGenSuccess();
  }

  bool handle_ld(const InlineAsmInstruction *Inst) override {
    if (Inst->getNumInputOperands() != 1)
      return SYCLGenError();
    llvm::SaveAndRestore<const InlineAsmInstruction *> Store(CurrInst);
    CurrInst = Inst;
    const auto *Src =
        dyn_cast_or_null<InlineAsmAddressExpr>(Inst->getInputOperand(0));
    const auto *Dst = Inst->getOutputOperand();

    if (!Src)
      return false;
    std::string Type;
    if (tryEmitType(Type, Inst->getType(0)))
      return SYCLGenError();
    if (emitStmt(Dst))
      return SYCLGenError();
    OS() << " = ";
    if (emitStmt(Src))
      return SYCLGenError();
    endstmt();
    return SYCLGenSuccess();
  }

  bool handle_atom(const InlineAsmInstruction *Inst) override {
    // FIXME: CAS operation was not supported now.
    if (Inst->getNumInputOperands() > 2)
      return SYCLGenError();
    if (emitStmt(Inst->getOutputOperand()))
      return SYCLGenError();
    OS() << " = " << MapNames::getDpctNamespace() << "atomic_fetch_";
    if (Inst->hasAttr(InstAttr::add))
      OS() << "add";
    else if (Inst->hasAttr(InstAttr::min))
      OS() << "min";
    else if (Inst->hasAttr(InstAttr::max))
      OS() << "max";
    else
      return SYCLGenError();
    OS() << '(';
    llvm::SaveAndRestore<const InlineAsmInstruction *> Save(CurrInst);
    CurrInst = Inst;
    for (const auto &[I, Op] : llvm::enumerate(Inst->input_operands())) {
      if (emitStmt(Op))
        return SYCLGenError();
      if (I != Inst->getNumInputOperands() - 1)
        OS() << ", ";
    }
    OS() << ')';
    endstmt();
    insertHeader(HeaderType::HT_DPCT_Atomic);
    return SYCLGenSuccess();
  }
};

/// Clean the special character in identifier.
/// Rule: 1. replace '$' with '_d_'
///       2. replace '%' with '_p_'
///       3. add a '_' character to escape '_d_' and '_p_'
/// %r -> _p_r
/// %$r -> _p__d_r
/// %r$ -> _p_r_d_
/// %r% -> _p_r_p_
/// %%r -> _p__p_r
/// %r%_d_ -> _p_r_p__d_
/// %r_d_% => _p_r__d__p_
struct SYCLIdentiferHandler : public IdentifierHandler {
  bool HandleIdentifier(SmallVectorImpl<char> &Buf,
                        InlineAsmToken &Identifier) override {
    assert(Identifier.is(asmtok::raw_identifier) &&
           "Must handle an raw identifier");
    StringRef Input = Identifier.getRawIdentifier();
    if (!Identifier.needsCleaning() && !Input.contains("_d_") &&
        !Input.contains("_p_"))
      return false;
    Buf.clear();
    auto Push = [&Buf](char C) {
      Buf.push_back('_');
      Buf.push_back(C);
      Buf.push_back('_');
    };

    for (const char *Ptr = Input.begin(); Ptr != Input.end(); ++Ptr) {
      char C = *Ptr;
      StringRef SubStr(Ptr, Input.end() - Ptr);
      switch (C) {
      case '$':
        Push('d');
        break;
      case '%':
        Push('p');
        break;
      case '_':
        if (SubStr.starts_with("_d_") || SubStr.starts_with("_p_"))
          Buf.push_back('_');
        [[fallthrough]];
      default:
        Buf.push_back(C);
        break;
      }
    }
    return true;
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

static TextModification *
OptimizeMigrationForCUDABackend(const GCCAsmStmt *GAS, StringRef Replacement) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto &Ctx = DpctGlobalInfo::getContext();
  unsigned LastTokLen = Lexer::MeasureTokenLength(
      SM.getSpellingLoc(GAS->getEndLoc()), SM, Ctx.getLangOpts());
  auto getBufferData = [&](SourceLocation L) {
    bool Invalid = false;
    const char *Ptr = SM.getCharacterData(L, &Invalid);
    return Invalid ? nullptr : Ptr;
  };

  const char *AsmStmtBegin =
      getBufferData(SM.getSpellingLoc(GAS->getBeginLoc()));
  const char *AsmStmtEnd = getBufferData(
      SM.getSpellingLoc(GAS->getEndLoc()).getLocWithOffset(LastTokLen));

  if (!AsmStmtBegin || !AsmStmtEnd)
    return nullptr;

  std::string Buffer, MigratedReplacement;
  SourceLocation BeginLoc, EndLoc;
  llvm::raw_string_ostream NewOS(Buffer);
  NewOS << "#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)" << getNL();
  auto FileID = SM.getFileID(SM.getSpellingLoc(GAS->getBeginLoc()));
  auto FileStartLoc = SM.getLocForStartOfFile(FileID);
  if (isInMacroDefinition(GAS->getBeginLoc(), GAS->getEndLoc())) {
    auto FindMacroBeginLoc = [&]() {
      auto Iter = DpctGlobalInfo::getMacroTokenToMacroDefineLoc().find(
          getHashStrFromLoc(SM.getSpellingLoc(GAS->getBeginLoc())));
      if (Iter != DpctGlobalInfo::getMacroTokenToMacroDefineLoc().end()) {
        auto MacroStartLoc =
            FileStartLoc.getLocWithOffset(Iter->second->Offset);
        auto LineBeginLoc = FileStartLoc.getLocWithOffset(
            getOffsetOfLineBegin(MacroStartLoc, SM));
        return LineBeginLoc;
      }
      return SourceLocation();
    };

    auto FindMacroEndLoc = [&]() {
      auto Iter = DpctGlobalInfo::getExpansionRangeToMacroRecord().find(
          getCombinedStrFromLoc(SM.getSpellingLoc(GAS->getBeginLoc())));
      if (Iter != DpctGlobalInfo::getExpansionRangeToMacroRecord().end()) {
        unsigned EndOffset = Iter->second->ReplaceTokenEndOffset;
        auto MacroEndLoc = FileStartLoc.getLocWithOffset(EndOffset);
        unsigned Len =
            Lexer::MeasureTokenLength(MacroEndLoc, SM, Ctx.getLangOpts());
        return MacroEndLoc.getLocWithOffset(Len);
      }
      return SourceLocation();
    };

    auto MacroBeginLoc = FindMacroBeginLoc();
    auto MacroEndLoc = FindMacroEndLoc();
    const char *MacroBegin = getBufferData(MacroBeginLoc);
    const char *MacroEnd = getBufferData(MacroEndLoc);
    StringRef MacroStr(MacroBegin, MacroEnd - MacroBegin);
    NewOS << MacroStr;
    BeginLoc = MacroBeginLoc;
    EndLoc = MacroEndLoc;
    std::string Prefix(MacroBegin, AsmStmtBegin);
    std::string Suffix(AsmStmtEnd, MacroEnd);
    MigratedReplacement = Prefix + Replacement.str() + Suffix;
  } else if (SM.isMacroArgExpansion(GAS->getBeginLoc()) &&
             SM.isMacroArgExpansion(GAS->getEndLoc())) {
    auto ExpansionRange = SM.getExpansionRange(GAS->getBeginLoc());
    auto LineStartLoc = FileStartLoc.getLocWithOffset(
        getOffsetOfLineBegin(ExpansionRange.getBegin(), SM));
    const char *LineStart = getBufferData(LineStartLoc);
    if (!LineStart)
      return nullptr;
    auto SemiTok =
        Lexer::findNextToken(ExpansionRange.getEnd(), SM, Ctx.getLangOpts());
    // Can't find the trailing semicolon.
    if (!SemiTok || SemiTok->isNot(tok::semi))
      return nullptr;
    const char *ExpansionEnd = getBufferData(SemiTok->getEndLoc());
    NewOS << StringRef(LineStart, ExpansionEnd - LineStart);
    BeginLoc = LineStartLoc;
    EndLoc = LineStartLoc.getLocWithOffset(ExpansionEnd - LineStart);
    std::string Prefix(LineStart, AsmStmtBegin);
    std::string Suffix(AsmStmtEnd, ExpansionEnd);
    MigratedReplacement = Prefix + Replacement.str() + Suffix;
  } else {
    auto LineStartLoc = FileStartLoc.getLocWithOffset(
        getOffsetOfLineBegin(GAS->getBeginLoc(), SM));
    const char *LineStart = getBufferData(LineStartLoc);
    if (!LineStart)
      return nullptr;
    auto SemiTok = Lexer::findNextToken(SM.getSpellingLoc(GAS->getEndLoc()), SM,
                                        Ctx.getLangOpts());
    // Can't find the trailing semicolon.
    if (!SemiTok || SemiTok->isNot(tok::semi))
      return nullptr;
    const char *StmtEnd = getBufferData(SemiTok->getEndLoc());
    NewOS << StringRef(LineStart, StmtEnd - LineStart);
    MigratedReplacement =
        std::string(LineStart, AsmStmtBegin) + Replacement.str();
    BeginLoc = LineStartLoc;
    EndLoc = SemiTok->getEndLoc();
  }
  NewOS << getNL() << "#else" << getNL() << MigratedReplacement << getNL()
        << "#endif";
  NewOS.flush();
  auto *Repl = new ReplaceText(BeginLoc, EndLoc, std::move(Buffer));
  Repl->setBlockLevelFormatFlag();
  return Repl;
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
  SYCLIdentiferHandler Handle;
  InlineAsmParser Parser(Context, Mgr);
  Parser.getLexer().setIdentifierHandler(&Handle);
  std::string ReplaceString;
  llvm::raw_string_ostream OS(ReplaceString);
  SYCLGen CodeGen(OS, Context, GAS);
  StringRef Indent = getIndent(SM.getSpellingLoc(GAS->getBeginLoc()),
                               DpctGlobalInfo::getSourceManager());

  CodeGen.setIndentUnit(Indent);
  CodeGen.incIndent();
  if (isInMacroDefinition(GAS->getBeginLoc(), GAS->getEndLoc()))
    CodeGen.setInMacroDefine();

  auto getReplaceString = [&](const Expr *E) {
    ArgumentAnalysis AA(CodeGen.isInMacroDefine());
    AA.setCallSpelling(SM.getSpellingLoc(GAS->getBeginLoc()),
                       SM.getSpellingLoc(GAS->getEndLoc()));
    AA.analyze(E);
    if (needExtraParens(E) && !isa<UnaryOperator>(E) &&
        !isa<UnaryExprOrTypeTraitExpr>(E))
      return "(" + AA.getRewriteString() + ")";
    return AA.getRewriteString();
  };
  Parser.addBuiltinIdentifier();
  for (unsigned I = 0, E = GAS->getNumOutputs(); I != E; ++I)
    Parser.addInlineAsmOperands(GAS->getOutputExpr(I),
                                getReplaceString(GAS->getOutputExpr(I)),
                                GAS->getOutputConstraint(I));

  for (unsigned I = 0, E = GAS->getNumInputs(); I != E; ++I)
    Parser.addInlineAsmOperands(GAS->getInputExpr(I),
                                getReplaceString(GAS->getInputExpr(I)),
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
    if (CodeGen.isMigrationStopped())
      return;
  } while (!Parser.getCurToken().is(asmtok::eof));

  StringRef Ref = ReplaceString;
  Ref = Ref.trim();
  if (CodeGen.isInMacroDefine() && Ref.ends_with("\\"))
    Ref = Ref.drop_back();
  if (Ref.size() > 2 && Ref.back() == ';' && Ref.front() == '{' &&
      Ref.drop_back().back() == '}')
    Ref = Ref.drop_back();
  if (SM.isMacroArgExpansion(GAS->getBeginLoc()) &&
      SM.isMacroArgExpansion(GAS->getEndLoc()) && Ref.back() == ';')
    Ref = Ref.drop_back();

  if (DpctGlobalInfo::isOptimizeMigration()) {
    if (auto *Repl = OptimizeMigrationForCUDABackend(GAS, Ref)) {
      emplaceTransformation(Repl);
      return;
    }
  }

  auto *Repl = new ReplaceStmt(GAS, std::move(Ref.str()));
  Repl->setBlockLevelFormatFlag();
  emplaceTransformation(Repl);

  auto Range = getDefinitionRange(GAS->getBeginLoc(), GAS->getEndLoc());
  auto KELoc =
      getTheLastCompleteImmediateRange(Range.getBegin(), Range.getEnd()).second;
  auto Tok = Lexer::findNextToken(KELoc, SM, LangOptions()).value();
  if (Tok.is(tok::TokenKind::semi))
    emplaceTransformation(new ReplaceToken(Tok.getLocation(), ""));
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
