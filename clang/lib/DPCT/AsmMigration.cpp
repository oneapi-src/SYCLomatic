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
#include "Diagnostics.h"
#include "MapNames.h"
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
  SmallString<4> IndentUnit{"  "};
  SmallString<16> Indent;
  SmallString<4> NewLine;
  SmallVector<SmallString<10>, 4> VecExprTypeRecord;
  raw_ostream *Stream;

protected:
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
  bool emitVectorExpr(const InlineAsmVectorExpr *E);
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
  switch (I->getOpcodeID()->getTokenID()) {
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
    std::string Fmt =
        MapNames::getClNamespace() + "bit_cast<{0}>({1}(0x{2}{3}))";
    if (const auto *T = dyn_cast<InlineAsmBuiltinType>(Fp->getType())) {
      switch (T->getKind()) {
      case InlineAsmBuiltinType::TK_f32:
        OS() << llvm::formatv(Fmt.c_str(), "float", "uint32_t",
                              Fp->getLiteral(), "U");
        break;
      case InlineAsmBuiltinType::TK_f64:
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
  case InlineAsmBuiltinType::TK_f16:    OS() << MapNames::getClNamespace() + "half"; break;
  case InlineAsmBuiltinType::TK_f32:    OS() << "float"; break;
  case InlineAsmBuiltinType::TK_f64:    OS() << "double"; break;
  case InlineAsmBuiltinType::TK_byte:   OS() << "uint8_t"; break;
  case InlineAsmBuiltinType::TK_4byte:  OS() << "uint32_t"; break;
  case InlineAsmBuiltinType::TK_pred:   OS() << "bool"; break;
  case InlineAsmBuiltinType::TK_s16x2:  OS() << MapNames::getClNamespace() + "short2"; break;
  case InlineAsmBuiltinType::TK_u16x2:  OS() << MapNames::getClNamespace() + "ushort2"; break;
  case InlineAsmBuiltinType::TK_bf16:
  case InlineAsmBuiltinType::TK_e4m3:
  case InlineAsmBuiltinType::TK_e5m2:
  case InlineAsmBuiltinType::TK_tf32:
  case InlineAsmBuiltinType::TK_f16x2:
  case InlineAsmBuiltinType::TK_bf16x2:
  case InlineAsmBuiltinType::TK_e4m3x2:
  case InlineAsmBuiltinType::TK_e5m2x2:
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
  unsigned getBitSizeTypeWidth(const InlineAsmBuiltinType *T) const {
    switch (T->getKind()) {
    case InlineAsmBuiltinType::TK_b8:
      return 8;
    case InlineAsmBuiltinType::TK_b16:
      return 16;
    case InlineAsmBuiltinType::TK_b32:
      return 32;
    case InlineAsmBuiltinType::TK_b64:
      return 64;
    default:
      return 0;
    }
  }

  const char *unpackBitSizeType(const InlineAsmBuiltinType *T,
                                unsigned N) const {
    assert(T->isBitSize());
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
      if (!Type || !Type->isBitSize())
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
      if (!Type || !Type->isBitSize())
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

    bool IsInteger = T->isSignedInt() || T->isUnsignedInt() || T->isBitSize();

    for (const auto A : I->attrs()) {
      switch (A) {
      case InstAttr::eq:
        if (IsInteger)
          ExprCmp("==");
        else if (T->isFloating())
          DpctCmp("compare<{0}>({1}, {2}, std::equal_to<>())");
        else
          return SYCLGenError();
        break;
      case InstAttr::ne:
        if (IsInteger)
          ExprCmp("!=");
        else if (T->isFloating())
          DpctCmp("compare<{0}>({1}, {2}, std::not_equal_to<>())");
        else
          return SYCLGenError();
        break;
      case InstAttr::lt:
        if (IsInteger)
          ExprCmp("<");
        else if (T->isFloating())
          DpctCmp("compare<{0}>({1}, {2}, std::less<>())");
        else
          return SYCLGenError();
        break;
      case InstAttr::le:
        if (IsInteger)
          ExprCmp("<=");
        else if (T->isFloating())
          DpctCmp("compare<{0}>({1}, {2}, std::less_equal<>())");
        else
          return SYCLGenError();
        break;
      case InstAttr::gt:
        if (IsInteger)
          ExprCmp(">");
        else if (T->isFloating())
          DpctCmp("compare<{0}>({1}, {2}, std::greater<>())");
        else
          return SYCLGenError();
        break;
      case InstAttr::ge:
        if (IsInteger)
          ExprCmp(">=");
        else if (T->isFloating())
          DpctCmp("compare<{0}>({1}, {2}, std::greater_equal<>())");
        else
          return SYCLGenError();
        break;
      case InstAttr::lo:
        if (T->isUnsignedInt())
          ExprCmp("<");
        else
          return SYCLGenError();
        break;
      case InstAttr::ls:
        if (T->isUnsignedInt())
          ExprCmp("<=");
        else
          return SYCLGenError();
        break;
      case InstAttr::hi:
        if (T->isUnsignedInt())
          ExprCmp(">");
        else
          return SYCLGenError();
        break;
      case InstAttr::hs:
        if (T->isUnsignedInt())
          ExprCmp(">=");
        else
          return SYCLGenError();
        break;
      case InstAttr::equ:
        if (T->isFloating())
          DpctCmp("unordered_compare<{0}>({1}, {2}, std::equal_to<>())");
        else
          return SYCLGenError();
        break;
      case InstAttr::neu:
        if (T->isFloating())
          DpctCmp("unordered_compare<{0}>({1}, {2}, std::not_equal_to<>())");
        else
          return SYCLGenError();
        break;
      case InstAttr::ltu:
        if (T->isFloating())
          DpctCmp("unordered_compare<{0}>({1}, {2}, std::less<>())");
        else
          return SYCLGenError();
        break;
      case InstAttr::leu:
        if (T->isFloating())
          DpctCmp("unordered_compare<{0}>({1}, {2}, std::less_equal<>())");
        else
          return SYCLGenError();
        break;
      case InstAttr::gtu:
        if (T->isFloating())
          DpctCmp("unordered_compare<{0}>({1}, {2}, std::greater<>())");
        else
          return SYCLGenError();
        break;
      case InstAttr::geu:
        if (T->isFloating())
          DpctCmp("unordered_compare<{0}>({1}, {2}, std::greater_equal<>())");
        else
          return SYCLGenError();
        break;
      case InstAttr::num:
        if (T->isFloating()) {
          OS() << '!' << MapNames::getClNamespace();
          OS() << "isnan(" << Op[0] << ")";
          OS() << " && ";
          OS() << '!' << MapNames::getClNamespace();
          OS() << "isnan(" << Op[1] << ")";
        } else
          return SYCLGenError();
        break;
      case InstAttr::nan:
        if (T->isFloating()) {
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

  bool CheckAddSubMinMaxType(const InlineAsmInstruction *Inst, bool &isVec) {
    if (const auto *BI = dyn_cast<InlineAsmBuiltinType>(Inst->getType(0))) {
      if (Inst->hasAttr(InstAttr::sat) &&
          BI->getKind() != InlineAsmBuiltinType::TK_s32)
        return false;
      if (BI->getKind() != InlineAsmBuiltinType::TK_s16 &&
          BI->getKind() != InlineAsmBuiltinType::TK_u16 &&
          BI->getKind() != InlineAsmBuiltinType::TK_s32 &&
          BI->getKind() != InlineAsmBuiltinType::TK_u32 &&
          BI->getKind() != InlineAsmBuiltinType::TK_s64 &&
          BI->getKind() != InlineAsmBuiltinType::TK_u64 &&
          BI->getKind() != InlineAsmBuiltinType::TK_s16x2 &&
          BI->getKind() != InlineAsmBuiltinType::TK_u16x2)
        return false;
      isVec = BI->getKind() == InlineAsmBuiltinType::TK_s16x2 ||
              BI->getKind() == InlineAsmBuiltinType::TK_u16x2;
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

  bool handle_sub(const InlineAsmInstruction *Inst) override {
    return HandleAddSub(Inst);
  }

  StringRef GetWiderTypeAsString(const InlineAsmBuiltinType *Type) const {
    switch (Type->getKind()) {
    case InlineAsmBuiltinType::TK_s16:
      return "int32_t";
    case InlineAsmBuiltinType::TK_u16:
      return "uint32_t";
    case InlineAsmBuiltinType::TK_s32:
      return "int64_t";
    case InlineAsmBuiltinType::TK_u32:
      return "uint64_t";
    default:
      assert(false && "Can not find wide type");
    }
    return "";
  }

  bool CheckMulMadType(const InlineAsmBuiltinType *Type) const {
    return Type->getKind() == InlineAsmBuiltinType::TK_s16 ||
           Type->getKind() == InlineAsmBuiltinType::TK_u16 ||
           Type->getKind() == InlineAsmBuiltinType::TK_s32 ||
           Type->getKind() == InlineAsmBuiltinType::TK_u32 ||
           Type->getKind() == InlineAsmBuiltinType::TK_s64 ||
           Type->getKind() == InlineAsmBuiltinType::TK_u64;
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
        (Type->getKind() == InlineAsmBuiltinType::TK_s64 ||
         Type->getKind() == InlineAsmBuiltinType::TK_u64))
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
        (Type->getKind() == InlineAsmBuiltinType::TK_s64 ||
         Type->getKind() == InlineAsmBuiltinType::TK_u64))
      return SYCLGenError();

    // The attribute .sat only work with .hi mode.
    if (Inst->hasAttr(InstAttr::sat) &&
        (!Inst->hasAttr(InstAttr::hi) ||
         Type->getKind() != InlineAsmBuiltinType::TK_s32))
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

    if (Type->getKind() != InlineAsmBuiltinType::TK_s16 &&
        Type->getKind() != InlineAsmBuiltinType::TK_s32 &&
        Type->getKind() != InlineAsmBuiltinType::TK_s64) {
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
    if (!Type || (Type->getKind() != InlineAsmBuiltinType::TK_b32 &&
                  Type->getKind() != InlineAsmBuiltinType::TK_b64))
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
    if (!BI || (BI->getKind() != InlineAsmBuiltinType::TK_b16 &&
                BI->getKind() != InlineAsmBuiltinType::TK_b32 &&
                BI->getKind() != InlineAsmBuiltinType::TK_b64 &&
                (IsShift && BI->getKind() == InlineAsmBuiltinType::TK_pred)))
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
    if (!BI || (BI->getKind() != InlineAsmBuiltinType::TK_b16 &&
                BI->getKind() != InlineAsmBuiltinType::TK_b32 &&
                BI->getKind() != InlineAsmBuiltinType::TK_b64 &&
                (Inst->is(asmtok::op_cnot) &&
                 BI->getKind() == InlineAsmBuiltinType::TK_pred)))
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

    std::string TypeString;
    if (tryEmitType(TypeString, Inst->getType(0)))
      return SYCLGenError();

    std::string ReplaceString =
        MapNames::getClNamespace() + MathFn.str() + "<" + TypeString + ">(";
    if (Inst->getOpcode() == asmtok::op_ex2)
      ReplaceString += "2, ";
    ReplaceString += Op + ")";
    if (Inst->hasAttr(InstAttr::rn, InstAttr::rz, InstAttr::rm, InstAttr::rp))
      report(Diagnostics::ROUNDING_MODE_UNSUPPORTED, true);
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
    std::string Op[3], TypeString;
    for (int i = 0; i < 3; ++i)
      if (tryEmitStmt(Op[i], Inst->getInputOperand(i)))
        return SYCLGenError();
    if (tryEmitType(TypeString, Inst->getType(0)))
      return SYCLGenError();

    OS() << MapNames::getClNamespace() << "abs_diff<" << TypeString << ">("
         << Op[0] << ", " << Op[1] << ") + " << Op[2];
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

  // Handle the 1 element vadd/vsub/vmin/vmax/vabsdiff video instructions.
  bool HandleOneElementAddSubMinMax(const InlineAsmInstruction *Inst,
                                    StringRef Fn) {
    if (Inst->getNumInputOperands() < 2 || Inst->getNumTypes() != 3)
      return SYCLGenError();

    // Arguments mismatch for instruction, which has a secondary arithmetic
    // operation.
    if (Inst->hasAttr(InstAttr::add, InstAttr::min, InstAttr::max) &&
        Inst->getNumInputOperands() < 3)
      return SYCLGenError();

    // The type of operands must be one of s32/u32.
    for (const auto *T : Inst->types()) {
      if (const auto *BI = dyn_cast<InlineAsmBuiltinType>(T)) {
        if (BI->getKind() == InlineAsmBuiltinType::TK_s32 ||
            BI->getKind() == InlineAsmBuiltinType::TK_u32)
          continue;
      }
      return SYCLGenError();
    }

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

  // Handle the 2/4 element video instructions.
  bool handleMultiElementAddSubMinMax(const InlineAsmInstruction *Inst,
                                      StringRef Fn) {
    if (Inst->getNumInputOperands() < 3 || Inst->getNumTypes() != 3)
      return SYCLGenError();

    // The type of operands must be one of s32/u32.
    for (const auto *T : Inst->types()) {
      if (const auto *BI = dyn_cast<InlineAsmBuiltinType>(T)) {
        if (BI->getKind() == InlineAsmBuiltinType::TK_s32 ||
            BI->getKind() == InlineAsmBuiltinType::TK_u32)
          continue;
      }
      return SYCLGenError();
    }

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

  bool handle_bfe(const InlineAsmInstruction *Inst) override {
    if (Inst->getNumInputOperands() != 3)
      return SYCLGenError();
    const auto *Type = dyn_cast<InlineAsmBuiltinType>(Inst->getType(0));
    if (!Type || (Type->getKind() != InlineAsmBuiltinType::TK_s32 &&
                  Type->getKind() != InlineAsmBuiltinType::TK_s64 &&
                  Type->getKind() != InlineAsmBuiltinType::TK_u32 &&
                  Type->getKind() != InlineAsmBuiltinType::TK_u64))
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
    if (!Type || (Type->getKind() != InlineAsmBuiltinType::TK_b32 &&
                  Type->getKind() != InlineAsmBuiltinType::TK_b64))
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
    if (!Type || (Type->getKind() != InlineAsmBuiltinType::TK_b32 &&
                  Type->getKind() != InlineAsmBuiltinType::TK_b64))
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
    if (!BIType || (BIType->getKind() != InlineAsmBuiltinType::TK_s32 &&
                    BIType->getKind() != InlineAsmBuiltinType::TK_u32))
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
  InlineAsmParser Parser(Context, Mgr);
  std::string ReplaceString;
  llvm::raw_string_ostream OS(ReplaceString);
  SYCLGen CodeGen(OS, GAS);
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
