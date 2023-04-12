//===------------------------- AsmParser.cpp --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AsmParser.h"
#include "Asm/AsmLexer.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace clang::dpct;

ptx::InstKind ptx::FindInstructionKindFromName(StringRef InstName) {
  return llvm::StringSwitch<ptx::InstKind>(InstName)
      .Case("cvt", ptx::Cvt)
      .Case("mov", ptx::Mov)
      .Case("bfe", ptx::Bfe)
      .Case("setp", Setp)
      .Default(ptx::Invalid);
}

PtxVariableDecl *PtxScope::LookupSymbol(StringRef Symbol) const {
  for (const auto &S : decls()) {
    if (S->getDeclName() == Symbol)
      return S;
  }

  if (this->AnyParent) {
    return this->AnyParent->LookupSymbol(Symbol);
  }
  return nullptr;
}

PtxFundamentalType *PtxContext::GetOrCreateFundamentalType(StringRef TypeName) {
  if (TypeName == ".s32")
    return GetOrCreateFundamentalType(PtxFundamentalType::TK_S32);
  if (TypeName == ".s64")
    return GetOrCreateFundamentalType(PtxFundamentalType::TK_S64);
  if (TypeName == ".u32")
    return GetOrCreateFundamentalType(PtxFundamentalType::TK_U32);
  if (TypeName == ".u64")
    return GetOrCreateFundamentalType(PtxFundamentalType::TK_U64);
  if (TypeName == ".pred")
    return GetOrCreateFundamentalType(PtxFundamentalType::TK_Pred);
  return nullptr;
}

PtxTupleType *
PtxContext::CreateTupleType(const PtxTupleType::ElementList &ElementType) {
  return ::new (*this) PtxTupleType(ElementType);
}

PtxAnyType *PtxContext::GetOrCreateAnyType() {
  if (!AnyType)
    AnyType = ::new (*this) PtxAnyType;
  return AnyType;
}

PtxFundamentalType *
PtxContext::GetOrCreateFundamentalType(PtxFundamentalType::TypeKind Kind) {
  if (FundamentalTypes.contains(Kind))
    return FundamentalTypes[Kind];

  PtxFundamentalType *NewType = ::new (*this) PtxFundamentalType(Kind);
  FundamentalTypes[Kind] = NewType;
  return NewType;
}

PtxType *PtxContext::GetTypeFromConstraint(StringRef Constraint) {
  if (Constraint.size() != 1)
    return nullptr;
  switch (Constraint[0]) {
  case 'h':
    return GetOrCreateFundamentalType(PtxFundamentalType::TK_U16);
  case 'r':
    return GetOrCreateFundamentalType(PtxFundamentalType::TK_U32);
  case 'l':
    return GetOrCreateFundamentalType(PtxFundamentalType::TK_U64);
  case 'f':
    return GetOrCreateFundamentalType(PtxFundamentalType::TK_F32);
  case 'd':
    return GetOrCreateFundamentalType(PtxFundamentalType::TK_F64);
  default:
    break;
  }
  return nullptr;
}

PtxVectorType *PtxContext::CreateVectorType(PtxVectorType::TypeKind Kind, const PtxFundamentalType *Base) {
  return  ::new (*this) PtxVectorType(Kind, Base);
}

PtxDeclStmt *PtxContext::CreateDeclStmt(const SmallVector<const PtxDecl *> &DeclGroup) {
  return ::new (*this) PtxDeclStmt(DeclGroup);
}

PtxVariableDecl *PtxContext::CreateVariableDecl(StringRef Name,
                                                const PtxType *Type) {
  return ::new (*this) PtxVariableDecl(Name, Type);
}

PtxCompoundStmt *
PtxContext::CreateCompoundStmt(const SmallVector<PtxStmt *> &Stmts) {
  return ::new (*this) PtxCompoundStmt(Stmts);
}

PtxDeclRefExpr *PtxContext::CreateDeclRefExpr(const PtxVariableDecl *Var) {
  return ::new (*this) PtxDeclRefExpr(Var);
}

PtxTupleExpr *
PtxContext::CreateTupleExpr(const PtxTupleType *Type,
                            const PtxTupleExpr::ElementList &Elements) {
  return ::new (*this) PtxTupleExpr(Type, Elements);
}

PtxSinkExpr *PtxContext::CreateSinkExpr() {
  return ::new (*this) PtxSinkExpr(GetOrCreateAnyType());
}

PtxInstruction *
PtxContext::CreateInstruction(ptx::InstKind Op,
                              const PtxInstruction::OperandList &Operands,
                              const PtxExpr *PredOut) {
  return ::new (*this) PtxInstruction(Op, Operands, PredOut);
}

PtxGuardInstruction *
PtxContext::CreateGuardInstruction(bool isNeg, const PtxExpr *Pred,
                                   const PtxInstruction *Inst) {
  return ::new (*this) PtxGuardInstruction(isNeg, Pred, Inst);
}

PtxUnaryOperator *PtxContext::CreateUnaryOperator(PtxUnaryOperator::Opcode Op,
                                                  const PtxExpr *Operand) {
  return ::new (*this) PtxUnaryOperator(Op, Operand, Operand->getType());
}

PtxBinaryOperator *
PtxContext::CreateBinaryOperator(PtxBinaryOperator::Opcode Op,
                                 const PtxExpr *LHS, const PtxExpr *RHS) {
  return ::new (*this) PtxBinaryOperator(Op, LHS, RHS, nullptr);
}

PtxConditionalOperator *
PtxContext::CreateConditionalOperator(const PtxExpr *Cond, const PtxExpr *LHS,
                                      const PtxExpr *RHS) {
  return ::new (*this) PtxConditionalOperator(Cond, LHS, RHS, LHS->getType());
}

PtxCastExpr *
PtxContext::CreateCastExpression(const PtxFundamentalType *CastType,
                                 const PtxExpr *SubExpr) {
  return ::new (*this) PtxCastExpr(CastType, SubExpr);
}

PtxIntegerLiteral *PtxContext::CreateIntegerLiteral(const PtxType *Type,
                                                    llvm::APInt Val) {
  return ::new (*this) PtxIntegerLiteral(Type, Val);
}

PtxFloatingLiteral *PtxContext::CreateFloatLiteral(const PtxType *Type,
                                                   llvm::APFloat Val) {
  return ::new (*this) PtxFloatingLiteral(Type, Val);
}

PtxType::~PtxType() = default;
PtxDecl::~PtxDecl() = default;
PtxStmt::~PtxStmt() = default;

PtxParser::~PtxParser() { ExitScope(); }

PtxDeclResult PtxParser::AddBuiltinSymbol(StringRef Name, const PtxType *Type) {
  PtxVariableDecl *D = getContext().CreateVariableDecl(Name, Type);
  getCurScope()->AddDecl(D);
  return D;
}

PtxDeclResult PtxParser::AddInlineAsmOperands(StringRef Name,
                                              StringRef Constraint) {
  PtxType *Type = getContext().GetTypeFromConstraint(Constraint);
  if (!Type)
    return true;

  return getContext().CreateVariableDecl(Name, Type);
}

const AsmToken &PtxParser::Lex() {
  if (Lexer.getTok().is(AsmToken::Error))
    return Lexer.getTok();

  const AsmToken *tok = &Lexer.Lex();

  // Parse comments here to be deferred until end of next statement.
  while (tok->is(AsmToken::Comment)) {
    tok = &Lexer.Lex();
  }

  return *tok;
}

PtxStmtResult PtxParser::ParseStatement() {
  switch (getTok().getKind()) {
  case AsmToken::LCurly:
    return ParseCompoundStatement();
  case AsmToken::At:
    return ParseGuardInstruction();
  case AsmToken::DotIdentifier:
    return ParseDeclStmt();
  default:
    break;
  }
  return ParseInstruction();
}

PtxStmtResult PtxParser::ParseCompoundStatement() {
  assert(getTok().is(AsmToken::LCurly));
  Lex(); // eat '{'

  ParseScope BlockScope(this);

  SmallVector<PtxStmt *> Stmts;
  while (getTok().isNot(AsmToken::RCurly)) {
    if (getTok().isVarAttributes()) {
      PtxStmtResult Res = ParseDeclStmt();
      if (Res.isInvalid())
        return true;
      Stmts.push_back(Res.get());
    } else {
      PtxStmtResult Res = ParseStatement();
      if (Res.isInvalid())
        return Res;
      Stmts.push_back(Res.get());
    }
  }
  return getContext().CreateCompoundStmt(Stmts);
}

PtxStmtResult PtxParser::ParseGuardInstruction() {
  assert(getTok().is(AsmToken::At));
  Lex(); // eat '@'

  bool isNeg = false;
  if (getTok().is(AsmToken::Exclaim)) {
    isNeg = true;
    Lex(); // eat '!'
  }

  PtxExprResult Pred = ParsePrimaryExpression();
  if (Pred.isInvalid())
    return true;

  PtxStmtResult SubInst = ParseInstruction();
  if (SubInst.isInvalid())
    return true;

  return getContext().CreateGuardInstruction(
      isNeg, Pred.get(), (const PtxInstruction *)SubInst.get());
}

PtxStmtResult PtxParser::ParseInstruction() {
  if (getTok().isNot(AsmToken::Identifier))
    return true;

  ptx::InstKind Opcode = ptx::FindInstructionKindFromName(getTok().getString());
  if (Opcode == ptx::Invalid)
    return true; // Parseing an invalid or unsupported instruction
  
  Lex(); // eat opcode

  PtxInstruction::Attribute Attr;
  if (ParseInstructionFlags(Attr))
    return true; // Parsed an invalid instruction attributes

  PtxExprResult DestOperand = ParseInstructionDestOperand();
  if (DestOperand.isInvalid())
    return true;

  bool HasPredOutput = getTok().is(AsmToken::Pipe);
  PtxExprResult PredOutput;

  if (HasPredOutput) {
    PredOutput = ParsePredOutput();
    if (PredOutput.isInvalid())
      return true;
  }

  PtxInstruction::OperandList Operands;
  Operands.push_back(DestOperand.get());

  while (getTok().is(AsmToken::Comma)) {
    Lex(); // eat ','
    PtxExprResult Operand = ParseInstructionSrcOperand();
    if (Operand.isInvalid())
      return true;
    Operands.push_back(Operand.get());
  }

  if (getTok().isNot(AsmToken::EndOfStatement))
    return true;
  Lex(); // eat ';'

  return getContext().CreateInstruction(
      Opcode, Operands, HasPredOutput ? PredOutput.get() : nullptr);
}

bool PtxParser::ParseInstructionFlags(PtxInstruction::Attribute &Attr) {
  while (getTok().is(AsmToken::DotIdentifier)) {
    if (getTok().isTypeName())
      Attr.Types.push_back(
          getContext().GetOrCreateFundamentalType(getTok().getString()));

    if (getTok().getString() == ".eq")
      Attr.setComparisonOp(ptx::CO_Eq);
    if (getTok().getString() == ".ne")
      Attr.setComparisonOp(ptx::CO_Ne);
    if (getTok().getString() == ".lt")
      Attr.setComparisonOp(ptx::CO_Le);
    if (getTok().getString() == ".gt")
      Attr.setComparisonOp(ptx::CO_Gt);
    if (getTok().getString() == ".ge")
      Attr.setComparisonOp(ptx::CO_Ge);

    /// TODO: Parse Other modifiers
    Lex(); // eat a 'dot identifier
  }
  return false;
}

PtxExprResult PtxParser::ParsePredOutput() {
  assert(getTok().is(AsmToken::Pipe));
  // Parse predicate output
  Lex(); // eat '|'
  switch (getTok().getKind()) {
  case AsmToken::Identifier:
    if (PtxVariableDecl *S = getCurScope()->LookupSymbol(getTok().getString()))
      return getContext().CreateDeclRefExpr(S);
    break;
  case AsmToken::Sink:
    return getContext().CreateSinkExpr();
  default:
    break;
  }

  // expect an identifier or '_'
  return true;
}

PtxExprResult PtxParser::ParseTuple() {
  Lex(); // eat '{'
  PtxTupleExpr::ElementList List;
  do {
    if (getTok().is(AsmToken::Comma))
      Lex(); // eat ','
    switch (getTok().getKind()) {
    case AsmToken::Identifier:
      if (PtxVariableDecl *S =
              getCurScope()->LookupSymbol(getTok().getString()))
        List.push_back(getContext().CreateDeclRefExpr(S));
      else
        return true;
      break;
    case AsmToken::Sink:
      List.push_back(getContext().CreateSinkExpr());
      break;
    default:
      auto Const = ParseConstantExpression();
      if (Const.isInvalid())
        return true;
      List.push_back(Const.get());
    }
  } while (getTok().is(AsmToken::Comma));
  if (getTok().isNot(AsmToken::RBrac))
    return true;

  PtxTupleType::ElementList ElementTypes;
  for (const auto *E : List) {
    ElementTypes.push_back(E->getType());
  }

  return getContext().CreateTupleExpr(
      getContext().CreateTupleType(ElementTypes), List);
}

PtxExprResult PtxParser::ParseInstructionDestOperand() {
  PtxExprResult FirstOp;
  switch (getTok().getKind()) {
  case AsmToken::Identifier:
    if (PtxVariableDecl *S = getCurScope()->LookupSymbol(getTok().getString()))
      FirstOp = getContext().CreateDeclRefExpr(S);
    else
      return true;
    Lex(); // eat identifier
    break;
  case AsmToken::Sink:
    FirstOp = getContext().CreateSinkExpr();
    Lex(); // eat '_'
    break;
  case AsmToken::LBrac:
    FirstOp = ParseTuple();
    break;
  default:
    return true;
  }

  return FirstOp;
}

PtxExprResult PtxParser::ParseInstructionPrimaryOperand() {
  if (getTok().is(AsmToken::Identifier)) {
    auto ID = getTok();
    Lex(); // eat identifier
    if (PtxVariableDecl *S = getCurScope()->LookupSymbol(ID.getString()))
      return getContext().CreateDeclRefExpr(S);
    return true;
  }

  if (getTok().is(AsmToken::LCurly))
    return ParseTuple();

  return ParseConstantExpression();
}

PtxExprResult PtxParser::ParseInstructionUnaryOperand() {
  AsmToken Tok = getTok();
  PtxExprResult Operand = ParseInstructionPrimaryOperand();
  if (Operand.isInvalid())
    return true;
  if (Tok.is(AsmToken::Exclaim, AsmToken::Minus)) {
    switch (Tok.getKind()) {
    case AsmToken::Exclaim:
      return getContext().CreateUnaryOperator(PtxUnaryOperator::LNot,
                                              Operand.get());
    case AsmToken::Minus:
      return getContext().CreateUnaryOperator(PtxUnaryOperator::Minus,
                                              Operand.get());
    default:
      break;
    }
    return true;
  }
  return Operand;
}

PtxExprResult PtxParser::ParseInstructionSrcOperand() {
  PtxExprResult Operand = ParseInstructionUnaryOperand();
  if (Operand.isInvalid())
    return true;

  /// TODO: Parse operand postfix here, e.g. var[Imm], [Imm], ...
  return Operand;
}

PtxStmtResult PtxParser::ParseDeclStmt() {
  PtxVariableDecl::Attribute Attr;

  PtxTypeResult Declspec = ParseVarDeclspec(Attr);
  if (Declspec.isInvalid())
    return true;

  SmallVector<const PtxDecl *> DeclGroup;
  bool FirstDecl = true;
  while (getTok().isNot(AsmToken::EndOfStatement)) {
    if (!FirstDecl && getTok().is(AsmToken::Comma))
      return true;
    
    if (FirstDecl)
      FirstDecl = false;
    
    PtxDeclResult D = ParseVariableDecl(Declspec.get());
    if (D.isInvalid())
      return true;
    getCurScope()->AddDecl(dyn_cast<PtxVariableDecl>(D.get()));
    DeclGroup.push_back(D.get());
  }

  Lex(); // eat ';'
  return getContext().CreateDeclStmt(DeclGroup);
}

PtxTypeResult PtxParser::ParseVarDeclspec(PtxVariableDecl::Attribute Attr) {

  if (!getTok().isStorageClass()) // unexpected dot identifier
    return true;
  
  Lex(); // eat storage class

  bool HasAlign = getTok().getString() == ".align";

  if (HasAlign) {
    Lex(); // eat '.align'
    if (getTok().isNot(AsmToken::Integer))
      return true; // expected an integer
    Attr.Align = getTok().getIntVal();
    Lex(); // eat integer
  }

  bool IsV2 = false, IsV4 = false;
  if (getTok().getString() == ".v2") {
    IsV2 = true;
    Lex(); // eat '.v2'
  } else if (getTok().getString() == ".v4") {
    IsV4 = true;
    Lex(); // eat '.v4'
  }

  PtxFundamentalType *BaseType = getContext().GetOrCreateFundamentalType(getTok().getString());
  PtxType *VarType = BaseType;
  if (IsV2) {
    VarType = getContext().CreateVectorType(PtxVectorType::V2, BaseType);
  } else if (IsV4) {
    VarType = getContext().CreateVectorType(PtxVectorType::V4, BaseType);
  }

  Lex(); // eat a type name

  return VarType;
}

PtxDeclResult PtxParser::ParseVariableDecl(const PtxType *Type) {
  if (getTok().isNot(AsmToken::Identifier))
    return true;  // expected an identifier
  
  StringRef VarName = getTok().getIdentifier();

  Lex(); // eat identifier

  // Parse Parameterized Variable Names
  if (getTok().is(AsmToken::Less)) {
    /// TODO: Parameterized Variable Names
  }

  // Parse array
  if (getTok().is(AsmToken::LBrac)) {
    /// TODO: Parse array
  }

  return getContext().CreateVariableDecl(VarName, Type);
}

PtxExprResult PtxParser::ParseConstantExpression() {
  return ParseConditionalExpression();
}

PtxExprResult PtxParser::ParseConditionalExpression() {
  PtxExprResult LogicOrExpr = ParseLogicOrExpression();
  if (LogicOrExpr.isInvalid())
    return true;
  if (getTok().is(AsmToken::Question)) {
    Lex(); // eat '?'
    PtxExprResult Then = ParseConditionalExpression();
    if (Then.isInvalid() || getTok().isNot(AsmToken::Colon))
      return true;
    Lex(); // eat ':'
    PtxExprResult Else = ParseConditionalExpression();
    if (Else.isInvalid())
      return true;
    return getContext().CreateConditionalOperator(LogicOrExpr.get(), Then.get(),
                                                  Else.get());
  }
  return LogicOrExpr;
}

PtxExprResult PtxParser::ParseLogicOrExpression() {
  PtxExprResult LHS = ParseLogicAndExpression();
  if (LHS.isInvalid())
    return true;
  while (getTok().is(AsmToken::PipePipe)) {
    Lex(); // eat '||'
    PtxExprResult RHS = ParseLogicAndExpression();
    if (RHS.isInvalid())
      return true;
    LHS = getContext().CreateBinaryOperator(PtxBinaryOperator::LOr, LHS.get(),
                                            RHS.get());
  }
  return LHS;
}

PtxExprResult PtxParser::ParseLogicAndExpression() {
  PtxExprResult LHS = ParseInclusiveOrExpression();
  if (LHS.isInvalid())
    return true;
  while (getTok().is(AsmToken::AmpAmp)) {
    Lex(); // eat '&&'
    PtxExprResult RHS = ParseInclusiveOrExpression();
    if (RHS.isInvalid())
      return true;
    LHS = getContext().CreateBinaryOperator(PtxBinaryOperator::LAnd, LHS.get(),
                                            RHS.get());
  }
  return LHS;
}

PtxExprResult PtxParser::ParseInclusiveOrExpression() {
  PtxExprResult LHS = ParseExclusiveOrExpression();
  if (LHS.isInvalid())
    return true;
  while (getTok().is(AsmToken::Pipe)) {
    Lex(); // eat '|'
    PtxExprResult RHS = ParseInclusiveOrExpression();
    if (RHS.isInvalid())
      return true;
    LHS = getContext().CreateBinaryOperator(PtxBinaryOperator::Or, LHS.get(),
                                            RHS.get());
  }
  return LHS;
}

PtxExprResult PtxParser::ParseExclusiveOrExpression() {
  PtxExprResult LHS = ParseAndExpression();
  if (LHS.isInvalid())
    return true;
  while (getTok().is(AsmToken::Caret)) {
    Lex(); // eat '^'
    PtxExprResult RHS = ParseAndExpression();
    if (RHS.isInvalid())
      return true;
    LHS = getContext().CreateBinaryOperator(PtxBinaryOperator::Xor, LHS.get(),
                                            RHS.get());
  }
  return LHS;
}

PtxExprResult PtxParser::ParseAndExpression() {
  PtxExprResult LHS = ParseEqualityExpression();
  if (LHS.isInvalid())
    return true;
  while (getTok().is(AsmToken::Amp)) {
    Lex(); // eat '&'
    PtxExprResult RHS = ParseEqualityExpression();
    if (RHS.isInvalid())
      return true;
    LHS = getContext().CreateBinaryOperator(PtxBinaryOperator::And, LHS.get(),
                                            RHS.get());
  }
  return LHS;
}

PtxExprResult PtxParser::ParseEqualityExpression() {
  PtxExprResult LHS = ParseRelationExpression();
  if (LHS.isInvalid())
    return true;
  while (getTok().is(AsmToken::EqualEqual, AsmToken::ExclaimEqual)) {
    auto Opcode = getTok().getKind();
    Lex(); // eat '==' or '!='
    PtxExprResult RHS = ParseRelationExpression();
    if (RHS.isInvalid())
      return true;
    if (Opcode == AsmToken::EqualEqual)
      LHS = getContext().CreateBinaryOperator(PtxBinaryOperator::EQ, LHS.get(),
                                              RHS.get());
    else
      LHS = getContext().CreateBinaryOperator(PtxBinaryOperator::NE, LHS.get(),
                                              RHS.get());
  }
  return LHS;
}

PtxExprResult PtxParser::ParseRelationExpression() {
  PtxExprResult LHS = ParseShiftExpression();
  if (LHS.isInvalid())
    return true;
  while (getTok().is(AsmToken::Less, AsmToken::Greater, AsmToken::LessEqual,
                     AsmToken::GreaterEqual)) {
    auto Opcode = getTok().getKind();
    Lex(); // eat one of '<', '>', '<=' and '>='
    PtxExprResult RHS = ParseRelationExpression();
    if (RHS.isInvalid())
      return true;
    switch (Opcode) {
    case AsmToken::Less:
      LHS = getContext().CreateBinaryOperator(PtxBinaryOperator::LT, LHS.get(),
                                              RHS.get());
      break;
    case AsmToken::Greater:
      LHS = getContext().CreateBinaryOperator(PtxBinaryOperator::GT, LHS.get(),
                                              RHS.get());
      break;
    case AsmToken::LessEqual:
      LHS = getContext().CreateBinaryOperator(PtxBinaryOperator::LE, LHS.get(),
                                              RHS.get());
      break;
    case AsmToken::GreaterEqual:
      LHS = getContext().CreateBinaryOperator(PtxBinaryOperator::GE, LHS.get(),
                                              RHS.get());
      break;
    default:
      assert(false && "Invalid relation operator kind");
    }
  }
  return LHS;
}

PtxExprResult PtxParser::ParseShiftExpression() {
  PtxExprResult LHS = ParseAdditiveExpression();
  if (LHS.isInvalid())
    return true;
  while (getTok().is(AsmToken::LessLess, AsmToken::GreaterGreater)) {
    auto Opcode = getTok().getKind();
    Lex(); // eat '<<' or '>>'
    PtxExprResult RHS = ParseAdditiveExpression();
    if (RHS.isInvalid())
      return true;
    if (Opcode == AsmToken::LessLess)
      LHS = getContext().CreateBinaryOperator(PtxBinaryOperator::Shl, LHS.get(),
                                              RHS.get());
    else
      LHS = getContext().CreateBinaryOperator(PtxBinaryOperator::Shr, LHS.get(),
                                              RHS.get());
  }
  return LHS;
}

PtxExprResult PtxParser::ParseAdditiveExpression() {
  PtxExprResult LHS = ParseMultiplicativeExpression();
  if (LHS.isInvalid())
    return true;
  while (getTok().is(AsmToken::Plus, AsmToken::Minus)) {
    auto Opcode = getTok().getKind();
    Lex(); // eat '+' or '-'
    PtxExprResult RHS = ParseMultiplicativeExpression();
    if (RHS.isInvalid())
      return true;
    if (Opcode == AsmToken::Plus)
      LHS = getContext().CreateBinaryOperator(PtxBinaryOperator::Add, LHS.get(),
                                              RHS.get());
    else
      LHS = getContext().CreateBinaryOperator(PtxBinaryOperator::Sub, LHS.get(),
                                              RHS.get());
  }
  return LHS;
}

PtxExprResult PtxParser::ParseMultiplicativeExpression() {
  PtxExprResult LHS = ParseCastExpresion();
  if (LHS.isInvalid())
    return true;
  while (getTok().is(AsmToken::Star, AsmToken::Slash, AsmToken::Percent)) {
    auto Opcode = getTok().getKind();
    Lex(); // eat one of '*', '/' and '%'
    PtxExprResult RHS = ParseCastExpresion();
    if (RHS.isInvalid())
      return true;
    switch (Opcode) {
    case AsmToken::Star:
      LHS = getContext().CreateBinaryOperator(PtxBinaryOperator::Mul, LHS.get(),
                                              RHS.get());
      break;
    case AsmToken::Slash:
      LHS = getContext().CreateBinaryOperator(PtxBinaryOperator::Div, LHS.get(),
                                              RHS.get());
      break;
    case AsmToken::Percent:
      LHS = getContext().CreateBinaryOperator(PtxBinaryOperator::Rem, LHS.get(),
                                              RHS.get());
      break;
    default:
      assert(false && "Invalid multiplicative operator kind");
    }
  }
  return LHS;
}

PtxExprResult PtxParser::ParseCastExpresion() {
  if (getTok().is(AsmToken::LParen) && Lexer.peekTok().isTypeName()) {
    Lex();                     // eat '('
    AsmToken TypeName = Lex(); // eat typename
    if (getTok().isNot(AsmToken::RParen))
      return true;
    PtxExprResult SubExpr = ParseUnaryExpression();
    if (SubExpr.isInvalid())
      return true;
    PtxFundamentalType *CastType =
        getContext().GetOrCreateFundamentalType(TypeName.getString());
    return getContext().CreateCastExpression(CastType, SubExpr.get());
  }

  return ParseUnaryExpression();
}

PtxExprResult PtxParser::ParseUnaryExpression() {
  if (getTok().is(AsmToken::Plus, AsmToken::Minus, AsmToken::Exclaim,
                  AsmToken::Tilde)) {
    auto Opcode = getTok().getKind();
    Lex(); // eat unary operator
    PtxExprResult SubExpr = ParseCastExpresion();
    if (SubExpr.isInvalid())
      return true;
    switch (Opcode) {
    case AsmToken::Plus:
      return SubExpr;
    case AsmToken::Minus:
      return getContext().CreateUnaryOperator(PtxUnaryOperator::Minus,
                                              SubExpr.get());
    case AsmToken::Exclaim:
      return getContext().CreateUnaryOperator(PtxUnaryOperator::LNot,
                                              SubExpr.get());
    case AsmToken::Tilde:
      return getContext().CreateUnaryOperator(PtxUnaryOperator::Not,
                                              SubExpr.get());
    default:
      assert(false && "Invalid unary operator kind");
    }
  }
  return ParsePrimaryExpression();
}

PtxExprResult PtxParser::ParsePrimaryExpression() {
  if (getTok().is(AsmToken::Identifier) &&
      getTok().getString() == "WARP_SIZE") {
    auto *Symbol = getCurScope()->LookupSymbol(getTok().getString());
    return getContext().CreateDeclRefExpr(Symbol);
  }

  auto Tok = getTok();
  Lex();
  switch (Tok.getKind()) {
  case AsmToken::Identifier: {
    auto *Symbol = getCurScope()->LookupSymbol(Tok.getString());
    if (Symbol)
      return getContext().CreateDeclRefExpr(Symbol);
    return true;
  }
  case AsmToken::Integer:
    return getContext().CreateIntegerLiteral(
        getContext().GetOrCreateFundamentalType(PtxFundamentalType::TK_S64),
        llvm::APInt(64, Tok.getIntVal(), true));
  case AsmToken::Unsigned:
    return getContext().CreateIntegerLiteral(
        getContext().GetOrCreateFundamentalType(PtxFundamentalType::TK_U64),
        llvm::APInt(64, Tok.getUnsignedVal(), false));
  case AsmToken::Float:
    return getContext().CreateFloatLiteral(
        getContext().GetOrCreateFundamentalType(PtxFundamentalType::TK_F32),
        llvm::APFloat(Tok.getF32Val()));
  case AsmToken::Double:
    return getContext().CreateFloatLiteral(
        getContext().GetOrCreateFundamentalType(PtxFundamentalType::TK_F64),
        llvm::APFloat(Tok.getF64Val()));
  case AsmToken::LParen: {
    PtxExprResult Expr = ParseConstantExpression();
    if (getTok().isNot(AsmToken::RParen))
      return true;
    Lex(); // eat ')'
    return Expr;
  }
  default:
    break;
  }
  return true;
}
