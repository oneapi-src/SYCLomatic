//===------------------------- AsmParser.cpp --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AsmParser.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace clang::dpct;

AsmType *AsmContext::getScalarType(AsmType::TypeKind Kind) {
  if (Kind < AsmType::TK_B8 || Kind > AsmType::TK_Pred)
    return nullptr;
  if (ScalarTypes.find(Kind) == ScalarTypes.end()) {
    AsmType *Type = new (*this) AsmType;
    Type->Kind = Kind;
    ScalarTypes[Kind] = Type;
    return Type;
  }
  return ScalarTypes[Kind];
}

AsmType *AsmContext::getScalarTypeFromName(StringRef TypeName) {
  if (TypeName == ".s32")
    return getScalarType(AsmType::TK_S32);
  if (TypeName == ".s64")
    return getScalarType(AsmType::TK_S64);
  if (TypeName == ".u32")
    return getScalarType(AsmType::TK_U32);
  if (TypeName == ".u64")
    return getScalarType(AsmType::TK_U64);
  if (TypeName == ".pred")
    return getScalarType(AsmType::TK_Pred);
  return nullptr;
}

AsmSymbol *AsmScope::LookupSymbol(StringRef Symbol) const {
  for (const auto &S : decls()) {
    if (S->Name == Symbol)
      return S;
  }

  if (this->AnyParent) {
    return this->AnyParent->LookupSymbol(Symbol);
  }
  return nullptr;
}

AsmStatement *AsmContext::CreateStmt(AsmStatement::StmtKind Kind) {
  AsmStatement *Stmt = new (*this) AsmStatement(Kind);
  return Stmt;
}

AsmStatement *AsmContext::CreateIntegerConstant(AsmType *Type, int64_t Val) {
  AsmStatement *Const = CreateStmt(AsmStatement::SK_Integer);
  Const->i64 = Val;
  Const->Type = Type;
  return Const;
}

AsmStatement *AsmContext::CreateIntegerConstant(AsmType *Type, uint64_t Val) {
  AsmStatement *Const = CreateStmt(AsmStatement::SK_Unsigned);
  Const->u64 = Val;
  Const->Type = Type;
  return Const;
}

AsmStatement *AsmContext::CreateFloatConstant(AsmType *Type, float Val) {
  AsmStatement *Fp = CreateStmt(AsmStatement::SK_Float);
  Fp->f32 = Val;
  Fp->Type = Type;
  llvm::APFloat aaa(0.0);
  return Fp;
}

AsmStatement *AsmContext::CreateFloatConstant(AsmType *Type, double Val) {
  AsmStatement *Fp = CreateStmt(AsmStatement::SK_Double);
  Fp->f64 = Val;
  Fp->Type = Type;
  return Fp;
}

AsmStatement *AsmContext::CreateConditionalExpression(AsmStatement *Cond, AsmStatement *Then,
                                                 AsmStatement *Else) {
  AsmStatement *S = CreateStmt(AsmStatement::SK_Cond);
  S->Cond = Cond;
  S->Then = Then;
  S->Else = Else;
  return S;
}

AsmStatement *AsmContext::CreateBinaryOperator(AsmStatement::StmtKind Opcode,
                                          AsmStatement *LHS, AsmStatement *RHS) {
  AsmStatement *S = CreateStmt(Opcode);
  S->LHS = LHS;
  S->RHS = RHS;
  return S;
}

AsmStatement *AsmContext::CreateCastExpression(AsmType *Type, AsmStatement *SubExpr) {
  AsmStatement *S = CreateStmt(AsmStatement::SK_Cast);
  S->Type = Type;
  S->SubExpr = SubExpr;
  return S;
}

AsmStatement *AsmContext::CreateUnaryExpression(AsmStatement::StmtKind Opcode,
                                           AsmStatement *SubExpr) {
  AsmStatement *S = CreateStmt(Opcode);
  S->SubExpr = SubExpr;
  return S;
}

AsmStatement *AsmContext::CreateVariableRefExpression(AsmSymbol *Symbol) {
  AsmStatement *S = CreateStmt(AsmStatement::SK_Variable);
  S->Variable = Symbol;
  return S;
}

AsmStatement *AsmContext::GetOrCreateSinkExpression() {
  if (SinkExpression)
    return SinkExpression;
  return SinkExpression = CreateStmt(AsmStatement::SK_Sink);
}

AsmParser::~AsmParser() = default;

const AsmToken &AsmParser::Lex() {
  if (Lexer.getTok().is(AsmToken::Error))
    return Lexer.getTok();

  const AsmToken *tok = &Lexer.Lex();

  // Parse comments here to be deferred until end of next statement.
  while (tok->is(AsmToken::Comment)) {
    tok = &Lexer.Lex();
  }

  return *tok;
}

AsmStmtResult AsmParser::ParseStatement() {
  if (getTok().is(AsmToken::LBrac)) {
    return ParseCompoundStatement();
  }
  return ParseInstruction();
}

AsmStmtResult AsmParser::ParseCompoundStatement() {
  assert(getTok().is(AsmToken::LBrac));
  Lex(); // eat '{'
  AsmStatement *Block = Context.CreateStmt(AsmStatement::SK_Block);
  ParseScope BlockScope(this);

  SmallVector<AsmStatement *, 10> Stmts;
  while (getTok().isNot(AsmToken::RBrac)) {
    if (getTok().isVarAttributes()) {
      AsmStmtResult Res; /* = ParseDeclaration*/
      if (Res.isUsable())
        Stmts.push_back(Res.get());
      else
        return Res;
    } else {
      AsmStmtResult Res = ParseStatement();
      if (Res.isInvalid())
        return Res;
      Stmts.push_back(Res.get());
    }
  }
  return Block;
}

AsmStmtResult AsmParser::ParsePredicate() {
  return true;
}

AsmStmtResult AsmParser::ParseInstruction() {
  AsmStatement *Inst = Context.CreateStmt(AsmStatement::SK_Inst);
  if (getTok().is(AsmToken::At)) {
    AsmStmtResult PredExpr = ParsePredicate();
    if (PredExpr.isInvalid())
      return true;
    Inst->Pred = PredExpr.get();
  }

  AsmStmtResult UnGuardInst = ParseUnGuardInstruction();

  if (UnGuardInst.isInvalid())
    return true;

  Inst->Body = UnGuardInst.get();
  return Inst;
}

AsmStmtResult AsmParser::ParseUnGuardInstruction() {
  if (getTok().isNot(AsmToken::Identifier))
    return true;
  AsmStatement *Inst = Context.CreateStmt(AsmStatement::SK_Inst);
  Inst->InstructionAttr = ParseInstructionFlags();

  while (getTok().isNot(AsmToken::EndOfStatement)) {
    AsmStmtResult Operand = ParseInstructionOperand();
    if (Operand.isInvalid())
      return true;
    Inst->Operands.push_back(Operand.get());
  }
  return Inst;
}

InstAttr AsmParser::ParseInstructionFlags() {
  InstAttr Inst;
  return Inst;
}

AsmStmtResult AsmParser::ParseTuple() {
  Lex(); // eat '{'
  AsmStmtResult Tuple = Context.CreateStmt(AsmStatement::SK_Tuple);
  do {
    if (getTok().is(AsmToken::Comma))
      Lex(); // eat ','
    switch (getTok().getKind()) {
    if (AsmSymbol *S = getCurScope()->LookupSymbol(getTok().getString()))
      Tuple.get()->Tuple.push_back(Context.CreateVariableRefExpression(S));
      else
        return true;
      break;
    case AsmToken::Sink:
      Tuple.get()->Tuple.push_back(Context.GetOrCreateSinkExpression());
      break;
    default:
      auto Const = ParseConstantExpression();
      if (Const.isInvalid())
        return true;
      Tuple.get()->Tuple.push_back(Const.get());
    }
  } while (getTok().is(AsmToken::Comma));
  if (getTok().isNot(AsmToken::RBrac))
    return true;
  return Tuple;
}

AsmStmtResult AsmParser::ParseInstructionFirstOperand() {
  AsmStmtResult FirstOp;
  switch (getTok().getKind()) {
  case AsmToken::Identifier:
    if (AsmSymbol *S = getCurScope()->LookupSymbol(getTok().getString()))
      FirstOp = Context.CreateVariableRefExpression(S);
    else
      return true;
    break;
  case AsmToken::Sink:
    FirstOp = Context.GetOrCreateSinkExpression();
    break;
  case AsmToken::LBrac:
    FirstOp = ParseTuple();
    break;
  default:
    return true;
  }

  Lex(); // eat '_' or identifier

  // Parse predicate output
  if (getTok().is(AsmToken::Pipe)) {
    Lex(); // eat '|'
    switch (getTok().getKind()) {
    case AsmToken::Identifier:
      if (AsmSymbol *S = getCurScope()->LookupSymbol(getTok().getString()))
        FirstOp.get()->PredOutput = Context.CreateVariableRefExpression(S);
      else
        return true;
      break;
    case AsmToken::Sink:
      FirstOp.get()->PredOutput = Context.GetOrCreateSinkExpression();
      break;
    default:
      // expect an identifier or '_'
      return true;
    }
    Lex(); // eat identifier or '_'
  }

  return FirstOp;
}

AsmStmtResult AsmParser::ParseInstructionOperand() {
  return {};
}

AsmStmtResult AsmParser::ParseConstantExpression() {
  switch (getTok().getKind()) {
  case AsmToken::Float:
    return Context.CreateFloatConstant(Context.getScalarType(AsmType::TK_F32), getTok().getF32Val());
  case AsmToken::Double:
    return Context.CreateFloatConstant(Context.getScalarType(AsmType::TK_F64), getTok().getF64Val());
  default:
    break;
  }
  return ParseConditionalExpression();
}

AsmStmtResult AsmParser::ParseConditionalExpression() {
  AsmStmtResult LogicOrExpr = ParseLogicOrExpression();
  if (LogicOrExpr.isInvalid())
    return true;
  if (getTok().is(AsmToken::Question)) {
    Lex(); // eat '?'
    AsmStmtResult Then = ParseConditionalExpression();
    if (Then.isInvalid() || getTok().isNot(AsmToken::Colon))
      return true;
    AsmStmtResult Else = ParseConditionalExpression();
    if (Else.isInvalid())
      return true;
    return Context.CreateConditionalExpression(LogicOrExpr.get(), Then.get(),
                                               Else.get());
  }
  return LogicOrExpr;
}

AsmStmtResult AsmParser::ParseLogicOrExpression() {
  AsmStmtResult LHS = ParseLogicAndExpression();
  if (LHS.isInvalid())
    return true;
  while (getTok().is(AsmToken::PipePipe)) {
    Lex(); // eat '||'
    AsmStmtResult RHS = ParseLogicAndExpression();
    if (RHS.isInvalid())
      return true;
    LHS = Context.CreateBinaryOperator(AsmStatement::SK_Or, LHS.get(), RHS.get());
  }
  return LHS;
}

AsmStmtResult AsmParser::ParseLogicAndExpression() {
  AsmStmtResult LHS = ParseInclusiveOrExpression();
  if (LHS.isInvalid())
    return true;
  while (getTok().is(AsmToken::AmpAmp)) {
    Lex(); // eat '&&'
    AsmStmtResult RHS = ParseInclusiveOrExpression();
    if (RHS.isInvalid())
      return true;
    LHS = Context.CreateBinaryOperator(AsmStatement::SK_And, LHS.get(), RHS.get());
  }
  return LHS;
}

AsmStmtResult AsmParser::ParseInclusiveOrExpression() {
  AsmStmtResult LHS = ParseExclusiveOrExpression();
  if (LHS.isInvalid())
    return true;
  while (getTok().is(AsmToken::Pipe)) {
    Lex(); // eat '|'
    AsmStmtResult RHS = ParseInclusiveOrExpression();
    if (RHS.isInvalid())
      return true;
    LHS = Context.CreateBinaryOperator(AsmStatement::SK_BitOr, LHS.get(), RHS.get());
  }
  return LHS;
}

AsmStmtResult AsmParser::ParseExclusiveOrExpression() {
  AsmStmtResult LHS = ParseAndExpression();
  if (LHS.isInvalid())
    return true;
  while (getTok().is(AsmToken::Caret)) {
    Lex(); // eat '^'
    AsmStmtResult RHS = ParseAndExpression();
    if (RHS.isInvalid())
      return true;
    LHS =
        Context.CreateBinaryOperator(AsmStatement::SK_BitXor, LHS.get(), RHS.get());
  }
  return LHS;
}

AsmStmtResult AsmParser::ParseAndExpression() {
  AsmStmtResult LHS = ParseEqualityExpression();
  if (LHS.isInvalid())
    return true;
  while (getTok().is(AsmToken::Amp)) {
    Lex(); // eat '&'
    AsmStmtResult RHS = ParseEqualityExpression();
    if (RHS.isInvalid())
      return true;
    LHS =
        Context.CreateBinaryOperator(AsmStatement::SK_BitAnd, LHS.get(), RHS.get());
  }
  return LHS;
}

AsmStmtResult AsmParser::ParseEqualityExpression() {
  AsmStmtResult LHS = ParseRelationExpression();
  if (LHS.isInvalid())
    return true;
  while (getTok().is(AsmToken::EqualEqual, AsmToken::ExclaimEqual)) {
    auto Opcode = getTok().getKind();
    Lex(); // eat '==' or '!='
    AsmStmtResult RHS = ParseRelationExpression();
    if (RHS.isInvalid())
      return true;
    if (Opcode == AsmToken::EqualEqual)
      LHS = Context.CreateBinaryOperator(AsmStatement::SK_EQ, LHS.get(), RHS.get());
    else
      LHS = Context.CreateBinaryOperator(AsmStatement::SK_NE, LHS.get(), RHS.get());
    ;
  }
  return LHS;
}

AsmStmtResult AsmParser::ParseRelationExpression() {
  AsmStmtResult LHS = ParseShiftExpression();
  if (LHS.isInvalid())
    return true;
  while (getTok().is(AsmToken::Less, AsmToken::Greater, AsmToken::LessEqual,
                     AsmToken::GreaterEqual)) {
    auto Opcode = getTok().getKind();
    Lex(); // eat one of '<', '>', '<=' and '>='
    AsmStmtResult RHS = ParseRelationExpression();
    if (RHS.isInvalid())
      return true;
    switch (Opcode) {
    case AsmToken::Less:
      LHS = Context.CreateBinaryOperator(AsmStatement::SK_LT, LHS.get(), RHS.get());
      break;
    case AsmToken::Greater:
      LHS = Context.CreateBinaryOperator(AsmStatement::SK_GT, LHS.get(), RHS.get());
      break;
    case AsmToken::LessEqual:
      LHS = Context.CreateBinaryOperator(AsmStatement::SK_LE, LHS.get(), RHS.get());
      break;
    case AsmToken::GreaterEqual:
      LHS = Context.CreateBinaryOperator(AsmStatement::SK_GE, LHS.get(), RHS.get());
      break;
    default:
      assert(false && "Invalid relation operator kind");
    }
  }
  return LHS;
}

AsmStmtResult AsmParser::ParseShiftExpression() {
  AsmStmtResult LHS = ParseAdditiveExpression();
  if (LHS.isInvalid())
    return true;
  while (getTok().is(AsmToken::LessLess, AsmToken::GreaterGreater)) {
    auto Opcode = getTok().getKind();
    Lex(); // eat '<<' or '>>'
    AsmStmtResult RHS = ParseAdditiveExpression();
    if (RHS.isInvalid())
      return true;
    if (Opcode == AsmToken::LessLess)
      LHS = Context.CreateBinaryOperator(AsmStatement::SK_Shl, LHS.get(), RHS.get());
    else
      LHS = Context.CreateBinaryOperator(AsmStatement::SK_Shr, LHS.get(), RHS.get());
    ;
  }
  return LHS;
}

AsmStmtResult AsmParser::ParseAdditiveExpression() {
  AsmStmtResult LHS = ParseMultiplicativeExpression();
  if (LHS.isInvalid())
    return true;
  while (getTok().is(AsmToken::Plus, AsmToken::Minus)) {
    auto Opcode = getTok().getKind();
    Lex(); // eat '+' or '-'
    AsmStmtResult RHS = ParseMultiplicativeExpression();
    if (RHS.isInvalid())
      return true;
    if (Opcode == AsmToken::Plus)
      LHS = Context.CreateBinaryOperator(AsmStatement::SK_Add, LHS.get(), RHS.get());
    else
      LHS = Context.CreateBinaryOperator(AsmStatement::SK_Sub, LHS.get(), RHS.get());
  }
  return LHS;
}

AsmStmtResult AsmParser::ParseMultiplicativeExpression() {
  AsmStmtResult LHS = ParseCaseExpresion();
  if (LHS.isInvalid())
    return true;
  while (getTok().is(AsmToken::Star, AsmToken::Slash, AsmToken::Percent)) {
    auto Opcode = getTok().getKind();
    Lex(); // eat one of '*', '/' and '%'
    AsmStmtResult RHS = ParseCaseExpresion();
    if (RHS.isInvalid())
      return true;
    switch (Opcode) {
    case AsmToken::Star:
      LHS = Context.CreateBinaryOperator(AsmStatement::SK_Mul, LHS.get(), RHS.get());
      break;
    case AsmToken::Slash:
      LHS = Context.CreateBinaryOperator(AsmStatement::SK_Div, LHS.get(), RHS.get());
      break;
    case AsmToken::Percent:
      LHS = Context.CreateBinaryOperator(AsmStatement::SK_Mod, LHS.get(), RHS.get());
      break;
    default:
      assert(false && "Invalid multiplicative operator kind");
    }
  }
  return LHS;
}

AsmStmtResult AsmParser::ParseCaseExpresion() {
  if (getTok().is(AsmToken::LParen) && Lexer.peekTok().isTypeName()) {
    Lex();                     // eat '('
    AsmToken TypeName = Lex(); // eat typename
    if (getTok().isNot(AsmToken::RParen))
      return true;
    AsmStmtResult SubExpr = ParseUnaryExpression();
    if (SubExpr.isInvalid())
      return true;
    AsmType *CastType = getContext().getScalarTypeFromName(TypeName.getString());
    return Context.CreateCastExpression(CastType, SubExpr.get());
  }

  return ParseUnaryExpression();
}

AsmStmtResult AsmParser::ParseUnaryExpression() {
  if (getTok().is(AsmToken::Plus, AsmToken::Minus, AsmToken::Exclaim,
                  AsmToken::Tilde)) {
    auto Opcode = getTok().getKind();
    Lex(); // eat unary operator
    AsmStmtResult SubExpr = ParseCaseExpresion();
    if (SubExpr.isInvalid())
      return true;
    switch (Opcode) {
    case AsmToken::Plus:
      return SubExpr;
    case AsmToken::Minus:
      return Context.CreateUnaryExpression(AsmStatement::SK_Neg, SubExpr.get());
    case AsmToken::Exclaim:
      return Context.CreateUnaryExpression(AsmStatement::SK_Not, SubExpr.get());
    case AsmToken::Tilde:
      return Context.CreateUnaryExpression(AsmStatement::SK_BitNot, SubExpr.get());
    default:
      assert(false && "Invalid unary operator kind");
    }
  }
  return ParsePrimaryExpression();
}

AsmStmtResult AsmParser::ParsePrimaryExpression() {
  if (getTok().is(AsmToken::Identifier) && getTok().getString() == "WARP_SIZE") {
    auto *Symbol = getCurScope()->LookupSymbol(getTok().getString());
    return Context.CreateVariableRefExpression(Symbol);
  }
  switch (getTok().getKind()) {
  case AsmToken::Identifier: {
    if (getTok().getString() == "WARP_SIZE") {
      auto *Symbol = getCurScope()->LookupSymbol(getTok().getString());
      return Context.CreateVariableRefExpression(Symbol);
    }
    return true;
  }
  case AsmToken::Integer:
    return Context.CreateIntegerConstant(Context.getScalarType(AsmType::TK_S64), getTok().getIntVal());
  case AsmToken::Unsigned:
    return Context.CreateIntegerConstant(Context.getScalarType(AsmType::TK_U64), getTok().getUnsignedVal());
  case AsmToken::Double:
    return Context.CreateFloatConstant(Context.getScalarType(AsmType::TK_F64), getTok().getF64Val());
  case AsmToken::LParen: {
    Lex(); // eat '('
    AsmStmtResult Expr = ParseConstantExpression();
    if (getTok().is(AsmToken::RParen))
      return true;
    Lex(); // eat ')'
    return Expr;
  }
  default:
    break;
  }
  return true;
}
