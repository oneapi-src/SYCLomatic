//===------------------------- AsmParser.cpp --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AsmParser.h"
#include "Asm/AsmIdentifierTable.h"
#include "Asm/AsmLexer.h"
#include "Asm/AsmToken.h"
#include "Asm/AsmTokenKinds.h"
#include "clang/AST/Type.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Lex/LiteralSupport.h"
#include "clang/Sema/Ownership.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <limits>
#include <type_traits>

using namespace llvm;
using namespace clang::dpct;

// clang-format off
asm_precedence::Level getBinOpPrec(asmtok::TokenKind Kind) {
  switch (Kind) {
  default:                           return asm_precedence::Unknown;
  case asmtok::question:             return asm_precedence::Conditional;
  case asmtok::pipepipe:             return asm_precedence::LogicalOr;
  case asmtok::ampamp:               return asm_precedence::LogicalAnd;
  case asmtok::pipe:                 return asm_precedence::InclusiveOr;
  case asmtok::caret:                return asm_precedence::ExclusiveOr;
  case asmtok::amp:                  return asm_precedence::And;
  case asmtok::exclaimequal:
  case asmtok::equalequal:           return asm_precedence::Equality;
  case asmtok::lessequal:
  case asmtok::less:
  case asmtok::greater:
  case asmtok::greaterequal:         return asm_precedence::Relational;
  case asmtok::greatergreater:
  case asmtok::lessless:             return asm_precedence::Shift;
  case asmtok::plus:
  case asmtok::minus:                return asm_precedence::Additive;
  case asmtok::percent:
  case asmtok::slash:
  case asmtok::star:                 return asm_precedence::Multiplicative;
  }
}
// clang-format on

InlineAsmVariableDecl *
InlineAsmScope::lookupDecl(InlineAsmIdentifierInfo *II) const {
  if (!II)
    return nullptr;
  for (const auto &S : decls()) {
    if (S->getDeclName() == II)
      return S;
  }

  if (hasParent()) {
    return getParent()->lookupDecl(II);
  }
  return nullptr;
}

InlineAsmVariableDecl *
InlineAsmScope::lookupParameterizedNameDecl(InlineAsmIdentifierInfo *II,
                                            unsigned &Idx) const {
  if (!II)
    return nullptr;

  StringRef Name = II->getName().take_while(
      [](char C) -> bool { return isAlpha(C) || C == '_'; });
  StringRef Count = II->getName().drop_until(isDigit);

  if (Name.empty() || Count.empty() || Count.getAsInteger(10, Idx))
    return nullptr;

  InlineAsmIdentifierInfo *RealII = Parser.getLexer().getIdentifierInfo(Name);
  if (!RealII)
    return nullptr;

  InlineAsmVariableDecl *D = lookupDecl(RealII);
  if (D && D->isParameterizedNameDecl() && Idx < D->getNumParameterizedNames())
    return D;
  return nullptr;
}

void InlineAsmParser::addBuiltinIdentifier() {
#define BUILTIN_ID(X, Y, Z)                                                    \
  getCurScope()->addDecl(::new (Context) InlineAsmVariableDecl(                \
      getLexer().getIdentifierInfo(Y),                                         \
      Context.getBuiltinTypeFromTokenKind(asmtok::kw_##Z)));
#include "AsmTokenKinds.def"
}

InlineAsmBuiltinType *
InlineAsmContext::getBuiltinType(InlineAsmBuiltinType::TypeKind Kind) {
  assert(Kind > 0 && Kind < InlineAsmBuiltinType::NUM_TYPES && "Unknown Kind");
  if (AsmBuiltinTypes[Kind])
    return AsmBuiltinTypes[Kind];

  InlineAsmBuiltinType *NewType = ::new (*this) InlineAsmBuiltinType(Kind);
  AsmBuiltinTypes[Kind] = NewType;
  return NewType;
}

InlineAsmBuiltinType *
InlineAsmContext::getBuiltinTypeFromTokenKind(asmtok::TokenKind Kind) {
  switch (Kind) {
#define BUILTIN_TYPE(X, Y)                                                     \
  case asmtok::kw_##X:                                                         \
    return getBuiltinType(InlineAsmBuiltinType::TK_##X);
#include "AsmTokenKinds.def"
  default:
    break;
  }
  return nullptr;
}

InlineAsmDiscardType *InlineAsmContext::getDiscardType() {
  if (!DiscardType)
    DiscardType = ::new (*this) InlineAsmDiscardType;
  return DiscardType;
}

InlineAsmBuiltinType *
InlineAsmContext::getTypeFromConstraint(StringRef Constraint) {
  if (Constraint.size() != 1) {
    StringRef AllowedConstraint = "hrlfd";
    Constraint = Constraint.drop_until(
        [&](char C) -> bool { return AllowedConstraint.contains(C); });
    if (Constraint.empty())
      return nullptr;
  }
  switch (Constraint[0]) {
  case 'h':
    return getBuiltinType(InlineAsmBuiltinType::TK_u16);
  case 'r':
    return getBuiltinType(InlineAsmBuiltinType::TK_u32);
  case 'l':
    return getBuiltinType(InlineAsmBuiltinType::TK_u64);
  case 'f':
    return getBuiltinType(InlineAsmBuiltinType::TK_f32);
  case 'd':
    return getBuiltinType(InlineAsmBuiltinType::TK_f64);
  default:
    break;
  }
  return nullptr;
}

InlineAsmType::~InlineAsmType() = default;
InlineAsmDecl::~InlineAsmDecl() = default;
InlineAsmStmt::~InlineAsmStmt() = default;

InlineAsmDeclResult
InlineAsmParser::addInlineAsmOperands(StringRef Operand, StringRef Constraint) {
  unsigned Index = Context.addInlineAsmOperand(Operand);
  InlineAsmIdentifierInfo *II = Context.get(Index);
  InlineAsmType *Type = Context.getTypeFromConstraint(Constraint);
  if (!Type)
    return AsmDeclError();

  InlineAsmVariableDecl *VD = ::new (Context) InlineAsmVariableDecl(II, Type);
  getCurScope()->addDecl(VD);
  return VD;
}

InlineAsmStmtResult InlineAsmParser::ParseStatement() {
  switch (Tok.getKind()) {
  case asmtok::l_brace:
    return ParseCompoundStatement();
  case asmtok::at:
    return ParseConditionalInstruction();
#define STATE_SPACE(X, Y)                                                      \
  case asmtok::kw_##X:                                                         \
    return ParseDeclarationStatement();
#include "AsmTokenKinds.def"
#define INSTRUCTION(X)                                                         \
  case asmtok::op_##X:                                                         \
    return ParseInstruction();
#include "AsmTokenKinds.def"
  default:
    break;
  }
  return AsmStmtError();
}

InlineAsmStmtResult InlineAsmParser::ParseCompoundStatement() {
  ConsumeToken();

  ParseScope BlockScope(this);

  SmallVector<InlineAsmStmt *, 4> Stmts;
  while (Tok.isNot(asmtok::r_brace) && Tok.isNot(asmtok::eof)) {
    InlineAsmStmtResult Result = ParseStatement();
    if (Result.isInvalid())
      return AsmStmtError();
    Stmts.push_back(Result.get());
  }

  if (!TryConsumeToken(asmtok::r_brace))
    return AsmStmtError();
  return ::new (Context) InlineAsmCompoundStmt(Stmts);
}

InlineAsmStmtResult InlineAsmParser::ParseConditionalInstruction() {
  if (!TryConsumeToken(asmtok::at))
    return AsmStmtError();

  bool isNeg = false;
  if (TryConsumeToken(asmtok::exclaim)) {
    isNeg = true;
  }

  InlineAsmExprResult Pred = ParseExpression();

  if (Pred.isInvalid())
    return AsmStmtError();

  InlineAsmStmtResult SubInst = ParseInstruction();
  if (SubInst.isInvalid())
    return AsmStmtError();

  return ::new (Context) InlineAsmConditionalInstruction(
      isNeg, Pred.get(), SubInst.getAs<InlineAsmInstruction>());
}

InlineAsmStmtResult InlineAsmParser::ParseInstruction() {
  if (!Tok.getIdentifier() || !Tok.getIdentifier()->isInstruction())
    return AsmStmtError();

  InlineAsmIdentifierInfo *Opcode = Tok.getIdentifier();
  ConsumeToken();

  SmallVector<InlineAsmType *, 4> Types;
  SmallVector<InlineAsmIdentifierInfo *, 4> Attrs;
  while (Tok.startOfDot()) {
    if (Tok.getIdentifier()->isBuiltinType())
      Types.push_back(Context.getBuiltinTypeFromTokenKind(Tok.getKind()));
    else
      Attrs.push_back(Tok.getIdentifier());
    ConsumeToken(); // consume instruction attribute
  }

  InlineAsmExprResult OutputOperand = ParseExpression();
  if (OutputOperand.isInvalid())
    return AsmStmtError();

  bool HasPredOutput = TryConsumeToken(asmtok::pipe);
  InlineAsmExprResult PredOutput;
  if (HasPredOutput) {
    PredOutput = ParseExpression();
    if (PredOutput.isInvalid())
      return AsmStmtError();
  }

  SmallVector<InlineAsmExpr *, 4> InputOperands;

  while (TryConsumeToken(asmtok::comma)) {
    InlineAsmExprResult Operand = ParseExpression();
    if (Operand.isInvalid())
      return AsmStmtError();
    InputOperands.push_back(Operand.get());
  }

  if (!TryConsumeToken(asmtok::semi))
    return AsmStmtError();

  if (HasPredOutput)
    return ::new (Context)
        InlineAsmInstruction(Opcode, Types, Attrs, OutputOperand.get(),
                             InputOperands, PredOutput.get());
  return ::new (Context) InlineAsmInstruction(
      Opcode, Types, Attrs, OutputOperand.get(), InputOperands);
}

InlineAsmExprResult InlineAsmParser::ParseExpression() {
  return ParseAssignmentExpression();
}

InlineAsmExprResult InlineAsmParser::ParseAssignmentExpression() {
  InlineAsmExprResult LHS = ParseCastExpression();
  return ParseRHSOfBinaryExpression(LHS, asm_precedence::Assignment);
}

InlineAsmExprResult
InlineAsmParser::ParseRHSOfBinaryExpression(InlineAsmExprResult LHS,
                                            asm_precedence::Level MinPrec) {
  asm_precedence::Level NextTokPrec = getBinOpPrec(Tok.getKind());
  while (true) {
    if (NextTokPrec < MinPrec)
      return LHS;

    InlineAsmToken OpTok = Tok;
    ConsumeToken();

    // Special case handling for the ternary operator.
    bool isCondOp = false;
    InlineAsmExprResult TernaryMiddle(true);
    if (NextTokPrec == asm_precedence::Conditional) {
      isCondOp = true;
      if (Tok.isNot(asmtok::colon)) {
        TernaryMiddle = ParseExpression();
        if (TernaryMiddle.isInvalid())
          return AsmExprError();
      } else {
        return AsmExprError();
      }

      if (!TryConsumeToken(asmtok::colon)) {
        return AsmExprError();
      }
    }

    InlineAsmExprResult RHS = ParseCastExpression();
    if (RHS.isInvalid())
      return AsmExprError();

    asm_precedence::Level ThisPrec = NextTokPrec;
    NextTokPrec = getBinOpPrec(Tok.getKind());

    bool isRightAssoc = ThisPrec == asm_precedence::Conditional ||
                        ThisPrec == asm_precedence::Assignment;

    if (ThisPrec < NextTokPrec || (ThisPrec == NextTokPrec && isRightAssoc)) {
      RHS = ParseRHSOfBinaryExpression(
          RHS, static_cast<asm_precedence::Level>(ThisPrec + !isRightAssoc));
      if (RHS.isInvalid())
        return AsmExprError();
    }

    NextTokPrec = getBinOpPrec(Tok.getKind());

    if (isCondOp) {
      LHS = ActOnConditionalOp(LHS.get(), TernaryMiddle.get(), RHS.get());
    } else {
      LHS = ActOnBinaryOp(OpTok.getKind(), LHS.get(), RHS.get());
    }

    if (LHS.isInvalid())
      return AsmExprError();
  }
}

InlineAsmExprResult InlineAsmParser::ParseCastExpression() {
  InlineAsmExprResult Res;
  auto SavedKind = Tok.getKind();
  switch (SavedKind) {
  case asmtok::l_paren:
    ConsumeToken();
    if (Tok.isOneOf(asmtok::kw_s64, asmtok::kw_u64)) {
      InlineAsmBuiltinType *CastTy =
          Tok.is(asmtok::kw_s64) ? Context.getS64Type() : Context.getU64Type();
      if (!TryConsumeToken(asmtok::r_paren))
        return AsmExprError();
      InlineAsmExprResult SubExpr = ParseCastExpression();
      if (SubExpr.isInvalid())
        return AsmExprError();
      Res = ActOnTypeCast(CastTy, SubExpr.get());
    } else {
      InlineAsmExprResult SubExpr = ParseExpression();
      if (SubExpr.isInvalid())
        return AsmExprError();
      if (!TryConsumeToken(asmtok::r_paren))
        return AsmExprError();
      Res = ActOnParenExpr(SubExpr.get());
    }
    break;
  case asmtok::l_square:
    ConsumeToken();
    Res = ParseExpression();
    if (Res.isInvalid())
      return AsmExprError();
    if (!TryConsumeToken(asmtok::r_square))
      return AsmExprError();
    Res = ActOnAddressExpr(Res.get());
    break;
  case asmtok::l_brace: {
    ConsumeToken();
    SmallVector<InlineAsmExpr *, 4> Tuple;
    while (true) {
      Res = ParseExpression();
      if (Res.isInvalid())
        return AsmExprError();
      Tuple.push_back(Res.get());
      if (Tok.isNot(asmtok::comma))
        break;
    }

    if (!TryConsumeToken(asmtok::r_brace))
      return AsmExprError();
    Res = ActOnVectorExpr(Tuple);
    break;
  }
  case asmtok::underscore:
    Res = ActOnDiscardExpr();
    break;
  case asmtok::numeric_constant:
    Res = ActOnNumericConstant(Tok);
    ConsumeToken();
    break;
  case asmtok::identifier:
#define BUILTIN_ID(X, Y, Z) case asmtok::bi_##X:
#include "AsmTokenKinds.def"
    Res = ActOnIdExpr(Tok.getIdentifier());
    ConsumeToken();
    break;
  case asmtok::plus:    // unary-expression: '+' cast-expression
  case asmtok::minus:   // unary-expression: '-' cast-expression
  case asmtok::tilde:   // unary-expression: '~' cast-expression
  case asmtok::exclaim: // unary-expression: '!' cast-expression
    ConsumeToken();
    Res = ParseCastExpression();
    if (Res.isInvalid())
      return AsmExprError();
    Res = ActOnUnaryOp(SavedKind, Res.get());
    break;
  default:
    break;
  }
  return Res;
}

InlineAsmStmtResult InlineAsmParser::ParseDeclarationStatement() {
  InlineAsmDeclarationSpecifier DeclSpec;
  InlineAsmTypeResult Type = ParseDeclarationSpecifier(DeclSpec);
  if (Type.isInvalid())
    return AsmStmtError();

  SmallVector<InlineAsmDecl *, 4> Decls;
  while (true) {
    InlineAsmDeclResult DeclRes = ParseDeclarator(DeclSpec);
    if (DeclRes.isInvalid())
      return AsmStmtError();
    Decls.push_back(DeclRes.get());
    if (!TryConsumeToken(asmtok::comma))
      break;
  }

  if (!TryConsumeToken(asmtok::semi))
    return AsmStmtError();
  return ::new (Context) InlineAsmDeclStmt(DeclSpec, Decls);
}

InlineAsmTypeResult InlineAsmParser::ParseDeclarationSpecifier(
    InlineAsmDeclarationSpecifier &DeclSpec) {
  // Only support register variable
  DeclSpec.StateSpace = Tok.getKind();
  switch (Tok.getKind()) {
  case asmtok::kw_reg:
  case asmtok::kw_sreg:
    ConsumeToken();
    break;
  case asmtok::kw_const:
  case asmtok::kw_global:
  case asmtok::kw_local:
  case asmtok::kw_shared:
  case asmtok::kw_param:
  case asmtok::kw_tex:
  default:
    return AsmTypeError();
  }

  if (TryConsumeToken(asmtok::kw_align)) {
    InlineAsmExprResult AlignmentRes = ParseExpression();
    if (AlignmentRes.isInvalid())
      return AsmTypeError();
    AlignmentRes = ActOnAlignment(AlignmentRes.get());
    if (AlignmentRes.isInvalid())
      return AsmTypeError();
    DeclSpec.Alignment = cast<InlineAsmIntegerLiteral>(AlignmentRes.get());
  }

  if (Tok.isOneOf(asmtok::kw_v2, asmtok::kw_v4)) {
    DeclSpec.VectorTypeKind = Tok.getKind();
    ConsumeToken();
  }

  switch (Tok.getKind()) {
#define BUILTIN_TYPE(X, Y)                                                     \
  case asmtok::kw_##X:                                                         \
    DeclSpec.BaseType = Context.getBuiltinType(InlineAsmBuiltinType::TK_##X);  \
    ConsumeToken();                                                            \
    break;
#include "AsmTokenKinds.def"
  default:
    return AsmTypeError();
  }

  switch (DeclSpec.VectorTypeKind) {
  case asmtok::unknown:
    DeclSpec.Type = DeclSpec.BaseType;
    break;
  case asmtok::kw_v2:
    DeclSpec.Type = ::new (Context)
        InlineAsmVectorType(InlineAsmVectorType::TK_v2, DeclSpec.BaseType);
    break;
  case asmtok::kw_v4:
    DeclSpec.Type = ::new (Context)
        InlineAsmVectorType(InlineAsmVectorType::TK_v4, DeclSpec.BaseType);
    break;
  default:
    assert(0 && "unexpected vector type");
  }
  return DeclSpec.Type;
}

InlineAsmDeclResult InlineAsmParser::ParseDeclarator(
    const InlineAsmDeclarationSpecifier &DeclSpec) {
  if (Tok.isNot(asmtok::identifier))
    return AsmDeclError();
  auto *Name = Tok.getIdentifier();
  ConsumeToken();

  auto VarRes = ActOnVariableDecl(Name, DeclSpec.Type);
  if (VarRes.isInvalid())
    return AsmDeclError();

  InlineAsmVariableDecl *Decl = VarRes.getAs<InlineAsmVariableDecl>();

  if (DeclSpec.Alignment) {
    Decl->setAlign(DeclSpec.Alignment->getValue().getZExtValue());
  }

  switch (Tok.getKind()) {
  case asmtok::less: { // Parameterized variable declaration
    ConsumeToken();
    if (Tok.isNot(asmtok::numeric_constant))
      return AsmDeclError();
    InlineAsmExprResult NumRes = ActOnNumericConstant(Tok);
    ConsumeToken();
    if (!TryConsumeToken(asmtok::greater))
      return AsmDeclError();

    if (NumRes.isInvalid())
      return AsmDeclError();
    if (const auto *Int = dyn_cast<InlineAsmIntegerLiteral>(NumRes.get())) {
      unsigned Num = Int->getValue().getZExtValue();
      Decl->setNumParameterizedNames(Num);
      // Parameterized variable declaration dosen't support for arrays and init.
      return Decl;
    }
    return AsmDeclError();
  }
  case asmtok::l_square:
    /// FIXME: Support array declaration
    break;
  default:
    break;
  }

  if (Tok.is(asmtok::equal)) {
    /// FIXME: Support assignment and initializer init.
  }
  return Decl;
}

InlineAsmExprResult InlineAsmParser::ActOnDiscardExpr() {
  return ::new (Context) InlineAsmDiscardExpr(Context.getDiscardType());
}

InlineAsmExprResult InlineAsmParser::ActOnAddressExpr(InlineAsmExpr *SubExpr) {
  return ::new (Context) InlineAsmAddressExpr(Context.getF64Type(), SubExpr);
}

InlineAsmExprResult InlineAsmParser::ActOnIdExpr(InlineAsmIdentifierInfo *II) {
  if (auto *D = getCurScope()->lookupDecl(II)) {
    return ::new (Context) InlineAsmDeclRefExpr(D);
  }

  unsigned ParameterizedNameIdx;
  // Maybe this identifier is a parameterized variable name
  if (auto *D = getCurScope()->lookupParameterizedNameDecl(
          II, ParameterizedNameIdx)) {
    return ::new (Context) InlineAsmDeclRefExpr(D, ParameterizedNameIdx);
  }

  return AsmExprError();
}

InlineAsmExprResult InlineAsmParser::ActOnParenExpr(InlineAsmExpr *SubExpr) {
  return ::new (Context) InlineAsmParenExpr(SubExpr);
}

InlineAsmExprResult
InlineAsmParser::ActOnVectorExpr(ArrayRef<InlineAsmExpr *> Vec) {

  // Vector size must be 2, 4, or 8.
  InlineAsmVectorType::VecKind Kind;
  switch (Vec.size()) {
  case 2:
    Kind = InlineAsmVectorType::TK_v2;
    break;
  case 4:
    Kind = InlineAsmVectorType::TK_v4;
    break;
  case 8:
    Kind = InlineAsmVectorType::TK_v8;
    break;
  default:
    return AsmExprError();
  }

  InlineAsmBuiltinType *ElementType = nullptr;
  // The type of each element must have the same non-predicate builtin type.
  for (auto *E : Vec) {
    if (auto *T = dyn_cast<InlineAsmBuiltinType>(E->getType())) {
      if (T->getKind() == InlineAsmBuiltinType::TK_pred)
        return AsmExprError();
      if (ElementType && ElementType->getKind() != T->getKind())
        return AsmExprError();
      if (!ElementType)
        ElementType = T;
    } else {
      return AsmExprError();
    }
  }

  InlineAsmVectorType *Type =
      ::new (Context) InlineAsmVectorType(Kind, ElementType);
  return ::new (Context) InlineAsmVectorExpr(Type, Vec);
}

InlineAsmExprResult InlineAsmParser::ActOnTypeCast(InlineAsmBuiltinType *CastTy,
                                                   InlineAsmExpr *SubExpr) {
  return ::new (Context) InlineAsmCastExpr(CastTy, SubExpr);
}

InlineAsmExprResult InlineAsmParser::ActOnUnaryOp(asmtok::TokenKind OpTok,
                                                  InlineAsmExpr *SubExpr) {
  InlineAsmUnaryOperator::Opcode Opcode;
  switch (OpTok) {
  case asmtok::plus:
    Opcode = InlineAsmUnaryOperator::Plus;
    break;
  case asmtok::minus:
    Opcode = InlineAsmUnaryOperator::Minus;
    break;
  case asmtok::tilde:
    Opcode = InlineAsmUnaryOperator::Not;
    break;
  case asmtok::exclaim:
    Opcode = InlineAsmUnaryOperator::LNot;
    break;
  default:
    assert(0 && "unexpected op token");
  }

  return ::new (Context)
      InlineAsmUnaryOperator(Opcode, SubExpr, SubExpr->getType());
}

// clang-format off
static InlineAsmBinaryOperator::Opcode ConvertTokenKindToBinaryOpcode(asmtok::TokenKind Kind) {
  InlineAsmBinaryOperator::Opcode Opc;
  switch (Kind) {
  default: assert(0 && "Unknown binop!");
  case asmtok::star:                 Opc = InlineAsmBinaryOperator::Mul; break;
  case asmtok::slash:                Opc = InlineAsmBinaryOperator::Div; break;
  case asmtok::percent:              Opc = InlineAsmBinaryOperator::Rem; break;
  case asmtok::plus:                 Opc = InlineAsmBinaryOperator::Add; break;
  case asmtok::minus:                Opc = InlineAsmBinaryOperator::Sub; break;
  case asmtok::lessless:             Opc = InlineAsmBinaryOperator::Shl; break;
  case asmtok::greatergreater:       Opc = InlineAsmBinaryOperator::Shr; break;
  case asmtok::lessequal:            Opc = InlineAsmBinaryOperator::LE; break;
  case asmtok::less:                 Opc = InlineAsmBinaryOperator::LT; break;
  case asmtok::greaterequal:         Opc = InlineAsmBinaryOperator::GE; break;
  case asmtok::greater:              Opc = InlineAsmBinaryOperator::GT; break;
  case asmtok::exclaimequal:         Opc = InlineAsmBinaryOperator::NE; break;
  case asmtok::equalequal:           Opc = InlineAsmBinaryOperator::EQ; break;
  case asmtok::amp:                  Opc = InlineAsmBinaryOperator::And; break;
  case asmtok::caret:                Opc = InlineAsmBinaryOperator::Xor; break;
  case asmtok::pipe:                 Opc = InlineAsmBinaryOperator::Or; break;
  case asmtok::ampamp:               Opc = InlineAsmBinaryOperator::LAnd; break;
  case asmtok::pipepipe:             Opc = InlineAsmBinaryOperator::LOr; break;
  case asmtok::equal:                Opc = InlineAsmBinaryOperator::Assign; break;
  }
  return Opc;
}
// clang-format on

InlineAsmExprResult InlineAsmParser::ActOnBinaryOp(asmtok::TokenKind OpTok,
                                                   InlineAsmExpr *LHS,
                                                   InlineAsmExpr *RHS) {
  InlineAsmBinaryOperator::Opcode Opcode =
      ConvertTokenKindToBinaryOpcode(OpTok);
  /// TODO: Compute the type of binary operator
  return ::new (Context)
      InlineAsmBinaryOperator(Opcode, LHS, RHS, LHS->getType());
}

InlineAsmExprResult InlineAsmParser::ActOnConditionalOp(InlineAsmExpr *Cond,
                                                        InlineAsmExpr *LHS,
                                                        InlineAsmExpr *RHS) {
  /// TODO: Compute the type of conditional operator
  return ::new (Context)
      InlineAsmConditionalOperator(Cond, LHS, RHS, LHS->getType());
}

namespace {
/// AsmNumericLiteralParser - This performs strict semantic analysis of the
/// content of a ppnumber, classifying it as either integer, floating, or
/// erroneous, determines the radix of the value and can convert it to a useful
/// value.
class AsmNumericLiteralParser {
  const char *const ThisTokBegin;
  const char *const ThisTokEnd;
  const char *DigitsBegin, *SuffixBegin; // markers
  const char *s;                         // cursor

  unsigned radix;

  bool saw_exponent, saw_period;

public:
  AsmNumericLiteralParser(StringRef TokSpelling);
  bool hadError : 1;
  bool isUnsigned : 1;
  bool isFloat : 1;              // 1.0f
  bool isExactMachineFloat : 1;  // 0[fF]{hexdigit}{8}
  bool isExactMachineDouble : 1; // 0[dD]{hexdigit}{16}

  bool isIntegerLiteral() const {
    return !saw_period && !saw_exponent && !isExactMachineFloat ||
           isExactMachineDouble;
  }
  bool isFloatingLiteral() const {
    return (saw_period || saw_exponent || isExactMachineFloat ||
            isExactMachineDouble);
  }

  unsigned getRadix() const { return radix; }

  /// GetIntegerValue - Convert this numeric literal value to an APInt that
  /// matches Val's input width.  If there is an overflow (i.e., if the unsigned
  /// value read is larger than the APInt's bits will hold), set Val to the low
  /// bits of the result and return true.  Otherwise, return false.
  bool GetIntegerValue(llvm::APInt &Val);

  /// GetFloatValue - Convert this numeric literal to a floating value, using
  /// the specified APFloat fltSemantics (specifying float, double, etc).
  /// The optional bool isExact (passed-by-reference) has its value
  /// set to true if the returned APFloat can represent the number in the
  /// literal exactly, and false otherwise.
  llvm::APFloat::opStatus GetFloatValue(llvm::APFloat &Result);

  /// Get the digits that comprise the literal. This excludes any prefix or
  /// suffix associated with the literal.
  StringRef getLiteralDigits() const {
    assert(!hadError && "cannot reliably get the literal digits with an error");
    return StringRef(DigitsBegin, SuffixBegin - DigitsBegin);
  }

  /// Get the digits that comprise the literal. This excludes any prefix or
  /// suffix associated with the literal.
  StringRef getExactMachineFloatingHexLiteralDigits() const {
    assert(!hadError && (isExactMachineFloat || isExactMachineDouble) &&
           "cannot reliably get the literal digits with an error");
    return StringRef(DigitsBegin, SuffixBegin - DigitsBegin);
  }

private:
  void ParseNumberStartingWithZero();
  void ParseDecimalOrOctalCommon();

  /// Determine whether the sequence of characters [Start, End) contains
  /// any real digits (not digit separators).
  bool containsDigits(const char *Start, const char *End) {
    return Start != End;
  }

  enum CheckSeparatorKind { CSK_BeforeDigits, CSK_AfterDigits };

  /// SkipHexDigits - Read and skip over any hex digits, up to End.
  /// Return a pointer to the first non-hex digit or End.
  const char *SkipHexDigits(const char *ptr) {
    while (ptr != ThisTokEnd && (isHexDigit(*ptr)))
      ptr++;
    return ptr;
  }

  /// SkipOctalDigits - Read and skip over any octal digits, up to End.
  /// Return a pointer to the first non-hex digit or End.
  const char *SkipOctalDigits(const char *ptr) {
    while (ptr != ThisTokEnd && ((*ptr >= '0' && *ptr <= '7')))
      ptr++;
    return ptr;
  }

  /// SkipDigits - Read and skip over any digits, up to End.
  /// Return a pointer to the first non-hex digit or End.
  const char *SkipDigits(const char *ptr) {
    while (ptr != ThisTokEnd && (isDigit(*ptr)))
      ptr++;
    return ptr;
  }

  /// SkipBinaryDigits - Read and skip over any binary digits, up to End.
  /// Return a pointer to the first non-binary digit or End.
  const char *SkipBinaryDigits(const char *ptr) {
    while (ptr != ThisTokEnd && (*ptr == '0' || *ptr == '1'))
      ptr++;
    return ptr;
  }
};

///       integer-constant: [C99 6.4.4.1]
///         decimal-constant integer-suffix
///         octal-constant integer-suffix
///         hexadecimal-constant integer-suffix
///         binary-literal integer-suffix [GNU, C++1y]
///       user-defined-integer-literal: [C++11 lex.ext]
///         decimal-literal ud-suffix
///         octal-literal ud-suffix
///         hexadecimal-literal ud-suffix
///         binary-literal ud-suffix [GNU, C++1y]
///       decimal-constant:
///         nonzero-digit
///         decimal-constant digit
///       octal-constant:
///         0
///         octal-constant octal-digit
///       hexadecimal-constant:
///         hexadecimal-prefix hexadecimal-digit
///         hexadecimal-constant hexadecimal-digit
///       hexadecimal-prefix: one of
///         0x 0X
///       binary-literal:
///         0b binary-digit
///         0B binary-digit
///         binary-literal binary-digit
///       integer-suffix:
///         unsigned-suffix [long-suffix]
///         unsigned-suffix [long-long-suffix]
///         long-suffix [unsigned-suffix]
///         long-long-suffix [unsigned-sufix]
///       nonzero-digit:
///         1 2 3 4 5 6 7 8 9
///       octal-digit:
///         0 1 2 3 4 5 6 7
///       hexadecimal-digit:
///         0 1 2 3 4 5 6 7 8 9
///         a b c d e f
///         A B C D E F
///       binary-digit:
///         0
///         1
///       unsigned-suffix: one of
///         u U
///
///       floating-constant: [C99 6.4.4.2]
///         TODO: add rules...
///
AsmNumericLiteralParser::AsmNumericLiteralParser(StringRef TokSpelling)
    : ThisTokBegin(TokSpelling.begin()), ThisTokEnd(TokSpelling.end()) {
  s = DigitsBegin = ThisTokBegin;
  saw_exponent = false;
  saw_period = false;
  isUnsigned = false;
  isFloat = false;
  hadError = false;
  isExactMachineFloat = false;
  isExactMachineDouble = false;

  // This routine assumes that the range begin/end matches the regex for integer
  // and FP constants (specifically, the 'pp-number' regex), and assumes that
  // the byte at "*end" is both valid and not part of the regex.  Because of
  // this, it doesn't have to check for 'overscan' in various places.
  if (clang::isPreprocessingNumberBody(*ThisTokEnd)) {
    hadError = true;
    return;
  }

  if (*s == '0') { // parse radix
    ParseNumberStartingWithZero();
    if (hadError)
      return;
  } else { // the first digit is non-zero
    radix = 10;
    s = SkipDigits(s);
    if (s == ThisTokEnd) {
      // Done.
    } else {
      ParseDecimalOrOctalCommon();
      if (hadError)
        return;
    }
  }

  SuffixBegin = s;

  bool isFPConstant = isFloatingLiteral();

  if (*s == 'U') {
    if (isFPConstant) 
      hadError = true; // Error for floating constant.
    isUnsigned = true;
    ++s;
  }

  if (s != ThisTokEnd)
    hadError = true;
}

/// ParseDecimalOrOctalCommon - This method is called for decimal or octal
/// numbers. It issues an error for illegal digits, and handles floating point
/// parsing. If it detects a floating point number, the radix is set to 10.
void AsmNumericLiteralParser::ParseDecimalOrOctalCommon() {
  assert((radix == 8 || radix == 10) && "Unexpected radix");

  // If we have a hex digit other than 'e' (which denotes a FP exponent) then
  // the code is using an incorrect base.
  if (isHexDigit(*s) && *s != 'e' && *s != 'E') {
    hadError = true;
    return;
  }

  if (*s == '.') {
    s++;
    radix = 10;
    saw_period = true;
    s = SkipDigits(s); // Skip suffix.
  }
  if (*s == 'e' || *s == 'E') { // exponent
    s++;
    radix = 10;
    saw_exponent = true;
    if (s != ThisTokEnd && (*s == '+' || *s == '-'))
      s++; // sign
    const char *first_non_digit = SkipDigits(s);
    if (containsDigits(s, first_non_digit)) {
      s = first_non_digit;
    } else {
      if (!hadError) {
        hadError = true;
      }
      return;
    }
  }
}

/// ParseNumberStartingWithZero - This method is called when the first character
/// of the number is found to be a zero.  This means it is either an octal
/// number (like '04') or a hex number ('0x123a') a binary number ('0b1010') or
/// a floating point number (01239.123e4).  Eat the prefix, determining the
/// radix etc.
void AsmNumericLiteralParser::ParseNumberStartingWithZero() {
  assert(s[0] == '0' && "Invalid method call");
  s++;

  int c1 = s[0];

  // Handle a hex number like 0x1234.
  if ((c1 == 'x' || c1 == 'X') && (isHexDigit(s[1]) || s[1] == '.')) {
    s++;
    assert(s < ThisTokEnd && "didn't maximally munch?");
    radix = 16;
    DigitsBegin = s;
    s = SkipHexDigits(s);
    bool HasSignificandDigits = containsDigits(DigitsBegin, s);
    if (s == ThisTokEnd) {
      // Done.
    } else if (*s == '.') {
      s++;
      saw_period = true;
      const char *floatDigitsBegin = s;
      s = SkipHexDigits(s);
      if (containsDigits(floatDigitsBegin, s))
        HasSignificandDigits = true;
    }

    if (!HasSignificandDigits) {
      hadError = true;
      return;
    }

    // A binary exponent can appear with or with a '.'. If dotted, the
    // binary exponent is required.
    if (*s == 'p' || *s == 'P') {
      s++;
      saw_exponent = true;
      if (s != ThisTokEnd && (*s == '+' || *s == '-'))
        s++; // sign
      const char *first_non_digit = SkipDigits(s);
      if (!containsDigits(s, first_non_digit)) {
        if (!hadError) {
          hadError = true;
        }
        return;
      }
      s = first_non_digit;

    } else if (saw_period) {
      hadError = true;
    }
    return;
  }

  // Handle the exact machine representation single-precision floating point
  // constant.
  if ((c1 == 'f' || c1 == 'F') && (isHexDigit(s[1]))) {
    ++s;
    DigitsBegin = s;
    isExactMachineFloat = true;
    s = SkipHexDigits(s);
    if (s != ThisTokEnd || s - DigitsBegin != 8) {
      hadError = true;
    }
    return;
  }

  // Handle the exact machine representation double-precision floating point
  // constant.
  if ((c1 == 'd' || c1 == 'D') && (isHexDigit(s[1]))) {
    ++s;
    DigitsBegin = s;
    isExactMachineDouble = true;
    s = SkipHexDigits(s);
    if (s != ThisTokEnd || s - DigitsBegin != 16) {
      hadError = true;
    }
    return;
  }

  // Handle simple binary numbers 0b01010
  if ((c1 == 'b' || c1 == 'B') && (s[1] == '0' || s[1] == '1')) {
    ++s;
    radix = 2;
    DigitsBegin = s;
    s = SkipBinaryDigits(s);
    if (s == ThisTokEnd) {
      // Done.
    } else if (isHexDigit(*s)) {
      hadError = true;
    }
    // Other suffixes will be diagnosed by the caller.
    return;
  }

  // For now, the radix is set to 8. If we discover that we have a
  // floating point constant, the radix will change to 10. Octal floating
  // point constants are not permitted (only decimal and hexadecimal).
  radix = 8;
  const char *PossibleNewDigitStart = s;
  s = SkipOctalDigits(s);
  // When the value is 0 followed by a suffix (like 0wb), we want to leave 0
  // as the start of the digits. So if skipping octal digits does not skip
  // anything, we leave the digit start where it was.
  if (s != PossibleNewDigitStart)
    DigitsBegin = PossibleNewDigitStart;

  if (s == ThisTokEnd)
    return; // Done, simple octal number like 01234

  // If we have some other non-octal digit that *is* a decimal digit, see if
  // this is part of a floating point number like 094.123 or 09e1.
  if (isDigit(*s)) {
    const char *EndDecimal = SkipDigits(s);
    if (EndDecimal[0] == '.' || EndDecimal[0] == 'e' || EndDecimal[0] == 'E') {
      s = EndDecimal;
      radix = 10;
    }
  }

  ParseDecimalOrOctalCommon();
}

static bool alwaysFitsInto64Bits(unsigned Radix, unsigned NumDigits) {
  switch (Radix) {
  case 2:
    return NumDigits <= 64;
  case 8:
    return NumDigits <= 64 / 3; // Digits are groups of 3 bits.
  case 10:
    return NumDigits <= 19; // floor(log10(2^64))
  case 16:
    return NumDigits <= 64 / 4; // Digits are groups of 4 bits.
  default:
    assert(0 && "impossible Radix");
  }
}

/// GetIntegerValue - Convert this numeric literal value to an APInt that
/// matches Val's input width.  If there is an overflow, set Val to the low bits
/// of the result and return true.  Otherwise, return false.
bool AsmNumericLiteralParser::GetIntegerValue(llvm::APInt &Val) {
  // Fast path: Compute a conservative bound on the maximum number of
  // bits per digit in this radix. If we can't possibly overflow a
  // uint64 based on that bound then do the simple conversion to
  // integer. This avoids the expensive overflow checking below, and
  // handles the common cases that matter (small decimal integers and
  // hex/octal values which don't overflow).
  const unsigned NumDigits = SuffixBegin - DigitsBegin;
  if (alwaysFitsInto64Bits(radix, NumDigits)) {
    uint64_t N = 0;
    for (const char *Ptr = DigitsBegin; Ptr != SuffixBegin; ++Ptr)
      N = N * radix + llvm::hexDigitValue(*Ptr);

    // This will truncate the value to Val's input width. Simply check
    // for overflow by comparing.
    Val = N;
    return Val.getZExtValue() != N;
  }

  Val = 0;
  const char *Ptr = DigitsBegin;

  llvm::APInt RadixVal(Val.getBitWidth(), radix);
  llvm::APInt CharVal(Val.getBitWidth(), 0);
  llvm::APInt OldVal = Val;

  bool OverflowOccurred = false;
  while (Ptr < SuffixBegin) {

    unsigned C = llvm::hexDigitValue(*Ptr++);

    // If this letter is out of bound for this radix, reject it.
    assert(C < radix &&
           "AsmNumericLiteralParser ctor should have rejected this");

    CharVal = C;

    // Add the digit to the value in the appropriate radix.  If adding in digits
    // made the value smaller, then this overflowed.
    OldVal = Val;

    // Multiply by radix, did overflow occur on the multiply?
    Val *= RadixVal;
    OverflowOccurred |= Val.udiv(RadixVal) != OldVal;

    // Add value, did overflow occur on the value?
    //   (a + b) ult b  <=> overflow
    Val += CharVal;
    OverflowOccurred |= Val.ult(CharVal);
  }
  return OverflowOccurred;
}

template <class T> union ExactMachineFloatingConverter;

template <> union ExactMachineFloatingConverter<float> {
  uint32_t IntValue;
  float FpValue;
};

template <> union ExactMachineFloatingConverter<double> {
  uint64_t IntValue;
  double FpValue;
};

llvm::APFloat::opStatus
AsmNumericLiteralParser::GetFloatValue(llvm::APFloat &Result) {
  using llvm::APFloat;

  if (isExactMachineFloat) {
    assert(getLiteralDigits().size() == 8 &&
           "AsmNumericLiteralParser ctor should have rejected this");
    ExactMachineFloatingConverter<float> Value;
    if (getLiteralDigits().getAsInteger(16, Value.IntValue)) {
      return APFloat::opInvalidOp;
    }
    Result = APFloat(Value.FpValue);
    return APFloat::opOK;
  }

  if (isExactMachineDouble) {
    assert(getLiteralDigits().size() == 16 &&
           "AsmNumericLiteralParser ctor should have rejected this");
    ExactMachineFloatingConverter<double> Value;
    if (getLiteralDigits().getAsInteger(16, Value.IntValue)) {
      return APFloat::opInvalidOp;
    }
    Result = APFloat(Value.FpValue);
    return APFloat::opOK;
  }

  unsigned n = std::min(SuffixBegin - ThisTokBegin, ThisTokEnd - ThisTokBegin);

  StringRef Str(ThisTokBegin, n);

  auto StatusOrErr =
      Result.convertFromString(Str, APFloat::rmNearestTiesToEven);
  assert(StatusOrErr && "Invalid floating point representation");
  return !errorToBool(StatusOrErr.takeError()) ? *StatusOrErr
                                               : APFloat::opInvalidOp;
}
} // namespace

InlineAsmExprResult
InlineAsmParser::ActOnNumericConstant(const InlineAsmToken &Tok) {
  assert(Tok.is(asmtok::numeric_constant) && Tok.getLength() >= 1);
  StringRef LiteralData(Tok.getLiteralData(), Tok.getLength());
  if (Tok.getLength() == 1 && isDigit(LiteralData[0])) {
    InlineAsmBuiltinType *Type = Context.getS64Type();
    llvm::APInt Val(64, LiteralData[0] - '0', true);
    return ::new (Context) InlineAsmIntegerLiteral(Type, Val, LiteralData);
  }

  AsmNumericLiteralParser LiteralParser(LiteralData);
  if (LiteralParser.hadError)
    return AsmExprError();

  if (LiteralParser.isFloatingLiteral()) {
    if (LiteralParser.isExactMachineFloat) {
      APFloat Float(APFloat::IEEEsingle());
      auto Status = LiteralParser.GetFloatValue(Float);
      if (Status != APFloat::opOK)
        return AsmExprError();
      return ::new (Context) InlineAsmFloatingLiteral(
          Context.getF32Type(), Float,
          LiteralParser.getExactMachineFloatingHexLiteralDigits(),
          /*IsExactMachineFloatingLiteral*/ true);
    }

    if (LiteralParser.isExactMachineDouble) {
      APFloat Float(APFloat::IEEEdouble());
      auto Status = LiteralParser.GetFloatValue(Float);
      if (Status != APFloat::opOK)
        return AsmExprError();
      return ::new (Context) InlineAsmFloatingLiteral(
          Context.getF64Type(), Float,
          LiteralParser.getExactMachineFloatingHexLiteralDigits(),
          /*IsExactMachineFloatingLiteral*/ true);
    }

    APFloat Float(APFloat::IEEEdouble());
    auto Status = LiteralParser.GetFloatValue(Float);
    if ((Status & APFloat::opOverflow) ||
        ((Status & APFloat::opUnderflow) && Float.isZero()))
      return AsmExprError();
    return ::new (Context)
        InlineAsmFloatingLiteral(Context.getF64Type(), Float, LiteralData);
  }

  APInt Int(64, 0, /*isSigned*/ true);
  InlineAsmBuiltinType *Type = Context.getS64Type();
  if (LiteralParser.GetIntegerValue(Int)) {
    // Overflow occurred, promote integer type to u64.
    Int = APInt(64, 0);
    if (LiteralParser.GetIntegerValue(Int)) {
      // Integer too large
      return AsmExprError();
    }
    Type = Context.getU64Type();
  }

  return ::new (Context) InlineAsmIntegerLiteral(Type, Int, LiteralData);
}

InlineAsmExprResult InlineAsmParser::ActOnAlignment(InlineAsmExpr *Alignment) {
  if (auto *Int = dyn_cast<InlineAsmIntegerLiteral>(Alignment)) {
    return Int;
  }
  return AsmExprError();
}

InlineAsmDeclResult
InlineAsmParser::ActOnVariableDecl(InlineAsmIdentifierInfo *Name,
                                   InlineAsmType *Type) {
  InlineAsmVariableDecl *D = ::new (Context) InlineAsmVariableDecl(Name, Type);
  getCurScope()->addDecl(D);
  return D;
}
