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
asmprec::Level getBinOpPrec(asmtok::TokenKind Kind) {
  switch (Kind) {
  default:                           return asmprec::Unknown;
  case asmtok::question:             return asmprec::Conditional;
  case asmtok::pipepipe:             return asmprec::LogicalOr;
  case asmtok::ampamp:               return asmprec::LogicalAnd;
  case asmtok::pipe:                 return asmprec::InclusiveOr;
  case asmtok::caret:                return asmprec::ExclusiveOr;
  case asmtok::amp:                  return asmprec::And;
  case asmtok::exclaimequal:
  case asmtok::equalequal:           return asmprec::Equality;
  case asmtok::lessequal:
  case asmtok::less:
  case asmtok::greater:
  case asmtok::greaterequal:         return asmprec::Relational;
  case asmtok::greatergreater:
  case asmtok::lessless:             return asmprec::Shift;
  case asmtok::plus:
  case asmtok::minus:                return asmprec::Additive;
  case asmtok::percent:
  case asmtok::slash:
  case asmtok::star:                 return asmprec::Multiplicative;
  }
}
// clang-format on

DpctAsmVariableDecl *DpctAsmScope::lookupDecl(DpctAsmIdentifierInfo *II) const {
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

DpctAsmBuiltinType *
DpctAsmContext::getBuiltinType(DpctAsmBuiltinType::TypeKind Kind) {
  if (AsmBuiltinTypes.contains(Kind))
    return AsmBuiltinTypes[Kind];

  DpctAsmBuiltinType *NewType = ::new (*this) DpctAsmBuiltinType(Kind);
  AsmBuiltinTypes[Kind] = NewType;
  return NewType;
}

DpctAsmBuiltinType *
DpctAsmContext::getBuiltinTypeFromTokenKind(asmtok::TokenKind Kind) {
  switch (Kind) {
#define BUILTIN_TYPE(X, Y)                                                     \
  case asmtok::kw_##X:                                                         \
    return getBuiltinType(DpctAsmBuiltinType::TK_##X);
#include "AsmTokenKinds.def"
  default:
    break;
  }
  return nullptr;
}

DpctAsmDiscardType *DpctAsmContext::getDiscardType() {
  if (!DiscardType)
    DiscardType = ::new (*this) DpctAsmDiscardType;
  return DiscardType;
}

DpctAsmBuiltinType *
DpctAsmContext::getTypeFromConstraint(StringRef Constraint) {
  if (Constraint.size() != 1) {
    StringRef AllowedConstraint = "hrlfd";
    Constraint = Constraint.drop_until(
        [&](char C) -> bool { return AllowedConstraint.contains(C); });
    if (Constraint.empty())
      return nullptr;
  }
  switch (Constraint[0]) {
  case 'h':
    return getBuiltinType(DpctAsmBuiltinType::TK_u16);
  case 'r':
    return getBuiltinType(DpctAsmBuiltinType::TK_u32);
  case 'l':
    return getBuiltinType(DpctAsmBuiltinType::TK_u64);
  case 'f':
    return getBuiltinType(DpctAsmBuiltinType::TK_f32);
  case 'd':
    return getBuiltinType(DpctAsmBuiltinType::TK_f64);
  default:
    break;
  }
  return nullptr;
}

DpctAsmType::~DpctAsmType() = default;
DpctAsmDecl::~DpctAsmDecl() = default;
DpctAsmStmt::~DpctAsmStmt() = default;

DpctAsmDeclResult DpctAsmParser::addInlineAsmOperands(StringRef Operand,
                                                      StringRef Constraint) {
  unsigned Index = Context.addInlineAsmOperand(Operand);
  DpctAsmIdentifierInfo *II = Context.get(Index);
  DpctAsmType *Type = Context.getTypeFromConstraint(Constraint);
  if (!Type)
    return AsmDeclError();

  DpctAsmVariableDecl *VD = ::new (Context) DpctAsmVariableDecl(II, Type);
  getCurScope()->addDecl(VD);
  return VD;
}

bool DpctAsmParser::ExpectAndConsume(asmtok::TokenKind ExpectedTok) {
  if (Tok.is(ExpectedTok)) {
    ConsumeAnyToken();
    return false;
  }
  return true;
}

DpctAsmStmtResult DpctAsmParser::ParseStatement() {
  switch (Tok.getKind()) {
  case asmtok::l_brace:
    return ParseCompoundStatement();
  case asmtok::at:
    return ParseGuardInstruction();
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

DpctAsmStmtResult DpctAsmParser::ParseCompoundStatement() {
  ConsumeBrace();

  ParseScope BlockScope(this);

  SmallVector<DpctAsmStmt *, 4> Stmts;
  while (Tok.isNot(asmtok::r_brace) && Tok.isNot(asmtok::eof)) {
    DpctAsmStmtResult Result = ParseStatement();
    if (Result.isInvalid())
      return AsmStmtError();
    Stmts.push_back(Result.get());
  }

  if (ExpectAndConsume(asmtok::r_brace))
    return AsmStmtError();
  return ::new (Context) DpctAsmCompoundStmt(Stmts);
}

DpctAsmStmtResult DpctAsmParser::ParseGuardInstruction() {
  if (ExpectAndConsume(asmtok::at))
    return AsmStmtError();

  bool isNeg = false;
  if (TryConsumeToken(asmtok::exclaim)) {
    isNeg = true;
  }

  DpctAsmExprResult Pred = ParseExpression();

  if (Pred.isInvalid())
    return AsmStmtError();

  DpctAsmStmtResult SubInst = ParseInstruction();
  if (SubInst.isInvalid())
    return AsmStmtError();

  return ::new (Context) DpctAsmGuardInstruction(
      isNeg, Pred.get(), SubInst.getAs<DpctAsmInstruction>());
}

DpctAsmStmtResult DpctAsmParser::ParseInstruction() {
  if (!Tok.getIdentifier() || !Tok.getIdentifier()->isInstruction())
    return AsmStmtError();

  DpctAsmIdentifierInfo *Opcode = Tok.getIdentifier();
  ConsumeToken();

  SmallVector<DpctAsmType *, 4> Types;
  SmallVector<DpctAsmIdentifierInfo *, 4> Attrs;
  while (Tok.startOfDot()) {
    if (Tok.getIdentifier()->isBuiltinType())
      Types.push_back(Context.getBuiltinTypeFromTokenKind(Tok.getKind()));
    else
      Attrs.push_back(Tok.getIdentifier());
    ConsumeToken(); // consume instruction attribute
  }

  DpctAsmExprResult OutputOperand = ParseExpression();
  if (OutputOperand.isInvalid())
    return AsmStmtError();

  bool HasPredOutput = TryConsumeToken(asmtok::pipe);
  DpctAsmExprResult PredOutput;
  if (HasPredOutput) {
    PredOutput = ParseExpression();
    if (PredOutput.isInvalid())
      return AsmStmtError();
  }

  if (ExpectAndConsume(asmtok::comma))
    return AsmStmtError();

  SmallVector<DpctAsmExpr *, 4> InputOperands;

  while (true) {
    DpctAsmExprResult Operand = ParseExpression();
    if (Operand.isInvalid())
      return true;
    InputOperands.push_back(Operand.get());
    if (!TryConsumeToken(asmtok::comma))
      break;
  }

  if (ExpectAndConsume(asmtok::semi))
    return AsmStmtError();

  if (HasPredOutput)
    return ::new (Context)
        DpctAsmInstruction(Opcode, Types, Attrs, OutputOperand.get(),
                           InputOperands, PredOutput.get());
  return ::new (Context) DpctAsmInstruction(Opcode, Types, Attrs,
                                            OutputOperand.get(), InputOperands);
}

DpctAsmExprResult DpctAsmParser::ParseExpression() {
  return ParseAssignmentExpression();
}

DpctAsmExprResult DpctAsmParser::ParseAssignmentExpression() {
  DpctAsmExprResult LHS = ParseCastExpression();
  return ParseRHSOfBinaryExpression(LHS, asmprec::Assignment);
}

DpctAsmExprResult
DpctAsmParser::ParseRHSOfBinaryExpression(DpctAsmExprResult LHS,
                                          asmprec::Level MinPrec) {
  asmprec::Level NextTokPrec = getBinOpPrec(Tok.getKind());
  while (true) {
    if (NextTokPrec < MinPrec)
      return LHS;

    DpctAsmToken OpTok = Tok;
    ConsumeToken();

    // Special case handling for the ternary operator.
    bool isCondOp = false;
    DpctAsmExprResult TernaryMiddle(true);
    if (NextTokPrec == asmprec::Conditional) {
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

    DpctAsmExprResult RHS = ParseCastExpression();
    if (RHS.isInvalid())
      return AsmExprError();

    asmprec::Level ThisPrec = NextTokPrec;
    NextTokPrec = getBinOpPrec(Tok.getKind());

    bool isRightAssoc = ThisPrec == asmprec::Conditional;

    if (ThisPrec < NextTokPrec || (ThisPrec == NextTokPrec && isRightAssoc)) {
      RHS = ParseRHSOfBinaryExpression(
          RHS, static_cast<asmprec::Level>(ThisPrec + !isRightAssoc));
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

DpctAsmExprResult DpctAsmParser::ParseCastExpression() {
  DpctAsmExprResult Res;
  auto SavedKind = Tok.getKind();
  switch (SavedKind) {
  case asmtok::l_paren:
    ConsumeParen();
    if (Tok.isOneOf(asmtok::kw_s64, asmtok::kw_u64)) {
      DpctAsmBuiltinType *CastTy =
          Tok.is(asmtok::kw_s64) ? Context.getS64Type() : Context.getU64Type();
      ConsumeParen();
      DpctAsmExprResult SubExpr = ParseCastExpression();
      if (SubExpr.isInvalid())
        return AsmExprError();
      Res = ActOnTypeCast(CastTy, SubExpr.get());
    } else {
      DpctAsmExprResult SubExpr = ParseExpression();
      if (SubExpr.isInvalid())
        return AsmExprError();
      ConsumeParen();
      Res = ActOnParenExpr(SubExpr.get());
    }
    break;
  case asmtok::l_square:
    ConsumeBracket();
    Res = ParseExpression();
    if (Res.isInvalid())
      return AsmExprError();
    ConsumeBracket();
    Res = ActOnAddressExpr(Res.get());
    break;
  case asmtok::l_brace: {
    ConsumeBrace();
    SmallVector<DpctAsmExpr *, 4> Tuple;
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
    Res = ActOnTupleExpr(Tuple);
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

DpctAsmExprResult
DpctAsmParser::ParsePostfixExpressionSuffix(DpctAsmExprResult LHS) {
  while (true) {
    switch (Tok.getKind()) {
    default:
      return LHS;
    case asmtok::l_square: {
      ConsumeBracket();
      DpctAsmExprResult Idx = ParseExpression();
      if (Idx.isInvalid())
        return AsmExprError();
      ConsumeBracket();
    }
    }
  }
}

DpctAsmStmtResult DpctAsmParser::ParseDeclarationStatement() {
  DpctAsmDeclarationSpecifier DeclSpec;
  DpctAsmTypeResult Type = ParseDeclarationSpecifier(DeclSpec);
  if (Type.isInvalid())
    return AsmStmtError();

  SmallVector<DpctAsmDecl *, 4> Decls;
  while (true) {
    DpctAsmDeclResult DeclRes = ParseDeclarator(DeclSpec);
    if (DeclRes.isInvalid())
      return AsmStmtError();
    Decls.push_back(DeclRes.get());
    if (!TryConsumeToken(asmtok::comma))
      break;
  }

  if (!TryConsumeToken(asmtok::semi))
    return AsmStmtError();
  return ::new (Context) DpctAsmDeclStmt(DeclSpec.Type, Decls);
}

DpctAsmTypeResult DpctAsmParser::ParseDeclarationSpecifier(
    DpctAsmDeclarationSpecifier &DeclSpec) {
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
    DpctAsmExprResult AlignmentRes = ParseExpression();
    if (AlignmentRes.isInvalid())
      return AsmTypeError();
    AlignmentRes = ActOnAlignment(AlignmentRes.get());
    if (AlignmentRes.isInvalid())
      return AsmTypeError();
    DeclSpec.Alignment = cast<DpctAsmIntegerLiteral>(AlignmentRes.get());
  }

  if (Tok.isOneOf(asmtok::kw_v2, asmtok::kw_v4)) {
    DeclSpec.VectorType = Tok.getKind();
    ConsumeToken();
  }

  switch (Tok.getKind()) {
#define BUILTIN_TYPE(X, Y)                                                     \
  case asmtok::kw_##X:                                                         \
    DeclSpec.BaseType = Context.getBuiltinType(DpctAsmBuiltinType::TK_##X);    \
    ConsumeToken();                                                            \
    break;
#include "AsmTokenKinds.def"
  default:
    return AsmTypeError();
  }

  switch (DeclSpec.VectorType) {
  case asmtok::unknown:
    DeclSpec.Type = DeclSpec.BaseType;
    break;
  case asmtok::kw_v2:
    DeclSpec.Type = ::new (Context)
        DpctAsmVectorType(DpctAsmVectorType::TK_v2, DeclSpec.BaseType);
    break;
  case asmtok::kw_v4:
    DeclSpec.Type = ::new (Context)
        DpctAsmVectorType(DpctAsmVectorType::TK_v4, DeclSpec.BaseType);
    break;
  default:
    llvm_unreachable("unexpected vector type");
  }
  return DeclSpec.Type;
}

DpctAsmDeclResult
DpctAsmParser::ParseDeclarator(const DpctAsmDeclarationSpecifier &DeclSpec) {
  if (Tok.isNot(asmtok::identifier))
    return AsmDeclError();
  auto *Name = Tok.getIdentifier();
  ConsumeToken();
  return ActOnVariableDecl(Name, DeclSpec.Type);
}

DpctAsmExprResult DpctAsmParser::ActOnDiscardExpr() {
  return ::new (Context) DpctAsmDiscardExpr(Context.getDiscardType());
}

DpctAsmExprResult DpctAsmParser::ActOnAddressExpr(DpctAsmExpr *SubExpr) {
  return ::new (Context) DpctAsmAddressExpr(Context.getF64Type(), SubExpr);
}

DpctAsmExprResult DpctAsmParser::ActOnIdExpr(DpctAsmIdentifierInfo *II) {
  if (auto *D = getCurScope()->lookupDecl(II)) {
    return ::new (Context) DpctAsmDeclRefExpr(D);
  }
  return AsmExprError();
}

DpctAsmExprResult DpctAsmParser::ActOnParenExpr(DpctAsmExpr *SubExpr) {
  return ::new (Context) DpctAsmParenExpr(SubExpr);
}

DpctAsmExprResult DpctAsmParser::ActOnTupleExpr(ArrayRef<DpctAsmExpr *> Tuple) {
  SmallVector<DpctAsmType *, 4> ElementTypes;
  for (auto *E : Tuple) {
    ElementTypes.push_back(E->getType());
  }
  DpctAsmTupleType *Type = ::new (Context) DpctAsmTupleType(ElementTypes);
  return ::new (Context) DpctAsmTupleExpr(Type, Tuple);
}

DpctAsmExprResult DpctAsmParser::ActOnTypeCast(DpctAsmBuiltinType *CastTy,
                                               DpctAsmExpr *SubExpr) {
  return ::new (Context) DpctAsmCastExpr(CastTy, SubExpr);
}

DpctAsmExprResult DpctAsmParser::ActOnUnaryOp(asmtok::TokenKind OpTok,
                                              DpctAsmExpr *SubExpr) {
  DpctAsmUnaryOperator::Opcode Opcode;
  switch (OpTok) {
  case asmtok::plus:
    Opcode = DpctAsmUnaryOperator::Plus;
    break;
  case asmtok::minus:
    Opcode = DpctAsmUnaryOperator::Minus;
    break;
  case asmtok::tilde:
    Opcode = DpctAsmUnaryOperator::Not;
    break;
  case asmtok::exclaim:
    Opcode = DpctAsmUnaryOperator::LNot;
    break;
  default:
    llvm_unreachable("unexpected op token");
  }

  return ::new (Context)
      DpctAsmUnaryOperator(Opcode, SubExpr, SubExpr->getType());
}

// clang-format off
static DpctAsmBinaryOperator::Opcode ConvertTokenKindToBinaryOpcode(asmtok::TokenKind Kind) {
  DpctAsmBinaryOperator::Opcode Opc;
  switch (Kind) {
  default: llvm_unreachable("Unknown binop!");
  case asmtok::star:                 Opc = DpctAsmBinaryOperator::Mul; break;
  case asmtok::slash:                Opc = DpctAsmBinaryOperator::Div; break;
  case asmtok::percent:              Opc = DpctAsmBinaryOperator::Rem; break;
  case asmtok::plus:                 Opc = DpctAsmBinaryOperator::Add; break;
  case asmtok::minus:                Opc = DpctAsmBinaryOperator::Sub; break;
  case asmtok::lessless:             Opc = DpctAsmBinaryOperator::Shl; break;
  case asmtok::greatergreater:       Opc = DpctAsmBinaryOperator::Shr; break;
  case asmtok::lessequal:            Opc = DpctAsmBinaryOperator::LE; break;
  case asmtok::less:                 Opc = DpctAsmBinaryOperator::LT; break;
  case asmtok::greaterequal:         Opc = DpctAsmBinaryOperator::GE; break;
  case asmtok::greater:              Opc = DpctAsmBinaryOperator::GT; break;
  case asmtok::exclaimequal:         Opc = DpctAsmBinaryOperator::NE; break;
  case asmtok::equalequal:           Opc = DpctAsmBinaryOperator::EQ; break;
  case asmtok::amp:                  Opc = DpctAsmBinaryOperator::And; break;
  case asmtok::caret:                Opc = DpctAsmBinaryOperator::Xor; break;
  case asmtok::pipe:                 Opc = DpctAsmBinaryOperator::Or; break;
  case asmtok::ampamp:               Opc = DpctAsmBinaryOperator::LAnd; break;
  case asmtok::pipepipe:             Opc = DpctAsmBinaryOperator::LOr; break;
  case asmtok::equal:                Opc = DpctAsmBinaryOperator::Assign; break;
  }
  return Opc;
}
// clang-format on

DpctAsmExprResult DpctAsmParser::ActOnBinaryOp(asmtok::TokenKind OpTok,
                                               DpctAsmExpr *LHS,
                                               DpctAsmExpr *RHS) {
  DpctAsmBinaryOperator::Opcode Opcode = ConvertTokenKindToBinaryOpcode(OpTok);
  /// TODO: Compute the type of binary operator
  return ::new (Context)
      DpctAsmBinaryOperator(Opcode, LHS, RHS, LHS->getType());
}

DpctAsmExprResult DpctAsmParser::ActOnConditionalOp(DpctAsmExpr *Cond,
                                                    DpctAsmExpr *LHS,
                                                    DpctAsmExpr *RHS) {
  /// TODO: Compute the type of conditional operator
  return ::new (Context)
      DpctAsmConditionalOperator(Cond, LHS, RHS, LHS->getType());
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
    return Start != End && (Start + 1 != End);
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
///       long-suffix: one of
///         l L
///       long-long-suffix: one of
///         ll LL
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

  // Loop over all of the characters of the suffix.  If we see something bad,
  // we break out of the loop.
  for (; s != ThisTokEnd; ++s) {
    switch (*s) {
    case 'u':
    case 'U':
      if (isFPConstant)
        break; // Error for floating constant.
      if (isUnsigned)
        break; // Cannot be repeated.
      isUnsigned = true;
      continue; // Success.
    }

    if (s != ThisTokEnd) {
      hadError = true;
    }
  }
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
    llvm_unreachable("impossible Radix");
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

DpctAsmExprResult DpctAsmParser::ActOnNumericConstant(const DpctAsmToken &Tok) {
  assert(Tok.is(asmtok::numeric_constant) && Tok.getLength() >= 1);
  StringRef LiteralData(Tok.getLiteralData(), Tok.getLength());
  if (Tok.getLength() == 1 && isDigit(LiteralData[0])) {
    DpctAsmBuiltinType *Type = Context.getS64Type();
    llvm::APInt Val(64, LiteralData[0] - '0', true);
    return ::new (Context) DpctAsmIntegerLiteral(Type, Val);
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
      return ::new (Context) DpctAsmExactMachineFloatingLiteral(
          Context.getF32Type(), Float,
          LiteralParser.getExactMachineFloatingHexLiteralDigits());
    }

    if (LiteralParser.isExactMachineDouble) {
      APFloat Float(APFloat::IEEEdouble());
      auto Status = LiteralParser.GetFloatValue(Float);
      if (Status != APFloat::opOK)
        return AsmExprError();
      return ::new (Context) DpctAsmExactMachineFloatingLiteral(
          Context.getF64Type(), Float,
          LiteralParser.getExactMachineFloatingHexLiteralDigits());
    }

    APFloat Float(APFloat::IEEEdouble());
    auto Status = LiteralParser.GetFloatValue(Float);
    if (Status != APFloat::opOK)
      return AsmExprError();
    return ::new (Context) DpctAsmFloatingLiteral(Context.getF64Type(), Float);
  }

  APInt Int(64, 0);
  if (LiteralParser.GetIntegerValue(Int))
    return AsmExprError();

  DpctAsmBuiltinType *IntType =
      LiteralParser.isUnsigned ? Context.getU64Type() : Context.getS64Type();

  return ::new (Context) DpctAsmIntegerLiteral(IntType, Int);
}

DpctAsmExprResult DpctAsmParser::ActOnAlignment(DpctAsmExpr *Alignment) {
  if (auto *Int = dyn_cast<DpctAsmIntegerLiteral>(Alignment)) {
    return Int;
  }
  return AsmExprError();
}

DpctAsmDeclResult DpctAsmParser::ActOnVariableDecl(DpctAsmIdentifierInfo *Name,
                                                   DpctAsmType *Type) {
  DpctAsmVariableDecl *D = ::new (Context) DpctAsmVariableDecl(Name, Type);
  getCurScope()->addDecl(D);
  return D;
}
