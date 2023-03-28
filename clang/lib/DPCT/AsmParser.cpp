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

AsmStmt *AsmContext::CreateStmt(AsmStmt::StmtKind Kind) {
  AsmStmt *Stmt = new (*this) AsmStmt(Kind);
  return Stmt;
}

AsmStmt *AsmContext::CreateIntegerConstant(AsmType *Type, APInt Val) {
  AsmStmt *Const = CreateStmt(AsmStmt::SK_Integer);
  Const->Integer = Val;
  Const->Type = Type;
  return Const;
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

// // .align uint
// AsmStmtResult AsmParser::ParseAlign() {
//   assert(getTok().getString() == ".align");
//   Lex(); // eat '.align'
//   if (getTok().isNot(AsmToken::Integer)) {
//     // Parse error, expect an integer constant
//     return AsmStmtResult(true);
//   }
  
//   AsmToken ConstTok = getTok();

//   return AsmStmtResult(Context.CreateIntegerConstant(AsmType::getU64(), ConstTok.getAPIntVal()));
// }

// AsmStmtResult AsmParser::ParseDeclType() {
//   if (getTok().isVarAttributes()) {
//     if (getTok().getString() == ".align")

//   }
// }

AsmStmtResult AsmParser::ParseStatement() {
  if (getTok().is(AsmToken::LBrac)) {
    return ParseCompoundStatement();
  }
  return ParseInstruction();
}

AsmStmtResult AsmParser::ParseCompoundStatement() {
  assert(getTok().is(AsmToken::LBrac));
  Lex(); // eat '{'
  AsmStmt *Block = Context.CreateStmt(AsmStmt::SK_Block);
  ParseScope BlockScope(this);

  SmallVector<AsmStmt *, 10> Stmts;
  while (getTok().isNot(AsmToken::RBrac)) {
    if (getTok().isVarAttributes()) {
      AsmStmtResult Res;/* = ParseDeclaration*/
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

AsmStmtResult AsmParser::ParseInstruction() {
  AsmStmt *Inst = Context.CreateStmt(AsmStmt::SK_Inst);
  if (getTok().is(AsmToken::At)) {
    AsmStmtResult PredExpr = ParsePredicate();
    if (PredExpr.isInvalid())
      return false;
    Inst->Pred = PredExpr.get();
  }
  
  AsmStmtResult UnGuardInst = ParseUnGuardInstruction();

  if (UnGuardInst.isInvalid())
    return false;
  
  Inst->Body = UnGuardInst.get();
  return Inst;
}
