//===---------------------------- AsmParser.h -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DPCT_INLINE_ASM_PARSER_H
#define CLANG_DPCT_INLINE_ASM_PARSER_H

#include "AsmNodes.h"
#include "AsmIdentifierTable.h"
#include "AsmToken.h"
#include "AsmTokenKinds.h"
#include "AsmLexer.h"
#include "clang/Sema/Ownership.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/PointerLikeTypeTraits.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"

namespace clang::dpct {

using llvm::BumpPtrAllocator;
using llvm::SmallPtrSet;
using llvm::SmallSet;
using llvm::SMLoc;
using llvm::SourceMgr;

/// Holds long-lived AST nodes (such as types and decls) and predefined
/// identifiers that can be referred to throughout the semantic analysis of a
/// file.
class InlineAsmContext : public InlineAsmIdentifierInfoLookup {

  /// The allocator used to create AST objects.
  BumpPtrAllocator Allocator;

  /// The identifier table used to store predefined identifiers.
  InlineAsmIdentifierTable AsmBuiltinIdTable;

  /// This references identifiers in the predefined identifier table and
  /// provides the ability to index by subscript.
  SmallVector<InlineAsmIdentifierInfo *, 4> InlineAsmOperands;

  /// This array used to cache the builtin types.
  InlineAsmBuiltinType *AsmBuiltinTypes[InlineAsmBuiltinType::NUM_TYPES] = {0};

  /// This used to cache the discard type.
  InlineAsmDiscardType *DiscardType = nullptr;

public:
  void *allocate(unsigned Size, unsigned Align = 8) {
    return Allocator.Allocate(Size, Align);
  }

  void deallocate(void *Ptr) {}

  /// Add predefined inline asm operand, e.g. %0
  unsigned addInlineAsmOperand(StringRef Name) {
    InlineAsmIdentifierInfo &II = AsmBuiltinIdTable.get(Name);
    InlineAsmOperands.push_back(&II);
    return InlineAsmOperands.size() - 1;
  }

  /// Lookup predefined identifiers. A predefined identifier must start with
  /// '%', and the length must greater than 2. If the character after '%' are
  /// all digits, e.g. '%1', then this identifier maybe a inline asm
  /// placeholder, else the identifier maybe a device asm predefined identifier,
  /// e.g. '%laneid'.
  InlineAsmIdentifierInfo *get(StringRef Name) override {

    // Predefined identifiers must start with '%', and the length must greater
    // than 2 characters.
    if (Name.size() < 2 || Name[0] != '%')
      return nullptr;

    // This identifier is an inline asm placeholder, e.g. %0, %1, etc.
    if (isDigit(Name[1])) {
      unsigned Num;
      if (Name.drop_front().getAsInteger(10, Num) ||
          Num >= InlineAsmOperands.size()) {
        return nullptr;
      }
      return InlineAsmOperands[Num];
    }

    StringRef BuiltinName = Name.drop_front();
    // This identifier is an builtin.
    if (AsmBuiltinIdTable.contains(BuiltinName)) {
      return &AsmBuiltinIdTable.get(BuiltinName);
    }

    return nullptr;
  }

  /// Get the placeholder identifier, e.g. %1, %2, ..., etc.
  InlineAsmIdentifierInfo *get(unsigned Index) {
    if (Index >= InlineAsmOperands.size())
      return nullptr;
    return InlineAsmOperands[Index];
  }

  InlineAsmBuiltinType *getTypeFromConstraint(StringRef Constraint);
  InlineAsmBuiltinType *getTypeFromClangType(const Type *E);
  InlineAsmBuiltinType *getBuiltinType(StringRef TypeName);
  InlineAsmBuiltinType *getBuiltinType(InlineAsmBuiltinType::TypeKind Kind);
  InlineAsmBuiltinType *getBuiltinTypeFromTokenKind(asmtok::TokenKind Kind);
  InlineAsmDiscardType *getDiscardType();

  InlineAsmBuiltinType *getS64Type() {
    return getBuiltinType(InlineAsmBuiltinType::s64);
  }

  InlineAsmBuiltinType *getU64Type() {
    return getBuiltinType(InlineAsmBuiltinType::u64);
  }

  InlineAsmBuiltinType *getF32Type() {
    return getBuiltinType(InlineAsmBuiltinType::f32);
  }

  InlineAsmBuiltinType *getF64Type() {
    return getBuiltinType(InlineAsmBuiltinType::f64);
  }
};

/// Introduces a new scope for parsing when meet compound statement.
/// e.g. { stmt stmt }
class InlineAsmScope {
  using DeclSetTy = SmallPtrSet<InlineAsmVarDecl *, 32>;
  InlineAsmParser &Parser;

  /// Parent scope.
  InlineAsmScope *Parent;

  // Declarations in this scope.
  DeclSetTy DeclsInScope;

  // Used to represents the depth between the toplevel scope and current scope.
  unsigned Depth;

public:
  InlineAsmScope(InlineAsmParser &Parser, InlineAsmScope *Parent)
      : Parser(Parser), Parent(Parent), Depth(Parent ? Parent->Depth + 1 : 0) {}

  bool hasParent() const { return Parent; }
  const InlineAsmScope *getParent() const { return Parent; }
  InlineAsmScope *getParent() { return Parent; }
  unsigned getDepth() const { return Depth; }

  using decl_range = llvm::iterator_range<DeclSetTy::iterator>;

  decl_range decls() const {
    return decl_range(DeclsInScope.begin(), DeclsInScope.end());
  }

  void addDecl(InlineAsmVarDecl *D) { DeclsInScope.insert(D); }

  bool isDeclScope(const InlineAsmVarDecl *D) const {
    return DeclsInScope.contains(D);
  }

  bool contains(const InlineAsmScope &rhs) const { return Depth < rhs.Depth; }

  /// Lookup a simple declaration in current scope and parent scopes.
  InlineAsmVarDecl *lookupDecl(InlineAsmIdentifierInfo *II) const;

  /// Lookup a parameterized name declaration in current scope and parent
  /// scopes.
  InlineAsmVarDecl *
  lookupParameterizedNameDecl(InlineAsmIdentifierInfo *II, unsigned &Idx) const;
};

using InlineAsmTypeResult = clang::ActionResult<InlineAsmType *>;
using InlineAsmDeclResult = clang::ActionResult<InlineAsmDecl *>;
using InlineAsmStmtResult = clang::ActionResult<InlineAsmStmt *>;
using InlineAsmExprResult = clang::ActionResult<InlineAsmExpr *>;

inline InlineAsmExprResult AsmExprError() { return InlineAsmExprResult(true); }
inline InlineAsmStmtResult AsmStmtError() { return InlineAsmStmtResult(true); }
inline InlineAsmTypeResult AsmTypeError() { return InlineAsmTypeResult(true); }
inline InlineAsmDeclResult AsmDeclError() { return InlineAsmDeclResult(true); }

// clang-format off
namespace asm_precedence {

/// These are precedences for the binary/ternary operators in the C99 grammar.
/// These have been named to relate with the C99 grammar productions.  Low
/// precedences numbers bind more weakly than high numbers.
enum Level {
  Unknown = 0,     // Not binary operator.
  Assignment,      // =
  Conditional,     // ?
  LogicalOr,       // ||
  LogicalAnd,      // &&
  InclusiveOr,     // |
  ExclusiveOr,     // ^
  And,             // &
  Equality,        // ==, !=
  Relational,      //  >=, <=, >, <
  Shift,           // <<, >>
  Additive,        // -, +
  Multiplicative   // *, /, %
};

/// Return the precedence of the specified binary operator token.
asm_precedence::Level getBinOpPrec(asmtok::TokenKind Kind);
} // namespace asm_precedence
// clang-format on

/// Parser - This implements a parser for the device asm language.
class InlineAsmParser {
  InlineAsmLexer Lexer;
  InlineAsmContext &Context;
  SourceMgr &SrcMgr;
  InlineAsmScope *CurScope;

  /// Tok - The current token we are peeking ahead.  All parsing methods assume
  /// that this is valid.
  InlineAsmToken Tok;

  /// ScopeCache - Cache scopes to reduce malloc traffic.
  enum { ScopeCacheSize = 16 };
  unsigned NumCachedScopes = 0;
  InlineAsmScope *ScopeCache[ScopeCacheSize];

  class ParseScope {
    InlineAsmParser *Self;
    ParseScope(const ParseScope &) = delete;
    void operator=(const ParseScope &) = delete;

  public:
    ParseScope(InlineAsmParser *Self) : Self(Self) { Self->EnterScope(); }

    ~ParseScope() {
      Self->ExitScope();
      Self = nullptr;
    }
  };

public:
  InlineAsmParser(InlineAsmContext &Ctx, SourceMgr &Mgr)
      : Lexer(*Mgr.getMemoryBuffer(Mgr.getMainFileID())), Context(Ctx),
        SrcMgr(Mgr), CurScope(nullptr) {
    Lexer.getIdentifiertable().setExternalIdentifierLookup(&Context);
    Tok.startToken();
    Tok.setKind(asmtok::eof);
    ConsumeToken();
    EnterScope();
  }
  InlineAsmParser(const InlineAsmParser &) = delete;
  InlineAsmParser &operator=(const InlineAsmParser &) = delete;
  ~InlineAsmParser() { ExitScope(); }

  SourceMgr &getSourceManager() { return SrcMgr; }
  InlineAsmLexer &getLexer() { return Lexer; }
  InlineAsmContext &getContext() { return Context; }

  const InlineAsmToken &getCurToken() const { return Tok; }

  /// ConsumeToken - Consume the current 'peek token' and lex the next one.
  void ConsumeToken() { Lexer.lex(Tok); }

  bool TryConsumeToken(asmtok::TokenKind Expected) {
    if (Tok.isNot(Expected))
      return false;
    Lexer.lex(Tok);
    return true;
  }

  bool TryConsumeToken(asmtok::TokenKind Expected, SMLoc &Loc) {
    if (!TryConsumeToken(Expected))
      return false;
    return true;
  }

  InlineAsmDeclResult addInlineAsmOperands(const Expr *E, StringRef Operand,
                                           StringRef Constraint);
  void addBuiltinIdentifier();
  InlineAsmScope *getCurScope() const { return CurScope; }

  /// EnterScope - Start a new scope.
  void EnterScope() {
    if (NumCachedScopes) {
      InlineAsmScope *N = ScopeCache[--NumCachedScopes];
      CurScope = new (N) InlineAsmScope(*this, getCurScope());
    } else {
      CurScope = new InlineAsmScope(*this, getCurScope());
    }
  }

  /// ExitScope - Pop a scope off the scope stack.
  void ExitScope() {
    assert(getCurScope());
    InlineAsmScope *OldScope = getCurScope();
    CurScope = OldScope->getParent();
    if (NumCachedScopes == ScopeCacheSize) {
      delete OldScope;
    } else {
      OldScope->~InlineAsmScope();
      ScopeCache[NumCachedScopes++] = OldScope;
    }
  }

  /// Return true if current token is start of dot and is an instruction
  /// attributes, e.g. types, comparsion operator, rounding modifier.
  bool isInstructionAttribute();

  /// statement:
  ///     compound-statement
  ///     declaration
  ///     instruction
  ///     condition-instruction
  InlineAsmStmtResult ParseStatement();

  /// compound-statement:
  ///      { block-item-listopt }
  ///  block-item-list:
  ///          block-item
  ///          block-item-list block-item
  ///  block-item:
  ///          statement
  InlineAsmStmtResult ParseCompoundStatement();

  /// conditional-instruction:
  ///       @ expression instruction
  ///       @ ! expression instruction
  InlineAsmStmtResult ParseConditionalInstruction();

  /// instruction:
  ///       opcode attribute-list output-operand, input-operand-list ;
  ///       opcode attribute-list output-operand | pred-output,
  ///       input-operand-list ;
  ///   attribute-list:
  ///         attribute
  ///         attribute-list attribute
  ///   output-operand:
  ///         expression
  ///   input-operand:
  ///         expression
  ///   input-operand-list:
  ///         input-operand
  ///         input-operand-list , input-operand
  ///   pred-output:
  ///         expression
  ///   opcode:
  ///         mov setp cvt ...
  InlineAsmStmtResult ParseInstruction();

  /// multiplicative-expression:
  ///   cast-expression
  ///   multiplicative-expression '*' cast-expression
  ///   multiplicative-expression '/' cast-expression
  ///   multiplicative-expression '%' cast-expression
  ///
  /// additive-expression:
  ///   multiplicative-expression
  ///   additive-expression '+' multiplicative-expression
  ///   additive-expression '-' multiplicative-expression
  ///
  /// shift-expression:
  ///   additive-expression
  ///   shift-expression '<<' additive-expression
  ///   shift-expression '>>' additive-expression
  ///
  /// relational-expression:
  ///   shift-expression
  ///   relational-expression '<' shift-expression
  ///   relational-expression '>' shift-expression
  ///   relational-expression '<=' shift-expression
  ///   relational-expression '>=' shift-expression
  ///
  /// equality-expression:
  ///   relational-expression
  ///   equality-expression '==' relational-expression
  ///   equality-expression '!=' relational-expression
  ///
  /// AND-expression:
  ///   equality-expression
  ///   AND-expression '&' equality-expression
  ///
  /// exclusive-OR-expression:
  ///   AND-expression
  ///   exclusive-OR-expression '^' AND-expression
  ///
  /// inclusive-OR-expression:
  ///   exclusive-OR-expression
  ///   inclusive-OR-expression '|' exclusive-OR-expression
  ///
  /// logical-AND-expression:
  ///   inclusive-OR-expression
  ///   logical-AND-expression '&&' inclusive-OR-expression
  ///
  /// logical-OR-expression:
  ///   logical-AND-expression
  ///   logical-OR-expression '||' logical-AND-expression
  ///
  /// conditional-expression:
  ///   logical-OR-expression
  ///   logical-OR-expression '?' expression ':' conditional-expression
  ///
  /// assignment-expression:
  ///   conditional-expression
  ///   unary-expression assignment-operator assignment-expression
  ///
  /// assignment-operator:
  ///   =
  ///
  /// expression:
  ///   assignment-expression ...[opt]
  InlineAsmExprResult ParseExpression();

  /// Parse a cast-expression, unary-expression or primary-expression
  /// cast-expression:
  ///   unary-expression
  ///   '(' type-name ')' cast-expression
  ///
  /// unary-expression:
  ///   postfix-expression
  ///   unary-operator cast-expression
  ///
  /// unary-operator: one of
  ///   '+'  '-'  '~'  '!'
  ///
  /// primary-expression:
  ///   identifier
  ///   constant
  ///   '(' expression ')'
  ///
  /// constant: [C99 6.4.4]
  ///   integer-constant
  ///   floating-constant
  InlineAsmExprResult ParseCastExpression();
  InlineAsmExprResult ParseAssignmentExpression();

  /// primary-expression:
  ///   '(' expression ')'
  /// cast-expression:
  ///   '(' type-name ')' cast-expression
  InlineAsmExprResult ParseParenExpression(InlineAsmBuiltinType *&CastTy);

  /// Parse a binary expression that starts with \p LHS and has a
  /// precedence of at least \p MinPrec.
  InlineAsmExprResult ParseRHSOfBinaryExpression(InlineAsmExprResult LHS,
                                                 asm_precedence::Level MinPrec);

  /// Once the leading part of a postfix-expression is parsed, this
  /// method parses any suffixes that apply.
  /// FIXME: Postfix expression is not supported now.
  ///
  /// postfix-expression:
  ///   primary-expression
  ///   postfix-expression '[' expression ']'
  ///   postfix-expression '.' identifier
  InlineAsmExprResult ParsePostfixExpressionSuffix(InlineAsmExprResult LHS);

  /// FIXME: Assignment and initializer init are not supported now.
  /// declaration:
  ///       declaration-specifiers init-declarator-list[opt] ';'
  ///
  ///   init-declarator-list:
  ///           init-declarator
  ///           init-declarator-list , init-declarator
  ///
  ///   init-declarator:
  ///           declarator
  ///           declarator '=' initializer
  ///           declarator initializer[opt]
  ///
  ///   initializer:
  ///           braced-init-list
  ///
  ///   braced-init-list:
  ///           { initializer-list }
  ///
  ///   initializer-list:
  ///           constant-expression initializer
  ///           initializer-list , constant-expression[opt] initializer
  InlineAsmStmtResult ParseDeclarationStatement();

  /// FIXME: Only support .reg state space now.
  ///   declaration-specifiers:
  ///           state-space-specifier declaration-specifiers[opt]
  ///           vector-specifier declaration-specifiers[opt]
  ///           type-specifier declaration-specifiers[opt]
  ///           alignment-specifier integer-constant declaration-specifiers[opt]
  ///   state-space-specifier: one of
  ///           .reg .sreg .const .local .param .shared .tex
  ///
  ///   vector-specifier: one of
  ///           .v2 .v4 .v8
  ///
  ///   type-specifier: one of
  ///           .b8 .b16 .b32 .b64 .s8 .s16 .s32 .s64
  ///           .u8 .u16 .u32 .u64 .f16 .f32 .f64
  ///           ...
  ///
  ///   alignment-specifier:
  ///           .align
  InlineAsmTypeResult
  ParseDeclarationSpecifier(InlineAsmDeclarationSpecifier &DeclSpec);

  /// declarator:
  ///   identifier
  ///   declarator '[' constant-expression[opt] ']' FIXME: Array declaration is
  ///   not supported.
  InlineAsmDeclResult
  ParseDeclarator(const InlineAsmDeclarationSpecifier &DeclSpec);

  // Sema
  InlineAsmExprResult ActOnTypeCast(InlineAsmBuiltinType *CastTy,
                                    InlineAsmExpr *SubExpr);
  InlineAsmExprResult ActOnAddressExpr(InlineAsmExpr *SubExpr);
  InlineAsmExprResult ActOnDiscardExpr();
  InlineAsmExprResult ActOnParenExpr(InlineAsmExpr *SubExpr);
  InlineAsmExprResult ActOnIdExpr(InlineAsmIdentifierInfo *II);
  InlineAsmExprResult ActOnVectorExpr(ArrayRef<InlineAsmExpr *> Tuple);
  InlineAsmExprResult ActOnUnaryOp(asmtok::TokenKind OpTok,
                                   InlineAsmExpr *SubExpr);
  InlineAsmExprResult ActOnBinaryOp(asmtok::TokenKind OpTok, InlineAsmExpr *LHS,
                                    InlineAsmExpr *RHS);
  InlineAsmExprResult ActOnConditionalOp(InlineAsmExpr *Cond,
                                         InlineAsmExpr *LHS,
                                         InlineAsmExpr *RHS);
  InlineAsmExprResult ActOnNumericConstant(const InlineAsmToken &Tok);
  InlineAsmExprResult ActOnAlignment(InlineAsmExpr *Alignment);
  InlineAsmDeclResult ActOnVariableDecl(InlineAsmIdentifierInfo *Name,
                                        AsmStateSpace StateSpace,
                                        InlineAsmType *Type);
};
} // namespace clang::dpct

/// Below template specification was used for llvm data structures and
/// utilities.
namespace llvm {

template <typename T> struct PointerLikeTypeTraits;
template <> struct PointerLikeTypeTraits<::clang::dpct::InlineAsmType *> {
  static inline void *getAsVoidPointer(::clang::Type *P) { return P; }

  static inline ::clang::dpct::InlineAsmType *getFromVoidPointer(void *P) {
    return static_cast<::clang::dpct::InlineAsmType *>(P);
  }

  static constexpr int NumLowBitsAvailable = clang::TypeAlignmentInBits;
};

template <> struct PointerLikeTypeTraits<::clang::dpct::InlineAsmDecl *> {
  static inline void *getAsVoidPointer(::clang::ExtQuals *P) { return P; }

  static inline ::clang::dpct::InlineAsmDecl *getFromVoidPointer(void *P) {
    return static_cast<::clang::dpct::InlineAsmDecl *>(P);
  }

  static constexpr int NumLowBitsAvailable = clang::TypeAlignmentInBits;
};

template <> struct PointerLikeTypeTraits<::clang::dpct::InlineAsmStmt *> {
  static inline void *getAsVoidPointer(::clang::ExtQuals *P) { return P; }

  static inline ::clang::dpct::InlineAsmStmt *getFromVoidPointer(void *P) {
    return static_cast<::clang::dpct::InlineAsmStmt *>(P);
  }

  static constexpr int NumLowBitsAvailable = clang::TypeAlignmentInBits;
};

} // namespace llvm

inline void *operator new(size_t Bytes, ::clang::dpct::InlineAsmContext &C,
                          size_t Alignment = 8) noexcept {
  return C.allocate(Bytes, Alignment);
}

inline void operator delete(void *Ptr, ::clang::dpct::InlineAsmContext &C,
                            size_t) noexcept {
  C.deallocate(Ptr);
}

inline void *operator new[](size_t Bytes, ::clang::dpct::InlineAsmContext &C,
                            size_t Alignment = 8) noexcept {
  return C.allocate(Bytes, Alignment);
}

inline void operator delete[](void *Ptr,
                              ::clang::dpct::InlineAsmContext &C) noexcept {
  C.deallocate(Ptr);
}

#endif // CLANG_DPCT_INLINE_ASM_PARSER_H
