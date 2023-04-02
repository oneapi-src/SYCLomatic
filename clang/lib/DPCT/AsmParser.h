//===---------------------------- AsmParser.h -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DPCT_INLINE_ASM_PARSER_H
#define CLANG_DPCT_INLINE_ASM_PARSER_H

#include "AsmLexer.h"
#include "clang/Basic/LLVM.h"
#include "clang/Sema/Ownership.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/PointerLikeTypeTraits.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

namespace clang::dpct {

using llvm::SMLoc;
using llvm::SMRange;
using llvm::SourceMgr;

// struct AsmType {
// public:
//   enum TypeClass {
//     T_PTR,
//     T_ARRAY,
//     T_VLA,
//   };

//   TypeClass Kind;

// };

// class AsmDecl {
// public:
//   enum DeclClass {
//     VariableDeclClass,
//     LabelDeclClass,
//   };
// private:
//   unsigned dClass;
//   AsmIdentifierInfo *Name;
// public:
//   unsigned getDeclClass() const {
//     return dClass;
//   }

//   AsmIdentifierInfo *getDeclName() const {
//     return Name;
//   }

//   void *operator new(size_t bytes) noexcept {
//     llvm_unreachable("AsmDecl cannot be allocated with regular 'new'.");
//   }

//   void operator delete(void *data) noexcept {
//     llvm_unreachable("AsmDecl cannot be released with regular 'delete'.");
//   }
// };

// class AsmExpr;

// class AsmStmt {
// public:
//   enum StmtClass {
//     NoStmtClass = 0,
//     NullStmtClass,
//     CompoundStmtClass,
//     DeclStmtClass,
//     LabelStmtClass,
//     InstructionStmtClass,
//     UnaryOpExprClass,
//     BinaryOpExprClass,
//     DeclRefExprClass,
//     ConstantExprClass,
//     GuardExprClass,
//     OperandExprClass
//   };

//   void *operator new(size_t bytes) noexcept {
//     llvm_unreachable("Stmts cannot be allocated with regular 'new'.");
//   }

//   void operator delete(void *data) noexcept {
//     llvm_unreachable("Stmts cannot be released with regular 'delete'.");
//   }

//   StmtClass getStmtClass() const {
//     return static_cast<StmtClass>(sClass);
//   }

// private:
//   unsigned sClass;
// protected:
//   AsmStmt(StmtClass SC) : sClass(SC) {}
// };

// class AsmNullStmt : public AsmStmt {
//   SMLoc SemiLoc;
// public:
//   AsmNullStmt(SMLoc Loc) : AsmStmt(NullStmtClass), SemiLoc(Loc) {}

//   SMLoc getSemiLoc() const {
//     return SemiLoc;
//   }
// };

// class AsmCompoundStmt : public AsmStmt {
//   SmallVector<AsmStmt *> Stmts;
//   SMLoc LParenLoc;
//   SMLoc RParenLoc;
// public:
//   AsmCompoundStmt(SmallVector<AsmStmt *> Stmts, SMLoc LParenLoc,
//                   SMLoc RParenLoc)
//       : AsmStmt(CompoundStmtClass), Stmts(Stmts), LParenLoc(LParenLoc),
//         RParenLoc(RParenLoc) {}

//   ArrayRef<AsmStmt *> getStmts() const {
//     return Stmts;
//   }

//   static bool classof(const AsmStmt *S) {
//     return S->getStmtClass() == CompoundStmtClass;
//   }
// };

// class AsmDeclStmt : public AsmStmt {
//   SmallVector<AsmDecl *> DeclGroup;
//   SMLoc StartLoc;
//   SMLoc EndLoc;
// public:
//   AsmDeclStmt(SmallVector<AsmStmt *> Stmts, SMLoc Start, SMLoc End)
//     : AsmStmt(DeclStmtClass), StartLoc(Start), EndLoc(End) {}

//   static bool classof(const AsmStmt *S) {
//     return S->getStmtClass() == DeclStmtClass;
//   }
// };

// class LabelStmt : public AsmStmt {
// public:
//   LabelStmt() : AsmStmt(LabelStmtClass) {}

//   static bool classof(const AsmStmt *S) {
//     return S->getStmtClass() == LabelStmtClass;
//   }
// };

// /// Base class for the full range of assembler expressions which are
// /// needed for parsing.
// class AsmExpr : public AsmStmt {

//   SMLoc Loc;

// protected:
//   explicit AsmExpr(StmtClass SC, SMLoc Loc, unsigned SubclassData = 0)
//       : AsmStmt(SC), Loc(Loc) {
//   }

// public:
//   AsmExpr(const AsmExpr &) = delete;
//   AsmExpr &operator=(const AsmExpr &) = delete;

//   SMLoc getLoc() const { return Loc; }

//   void print(raw_ostream &OS, bool InParens = false) const;
//   void dump() const;
// };

// inline raw_ostream &operator<<(raw_ostream &OS, const AsmExpr &E) {
//   E.print(OS);
//   return OS;
// }

// class AsmConstantExpr : public AsmExpr {
//   int64_t Value;
// public:

//   AsmConstantExpr(int64_t Value)
//       : AsmExpr(ConstantExprClass, SMLoc()),
//         Value(Value) {}

//   int64_t getValue() const { return Value; }
// };

// class AsmDeclRefExpr : public AsmExpr {
//   const AsmDecl *RefDecl;

// public:
//   AsmDeclRefExpr(const AsmDecl *D, SMLoc Loc = SMLoc())
//       : AsmExpr(DeclRefExprClass, Loc), RefDecl(D) {}
//   const AsmDecl &getDecl() const { return *RefDecl; }
// };

// class AsmUnaryExpr : public AsmExpr {
// public:
//   enum Opcode {
//     LNot,  ///< Logical negation.
//     Minus, ///< Unary minus.
//     Not,   ///< Bitwise negation.
//     Plus   ///< Unary plus.
//   };

// private:
//   const AsmExpr *Expr;
//   unsigned Op;
//   AsmUnaryExpr(Opcode Op, const AsmExpr *Expr, SMLoc Loc)
//       : AsmExpr(UnaryOpExprClass, Loc), Expr(Expr), Op(Op) {}

// public:

//   Opcode getOpcode() const { return (Opcode)Op; }

//   const AsmExpr *getSubExpr() const { return Expr; }
// };

// class AsmBinaryExpr : public AsmExpr {
// public:
//   enum Opcode {
//     Add,   ///< Addition.
//     And,   ///< Bitwise and.
//     Div,   ///< Signed division.
//     EQ,    ///< Equality comparison.
//     GT,    ///< Signed greater than comparison (result is either 0 or some
//            ///< target-specific non-zero value)
//     GTE,   ///< Signed greater than or equal comparison (result is either 0
//     or
//            ///< some target-specific non-zero value).
//     LAnd,  ///< Logical and.
//     LOr,   ///< Logical or.
//     LT,    ///< Signed less than comparison (result is either 0 or
//            ///< some target-specific non-zero value).
//     LTE,   ///< Signed less than or equal comparison (result is either 0 or
//            ///< some target-specific non-zero value).
//     Mod,   ///< Signed remainder.
//     Mul,   ///< Multiplication.
//     NE,    ///< Inequality comparison.
//     Or,    ///< Bitwise or.
//     OrNot, ///< Bitwise or not.
//     Shl,   ///< Shift left.
//     AShr,  ///< Arithmetic shift right.
//     LShr,  ///< Logical shift right.
//     Sub,   ///< Subtraction.
//     Xor    ///< Bitwise exclusive or.
//   };

// private:
//   const AsmExpr *LHS, *RHS;
//   unsigned Op;
//   AsmBinaryExpr(Opcode Op, const AsmExpr *LHS, const AsmExpr *RHS,
//                 SMLoc Loc = SMLoc())
//       : AsmExpr(BinaryOpExprClass, Loc), LHS(LHS), RHS(RHS), Op(Op) {}

// public:

//   /// Get the kind of this binary expression.
//   Opcode getOpcode() const { return (Opcode)Op; }

//   /// Get the left-hand side expression of the binary operator.
//   const AsmExpr *getLHS() const { return LHS; }

//   /// Get the right-hand side expression of the binary operator.
//   const AsmExpr *getRHS() const { return RHS; }
// };

// class AsmInstruction;

// class AsmGrandExpr : public AsmExpr {
//   AsmInstruction *SubInst;
//   AsmDeclRefExpr *Predicate;
// };

namespace RoundMod {

} // namespace RoundMod


struct AsmType;

struct AsmSymbol {
  StringRef Name;
  AsmType *Type;
  bool IsVariable;
  bool IsLabel;
};

struct VarAttr {
  uint64_t Align;
  bool IsShared;
};

struct InstAttr {
  unsigned RoundMod;
  unsigned SatMod;
  unsigned SatfMod;
  unsigned FtzMod;
  unsigned AbsMod;
  unsigned TypeMod;
  unsigned AppxMod;
  unsigned ClampMod;
  unsigned SyncMod;
  unsigned ShuffleMod;
  unsigned Vector;
  unsigned Comparison;
  unsigned BooleanOp;
  unsigned ArithmeticOp;
  unsigned StorageClass;
  SmallVector<AsmType *, 4> Types;
};

struct AsmStatement {
  enum StmtKind {
    SK_NullExpr,
    SK_Add,
    SK_Sub,
    SK_Mul,
    SK_Div,
    SK_Mod,
    SK_BitAnd,
    SK_BitOr,
    SK_BitXor,
    SK_BitNot,
    SK_Shl,
    SK_Shr,
    SK_EQ,
    SK_NE,
    SK_LT,
    SK_GT,
    SK_LE,
    SK_GE,
    SK_Not,
    SK_And,
    SK_Or,
    SK_Neg,
    SK_Assign,
    SK_Cond,
    SK_Addr,
    SK_Deref,
    SK_Block,
    SK_Label,
    SK_ExprStmt,
    SK_StmtExpr,
    SK_Variable,
    SK_VLAPtr,
    SK_Integer,
    SK_Unsigned,
    SK_Float,
    SK_Double,
    SK_Cast,
    SK_Inst,
    SK_Sink,
    SK_Tuple,
  };

  StmtKind Kind;
  AsmStatement *Next;
  AsmType *Type;
  AsmStatement *LHS;
  AsmStatement *RHS;
  AsmStatement *SubExpr;
  AsmStatement *Pred;
  AsmStatement *PredOutput;
  AsmStatement *Cond;
  AsmStatement *Then;
  AsmStatement *Else;
  AsmStatement *Init;
  AsmStatement *Body;
  StringRef Label;
  AsmStatement *Bar;
  AsmSymbol *Variable;

  union {
    uint64_t u64;
    int64_t i64;
    float f32;
    double f64;
  };

  InstAttr InstructionAttr;
  SmallVector<AsmStatement *, 4> Operands;
  SmallVector<AsmStatement *, 4> Tuple;

  AsmStatement(StmtKind K) : Kind(K) {}
};

struct AsmType {
  enum TypeKind {
    TK_B8,
    TK_B16,
    TK_B32,
    TK_B64,
    TK_B128,
    TK_S2,
    TK_S4,
    TK_S8,
    TK_S16,
    TK_S32,
    TK_S64,
    TK_U2,
    TK_U4,
    TK_U8,
    TK_U16,
    TK_U32,
    TK_U64,
    TK_F16,
    TK_F16x2,
    TK_F32,
    TK_F64,
    TK_E4m3,
    TK_E5m2,
    TK_E4m3x2,
    TK_E5m2x2,
    TK_Byte,
    TK_4Byte,
    TK_Pred,
    TK_V2,
    TK_V4,
    TK_Ptr,
    TK_Array,
    TK_VLA
  };

  TypeKind Kind;
  int Size;
  int Align;
  AsmType *Origin;
  AsmType *Base;
  AsmToken Name;
  size_t ArrayLength;
  bool IsFlexible;
};

using AsmStmtResult = ActionResult<dpct::AsmStatement *>;

class AsmContext {
  llvm::BumpPtrAllocator Allocator;
  std::map<AsmType::TypeKind, AsmType *> ScalarTypes;
  AsmStatement *SinkExpression;
public:
  void *allocate(unsigned Size, unsigned Align = 8) {
    return Allocator.Allocate(Size, Align);
  }

  void deallocate(void *Ptr) {}

  
  AsmType *getScalarType(AsmType::TypeKind Kind);
  AsmType *getScalarTypeFromName(StringRef TypeName);
  AsmType::TypeKind getScalarTypeKindFromName(StringRef TypeName);

  // AsmType *getB8();
  // AsmType *getB16();
  // AsmType *getB32();
  // AsmType *getB64();
  // AsmType *getB128();
  // AsmType *getS2();
  // AsmType *getS4();
  // AsmType *getS8();
  // AsmType *getS16();
  // AsmType *getS32();
  // AsmType *getS64();
  // AsmType *getU2();
  // AsmType *getU4();
  // AsmType *getU8();
  // AsmType *getU16();
  // AsmType *getU32();
  // AsmType *getU64();
  // AsmType *getF16();
  // AsmType *getF16x2();
  // AsmType *getF32();
  // AsmType *getF64();
  // AsmType *getE4m3();
  // AsmType *getE5m2();
  // AsmType *getE4m3x2();
  // AsmType *getF5m2x2();
  // AsmType *getByte();
  // AsmType *get4Byte();
  // AsmType *getPred();
  AsmType *PointTo(AsmType *Base);
  AsmType *ArrayOf(AsmType *Base, size_t Len);
  AsmType *VLAOf(AsmType *Base, AsmStatement *Expr);

  AsmStatement *CreateStmt(AsmStatement::StmtKind Kind);
  AsmStatement *CreateIntegerConstant(AsmType *Type, int64_t Val);
  AsmStatement *CreateIntegerConstant(AsmType *Type, uint64_t Val);
  AsmStatement *CreateFloatConstant(AsmType *Type, float Val);
  AsmStatement *CreateFloatConstant(AsmType *Type, double Val);
  AsmStatement *CreateConditionalExpression(AsmStatement *Cond, AsmStatement *Then, AsmStatement *Else);
  AsmStatement *CreateBinaryOperator(AsmStatement::StmtKind Opcode, AsmStatement *LHS, AsmStatement *RHS);
  AsmStatement *CreateUnaryExpression(AsmStatement::StmtKind Opcode, AsmStatement *SubExpr);
  AsmStatement *CreateCastExpression(AsmType *Type, AsmStatement *SubExpr);
  AsmStatement *CreateVariableRefExpression(AsmSymbol *Symbol);
  AsmStatement *GetOrCreateSinkExpression();
};

class AsmIdentifierInfo {
  friend class AsmIdentifierTable;
  unsigned TokenID;
  llvm::StringMapEntry<AsmIdentifierInfo *> *Entry = nullptr;

  AsmIdentifierInfo()
      : TokenID(AsmToken::Identifier) {}

public:
  AsmIdentifierInfo(const AsmIdentifierInfo &) = delete;
  AsmIdentifierInfo &operator=(const AsmIdentifierInfo &) = delete;
  AsmIdentifierInfo(AsmIdentifierInfo &&) = delete;
  AsmIdentifierInfo &operator=(AsmIdentifierInfo &&) = delete;

  const char *getNameStart() const { return Entry->getKeyData(); }
  unsigned getLength() const { return Entry->getKeyLength(); }
  StringRef getName() const { return StringRef(getNameStart(), getLength()); }

  template <std::size_t StrLen> bool isStr(const char (&Str)[StrLen]) const {
    return getLength() == StrLen - 1 &&
           memcmp(getNameStart(), Str, StrLen - 1) == 0;
  }

  bool isStr(llvm::StringRef Str) const {
    llvm::StringRef ThisStr(getNameStart(), getLength());
    return ThisStr == Str;
  }

  AsmToken::TokenKind getTokenID() const {
    return (AsmToken::TokenKind)TokenID;
  }
};

class AsmIdentifierTable {
  using HashTableTy =
      llvm::StringMap<AsmIdentifierInfo *, llvm::BumpPtrAllocator>;
  HashTableTy HashTable;

public:
  AsmIdentifierTable() = default;

  llvm::BumpPtrAllocator &getAllocator() { return HashTable.getAllocator(); }

  AsmIdentifierInfo &get(StringRef Name) {
    auto &Entry = *HashTable.try_emplace(Name, nullptr).first;

    AsmIdentifierInfo *&II = Entry.second;
    if (II)
      return *II;

    // Lookups failed, make a new IdentifierInfo.
    void *Mem = getAllocator().Allocate<AsmIdentifierInfo>();
    II = new (Mem) AsmIdentifierInfo();

    // Make sure getName() knows how to find the IdentifierInfo
    // contents.
    II->Entry = &Entry;

    return *II;
  }

  AsmIdentifierInfo &get(StringRef Name, AsmToken::TokenKind TokenCode) {
    AsmIdentifierInfo &II = get(Name);
    II.TokenID = TokenCode;
    assert(II.TokenID == (unsigned)TokenCode && "TokenCode too large");
    return II;
  }

  using iterator = HashTableTy::const_iterator;
  using const_iterator = HashTableTy::const_iterator;

  iterator begin() const { return HashTable.begin(); }
  iterator end() const { return HashTable.end(); }
  unsigned size() const { return HashTable.size(); }
};

class AsmScope {
  using DeclSetTy = llvm::SmallPtrSet<AsmSymbol *, 32>;
  AsmScope *AnyParent;
  DeclSetTy DeclsInScope;
  unsigned Depth;

public:
  AsmScope(AsmScope *Parent) : AnyParent(Parent), Depth(Parent->Depth + 1) {}

  const AsmScope *getParent() const { return AnyParent; }
  AsmScope *getParent() { return AnyParent; }
  unsigned getDepth() const { return Depth; }

  using decl_range = llvm::iterator_range<DeclSetTy::iterator>;

  decl_range decls() const {
    return decl_range(DeclsInScope.begin(), DeclsInScope.end());
  }

  void AddDecl(AsmSymbol *D) { DeclsInScope.insert(D); }

  bool isDeclScope(const AsmSymbol *D) const { return DeclsInScope.contains(D); }

  bool Contains(const AsmScope &rhs) const { return Depth < rhs.Depth; }

  AsmSymbol *LookupSymbol(StringRef Symbol) const;
};


class AsmParser {
public:
  struct PendingError {
    SMLoc Loc;
    std::string Msg;
    SMRange Range;
  };

private:
  AsmLexer Lexer;
  AsmContext &Context;
  SourceMgr &SrcMgr;
  AsmScope *CurScope;

  class ParseScope {
    AsmParser *Self;
    ParseScope(const ParseScope &) = delete;
    void operator=(const ParseScope &) = delete;

  public:
    ParseScope(AsmParser *Self) : Self(Self) { Self->EnterScope(); }

    ~ParseScope() {
      Self->ExitScope();
      Self = nullptr;
    }
  };

public:
  AsmParser(AsmContext &Ctx, SourceMgr &Mgr)
      : Lexer(), Context(Ctx), SrcMgr(Mgr) {
    unsigned MainFileID = Mgr.getMainFileID();
    StringRef BufferRef = Mgr.getMemoryBuffer(MainFileID)->getBuffer();
    Lexer.setBuffer(BufferRef);
  }
  ~AsmParser();

  SourceMgr &getSourceManager() { return SrcMgr; }
  AsmLexer &getLexer() { return Lexer; }
  AsmContext &getContext() { return Context; }

  const AsmLexer &getLexer() const {
    return const_cast<AsmParser *>(this)->getLexer();
  }

  const AsmToken &getTok() const { return getLexer().getTok(); }
  const AsmToken &Lex();

  AsmScope *getCurScope() const { return CurScope; }

  void EnterScope() { CurScope = new AsmScope(getCurScope()); }

  void ExitScope() {
    assert(getCurScope());
    AsmScope *OldScope = getCurScope();
    CurScope = OldScope->getParent();
    delete CurScope;
  }

  /// TODO: bool Parse();
  AsmStmtResult ParseStatement();
  AsmStmtResult ParseCompoundStatement();
  AsmStmtResult ParsePredicate();
  AsmStmtResult ParseUnGuardInstruction();
  AsmStmtResult ParseInstruction();
  InstAttr ParseInstructionFlags();
  AsmStmtResult ParseTuple();
  AsmStmtResult ParseInstructionFirstOperand();
  AsmStmtResult ParseInstructionOperand();

  AsmStmtResult ParseConstantExpression();
  // bool ParseExpression();
  AsmStmtResult ParsePrimaryExpression();
  // bool ParseAssignExpression();
  AsmStmtResult ParseConditionalExpression();
  AsmStmtResult ParseLogicOrExpression();
  AsmStmtResult ParseLogicAndExpression();
  AsmStmtResult ParseInclusiveOrExpression();
  AsmStmtResult ParseExclusiveOrExpression();
  AsmStmtResult ParseAndExpression();
  AsmStmtResult ParseEqualityExpression();
  AsmStmtResult ParseRelationExpression();
  AsmStmtResult ParseShiftExpression();
  AsmStmtResult ParseAdditiveExpression();
  AsmStmtResult ParseMultiplicativeExpression();
  AsmStmtResult ParseCaseExpresion();
  AsmStmtResult ParseUnaryExpression();
  // bool ParsePostfixExpression();
};

} // namespace clang::dpct

inline void *operator new(size_t Bytes, clang::dpct::AsmContext &C,
                          size_t Alignment = 8) noexcept {
  return C.allocate(Bytes, Alignment);
}

inline void operator delete(void *Ptr, clang::dpct::AsmContext &C,
                            size_t) noexcept {
  C.deallocate(Ptr);
}

inline void *operator new[](size_t Bytes, clang::dpct::AsmContext &C,
                            size_t Alignment = 8) noexcept {
  return C.allocate(Bytes, Alignment);
}

inline void operator delete[](void *Ptr, clang::dpct::AsmContext &C) noexcept {
  C.deallocate(Ptr);
}

#endif // CLANG_DPCT_INLINE_ASM_PARSER_H
