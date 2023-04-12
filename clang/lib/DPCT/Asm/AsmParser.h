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
#include "llvm/ADT/APFloat.h"
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

using llvm::SmallPtrSet;
using llvm::SMLoc;
using llvm::SMRange;
using llvm::SourceMgr;

namespace ptx {
enum InstKind {
  Abs,
  Activemask,
  Add,
  Addc,
  Alloca,
  And,
  Applypriority,
  Atom,
  Bar,
  Barrier,
  Bfe,
  Bfi,
  Bfind,
  Bmsk,
  Bra,
  Brev,
  Brkpt,
  Brx,
  Call,
  Clz,
  Cnot,
  Copysign,
  Cos,
  Cp,
  Createpolicy,
  Cvt,
  Cvta,
  Discard,
  Div,
  Dp2a,
  Dp4a,
  Elect,
  Ex2,
  Exit,
  Fence,
  Fma,
  Fns,
  Getctarank,
  Griddepcontrol,
  Isspacep,
  Istypep,
  Ld,
  Ldmatrix,
  Ldu,
  Lg2,
  Lop3,
  Mad,
  Mad24,
  Madc,
  Mapa,
  Match,
  Max,
  Mbarrier,
  Membar,
  Min,
  Mma,
  Mov,
  Movmatrix,
  Mul,
  Mul24,
  Multimem,
  Nanosleep,
  Neg,
  Not,
  Or,
  Pmevent,
  Popc,
  Prefetch,
  Prefetchu,
  Prmt,
  Rcp,
  Red,
  Redux,
  Rem,
  Ret,
  Rsqrt,
  Sad,
  Selp,
  Set,
  Setmaxnreg,
  Setp,
  Shf,
  Shfl,
  Shl,
  Shr,
  Sin,
  Slct,
  Sqrt,
  St,
  Stackrestore,
  Stacksave,
  Stmatrix,
  Sub,
  Subc,
  Suld,
  Suq,
  Sured,
  Sust,
  Szext,
  Tanh,
  Testp,
  Tex,
  Tld4,
  Trap,
  Txq,
  Vabsdiff,
  Vabsdiff2,
  Vabsdiff4,
  Vadd,
  Vadd2,
  Vadd4,
  Vavrg2,
  Vavrg4,
  Vmad,
  Vmax,
  Vmax2,
  Vmax4,
  Vmin,
  Vmin2,
  Vmin4,
  Vote,
  Vset,
  Vset2,
  Vset4,
  Vshl,
  Vshr,
  Vsub,
  Vsub2,
  Vsub4,
  Wgmma,
  Wmma,
  Xor,
  Invalid,
};

enum StorageClass {
  SC_Reg = 0x01,
  SC_Const = 0x02,
  SC_Global = 0x04,
  SC_Local = 0x08,
  SC_Param = 0x10,
  SC_Shared = 0x20,
  SC_Tex = 0x40
};

enum ComparisonOp {
  CO_Eq,
  CO_Ne,
  CO_Lt,
  CO_Le,
  CO_Gt,
  CO_Ge,
  CO_Lo,
  CO_Ls,
  CO_Hi,
  CO_Hs,
  CO_Equ,
  CO_Neu,
  CO_Ltu,
  CO_Leu,
  CO_Gtu,
  CO_Geu,
  CO_Num,
  CO_Nan
};

enum BooleanOp {
  BO_And,
  BO_Or,
  BO_Xor
};

enum ArithmeticOp {
  AO_Add,
  AO_Min,
  AO_Max,
  AO_Maxabs,
  AO_Popc
};

InstKind FindInstructionKindFromName(StringRef InstName);

} // namespace ptx

class PtxType;
class PtxDecl;
class PtxExpr;
class PtxStmt;

class PtxType {
public:
  enum TypeClass {
    FundamentalClass,
    TupleClass,
    ArrayClass,
    VectorClass,
    AnyClass
  };

private:
  TypeClass tClass;

protected:
  PtxType(TypeClass TC) : tClass(TC) {}

public:
  virtual ~PtxType();
  TypeClass getTypeClass() const { return tClass; }

  void *operator new(size_t bytes) noexcept {
    llvm_unreachable("PtxDecl cannot be allocated with regular 'new'.");
  }

  void operator delete(void *data) noexcept {
    llvm_unreachable("PtxDecl cannot be released with regular 'delete'.");
  }
};

class PtxFundamentalType : public PtxType {
public:
  enum TypeKind : int {
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
    TK_Pred
  };

private:
  TypeKind Kind;

public:
  PtxFundamentalType(TypeKind Kind)
      : PtxType(FundamentalClass), Kind(Kind) {}

  TypeKind getKind() const { return Kind; }

  static bool classof(const PtxType *T) {
    return T->getTypeClass() == FundamentalClass;
  }
};

class PtxVectorType : public PtxType {
public:
  enum TypeKind { V2, V4 };

private:
  TypeKind Kind;
  const PtxFundamentalType *BaseType;

public:
  PtxVectorType(TypeKind Kind, const PtxFundamentalType *Base)
      : PtxType(VectorClass), Kind(Kind), BaseType(Base) {}

  TypeKind getKind() const { return Kind; }
  const PtxFundamentalType *getBaseType() const { return BaseType; }

  static bool classof(const PtxType *T) {
    return T->getTypeClass() == VectorClass;
  }
};

class PtxTupleType : public PtxType {
public:
  typedef SmallVector<const PtxType *, 4> ElementList;
private:
  ElementList ElementTypes;

public:
  PtxTupleType(const ElementList &ElementTypes)
      : PtxType(TupleClass), ElementTypes(ElementTypes) {}

  const ElementList &getElementTypes() const {
    return ElementTypes;
  }

  const PtxType *getElementType(unsigned I) const { return ElementTypes[I]; }

  static bool classof(const PtxType *T) {
    return T->getTypeClass() == TupleClass;
  }
};

class PtxAnyType : public PtxType {
public:
  PtxAnyType() : PtxType(AnyClass) {}

  static bool classof(const PtxType *T) {
    return T->getTypeClass() == AnyClass;
  }
};

class PtxDecl {
public:
  enum DeclClass {
    VariableDeclClass,
    LabelDeclClass,
  };

private:
  DeclClass dClass;
  std::string Name;

protected:
  PtxDecl(DeclClass DC, StringRef Name) : dClass(DC), Name(Name.str()) {}

public:
  virtual ~PtxDecl();
  DeclClass getDeclClass() const { return dClass; }
  StringRef getDeclName() const { return Name; }

  void *operator new(size_t bytes) noexcept {
    llvm_unreachable("PtxDecl cannot be allocated with regular 'new'.");
  }

  void operator delete(void *data) noexcept {
    llvm_unreachable("PtxDecl cannot be released with regular 'delete'.");
  }
};

class PtxVariableDecl : public PtxDecl {
public:
  struct Attribute {
    std::optional<unsigned> Align;
    unsigned StorageClass;
  };

private:
  const PtxType *Type;
  Attribute Attr;

public:
  PtxVariableDecl(StringRef Name, const PtxType *Type)
      : PtxDecl(VariableDeclClass, Name), Type(Type) {}

  const Attribute &getAttributes() const { return Attr; }
  const PtxType *getType() const { return Type; }

  void setAlign(unsigned Align) {
    Attr.Align = Align;
  }

  bool hasAlign() const {
    return Attr.Align.has_value();
  }

  static bool classof(const PtxDecl *T) {
    return T->getDeclClass() == VariableDeclClass;
  }
};

class PtxStmt {
public:
  enum StmtClass {
    NoStmtClass = 0,
    DeclStmtClass,
    CompoundStmtClass,
    InstructionClass,
    GuardInstructionClass,
    UnaryOperatorClass,
    BinaryOperatorClass,
    ConditionalOperatorClass,
    TupleExprClass,
    SinkExprClass,
    CastExprClass,
    DeclRefExprClass,
    IntegerLiteralClass,
    FloatingLiteralClass,
  };

  PtxStmt(const PtxStmt &) = delete;
  PtxStmt &operator=(const PtxStmt &) = delete;
  virtual ~PtxStmt();

  void *operator new(size_t bytes) noexcept {
    llvm_unreachable("Stmts cannot be allocated with regular 'new'.");
  }

  void operator delete(void *data) noexcept {
    llvm_unreachable("Stmts cannot be released with regular 'delete'.");
  }

  StmtClass getStmtClass() const { return static_cast<StmtClass>(sClass); }

private:
  StmtClass sClass;

protected:
  PtxStmt(StmtClass SC) : sClass(SC) {}
};

class PtxCompoundStmt : public PtxStmt {
  SmallVector<PtxStmt *> Stmts;

public:
  PtxCompoundStmt(SmallVector<PtxStmt *> Stmts)
      : PtxStmt(CompoundStmtClass), Stmts(Stmts) {}

  ArrayRef<PtxStmt *> getStmts() const { return Stmts; }

  static bool classof(const PtxStmt *S) {
    return S->getStmtClass() == CompoundStmtClass;
  }
};

class PtxInstruction : public PtxStmt {
public:
  struct Attribute {
    unsigned RoundMod = 0;
    unsigned SatMod = 0;
    unsigned SatfMod = 0;
    unsigned FtzMod = 0;
    unsigned AbsMod = 0;
    unsigned TypeMod = 0;
    unsigned AppxMod = 0;
    unsigned ClampMod = 0;
    unsigned SyncMod = 0;
    unsigned ShuffleMod = 0;
    unsigned Vector = 0;
    unsigned ComparisonOp = 0;
    unsigned BooleanOp = 0;
    unsigned ArithmeticOp = 0;
    unsigned StorageClass = 0;
    SmallVector<PtxType *, 4> Types;

    void setComparisonOp(ptx::ComparisonOp Op) {
      ComparisonOp = Op;
    }

    void setBooleanOp(ptx::BooleanOp Op) {
      BooleanOp = Op;
    }

    void setArithmeticOp(ptx::ArithmeticOp Op) {
      ArithmeticOp = Op;
    }

    void setStorageClass(ptx::StorageClass SC) {
      StorageClass = SC;
    }
  };

  using OperandList = SmallVector<const PtxExpr *, 4>;
private:
  ptx::InstKind Op;
  Attribute Attr;
  OperandList Operands;
  const PtxExpr *PredOutput;
public:
  PtxInstruction(ptx::InstKind Op, const OperandList &Operands = {}, const PtxExpr *PredOut = nullptr)
      : PtxStmt(InstructionClass), Op(Op), Operands(Operands), PredOutput(PredOut) {}

  ptx::InstKind getOpcode() const { return Op; }

  const SmallVector<const PtxExpr *, 4> &getOperands() const {
    return Operands;
  }

  const PtxExpr *getOperand(unsigned I) const { return Operands[I]; }

  void addOperand(const PtxExpr *Operand) { Operands.push_back(Operand); }

  const PtxExpr *getPredOutput() const {
    return PredOutput;
  }

  static bool classof(const PtxStmt *S) {
    return InstructionClass <= S->getStmtClass();
  }
};

class PtxGuardInstruction : public PtxStmt {
  bool IsNeg;
  const PtxExpr *Pred;
  const PtxInstruction *Instruction;
public:
  PtxGuardInstruction(bool IsNeg, const PtxExpr *Pred,
                      const PtxInstruction *Inst)
      : PtxStmt(GuardInstructionClass), IsNeg(IsNeg), Pred(Pred),
        Instruction(Inst) {}

  const PtxExpr *getPred() const { return Pred; }
  const PtxInstruction *getInstruction() const { return Instruction; }
  bool isNeg() const { return IsNeg; }

  static bool classof(const PtxStmt *S) {
    return S->getStmtClass() == GuardInstructionClass;
  }
};

class PtxDeclStmt : public PtxStmt {
  SmallVector<const PtxDecl *> DeclGroup;
public:
  PtxDeclStmt(const SmallVector<const PtxDecl *> &Decls) : PtxStmt(DeclStmtClass), DeclGroup(Decls) {}

  static bool classof(const PtxStmt *S) {
    return S->getStmtClass() == DeclStmtClass;
  }
};

/// Base class for the full range of assembler expressions which are
/// needed for parsing.
class PtxExpr : public PtxStmt {
  const PtxType *Type;

protected:
  explicit PtxExpr(StmtClass SC, const PtxType *Type)
      : PtxStmt(SC), Type(Type) {}

public:

  const PtxType *getType() const { return Type; }

  static bool classof(const PtxStmt *S) {
    return UnaryOperatorClass <= S->getStmtClass() &&
           S->getStmtClass() <= FloatingLiteralClass;
  }

  void print(raw_ostream &OS, bool InParens = false) const;
  void dump() const;
};

inline raw_ostream &operator<<(raw_ostream &OS, const PtxExpr &E) {
  E.print(OS);
  return OS;
}

class PtxIntegerLiteral : public PtxExpr {
  llvm::APInt Value;

public:
  PtxIntegerLiteral(const PtxType *Type, llvm::APInt Value)
      : PtxExpr(IntegerLiteralClass, Type), Value(Value) {}

  llvm::APInt getValue() const { return Value; }

  static bool classof(const PtxStmt *S) {
    return S->getStmtClass() == IntegerLiteralClass;
  }
};

class PtxFloatingLiteral : public PtxExpr {
  llvm::APFloat Value;

public:
  PtxFloatingLiteral(const PtxType *Type, llvm::APFloat Value)
      : PtxExpr(FloatingLiteralClass, Type), Value(Value) {}

  llvm::APFloat getValue() const { return Value; }

  static bool classof(const PtxStmt *S) {
    return S->getStmtClass() == FloatingLiteralClass;
  }
};

class PtxDeclRefExpr : public PtxExpr {
  const PtxDecl *Decl;

public:
  PtxDeclRefExpr(const PtxVariableDecl *D)
      : PtxExpr(DeclRefExprClass, D->getType()), Decl(D) {}

  const PtxDecl &getSymbol() const { return *Decl; }

  static bool classof(const PtxStmt *S) {
    return S->getStmtClass() == DeclRefExprClass;
  }
};

class PtxTupleExpr : public PtxExpr {
public:
  typedef SmallVector<const PtxExpr *, 4> ElementList;
private:
  ElementList Elements;

public:
  PtxTupleExpr(const PtxTupleType *Type,
               const SmallVector<const PtxExpr *, 4> &Elements)
      : PtxExpr(TupleExprClass, Type), Elements(Elements) {}

  const SmallVector<const PtxExpr *, 4> &getElements() const {
    return Elements;
  }

  const PtxExpr *getElement(unsigned I) const { return Elements[I]; }

  static bool classof(const PtxStmt *S) {
    return S->getStmtClass() == TupleExprClass;
  }
};

class PtxSinkExpr : public PtxExpr {
public:
  PtxSinkExpr(const PtxAnyType *Any) : PtxExpr(SinkExprClass, Any) {}

  static bool classof(const PtxStmt *S) {
    return S->getStmtClass() == SinkExprClass;
  }
};

class PtxCastExpr : public PtxExpr {
  const PtxExpr *SubExpr;

public:
  PtxCastExpr(const PtxFundamentalType *Type, const PtxExpr *Op)
      : PtxExpr(CastExprClass, Type), SubExpr(Op) {}

  const PtxExpr *getSubExpr() const { return SubExpr; }

  static bool classof(const PtxStmt *S) {
    return S->getStmtClass() == CastExprClass;
  }
};

class PtxUnaryOperator : public PtxExpr {
public:
  enum Opcode {
    // Unary arithmetic
    Plus,  // +
    Minus, // -
    Not,   // ~
    LNot   // !
  };

private:
  const PtxExpr *SubExpr;
  unsigned Op;
public:
  PtxUnaryOperator(Opcode Op, const PtxExpr *Expr, const PtxType *Type)
      : PtxExpr(UnaryOperatorClass, Type), SubExpr(Expr), Op(Op) {}

public:
  Opcode getOpcode() const { return (Opcode)Op; }

  const PtxExpr *getSubExpr() const { return SubExpr; }

  static bool classof(const PtxStmt *S) {
    return S->getStmtClass() == UnaryOperatorClass;
  }
};

class PtxBinaryOperator : public PtxExpr {
public:
  enum Opcode {
    // Multiplicative operators.
    Mul, // *
    Div, // /
    Rem, // %

    // Additive operators.
    Add, // +
    Sub, // -

    // Bitwise shift operators.
    Shl, // <<
    Shr, // >>

    // Relational operators.
    LT, // <
    GT, // >
    LE, // <=
    GE, // >=

    // Equality operators.
    EQ, // ==
    NE, // !=

    // Bitwise AND operator.
    And, // &

    // Bitwise XOR operator.
    Xor, // ^

    // Bitwise OR operator.
    Or, // |

    // Logical AND operator.
    LAnd, // &&

    // Logical OR operator.
    LOr, // ||
  };

private:
  Opcode Op;
  const PtxExpr *LHS;
  const PtxExpr *RHS;
public:
  PtxBinaryOperator(Opcode Op, const PtxExpr *LHS, const PtxExpr *RHS,
                    const PtxType *Type)
      : PtxExpr(BinaryOperatorClass, Type), Op(Op), LHS(LHS), RHS(RHS) {}

public:
  Opcode getOpcode() const { return (Opcode)Op; }
  const PtxExpr *getLHS() const { return LHS; }
  const PtxExpr *getRHS() const { return RHS; }

  static bool classof(const PtxStmt *S) {
    return S->getStmtClass() == BinaryOperatorClass;
  }
};

class PtxConditionalOperator : public PtxExpr {
  const PtxExpr *Cond;
  const PtxExpr *LHS;
  const PtxExpr *RHS;

public:
  PtxConditionalOperator(const PtxExpr *C, const PtxExpr *L, const PtxExpr *R,
                         const PtxType *Type)
      : PtxExpr(ConditionalOperatorClass, Type), Cond(C), LHS(L), RHS(R) {}

  const PtxExpr *getCond() const { return Cond; }
  const PtxExpr *getLHS() const { return LHS; }
  const PtxExpr *getRHS() const { return RHS; }

  static bool classof(PtxStmt *S) {
    return S->getStmtClass() == ConditionalOperatorClass;
  }
};

class PtxContext {
  llvm::BumpPtrAllocator Allocator;
  llvm::DenseMap<int, PtxFundamentalType *>
      FundamentalTypes;
  PtxAnyType *AnyType;
public:
  void *allocate(unsigned Size, unsigned Align = 8) {
    return Allocator.Allocate(Size, Align);
  }

  void deallocate(void *Ptr) {}

  PtxType *GetTypeFromConstraint(StringRef Constraint);
  PtxFundamentalType *GetOrCreateFundamentalType(StringRef TypeName);
  PtxFundamentalType *GetOrCreateFundamentalType(PtxFundamentalType::TypeKind Kind);
  PtxAnyType *GetOrCreateAnyType();
  PtxTupleType *CreateTupleType(const PtxTupleType::ElementList &ElementType);
  PtxVectorType *CreateVectorType(PtxVectorType::TypeKind Kind, const PtxFundamentalType *Base);

  PtxDeclStmt *CreateDeclStmt(const SmallVector<const PtxDecl *> &DeclGroup);
  PtxVariableDecl *CreateVariableDecl(StringRef Name, const PtxType *Type);
  PtxCompoundStmt *CreateCompoundStmt(const SmallVector<PtxStmt *> &Stmts);
  PtxDeclRefExpr *CreateDeclRefExpr(const PtxVariableDecl *Var);
  PtxTupleExpr *CreateTupleExpr(const PtxTupleType *Type, const PtxTupleExpr::ElementList &Elements);
  PtxSinkExpr *CreateSinkExpr();
  PtxInstruction *CreateInstruction(ptx::InstKind Op, const PtxInstruction::OperandList &Operands = {}, const PtxExpr *PredOut = nullptr);
  PtxGuardInstruction *CreateGuardInstruction(bool isNeg, const PtxExpr *Pred, const PtxInstruction *Inst);

  PtxUnaryOperator *CreateUnaryOperator(PtxUnaryOperator::Opcode Op, const PtxExpr *Operand);
  PtxBinaryOperator *CreateBinaryOperator(PtxBinaryOperator::Opcode Op, const PtxExpr *LHS, const PtxExpr *RHS);
  PtxConditionalOperator *CreateConditionalOperator(const PtxExpr *Cond, const PtxExpr *LHS, const PtxExpr *RHS);
  PtxCastExpr *CreateCastExpression(const PtxFundamentalType *CastType, const PtxExpr *SubExpr);

  PtxIntegerLiteral *CreateIntegerLiteral(const PtxType *Type, llvm::APInt Val);
  PtxFloatingLiteral *CreateFloatLiteral(const PtxType *Type, llvm::APFloat Val);

};

class PtxScope {
  using DeclSetTy = llvm::SmallPtrSet<PtxVariableDecl *, 32>;
  PtxScope *AnyParent;
  DeclSetTy DeclsInScope;
  unsigned Depth;

public:
  PtxScope(PtxScope *Parent)
      : AnyParent(Parent), Depth(Parent ? Parent->Depth + 1 : 0) {}

  const PtxScope *getParent() const { return AnyParent; }
  PtxScope *getParent() { return AnyParent; }
  unsigned getDepth() const { return Depth; }

  using decl_range = llvm::iterator_range<DeclSetTy::iterator>;

  decl_range decls() const {
    return decl_range(DeclsInScope.begin(), DeclsInScope.end());
  }

  void AddDecl(PtxVariableDecl *D) { DeclsInScope.insert(D); }

  bool isDeclScope(const PtxVariableDecl *D) const { return DeclsInScope.contains(D); }

  bool Contains(const PtxScope &rhs) const { return Depth < rhs.Depth; }

  PtxVariableDecl *LookupSymbol(StringRef Symbol) const;
};

using PtxTypeResult = clang::ActionResult<PtxType *>;
using PtxDeclResult = clang::ActionResult<PtxDecl *>;
using PtxStmtResult = clang::ActionResult<PtxStmt *>;
using PtxExprResult = clang::ActionResult<PtxExpr *>;

class PtxParser {
public:
  struct PendingError {
    SMLoc Loc;
    std::string Msg;
    SMRange Range;
  };

private:
  PtxLexer Lexer;
  PtxContext &Context;
  SourceMgr &SrcMgr;
  PtxScope *CurScope;

  class ParseScope {
    PtxParser *Self;
    ParseScope(const ParseScope &) = delete;
    void operator=(const ParseScope &) = delete;

  public:
    ParseScope(PtxParser *Self) : Self(Self) { Self->EnterScope(); }

    ~ParseScope() {
      Self->ExitScope();
      Self = nullptr;
    }
  };

public:
  PtxParser(PtxContext &Ctx, SourceMgr &Mgr)
      : Lexer(), Context(Ctx), SrcMgr(Mgr), CurScope(nullptr) {
    unsigned MainFileID = Mgr.getMainFileID();
    StringRef BufferRef = Mgr.getMemoryBuffer(MainFileID)->getBuffer();
    Lexer.setBuffer(BufferRef);
    Lex();
    EnterScope();
  }
  ~PtxParser();

  SourceMgr &getSourceManager() { return SrcMgr; }
  PtxLexer &getLexer() { return Lexer; }
  PtxContext &getContext() { return Context; }

  const PtxLexer &getLexer() const {
    return const_cast<PtxParser *>(this)->getLexer();
  }

  const AsmToken &getTok() const { return getLexer().getTok(); }
  const AsmToken &Lex();

  PtxDeclResult AddBuiltinSymbol(StringRef Name, const PtxType *Type);
  PtxDeclResult AddInlineAsmOperands(StringRef Name, StringRef Constraint);

  PtxScope *getCurScope() const { return CurScope; }

  void EnterScope() { CurScope = new PtxScope(getCurScope()); }

  void ExitScope() {
    assert(getCurScope());
    PtxScope *OldScope = getCurScope();
    if (OldScope) {
      CurScope = OldScope->getParent();
      delete OldScope;
    } else {
      CurScope = nullptr;
    }
  }

  PtxStmtResult ParseStatement();
  PtxStmtResult ParseCompoundStatement();
  PtxStmtResult ParseGuardInstruction();
  PtxStmtResult ParseInstruction();
  bool ParseInstructionFlags(PtxInstruction::Attribute &Attr);

  PtxExprResult ParseTuple();
  PtxExprResult ParsePredOutput();
  PtxExprResult ParseInstructionDestOperand();
  PtxExprResult ParseInstructionSrcOperand();
  PtxExprResult ParseInstructionPrimaryOperand();
  PtxExprResult ParseInstructionUnaryOperand();
  PtxExprResult ParseInstructionPostfixOperand();

  PtxStmtResult ParseDeclStmt();
  PtxTypeResult ParseVarDeclspec(PtxVariableDecl::Attribute Attr);
  PtxDeclResult ParseVariableDecl(const PtxType *Type);

  /// Parse constant expression

  PtxExprResult ParseConstantExpression();
  PtxExprResult ParsePrimaryExpression();
  PtxExprResult ParseConditionalExpression();
  PtxExprResult ParseLogicOrExpression();
  PtxExprResult ParseLogicAndExpression();
  PtxExprResult ParseInclusiveOrExpression();
  PtxExprResult ParseExclusiveOrExpression();
  PtxExprResult ParseAndExpression();
  PtxExprResult ParseEqualityExpression();
  PtxExprResult ParseRelationExpression();
  PtxExprResult ParseShiftExpression();
  PtxExprResult ParseAdditiveExpression();
  PtxExprResult ParseMultiplicativeExpression();
  PtxExprResult ParseCastExpresion();
  PtxExprResult ParseUnaryExpression();
};

} // namespace clang::dpct

namespace llvm {

template <typename T> struct PointerLikeTypeTraits;
template <> struct PointerLikeTypeTraits<::clang::dpct::PtxType *> {
  static inline void *getAsVoidPointer(::clang::Type *P) { return P; }

  static inline ::clang::dpct::PtxType *getFromVoidPointer(void *P) {
    return static_cast<::clang::dpct::PtxType *>(P);
  }

  static constexpr int NumLowBitsAvailable = clang::TypeAlignmentInBits;
};

template <> struct PointerLikeTypeTraits<::clang::dpct::PtxDecl *> {
  static inline void *getAsVoidPointer(::clang::ExtQuals *P) { return P; }

  static inline ::clang::dpct::PtxDecl *getFromVoidPointer(void *P) {
    return static_cast<::clang::dpct::PtxDecl *>(P);
  }

  static constexpr int NumLowBitsAvailable = clang::TypeAlignmentInBits;
};

template <> struct PointerLikeTypeTraits<::clang::dpct::PtxStmt *> {
  static inline void *getAsVoidPointer(::clang::ExtQuals *P) { return P; }

  static inline ::clang::dpct::PtxStmt *getFromVoidPointer(void *P) {
    return static_cast<::clang::dpct::PtxStmt *>(P);
  }

  static constexpr int NumLowBitsAvailable = clang::TypeAlignmentInBits;
};

} // namespace llvm

inline void *operator new(size_t Bytes, ::clang::dpct::PtxContext &C,
                          size_t Alignment = 8) noexcept {
  return C.allocate(Bytes, Alignment);
}

inline void operator delete(void *Ptr, ::clang::dpct::PtxContext &C,
                            size_t) noexcept {
  C.deallocate(Ptr);
}

inline void *operator new[](size_t Bytes, ::clang::dpct::PtxContext &C,
                            size_t Alignment = 8) noexcept {
  return C.allocate(Bytes, Alignment);
}

inline void operator delete[](void *Ptr,
                              ::clang::dpct::PtxContext &C) noexcept {
  C.deallocate(Ptr);
}

#endif // CLANG_DPCT_INLINE_ASM_PARSER_H
