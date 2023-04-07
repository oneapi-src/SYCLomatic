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
    TK_Pred
  };

private:
  TypeKind Kind;
  size_t Size;

public:
  PtxFundamentalType(TypeKind Kind, size_t Size)
      : PtxType(FundamentalClass), Kind(Kind), Size(Size) {}

  TypeKind getKind() const { return Kind; }
  size_t getSize() const { return Size; }

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
  SmallVector<const PtxType *, 4> ElementTypes;

public:
  PtxTupleType(const SmallVector<const PtxType *, 4> &ElementTypes)
      : PtxType(TupleClass), ElementTypes(ElementTypes) {}

  const SmallVector<const PtxType *, 4> &getElementTypes() const {
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
    uint64_t Align;
    bool IsShared;
  };

private:
  const PtxType *Type;
  Attribute Attr;

public:
  PtxVariableDecl(StringRef Name) : PtxDecl(VariableDeclClass, Name) {}

  const Attribute &getAttributes() const { return Attr; }
  const PtxType *getType() const { return Type; }

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
    unsigned Opcode;
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
    SmallVector<PtxType *, 4> Types;
  };

public:
  ptx::InstKind Op;
  Attribute Attr;
  SmallVector<const PtxExpr *, 4> Operands;

protected:
  PtxInstruction(StmtClass SC, ptx::InstKind Op,
                 const SmallVector<const PtxExpr *, 4> &Operands)
      : PtxStmt(SC), Op(Op), Operands(Operands) {}

public:
  PtxInstruction(ptx::InstKind Op,
                 const SmallVector<const PtxExpr *, 4> &Operands = {})
      : PtxStmt(InstructionClass), Op(Op), Operands(Operands) {}

  ptx::InstKind getOpcode() const { return Op; }

  const SmallVector<const PtxExpr *, 4> &getOperands() const {
    return Operands;
  }

  const PtxExpr *getOperand(unsigned I) const { return Operands[I]; }

  void addOperand(const PtxExpr *Operand) { Operands.push_back(Operand); }

  static bool classof(const PtxStmt *S) {
    return InstructionClass <= S->getStmtClass() &&
           S->getStmtClass() <= GuardInstructionClass;
  }
};

class PtxGuardInstruction : public PtxInstruction {
  const PtxExpr *Pred;
  bool IsNeg;

public:
  PtxGuardInstruction(const PtxExpr *Pred, bool IsNeg, ptx::InstKind Op,
                      const SmallVector<const PtxExpr *, 4> &Operands)
      : PtxInstruction(GuardInstructionClass, Op, Operands), Pred(Pred),
        IsNeg(IsNeg) {}

  const PtxExpr *getPred() const { return Pred; }
  bool isNeg() const { return IsNeg; }

  static bool classof(const PtxStmt *S) {
    return S->getStmtClass() == GuardInstructionClass;
  }
};

class PtxDeclStmt : public PtxStmt {
  SmallVector<PtxDecl *> DeclGroup;

public:
  PtxDeclStmt(SmallVector<AsmStmt *> Stmts) : PtxStmt(DeclStmtClass) {}

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
  SmallVector<const PtxExpr *, 4> Elements;

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
  llvm::DenseMap<PtxFundamentalType::TypeKind, const PtxFundamentalType *>
      FundamentalTypes;

public:
  void *allocate(unsigned Size, unsigned Align = 8) {
    return Allocator.Allocate(Size, Align);
  }

  void deallocate(void *Ptr) {}
};

class PtxScope {
  using DeclSetTy = llvm::SmallPtrSet<PtxDecl *, 32>;
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

  void AddDecl(PtxDecl *D) { DeclsInScope.insert(D); }

  bool isDeclScope(const PtxDecl *D) const { return DeclsInScope.contains(D); }

  bool Contains(const PtxScope &rhs) const { return Depth < rhs.Depth; }

  PtxDecl *LookupSymbol(StringRef Symbol) const;
};

using PtxStmtResult = clang::ActionResult<PtxStmt *>;
using PtxExprResult = clang::ActionResult<PtxExpr *>;

// struct AsmType;

// struct AsmSymbol {
//   std::string Name;
//   AsmType *Type;
//   bool IsVariable;
// };

// struct VarAttr {
//   uint64_t Align;
//   bool IsShared;
// };

// struct InstAttr {
//   unsigned Opcode;
//   unsigned RoundMod;
//   unsigned SatMod;
//   unsigned SatfMod;
//   unsigned FtzMod;
//   unsigned AbsMod;
//   unsigned TypeMod;
//   unsigned AppxMod;
//   unsigned ClampMod;
//   unsigned SyncMod;
//   unsigned ShuffleMod;
//   unsigned Vector;
//   unsigned Comparison;
//   unsigned BooleanOp;
//   unsigned ArithmeticOp;
//   unsigned StorageClass;
//   SmallVector<AsmType *, 4> Types;
// };

// struct AsmStatement {
//   enum StmtKind {
//     SK_Add,
//     SK_Sub,
//     SK_Mul,
//     SK_Div,
//     SK_Mod,
//     SK_BitAnd,
//     SK_BitOr,
//     SK_BitXor,
//     SK_BitNot,
//     SK_Shl,
//     SK_Shr,
//     SK_EQ,
//     SK_NE,
//     SK_LT,
//     SK_GT,
//     SK_LE,
//     SK_GE,
//     SK_Not,
//     SK_And,
//     SK_Or,
//     SK_Neg,
//     SK_Assign,
//     SK_Cond,
//     SK_Addr,
//     SK_Deref,
//     SK_Block,
//     SK_Label,
//     SK_ExprStmt,
//     SK_StmtExpr,
//     SK_Variable,
//     SK_VLAPtr,
//     SK_Integer,
//     SK_Unsigned,
//     SK_Float,
//     SK_Double,
//     SK_Cast,
//     SK_Inst,
//     SK_Sink,
//     SK_Tuple,
//   };

//   StmtKind Kind;
//   AsmStatement *Next;
//   AsmType *Type;
//   AsmStatement *LHS;
//   AsmStatement *RHS;
//   AsmStatement *SubExpr;
//   AsmStatement *Pred;
//   AsmStatement *PredOutput;
//   AsmStatement *Cond;
//   AsmStatement *Then;
//   AsmStatement *Else;
//   AsmStatement *Init;
//   AsmStatement *Body;
//   StringRef Label;
//   AsmStatement *Bar;
//   AsmSymbol *Variable;
//   SmallPtrSet<AsmSymbol *, 32> DeclsInScope;

//   union {
//     uint64_t u64;
//     int64_t i64;
//     float f32;
//     double f64;
//   };

//   InstAttr InstructionAttr;
//   SmallVector<AsmStatement *, 4> Operands;
//   SmallVector<AsmStatement *, 4> Tuple;
//   SmallVector<AsmStatement *, 4> Block;

//   AsmStatement(StmtKind K) : Kind(K) {}
// };

// struct AsmType {
//   enum TypeKind {
//     TK_B8,
//     TK_B16,
//     TK_B32,
//     TK_B64,
//     TK_B128,
//     TK_S2,
//     TK_S4,
//     TK_S8,
//     TK_S16,
//     TK_S32,
//     TK_S64,
//     TK_U2,
//     TK_U4,
//     TK_U8,
//     TK_U16,
//     TK_U32,
//     TK_U64,
//     TK_F16,
//     TK_F16x2,
//     TK_F32,
//     TK_F64,
//     TK_E4m3,
//     TK_E5m2,
//     TK_E4m3x2,
//     TK_E5m2x2,
//     TK_Byte,
//     TK_4Byte,
//     TK_Pred,
//     TK_V2,
//     TK_V4,
//     TK_Ptr,
//     TK_Array,
//     TK_VLA
//   };

//   TypeKind Kind;
//   int Size;
//   int Align;
//   AsmType *Origin;
//   AsmType *Base;
//   AsmToken Name;
//   size_t ArrayLength;
//   bool IsFlexible;
// };

// using AsmStmtResult = ActionResult<dpct::AsmStatement *>;

// class AsmContext {
//   llvm::BumpPtrAllocator Allocator;
//   std::map<AsmType::TypeKind, AsmType *> ScalarTypes;
//   AsmStatement *SinkExpression;

// public:
//   void *allocate(unsigned Size, unsigned Align = 8) {
//     return Allocator.Allocate(Size, Align);
//   }

//   void deallocate(void *Ptr) {}

//   AsmType *getScalarType(AsmType::TypeKind Kind);
//   AsmType *getScalarTypeFromName(StringRef TypeName);
//   AsmType::TypeKind getScalarTypeKindFromName(StringRef TypeName);

//   AsmType *PointTo(AsmType *Base);
//   AsmType *ArrayOf(AsmType *Base, size_t Len);
//   AsmType *VLAOf(AsmType *Base, AsmStatement *Expr);

//   AsmSymbol *CreateSymbol(const std::string &Name, AsmType *Type,
//                           bool IsVar = true);
//   AsmStatement *CreateStmt(AsmStatement::StmtKind Kind);
//   AsmStatement *CreateIntegerConstant(AsmType *Type, int64_t Val);
//   AsmStatement *CreateIntegerConstant(AsmType *Type, uint64_t Val);
//   AsmStatement *CreateFloatConstant(AsmType *Type, float Val);
//   AsmStatement *CreateFloatConstant(AsmType *Type, double Val);
//   AsmStatement *CreateConditionalExpression(AsmStatement *Cond,
//                                             AsmStatement *Then,
//                                             AsmStatement *Else);
//   AsmStatement *CreateBinaryOperator(AsmStatement::StmtKind Opcode,
//                                      AsmStatement *LHS, AsmStatement *RHS);
//   AsmStatement *CreateUnaryExpression(AsmStatement::StmtKind Opcode,
//                                       AsmStatement *SubExpr);
//   AsmStatement *CreateCastExpression(AsmType *Type, AsmStatement *SubExpr);
//   AsmStatement *CreateVariableRefExpression(AsmSymbol *Symbol);
//   AsmStatement *GetOrCreateSinkExpression();
// };

// class AsmIdentifierInfo {
//   friend class AsmIdentifierTable;
//   unsigned TokenID;
//   llvm::StringMapEntry<AsmIdentifierInfo *> *Entry = nullptr;

//   AsmIdentifierInfo() : TokenID(AsmToken::Identifier) {}

// public:
//   AsmIdentifierInfo(const AsmIdentifierInfo &) = delete;
//   AsmIdentifierInfo &operator=(const AsmIdentifierInfo &) = delete;
//   AsmIdentifierInfo(AsmIdentifierInfo &&) = delete;
//   AsmIdentifierInfo &operator=(AsmIdentifierInfo &&) = delete;

//   const char *getNameStart() const { return Entry->getKeyData(); }
//   unsigned getLength() const { return Entry->getKeyLength(); }
//   StringRef getName() const { return StringRef(getNameStart(), getLength()); }

//   template <std::size_t StrLen> bool isStr(const char (&Str)[StrLen]) const {
//     return getLength() == StrLen - 1 &&
//            memcmp(getNameStart(), Str, StrLen - 1) == 0;
//   }

//   bool isStr(llvm::StringRef Str) const {
//     llvm::StringRef ThisStr(getNameStart(), getLength());
//     return ThisStr == Str;
//   }

//   AsmToken::TokenKind getTokenID() const {
//     return (AsmToken::TokenKind)TokenID;
//   }
// };

// class AsmIdentifierTable {
//   using HashTableTy =
//       llvm::StringMap<AsmIdentifierInfo *, llvm::BumpPtrAllocator>;
//   HashTableTy HashTable;

// public:
//   AsmIdentifierTable() = default;

//   llvm::BumpPtrAllocator &getAllocator() { return HashTable.getAllocator(); }

//   AsmIdentifierInfo &get(StringRef Name) {
//     auto &Entry = *HashTable.try_emplace(Name, nullptr).first;

//     AsmIdentifierInfo *&II = Entry.second;
//     if (II)
//       return *II;

//     // Lookups failed, make a new IdentifierInfo.
//     void *Mem = getAllocator().Allocate<AsmIdentifierInfo>();
//     II = new (Mem) AsmIdentifierInfo();

//     // Make sure getName() knows how to find the IdentifierInfo
//     // contents.
//     II->Entry = &Entry;

//     return *II;
//   }

//   AsmIdentifierInfo &get(StringRef Name, AsmToken::TokenKind TokenCode) {
//     AsmIdentifierInfo &II = get(Name);
//     II.TokenID = TokenCode;
//     assert(II.TokenID == (unsigned)TokenCode && "TokenCode too large");
//     return II;
//   }

//   using iterator = HashTableTy::const_iterator;
//   using const_iterator = HashTableTy::const_iterator;

//   iterator begin() const { return HashTable.begin(); }
//   iterator end() const { return HashTable.end(); }
//   unsigned size() const { return HashTable.size(); }
// };

// class AsmScope {
//   using DeclSetTy = llvm::SmallPtrSet<AsmSymbol *, 32>;
//   AsmScope *AnyParent;
//   DeclSetTy DeclsInScope;
//   unsigned Depth;

// public:
//   AsmScope(AsmScope *Parent)
//       : AnyParent(Parent), Depth(Parent ? Parent->Depth + 1 : 0) {}

//   const AsmScope *getParent() const { return AnyParent; }
//   AsmScope *getParent() { return AnyParent; }
//   unsigned getDepth() const { return Depth; }

//   using decl_range = llvm::iterator_range<DeclSetTy::iterator>;

//   decl_range decls() const {
//     return decl_range(DeclsInScope.begin(), DeclsInScope.end());
//   }

//   void AddDecl(AsmSymbol *D) { DeclsInScope.insert(D); }

//   bool isDeclScope(const AsmSymbol *D) const {
//     return DeclsInScope.contains(D);
//   }

//   bool Contains(const AsmScope &rhs) const { return Depth < rhs.Depth; }

//   AsmSymbol *LookupSymbol(StringRef Symbol) const;
// };

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

  void AddBuiltinSymbol(const std::string &Name, PtxType *Type);

  PtxScope *getCurScope() const { return CurScope; }

  void EnterScope() { CurScope = new AsmScope(getCurScope()); }

  void ExitScope() {
    assert(getCurScope());
    AsmScope *OldScope = getCurScope();
    if (OldScope) {
      CurScope = OldScope->getParent();
      delete OldScope;
    } else {
      CurScope = nullptr;
    }
  }

  /// TODO: bool Parse();
  PtxStmtResult ParseStatement();
  PtxStmtResult ParseCompoundStatement();
  PtxStmtResult ParseUnGuardInstruction();
  PtxStmtResult ParseInstruction();
  bool ParseInstructionFlags(PtxInstruction::Attribute &Attr);

  PtxExprResult ParseTuple();
  AsmStmtResult ParseInstructionFirstOperand();
  AsmStmtResult ParseInstructionOperand();
  AsmStmtResult ParseInstructionPrimaryOperand();
  AsmStmtResult ParseInstructionUnaryOperand();
  AsmStmtResult ParseInstructionPostfixOperand();

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
  AsmStmtResult ParseCastExpresion();
  AsmStmtResult ParseUnaryExpression();
  // bool ParsePostfixExpression();
};

} // namespace clang::dpct

// inline void *operator new(size_t Bytes, clang::dpct::AsmContext &C,
//                           size_t Alignment = 8) noexcept {
//   return C.allocate(Bytes, Alignment);
// }

// inline void operator delete(void *Ptr, clang::dpct::AsmContext &C,
//                             size_t) noexcept {
//   C.deallocate(Ptr);
// }

// inline void *operator new[](size_t Bytes, clang::dpct::AsmContext &C,
//                             size_t Alignment = 8) noexcept {
//   return C.allocate(Bytes, Alignment);
// }

// inline void operator delete[](void *Ptr, clang::dpct::AsmContext &C) noexcept
// {
//   C.deallocate(Ptr);
// }

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
