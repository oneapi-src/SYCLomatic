//===---------------------------- AsmParser.h -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DPCT_INLINE_ASM_PARSER_H
#define CLANG_DPCT_INLINE_ASM_PARSER_H

#include "Asm/AsmIdentifierTable.h"
#include "Asm/AsmToken.h"
#include "Asm/AsmTokenKinds.h"
#include "AsmLexer.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LLVM.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Ownership.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/PointerLikeTypeTraits.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <optional>

namespace clang::dpct {

using llvm::BumpPtrAllocator;
using llvm::DenseMap;
using llvm::SmallPtrSet;
using llvm::SMLoc;
using llvm::SourceMgr;

class DpctAsmType;
class DpctAsmDecl;
class DpctAsmExpr;
class DpctAsmStmt;
class DpctAsmParser;
class DpctAsmIntegerLiteral;

class DpctAsmType {
public:
  enum TypeClass {
    BuiltinClass,
    TupleClass,
    ConstantArrayClass,
    IncompleteArrayClass,
    VectorClass,
    DiscardClass
  };

private:
  TypeClass tClass;

protected:
  DpctAsmType(TypeClass TC) : tClass(TC) {}

public:
  virtual ~DpctAsmType();
  TypeClass getTypeClass() const { return tClass; }

  void *operator new(size_t bytes) noexcept {
    llvm_unreachable("PtxDecl cannot be allocated with regular 'new'.");
  }

  void operator delete(void *data) noexcept {
    llvm_unreachable("PtxDecl cannot be released with regular 'delete'.");
  }
};

class DpctAsmBuiltinType : public DpctAsmType {
public:
  enum TypeKind : uint8_t {
#define BUILTIN_TYPE(X, Y) TK_##X,
#include "AsmTokenKinds.def"
  };

private:
  TypeKind Kind;

public:
  DpctAsmBuiltinType(TypeKind Kind) : DpctAsmType(BuiltinClass), Kind(Kind) {}

  TypeKind getKind() const { return Kind; }

  bool isSignedInt() const {
    return getKind() == TK_s8 || getKind() == TK_s16 || getKind() == TK_s32 ||
           getKind() == TK_s64;
  }

  bool isUnsignedInt() const {
    return getKind() == TK_u8 || getKind() == TK_u16 || getKind() == TK_u32 ||
           getKind() == TK_u64;
  }

  bool isFloating() const {
    return getKind() == TK_f16 || getKind() == TK_f32 || getKind() == TK_f64;
  }

  bool isBitSize() const {
    return getKind() == TK_b8 || getKind() == TK_b16 || getKind() == TK_b32 ||
           getKind() == TK_b64;
  }

  static bool classof(const DpctAsmType *T) {
    return T->getTypeClass() == BuiltinClass;
  }
};

class DpctAsmVectorType : public DpctAsmType {
public:
  enum TypeKind : uint8_t {
#define VECTOR(X, Y) TK_##X,
#include "AsmTokenKinds.def"
  };

private:
  TypeKind Kind;
  DpctAsmBuiltinType *ElementType;

public:
  DpctAsmVectorType(TypeKind Kind, DpctAsmBuiltinType *ElementType)
      : DpctAsmType(VectorClass), Kind(Kind), ElementType(ElementType) {}

  TypeKind getKind() const { return Kind; }
  const DpctAsmBuiltinType *getElementType() const { return ElementType; }

  static bool classof(const DpctAsmType *T) {
    return T->getTypeClass() == VectorClass;
  }
};

class DpctAsmArrayType : public DpctAsmType {
  DpctAsmType *ElementType;

protected:
  DpctAsmArrayType(TypeClass TC, DpctAsmType *ElementType)
      : DpctAsmType(TC), ElementType(ElementType) {}

public:
  const DpctAsmType *getElementType() const { return ElementType; }

  static bool classof(const DpctAsmType *T) {
    return T->getTypeClass() == ConstantArrayClass ||
           T->getTypeClass() == IncompleteArrayClass;
  }
};

class DpctAsmConstantArrayType : public DpctAsmArrayType {
  DpctAsmIntegerLiteral *Size;

public:
  DpctAsmConstantArrayType(DpctAsmType *ElementType,
                           DpctAsmIntegerLiteral *Size)
      : DpctAsmArrayType(ConstantArrayClass, ElementType), Size(Size) {}

  const DpctAsmIntegerLiteral *getSize() const { return Size; }

  static bool classof(const DpctAsmType *T) {
    return T->getTypeClass() == ConstantArrayClass;
  }
};

class DpctAsmIncompleteArrayType : public DpctAsmArrayType {
public:
  DpctAsmIncompleteArrayType(DpctAsmType *ElementType)
      : DpctAsmArrayType(ConstantArrayClass, ElementType) {}

  static bool classof(const DpctAsmType *T) {
    return T->getTypeClass() == ConstantArrayClass;
  }
};

class DpctAsmTupleType : public DpctAsmType {
  SmallVector<DpctAsmType *, 4> ElementTypes;

public:
  DpctAsmTupleType(ArrayRef<DpctAsmType *> ElementTypes)
      : DpctAsmType(TupleClass), ElementTypes(ElementTypes) {}

  const DpctAsmType *getElementType(unsigned I) const {
    return ElementTypes[I];
  }

  static bool classof(const DpctAsmType *T) {
    return T->getTypeClass() == TupleClass;
  }
};

class DpctAsmDiscardType : public DpctAsmType {
public:
  DpctAsmDiscardType() : DpctAsmType(DiscardClass) {}

  static bool classof(const DpctAsmType *T) {
    return T->getTypeClass() == DiscardClass;
  }
};

class DpctAsmDecl {
public:
  enum DeclClass {
    VariableDeclClass,
    LabelDeclClass,
  };

private:
  DeclClass dClass;
  DpctAsmIdentifierInfo *Name;

protected:
  DpctAsmDecl(DeclClass DC, DpctAsmIdentifierInfo *Name)
      : dClass(DC), Name(Name) {}

public:
  virtual ~DpctAsmDecl();
  DeclClass getDeclClass() const { return dClass; }
  DpctAsmIdentifierInfo *getDeclName() const { return Name; }

  void *operator new(size_t bytes) noexcept {
    llvm_unreachable("PtxDecl cannot be allocated with regular 'new'.");
  }

  void operator delete(void *data) noexcept {
    llvm_unreachable("PtxDecl cannot be released with regular 'delete'.");
  }
};

class DpctAsmVariableDecl : public DpctAsmDecl {
  DpctAsmIdentifierInfo *StorageClass;
  DpctAsmType *Type;
  DpctAsmExpr *Align;

public:
  DpctAsmVariableDecl(DpctAsmIdentifierInfo *Name, DpctAsmType *Type)
      : DpctAsmDecl(VariableDeclClass, Name), Type(Type) {}

  const DpctAsmIdentifierInfo *getStorageClass() const { return StorageClass; }

  DpctAsmType *getType() { return Type; }
  const DpctAsmType *getType() const { return Type; }

  void setAlign(DpctAsmExpr *Align) { this->Align = Align; }

  static bool classof(const DpctAsmDecl *T) {
    return T->getDeclClass() == VariableDeclClass;
  }
};

class DpctAsmStmt {
public:
  enum StmtClass {
    DeclStmtClass,
    CompoundStmtClass,
    InstructionClass,
    GuardInstructionClass,
    UnaryOperatorClass,
    BinaryOperatorClass,
    ConditionalOperatorClass,
    ArraySubscriptExprClass,
    TupleExprClass,
    DiscardExprClass,
    AddressExprClass,
    CastExprClass,
    ParenExprClass,
    DeclRefExprClass,
    IntegerLiteralClass,
    FloatingLiteralClass,
    ExactMachineFloatingClass
  };

  DpctAsmStmt(const DpctAsmStmt &) = delete;
  DpctAsmStmt &operator=(const DpctAsmStmt &) = delete;
  virtual ~DpctAsmStmt();

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
  DpctAsmStmt(StmtClass SC) : sClass(SC) {}
};

class DpctAsmCompoundStmt : public DpctAsmStmt {
  SmallVector<DpctAsmStmt *, 4> Stmts;

public:
  DpctAsmCompoundStmt(ArrayRef<DpctAsmStmt *> Stmts)
      : DpctAsmStmt(CompoundStmtClass), Stmts(Stmts) {}

  using stmt_range =
      llvm::iterator_range<SmallVector<DpctAsmStmt *, 4>::const_iterator>;

  stmt_range stmts() const { return stmt_range(Stmts.begin(), Stmts.end()); }

  static bool classof(const DpctAsmStmt *S) {
    return S->getStmtClass() == CompoundStmtClass;
  }
};

class DpctAsmInstruction : public DpctAsmStmt {
  DpctAsmIdentifierInfo *Opcode;
  SmallVector<DpctAsmType *, 4> Types;
  SmallVector<DpctAsmIdentifierInfo *, 4> Attributes;
  DpctAsmExpr *OutputOperand;
  SmallVector<DpctAsmExpr *, 4> InputOperands;
  DpctAsmExpr *PredOutput = nullptr;

public:
  DpctAsmInstruction(DpctAsmIdentifierInfo *Op, ArrayRef<DpctAsmType *> Types,
                     ArrayRef<DpctAsmIdentifierInfo *> Attrs,
                     DpctAsmExpr *OutputOp, ArrayRef<DpctAsmExpr *> InputOps,
                     DpctAsmExpr *Pred)
      : DpctAsmStmt(InstructionClass), Opcode(Op), Types(Types),
        Attributes(Attrs), OutputOperand(OutputOp), InputOperands(InputOps),
        PredOutput(Pred) {}

  DpctAsmInstruction(DpctAsmIdentifierInfo *Op, ArrayRef<DpctAsmType *> Types,
                     ArrayRef<DpctAsmIdentifierInfo *> Attrs,
                     DpctAsmExpr *OutputOp, ArrayRef<DpctAsmExpr *> InputOps)
      : DpctAsmStmt(InstructionClass), Opcode(Op), Types(Types),
        Attributes(Attrs), OutputOperand(OutputOp), InputOperands(InputOps) {}

  using type_range =
      llvm::iterator_range<SmallVector<DpctAsmType *, 4>::const_iterator>;
  using attribute_range = llvm::iterator_range<
      SmallVector<DpctAsmIdentifierInfo *, 4>::const_iterator>;
  using input_operand_range =
      llvm::iterator_range<SmallVector<DpctAsmExpr *, 4>::const_iterator>;

  DpctAsmIdentifierInfo *getOpcode() const { return Opcode; }
  const DpctAsmIdentifierInfo *getAttribute(unsigned I) const {
    return Attributes[I];
  }
  const DpctAsmExpr *getOutputOperand() const { return OutputOperand; }
  const DpctAsmExpr *getPredOutputOperand() const { return PredOutput; }
  const DpctAsmExpr *getInputOperand(unsigned I) const {
    return InputOperands[I];
  }
  size_t getNumInputOperands() const { return InputOperands.size(); }

  size_t getNumTypes() const { return Types.size(); }
  DpctAsmType *getType(unsigned I) { return Types[I]; }
  const DpctAsmType *getType(unsigned I) const { return Types[I]; }

  type_range types() const { return type_range(Types.begin(), Types.end()); }

  attribute_range attrs() const {
    return attribute_range(Attributes.begin(), Attributes.end());
  }

  input_operand_range input_operands() const {
    return input_operand_range(InputOperands.begin(), InputOperands.end());
  }

  static bool classof(const DpctAsmStmt *S) {
    return InstructionClass <= S->getStmtClass();
  }
};

class DpctAsmGuardInstruction : public DpctAsmStmt {
  bool IsNeg;
  const DpctAsmExpr *Pred;
  const DpctAsmInstruction *Instruction;

public:
  DpctAsmGuardInstruction(bool IsNeg, const DpctAsmExpr *Pred,
                          const DpctAsmInstruction *Inst)
      : DpctAsmStmt(GuardInstructionClass), IsNeg(IsNeg), Pred(Pred),
        Instruction(Inst) {}

  const DpctAsmExpr *getPred() const { return Pred; }
  const DpctAsmInstruction *getInstruction() const { return Instruction; }
  bool isNeg() const { return IsNeg; }

  static bool classof(const DpctAsmStmt *S) {
    return S->getStmtClass() == GuardInstructionClass;
  }
};

class DpctAsmDeclStmt : public DpctAsmStmt {
  DpctAsmType *BaseType;
  SmallVector<DpctAsmDecl *, 4> DeclGroup;

public:
  DpctAsmDeclStmt(DpctAsmType *BaseType, ArrayRef<DpctAsmDecl *> Decls)
      : DpctAsmStmt(DeclStmtClass), BaseType(BaseType), DeclGroup(Decls) {}

  unsigned getNumDecl() const { return DeclGroup.size(); }
  const DpctAsmDecl *getDecl(unsigned I) const { return DeclGroup[I]; }

  using decl_range =
      llvm::iterator_range<SmallVector<DpctAsmDecl *, 4>::const_iterator>;

  decl_range decls() const {
    return decl_range(DeclGroup.begin(), DeclGroup.end());
  }

  const DpctAsmType *getBaseType() const { return BaseType; }

  static bool classof(const DpctAsmStmt *S) {
    return S->getStmtClass() == DeclStmtClass;
  }
};

/// Base class for the full range of assembler expressions which are
/// needed for parsing.
class DpctAsmExpr : public DpctAsmStmt {
  DpctAsmType *Type;

protected:
  explicit DpctAsmExpr(StmtClass SC, DpctAsmType *Type)
      : DpctAsmStmt(SC), Type(Type) {}

public:
  DpctAsmType *getType() { return Type; }
  const DpctAsmType *getType() const { return Type; }

  static bool classof(const DpctAsmStmt *S) {
    return UnaryOperatorClass <= S->getStmtClass() &&
           S->getStmtClass() <= FloatingLiteralClass;
  }

  void print(raw_ostream &OS, bool InParens = false) const;
  void dump() const;
};

inline raw_ostream &operator<<(raw_ostream &OS, const DpctAsmExpr &E) {
  E.print(OS);
  return OS;
}

class DpctAsmIntegerLiteral : public DpctAsmExpr {
  llvm::APInt Value;

public:
  DpctAsmIntegerLiteral(DpctAsmType *Type, llvm::APInt Value)
      : DpctAsmExpr(IntegerLiteralClass, Type), Value(Value) {}

  llvm::APInt getValue() const { return Value; }

  static bool classof(const DpctAsmStmt *S) {
    return S->getStmtClass() == IntegerLiteralClass;
  }
};

class DpctAsmFloatingLiteral : public DpctAsmExpr {
  llvm::APFloat Value;

protected:
  DpctAsmFloatingLiteral(StmtClass SC, DpctAsmType *Type, llvm::APFloat Value)
      : DpctAsmExpr(SC, Type), Value(Value) {}

public:
  DpctAsmFloatingLiteral(DpctAsmType *Type, llvm::APFloat Value)
      : DpctAsmExpr(FloatingLiteralClass, Type), Value(Value) {}

  llvm::APFloat getValue() const { return Value; }

  static bool classof(const DpctAsmStmt *S) {
    return S->getStmtClass() == FloatingLiteralClass ||
           S->getStmtClass() == ExactMachineFloatingClass;
  }
};

class DpctAsmExactMachineFloatingLiteral : public DpctAsmFloatingLiteral {
  SmallString<16> HexLiteral;

public:
  DpctAsmExactMachineFloatingLiteral(DpctAsmType *Type, llvm::APFloat Value,
                                     StringRef HexLiteral)
      : DpctAsmFloatingLiteral(ExactMachineFloatingClass, Type, Value),
        HexLiteral(HexLiteral) {
    assert((HexLiteral.size() == 8 || HexLiteral.size() == 16) &&
           "Hex literal length must be one of 8 or 16");
  }

  const StringRef getHexLiteral() const { return HexLiteral; }

  static bool classof(const DpctAsmStmt *S) {
    return S->getStmtClass() == FloatingLiteralClass ||
           S->getStmtClass() == ExactMachineFloatingClass;
  }
};

/// TODO: DpctAsmArraySubscriptExpr
// class DpctAsmArraySubscriptExpr : public DpctAsmExpr {
//   DpctAsmExpr *LHS;
//   DpctAsmExpr *RHS;
// public:
//   DpctAsmArraySubscriptExpr(DpctAsmExpr *LHS, DpctAsmExpr *RHS)
//     : DpctAsmExpr(ArraySubscriptExpr)
// };

class DpctAsmDeclRefExpr : public DpctAsmExpr {
  DpctAsmDecl *Decl;

public:
  DpctAsmDeclRefExpr(DpctAsmVariableDecl *D)
      : DpctAsmExpr(DeclRefExprClass, D->getType()), Decl(D) {}

  const DpctAsmDecl &getDecl() const { return *Decl; }

  static bool classof(const DpctAsmStmt *S) {
    return S->getStmtClass() == DeclRefExprClass;
  }
};

class DpctAsmTupleExpr : public DpctAsmExpr {
  SmallVector<DpctAsmExpr *, 4> Elements;

public:
  DpctAsmTupleExpr(DpctAsmTupleType *Type, ArrayRef<DpctAsmExpr *> Elements)
      : DpctAsmExpr(TupleExprClass, Type), Elements(Elements) {}

  using element_range =
      llvm::iterator_range<SmallVector<DpctAsmExpr *, 4>::const_iterator>;

  element_range elements() const {
    return element_range(Elements.begin(), Elements.end());
  }

  const DpctAsmExpr *getElement(unsigned I) const { return Elements[I]; }

  static bool classof(const DpctAsmStmt *S) {
    return S->getStmtClass() == TupleExprClass;
  }
};

class DpctAsmDiscardExpr : public DpctAsmExpr {
public:
  DpctAsmDiscardExpr(DpctAsmDiscardType *Any)
      : DpctAsmExpr(DiscardExprClass, Any) {}

  static bool classof(const DpctAsmStmt *S) {
    return S->getStmtClass() == DiscardExprClass;
  }
};

class DpctAsmAddressExpr : public DpctAsmExpr {
  DpctAsmExpr *SubExpr;

public:
  DpctAsmAddressExpr(DpctAsmBuiltinType *Type, DpctAsmExpr *SubExpr)
      : DpctAsmExpr(AddressExprClass, Type), SubExpr(SubExpr) {}

  const DpctAsmExpr *getSubExpr() const { return SubExpr; }

  static bool classof(const DpctAsmStmt *S) {
    return S->getStmtClass() == AddressExprClass;
  }
};

class DpctAsmCastExpr : public DpctAsmExpr {
  const DpctAsmExpr *SubExpr;

public:
  DpctAsmCastExpr(DpctAsmBuiltinType *Type, const DpctAsmExpr *Op)
      : DpctAsmExpr(CastExprClass, Type), SubExpr(Op) {}

  const DpctAsmExpr *getSubExpr() const { return SubExpr; }

  static bool classof(const DpctAsmStmt *S) {
    return S->getStmtClass() == CastExprClass;
  }
};

class DpctAsmParenExpr : public DpctAsmExpr {
  DpctAsmExpr *SubExpr;

public:
  DpctAsmParenExpr(DpctAsmExpr *SubExpr)
      : DpctAsmExpr(ParenExprClass, SubExpr->getType()), SubExpr(SubExpr) {}

  const DpctAsmExpr *getSubExpr() const { return SubExpr; }

  static bool classof(const DpctAsmStmt *S) {
    return S->getStmtClass() == ParenExprClass;
  }
};

class DpctAsmUnaryOperator : public DpctAsmExpr {
public:
  enum Opcode {
    // Unary arithmetic
    Plus,  // +
    Minus, // -
    Not,   // ~
    LNot   // !
  };

private:
  Opcode Op;
  DpctAsmExpr *SubExpr;

public:
  DpctAsmUnaryOperator(Opcode Op, DpctAsmExpr *Expr, DpctAsmType *Type)
      : DpctAsmExpr(UnaryOperatorClass, Type), Op(Op), SubExpr(Expr) {}

public:
  Opcode getOpcode() const { return (Opcode)Op; }

  const DpctAsmExpr *getSubExpr() const { return SubExpr; }

  static bool classof(const DpctAsmStmt *S) {
    return S->getStmtClass() == UnaryOperatorClass;
  }
};

class DpctAsmBinaryOperator : public DpctAsmExpr {
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

    // Assignment
    Assign // =
  };

private:
  Opcode Op;
  DpctAsmExpr *LHS;
  DpctAsmExpr *RHS;

public:
  DpctAsmBinaryOperator(Opcode Op, DpctAsmExpr *LHS, DpctAsmExpr *RHS,
                        DpctAsmType *Type)
      : DpctAsmExpr(BinaryOperatorClass, Type), Op(Op), LHS(LHS), RHS(RHS) {}

public:
  Opcode getOpcode() const { return Op; }
  const DpctAsmExpr *getLHS() const { return LHS; }
  const DpctAsmExpr *getRHS() const { return RHS; }

  static bool classof(const DpctAsmStmt *S) {
    return S->getStmtClass() == BinaryOperatorClass;
  }
};

class DpctAsmConditionalOperator : public DpctAsmExpr {
  DpctAsmExpr *Cond;
  DpctAsmExpr *LHS;
  DpctAsmExpr *RHS;

public:
  DpctAsmConditionalOperator(DpctAsmExpr *C, DpctAsmExpr *L, DpctAsmExpr *R,
                             DpctAsmType *Type)
      : DpctAsmExpr(ConditionalOperatorClass, Type), Cond(C), LHS(L), RHS(R) {}

  const DpctAsmExpr *getCond() const { return Cond; }
  const DpctAsmExpr *getLHS() const { return LHS; }
  const DpctAsmExpr *getRHS() const { return RHS; }

  static bool classof(const DpctAsmStmt *S) {
    return S->getStmtClass() == ConditionalOperatorClass;
  }
};

class DpctAsmContext : public DpctAsmIdentifierInfoLookup {

  BumpPtrAllocator Allocator;
  DpctAsmIdentifierTable AsmBuiltinIdTable;
  SmallVector<DpctAsmIdentifierInfo *, 4> InlineAsmOperands;
  llvm::DenseMap</*DpctAsmBuiltinType::TypeKind*/ int, DpctAsmBuiltinType *>
      AsmBuiltinTypes;
  DpctAsmDiscardType *DiscardType;

public:
  void *allocate(unsigned Size, unsigned Align = 8) {
    return Allocator.Allocate(Size, Align);
  }

  void deallocate(void *Ptr) {}

  unsigned addInlineAsmOperand(StringRef Name) {
    DpctAsmIdentifierInfo &II = AsmBuiltinIdTable.get(Name);
    InlineAsmOperands.push_back(&II);
    return InlineAsmOperands.size() - 1;
  }

  DpctAsmIdentifierInfo *get(StringRef Name) override {
    if (Name.size() < 2 || Name[0] != '%')
      return nullptr;

    // This identifier is an inline asm placeholder.
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

  DpctAsmIdentifierInfo *get(unsigned Index) {
    if (Index >= InlineAsmOperands.size())
      return nullptr;
    return InlineAsmOperands[Index];
  }

  DpctAsmBuiltinType *getTypeFromConstraint(StringRef Constraint);
  DpctAsmBuiltinType *getBuiltinType(StringRef TypeName);
  DpctAsmBuiltinType *getBuiltinType(DpctAsmBuiltinType::TypeKind Kind);
  DpctAsmBuiltinType *getBuiltinTypeFromTokenKind(asmtok::TokenKind Kind);
  DpctAsmDiscardType *getDiscardType();

  DpctAsmBuiltinType *getS64Type() {
    return getBuiltinType(DpctAsmBuiltinType::TK_s64);
  }

  DpctAsmBuiltinType *getU64Type() {
    return getBuiltinType(DpctAsmBuiltinType::TK_u64);
  }

  DpctAsmBuiltinType *getF32Type() {
    return getBuiltinType(DpctAsmBuiltinType::TK_f32);
  }

  DpctAsmBuiltinType *getF64Type() {
    return getBuiltinType(DpctAsmBuiltinType::TK_f64);
  }
};

class DpctAsmScope {
  using DeclSetTy = SmallPtrSet<DpctAsmVariableDecl *, 32>;
  DpctAsmScope *Parent;
  DeclSetTy DeclsInScope;
  unsigned Depth;

public:
  DpctAsmScope(DpctAsmScope *Parent)
      : Parent(Parent), Depth(Parent ? Parent->Depth + 1 : 0) {}

  bool hasParent() const { return Parent; }
  const DpctAsmScope *getParent() const { return Parent; }
  DpctAsmScope *getParent() { return Parent; }
  unsigned getDepth() const { return Depth; }

  using decl_range = llvm::iterator_range<DeclSetTy::iterator>;

  decl_range decls() const {
    return decl_range(DeclsInScope.begin(), DeclsInScope.end());
  }

  void addDecl(DpctAsmVariableDecl *D) { DeclsInScope.insert(D); }

  bool isDeclScope(const DpctAsmVariableDecl *D) const {
    return DeclsInScope.contains(D);
  }

  bool contains(const DpctAsmScope &rhs) const { return Depth < rhs.Depth; }

  DpctAsmVariableDecl *lookupDecl(DpctAsmIdentifierInfo *II) const;
};

using DpctAsmTypeResult = clang::ActionResult<DpctAsmType *>;
using DpctAsmDeclResult = clang::ActionResult<DpctAsmDecl *>;
using DpctAsmStmtResult = clang::ActionResult<DpctAsmStmt *>;
using DpctAsmExprResult = clang::ActionResult<DpctAsmExpr *>;

inline DpctAsmExprResult AsmExprError() { return DpctAsmExprResult(true); }
inline DpctAsmStmtResult AsmStmtError() { return DpctAsmStmtResult(true); }
inline DpctAsmTypeResult AsmTypeError() { return DpctAsmTypeResult(true); }
inline DpctAsmDeclResult AsmDeclError() { return DpctAsmDeclResult(true); }

// clang-format off
namespace asmprec {
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

asmprec::Level getBinOpPrec(asmtok::TokenKind Kind);
} // namespace asmprec
// clang-format on

class DpctAsmParser {
  DpctAsmLexer Lexer;
  DpctAsmContext &Context;
  SourceMgr &SrcMgr;
  DpctAsmScope *CurScope;

  /// Tok - The current token we are peeking ahead.  All parsing methods assume
  /// that this is valid.
  DpctAsmToken Tok;

  // PrevTokLocation - The location of the token we previously
  // consumed. This token is used for diagnostics where we expected to
  // see a token following another token (e.g., the ';' at the end of
  // a statement).
  SMLoc PrevTokLocation;

  unsigned short ParenCount = 0;
  unsigned short BracketCount = 0;
  unsigned short BraceCount = 0;

  /// ScopeCache - Cache scopes to reduce malloc traffic.
  enum { ScopeCacheSize = 16 };
  unsigned NumCachedScopes = 0;
  DpctAsmScope *ScopeCache[ScopeCacheSize];

  class ParseScope {
    DpctAsmParser *Self;
    ParseScope(const ParseScope &) = delete;
    void operator=(const ParseScope &) = delete;

  public:
    ParseScope(DpctAsmParser *Self) : Self(Self) { Self->EnterScope(); }

    ~ParseScope() {
      Self->ExitScope();
      Self = nullptr;
    }
  };

  struct DpctAsmDeclarationSpecifier {
    asmtok::TokenKind StateSpace = asmtok::unknown;
    asmtok::TokenKind VectorType = asmtok::unknown;
    DpctAsmIntegerLiteral *Alignment = nullptr;
    DpctAsmBuiltinType *BaseType = nullptr;
    DpctAsmType *Type = nullptr;
  };

public:
  DpctAsmParser(DpctAsmContext &Ctx, SourceMgr &Mgr)
      : Lexer(*Mgr.getMemoryBuffer(Mgr.getMainFileID())), Context(Ctx),
        SrcMgr(Mgr), CurScope(nullptr) {
    Lexer.getIdentifiertable().setExternalIdentifierLookup(&Context);
    Tok.startToken();
    Tok.setKind(asmtok::eof);
    ConsumeToken();
    EnterScope();
  }
  ~DpctAsmParser() { ExitScope(); }

  SourceMgr &getSourceManager() { return SrcMgr; }
  DpctAsmLexer &getLexer() { return Lexer; }
  DpctAsmContext &getContext() { return Context; }

  const DpctAsmToken &getCurToken() const { return Tok; }

  /// isTokenParen - Return true if the cur token is '(' or ')'.
  bool isTokenParen() const {
    return Tok.isOneOf(asmtok::l_paren, asmtok::r_paren);
  }
  /// isTokenBracket - Return true if the cur token is '[' or ']'.
  bool isTokenBracket() const {
    return Tok.isOneOf(asmtok::l_square, asmtok::r_square);
  }
  /// isTokenBrace - Return true if the cur token is '{' or '}'.
  bool isTokenBrace() const {
    return Tok.isOneOf(asmtok::l_brace, asmtok::r_brace);
  }

  /// isTokenSpecial - True if this token requires special consumption methods.
  bool isTokenSpecial() const {
    return isTokenParen() || isTokenBracket() || isTokenBrace();
  }

  /// ConsumeParen - This consume method keeps the paren count up-to-date.
  ///
  SMLoc ConsumeParen() {
    assert(isTokenParen() && "wrong consume method");
    if (Tok.getKind() == asmtok::l_paren)
      ++ParenCount;
    else if (ParenCount) {
      --ParenCount; // Don't let unbalanced )'s drive the count negative.
    }
    PrevTokLocation = Tok.getLocation();
    Lexer.lex(Tok);
    return PrevTokLocation;
  }

  /// ConsumeBracket - This consume method keeps the bracket count up-to-date.
  ///
  SMLoc ConsumeBracket() {
    assert(isTokenBracket() && "wrong consume method");
    if (Tok.getKind() == asmtok::l_square)
      ++BracketCount;
    else if (BracketCount) {
      --BracketCount; // Don't let unbalanced ]'s drive the count negative.
    }

    PrevTokLocation = Tok.getLocation();
    Lexer.lex(Tok);
    return PrevTokLocation;
  }

  /// ConsumeBrace - This consume method keeps the brace count up-to-date.
  ///
  SMLoc ConsumeBrace() {
    assert(isTokenBrace() && "wrong consume method");
    if (Tok.getKind() == asmtok::l_brace)
      ++BraceCount;
    else if (BraceCount) {
      --BraceCount; // Don't let unbalanced }'s drive the count negative.
    }

    PrevTokLocation = Tok.getLocation();
    Lexer.lex(Tok);
    return PrevTokLocation;
  }

  /// ConsumeToken - Consume the current 'peek token' and lex the next one.
  /// This does not work with special tokens: string literals, code completion,
  /// annotation tokens and balanced tokens must be handled using the specific
  /// consume methods.
  /// Returns the location of the consumed token.
  SMLoc ConsumeToken() {
    assert(!isTokenSpecial() &&
           "Should consume special tokens with Consume*Token");
    PrevTokLocation = Tok.getLocation();
    Lexer.lex(Tok);
    return PrevTokLocation;
  }

  bool TryConsumeToken(asmtok::TokenKind Expected) {
    if (Tok.isNot(Expected))
      return false;
    assert(!isTokenSpecial() &&
           "Should consume special tokens with Consume*Token");
    PrevTokLocation = Tok.getLocation();
    Lexer.lex(Tok);
    return true;
  }

  bool TryConsumeToken(asmtok::TokenKind Expected, SMLoc &Loc) {
    if (!TryConsumeToken(Expected))
      return false;
    Loc = PrevTokLocation;
    return true;
  }

  /// ConsumeAnyToken - Dispatch to the right Consume* method based on the
  /// current token type.  This should only be used in cases where the type of
  /// the token really isn't known, e.g. in error recovery.
  SMLoc ConsumeAnyToken() {
    if (isTokenParen())
      return ConsumeParen();
    if (isTokenBracket())
      return ConsumeBracket();
    if (isTokenBrace())
      return ConsumeBrace();
    return ConsumeToken();
  }

  // ExpectAndConsume - The parser expects that 'ExpectedTok' is next in the
  /// input.  If so, it is consumed and false is returned.
  ///
  /// If a trivial punctuator misspelling is encountered, a FixIt error
  /// diagnostic is issued and false is returned after recovery.
  ///
  /// If the input is malformed, this emits the specified diagnostic and true is
  /// returned.
  bool ExpectAndConsume(asmtok::TokenKind ExpectedTok);

  /// The parser expects a semicolon and, if present, will consume it.
  ///
  /// If the next token is not a semicolon, this emits the specified diagnostic,
  /// or, if there's just some closing-delimiter noise (e.g., ')' or ']') prior
  /// to the semicolon, consumes that extra token.
  bool ExpectAndConsumeSemi();

  DpctAsmDeclResult addInlineAsmOperands(StringRef Operand,
                                         StringRef Constraint);

  DpctAsmScope *getCurScope() const { return CurScope; }

  void EnterScope() {
    if (NumCachedScopes) {
      DpctAsmScope *N = ScopeCache[--NumCachedScopes];
      CurScope = new (N) DpctAsmScope(getCurScope());
    } else {
      CurScope = new DpctAsmScope(getCurScope());
    }
  }

  void ExitScope() {
    assert(getCurScope());
    DpctAsmScope *OldScope = getCurScope();
    if (OldScope) {
      CurScope = OldScope->getParent();
      if (NumCachedScopes == ScopeCacheSize)
        delete OldScope;
      else
        ScopeCache[NumCachedScopes++] = OldScope;
    } else {
      CurScope = nullptr;
    }
  }

  bool isInstructionAttribute();

  DpctAsmStmtResult ParseStatement();
  DpctAsmStmtResult ParseCompoundStatement();
  DpctAsmStmtResult ParseGuardInstruction();
  DpctAsmStmtResult ParseInstruction();

  DpctAsmExprResult ParseExpression();
  DpctAsmExprResult ParseCastExpression();
  DpctAsmExprResult ParseAssignmentExpression();
  DpctAsmExprResult ParseParenExpression(DpctAsmBuiltinType *&CastTy);
  DpctAsmExprResult ParseRHSOfBinaryExpression(DpctAsmExprResult LHS,
                                               asmprec::Level MinPrec);
  DpctAsmExprResult ParsePostfixExpressionSuffix(DpctAsmExprResult LHS);

  DpctAsmStmtResult ParseDeclarationStatement();
  DpctAsmTypeResult
  ParseDeclarationSpecifier(DpctAsmDeclarationSpecifier &DeclSpec);
  DpctAsmDeclResult
  ParseDeclarator(const DpctAsmDeclarationSpecifier &DeclSpec);

  // Sema
  DpctAsmExprResult ActOnTypeCast(DpctAsmBuiltinType *CastTy,
                                  DpctAsmExpr *SubExpr);
  DpctAsmExprResult ActOnAddressExpr(DpctAsmExpr *SubExpr);
  DpctAsmExprResult ActOnDiscardExpr();
  DpctAsmExprResult ActOnParenExpr(DpctAsmExpr *SubExpr);
  DpctAsmExprResult ActOnIdExpr(DpctAsmIdentifierInfo *II);
  DpctAsmExprResult ActOnTupleExpr(ArrayRef<DpctAsmExpr *> Tuple);
  DpctAsmExprResult ActOnUnaryOp(asmtok::TokenKind OpTok, DpctAsmExpr *SubExpr);
  DpctAsmExprResult ActOnBinaryOp(asmtok::TokenKind OpTok, DpctAsmExpr *LHS,
                                  DpctAsmExpr *RHS);
  DpctAsmExprResult ActOnConditionalOp(DpctAsmExpr *Cond, DpctAsmExpr *LHS,
                                       DpctAsmExpr *RHS);
  DpctAsmExprResult ActOnNumericConstant(const DpctAsmToken &Tok);
  DpctAsmExprResult ActOnAlignment(DpctAsmExpr *Alignment);
  DpctAsmDeclResult ActOnVariableDecl(DpctAsmIdentifierInfo *Name,
                                      DpctAsmType *Type);
};
} // namespace clang::dpct

namespace llvm {

template <typename T> struct PointerLikeTypeTraits;
template <> struct PointerLikeTypeTraits<::clang::dpct::DpctAsmType *> {
  static inline void *getAsVoidPointer(::clang::Type *P) { return P; }

  static inline ::clang::dpct::DpctAsmType *getFromVoidPointer(void *P) {
    return static_cast<::clang::dpct::DpctAsmType *>(P);
  }

  static constexpr int NumLowBitsAvailable = clang::TypeAlignmentInBits;
};

template <> struct PointerLikeTypeTraits<::clang::dpct::DpctAsmDecl *> {
  static inline void *getAsVoidPointer(::clang::ExtQuals *P) { return P; }

  static inline ::clang::dpct::DpctAsmDecl *getFromVoidPointer(void *P) {
    return static_cast<::clang::dpct::DpctAsmDecl *>(P);
  }

  static constexpr int NumLowBitsAvailable = clang::TypeAlignmentInBits;
};

template <> struct PointerLikeTypeTraits<::clang::dpct::DpctAsmStmt *> {
  static inline void *getAsVoidPointer(::clang::ExtQuals *P) { return P; }

  static inline ::clang::dpct::DpctAsmStmt *getFromVoidPointer(void *P) {
    return static_cast<::clang::dpct::DpctAsmStmt *>(P);
  }

  static constexpr int NumLowBitsAvailable = clang::TypeAlignmentInBits;
};

} // namespace llvm

inline void *operator new(size_t Bytes, ::clang::dpct::DpctAsmContext &C,
                          size_t Alignment = 8) noexcept {
  return C.allocate(Bytes, Alignment);
}

inline void operator delete(void *Ptr, ::clang::dpct::DpctAsmContext &C,
                            size_t) noexcept {
  C.deallocate(Ptr);
}

inline void *operator new[](size_t Bytes, ::clang::dpct::DpctAsmContext &C,
                            size_t Alignment = 8) noexcept {
  return C.allocate(Bytes, Alignment);
}

inline void operator delete[](void *Ptr,
                              ::clang::dpct::DpctAsmContext &C) noexcept {
  C.deallocate(Ptr);
}

#endif // CLANG_DPCT_INLINE_ASM_PARSER_H
