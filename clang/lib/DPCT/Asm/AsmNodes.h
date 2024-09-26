//===---------------------------- AsmNodes.h --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DPCT_ASM_NODES_H
#define CLANG_DPCT_ASM_NODES_H

#include "AsmIdentifierTable.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

namespace clang {
class Expr;
class Type;
namespace dpct {
class InlineAsmType;
class InlineAsmDecl;
class InlineAsmExpr;
class InlineAsmStmt;
class InlineAsmParser;
class InlineAsmIntegerLiteral;

using llvm::SmallSet;
using llvm::SmallVector;

enum class InstAttr {
#define MODIFIER(X, Y) X,
#include "Asm/AsmTokenKinds.def"
};

enum class AsmStateSpace {
#define STATE_SPACE(X, Y) S_ ## X,
#include "AsmTokenKinds.def"
  none
};

enum class AsmTarget {
#define TARGET(X) X,
#include "AsmTokenKinds.def"
  none
};

enum class AsmLinkage {
#define LINKAGE(X, Y) L_ ## X,
#include "AsmTokenKinds.def"
  none
};

/// The base class of the type hierarchy.
/// Types, once created, are immutable.
class InlineAsmType {
public:
  enum TypeClass { BuiltinClass, VectorClass, DiscardClass };

  virtual ~InlineAsmType();
  TypeClass getTypeClass() const { return tClass; }

  void *operator new(size_t bytes) noexcept {
    llvm_unreachable("InlineAsmType cannot be allocated with regular 'new'.");
  }

  void operator delete(void *data) noexcept {
    llvm_unreachable("InlineAsmType cannot be released with regular 'delete'.");
  }

protected:
  InlineAsmType(TypeClass TC) : tClass(TC) {}

private:
  TypeClass tClass;
};

/// This class is used for builtin types like 'u32'.
class InlineAsmBuiltinType : public InlineAsmType {
public:
  enum TypeKind {
#define BUILTIN_TYPE(X, Y) X,
#include "AsmTokenKinds.def"
    NUM_TYPES
  };

private:
  TypeKind Kind;

public:
  InlineAsmBuiltinType(TypeKind Kind)
      : InlineAsmType(BuiltinClass), Kind(Kind) {}

  TypeKind getKind() const { return Kind; }
  bool is(TypeKind K) const { return K == Kind; }
  template <class... Ks> bool isOneOf(Ks... K) const {
    return ((K == Kind) || ...);
  }
  template <class... Ks> bool isNot(Ks... K) { return ((K != Kind) && ...); }
  bool isBit() const { return isOneOf(b8, b16, b32, b64); }
  bool isSigned() const { return isOneOf(s8, s16, s32, s64); }
  bool isUnsigned() const { return isOneOf(u8, u16, u32, u64); }
  bool isInt() const { return isSigned() || isUnsigned(); }
  bool isFloat() const { return isOneOf(f16, f32, f64); }
  bool isScalar() const { return isInt() || isFloat(); }
  bool isVector() const { return isOneOf(f16x2, bf16x2, s16x2, u16x2); }
  unsigned getWidth() const;

  static bool classof(const InlineAsmType *T) {
    return T->getTypeClass() == BuiltinClass;
  }
};

// This class is used for device asm vector types.
class InlineAsmVectorType : public InlineAsmType {
public:
  enum VecKind { v2, v4, v8 };

private:
  VecKind Kind;
  InlineAsmBuiltinType *ElementType;

public:
  InlineAsmVectorType(VecKind Kind, InlineAsmBuiltinType *ElementType)
      : InlineAsmType(VectorClass), Kind(Kind), ElementType(ElementType) {}

  VecKind getKind() const { return Kind; }
  const InlineAsmBuiltinType *getElementType() const { return ElementType; }

  static bool classof(const InlineAsmType *T) {
    return T->getTypeClass() == VectorClass;
  }
};

/// This class is used for device asm '_' expression.
class InlineAsmDiscardType : public InlineAsmType {
public:
  InlineAsmDiscardType() : InlineAsmType(DiscardClass) {}

  static bool classof(const InlineAsmType *T) {
    return T->getTypeClass() == DiscardClass;
  }
};

/// This represents one declaration (or definition), e.g. a variable,
/// label, etc.
class InlineAsmDecl {
public:
  enum DeclClass {
    VariableDeclClass,

    /// FIXME: Label declaration do not support now.
    LabelDeclClass,
  };

private:
  DeclClass dClass;
protected:
  InlineAsmDecl(DeclClass DC)
      : dClass(DC) {}

public:
  virtual ~InlineAsmDecl();
  DeclClass getDeclClass() const { return dClass; }

  void *operator new(size_t bytes) noexcept {
    llvm_unreachable("InlineAsmDecl cannot be allocated with regular 'new'.");
  }

  void operator delete(void *data) noexcept {
    llvm_unreachable("InlineAsmDecl cannot be released with regular 'delete'.");
  }
};

class InlineAsmNamedDecl : public InlineAsmDecl {
  /// The declaration identifier.
  InlineAsmIdentifierInfo *Name;

protected:
  InlineAsmNamedDecl(DeclClass DC, InlineAsmIdentifierInfo *Name)
      : InlineAsmDecl(DC), Name(Name) {}

public:
  ~InlineAsmNamedDecl();
  InlineAsmIdentifierInfo *getDeclName() const { return Name; }

  static bool classof(InlineAsmDecl *D) {
    return D->getDeclClass() >= VariableDeclClass &&
           D->getDeclClass() <= LabelDeclClass;
  }
};

/// Represents a variable declaration.
class InlineAsmVarDecl : public InlineAsmNamedDecl {

  /// The state space of a variable, e.g. '.reg', '.global', '.local', etc.
  AsmStateSpace StateSpace;

  /// The type of this variable.
  InlineAsmType *Type = nullptr;

  /// Alignment of this variable, specificed by '.align' attribute.
  unsigned Align = 0;

  /// The num of parameterized names, e.g. %p<10>
  unsigned NumParameterizedNames = 0;

  /// Has '.align' attribute in this variable declaration.
  bool HasAlign = false;

  /// Has parameterized names in this variable declaration.
  bool IsParameterizedNameDecl = false;

  /// Is a inline asm statement operand.
  const Expr *InlineAsmOp = nullptr;

public:
  InlineAsmVarDecl(InlineAsmIdentifierInfo *Name, AsmStateSpace SS,
                   InlineAsmType *Type)
      : InlineAsmNamedDecl(VariableDeclClass, Name), StateSpace(SS),
        Type(Type) {}

  AsmStateSpace getStorageClass() const {
    return StateSpace;
  }

  void setInlineAsmOp(const Expr *Val) { InlineAsmOp = Val; }
  const Expr *getInlineAsmOp() const { return InlineAsmOp; }
  InlineAsmType *getType() { return Type; }
  const InlineAsmType *getType() const { return Type; }

  bool hasAlign() const { return HasAlign; }
  void setAlign(unsigned Align) {
    assert(!hasAlign() &&
           "This variable declaration already have a specific alignment");
    this->Align = Align;
  }
  unsigned getAlign() const {
    assert(hasAlign() && "This variable dose not have any specific alignment");
    return Align;
  }

  bool isParameterizedNameDecl() const { return IsParameterizedNameDecl; }
  void setNumParameterizedNames(unsigned Num) {
    assert(!isParameterizedNameDecl() &&
           "This variable declaration already is a parameterized name");
    IsParameterizedNameDecl = true;
    NumParameterizedNames = Num;
  }
  unsigned getNumParameterizedNames() const {
    assert(isParameterizedNameDecl() &&
           "This variable declaration was not a parameterized name");
    return NumParameterizedNames;
  }

  static bool classof(const InlineAsmDecl *T) {
    return T->getDeclClass() == VariableDeclClass;
  }
};

/// This represents one statement.
class InlineAsmStmt {
public:
  enum StmtClass {
#define STMT(CLASS, PARENT) CLASS##Class,
#define STMT_RANGE(BASE, FIRST, LAST)                                          \
  first##BASE##Constant = FIRST##Class, last##BASE##Constant = LAST##Class,
#define ABSTRACT_STMT(STMT)
#include "Asm/AsmNodes.def"
  };

  InlineAsmStmt(const InlineAsmStmt &) = delete;
  InlineAsmStmt &operator=(const InlineAsmStmt &) = delete;
  virtual ~InlineAsmStmt();

  void *operator new(size_t bytes) noexcept {
    llvm_unreachable("InlineAsmStmt cannot be allocated with regular 'new'.");
  }

  void operator delete(void *data) noexcept {
    llvm_unreachable("InlineAsmStmt cannot be released with regular 'delete'.");
  }

  StmtClass getStmtClass() const { return static_cast<StmtClass>(sClass); }

private:
  StmtClass sClass;

protected:
  InlineAsmStmt(StmtClass SC) : sClass(SC) {}
};

/// This represents a group of statements like { stmt stmt }.
class InlineAsmCompoundStmt : public InlineAsmStmt {
  SmallVector<InlineAsmStmt *, 4> Stmts;

public:
  InlineAsmCompoundStmt(ArrayRef<InlineAsmStmt *> Stmts)
      : InlineAsmStmt(CompoundStmtClass), Stmts(Stmts) {}

  using stmt_range =
      llvm::iterator_range<SmallVector<InlineAsmStmt *, 4>::const_iterator>;

  stmt_range stmts() const { return stmt_range(Stmts.begin(), Stmts.end()); }

  static bool classof(const InlineAsmStmt *S) {
    return S->getStmtClass() == CompoundStmtClass;
  }
};

/// This represents a device instruction.
/// opcode.attr1.attr2 dest-operand{|pred-output}, src1, src2, src3, ...;
class InlineAsmInstruction : public InlineAsmStmt {

  /// The opcode of instruction, must predefined in AsmTokenKinds.def
  /// e.g. asmtok::op_mov, asmtok::op_setp, etc.
  InlineAsmIdentifierInfo *Opcode = nullptr;

  std::optional<AsmStateSpace> StateSpace;

  /// This represents arrtibutes like: comparsion operator, rounding modifiers,
  /// ... e.g. instruction setp.eq.s32 has a comparsion operator 'eq'.
  SmallSet<InstAttr, 4> Attributes;

  /// This represents types in instruction, e.g. mov.u32.
  SmallVector<InlineAsmType *, 4> Types;

  // The output operand of instruction.
  InlineAsmExpr *OutputOp = nullptr;

  // The predicate output operand of instruction.
  // e.g. given shfl.sync.up.b32  Ry|p, Rx, 0x1,  0x0,
  // 0xffffffff; p is a predicate output.
  InlineAsmExpr *PredOutputOp = nullptr;

  /// The input operands of instruction. Operands[0] is output operand,
  /// If HasPredOutput is true, Operands[1] is pred output operand,
  /// therest are input operands.
  SmallVector<InlineAsmExpr *, 4> InputOps;

public:
  InlineAsmInstruction(InlineAsmIdentifierInfo *Op,
                       std::optional<AsmStateSpace> SS,
                       ArrayRef<InstAttr> Attrs,
                       ArrayRef<InlineAsmType *> Types, InlineAsmExpr *Out,
                       InlineAsmExpr *Pred, ArrayRef<InlineAsmExpr *> InOps)
      : InlineAsmStmt(InstructionClass), Opcode(Op), StateSpace(SS),
        Types(Types), OutputOp(Out), PredOutputOp(Pred), InputOps(InOps) {
    Attributes.insert(Attrs.begin(), Attrs.end());
  }

  using attr_range =
      llvm::iterator_range<SmallSet<InstAttr, 4>::const_iterator>;
  using type_range =
      llvm::iterator_range<SmallVector<InlineAsmType *, 4>::const_iterator>;
  using op_range =
      llvm::iterator_range<SmallVector<InlineAsmExpr *, 4>::const_iterator>;

  bool is(asmtok::TokenKind OpKind) const {
    return Opcode->getTokenID() == OpKind;
  }

  template <typename K, typename... Ks> bool is(K OpKind, Ks... OpKinds) const {
    return is(OpKind) || (is(OpKinds) || ...);
  }

  template <typename... Ts> bool hasAttr(Ts... Attrs) const {
    return (Attributes.contains(Attrs) || ...);
  }
  const InlineAsmIdentifierInfo *getOpcodeID() const { return Opcode; }
  asmtok::TokenKind getOpcode() const { return Opcode->getTokenID(); }
  ArrayRef<InlineAsmType *> getTypes() const { return Types; }
  const InlineAsmType *getType(unsigned I) const { return Types[I]; }
  unsigned getNumTypes() const { return Types.size(); }
  const InlineAsmExpr *getOutputOperand() const { return OutputOp; }
  const InlineAsmExpr *getPredOutputOperand() const { return PredOutputOp; }
  ArrayRef<InlineAsmExpr *> getInputOperands() const { return InputOps; }
  const InlineAsmExpr *getInputOperand(unsigned I) const {
    return getInputOperands()[I];
  }
  size_t getNumInputOperands() const { return InputOps.size(); }
  attr_range attrs() const {
    return attr_range(Attributes.begin(), Attributes.end());
  }
  type_range types() const { return type_range(Types.begin(), Types.end()); }
  op_range input_operands() const { return op_range(getInputOperands()); }
  static bool classof(const InlineAsmStmt *S) {
    return InstructionClass <= S->getStmtClass();
  }
};

/// This represents a device conditional instruction, e.g. instruction @%p
/// mov.s32 %0, 1; has a guard predicate '%p'.
class InlineAsmConditionalInstruction : public InlineAsmStmt {

  /// !Pred
  bool Not;

  /// Guard predicate expression.
  const InlineAsmExpr *Pred;

  /// The sub instruction.
  const InlineAsmInstruction *Instruction;

public:
  InlineAsmConditionalInstruction(bool IsNeg, const InlineAsmExpr *Pred,
                                  const InlineAsmInstruction *Inst)
      : InlineAsmStmt(ConditionalInstructionClass), Not(IsNeg), Pred(Pred),
        Instruction(Inst) {}

  const InlineAsmExpr *getPred() const { return Pred; }
  const InlineAsmInstruction *getInstruction() const { return Instruction; }
  bool hasNot() const { return Not; }

  static bool classof(const InlineAsmStmt *S) {
    return S->getStmtClass() == ConditionalInstructionClass;
  }
};

/// Captures information about "declaration specifiers".
struct InlineAsmDeclarationSpecifier {
  /// The state space, e.g. '.reg', '.global', '.local', etc.
  asmtok::TokenKind StateSpace = asmtok::unknown;

  /// The vector type kind to specific the fixed vector size, e.g. '.v2', '.v4',
  /// etc.
  asmtok::TokenKind VectorTypeKind = asmtok::unknown;

  /// The alignment specificed by '.align' attribute.
  InlineAsmIntegerLiteral *Alignment = nullptr;

  InlineAsmBuiltinType *BaseType = nullptr;

  /// The type represented by declaration specifier
  InlineAsmType *Type = nullptr;
};

/// Represents a variable declaration that came out of a declarator.
struct InlineAsmDeclarator {
  InlineAsmDeclarationSpecifier DeclSpec;
  bool isParameterizedNames = false;
};

/// DeclStmt - Adaptor class for mixing declarations with statements and
/// expressions. For example, CompoundStmt mixes statements, expressions
/// and declarations (variables, types).
class InlineAsmDeclStmt : public InlineAsmStmt {
  InlineAsmDeclarationSpecifier DeclSpec;
  SmallVector<InlineAsmDecl *, 4> DeclGroup;

public:
  InlineAsmDeclStmt(InlineAsmDeclarationSpecifier DS,
                    ArrayRef<InlineAsmDecl *> Decls)
      : InlineAsmStmt(DeclStmtClass), DeclSpec(DS), DeclGroup(Decls) {}

  unsigned getNumDecl() const { return DeclGroup.size(); }
  const InlineAsmDecl *getDecl(unsigned I) const { return DeclGroup[I]; }

  using decl_range =
      llvm::iterator_range<SmallVector<InlineAsmDecl *, 4>::const_iterator>;

  decl_range decls() const {
    return decl_range(DeclGroup.begin(), DeclGroup.end());
  }

  InlineAsmDeclarationSpecifier getDeclSpec() const { return DeclSpec; }

  static bool classof(const InlineAsmStmt *S) {
    return S->getStmtClass() == DeclStmtClass;
  }
};

/// Base class for the full range of assembler expressions which are
/// needed for parsing.
class InlineAsmExpr : public InlineAsmStmt {
  InlineAsmType *Type;

protected:
  explicit InlineAsmExpr(StmtClass SC, InlineAsmType *Type)
      : InlineAsmStmt(SC), Type(Type) {}

public:
  InlineAsmType *getType() { return Type; }
  const InlineAsmType *getType() const { return Type; }

  static bool classof(const InlineAsmStmt *S) {
    return firstExprConstant <= S->getStmtClass() &&
           lastExprConstant >= S->getStmtClass();
  }
};

/// This represents a binary, octal, decimal, or hexadecimal integer.
class InlineAsmIntegerLiteral : public InlineAsmExpr {
  /// Used to store the integer value.
  llvm::APInt Value;

  /// Used to store the user written integer literal.
  /// It's useful in sycl code generator to print the original
  /// literal.
  StringRef LiteralData;

public:
  InlineAsmIntegerLiteral(InlineAsmType *Type, llvm::APInt Value,
                          StringRef LiteralData)
      : InlineAsmExpr(IntegerLiteralClass, Type), Value(Value),
        LiteralData(LiteralData) {}

  llvm::APInt getValue() const { return Value; }

  StringRef getLiteral() const { return LiteralData; }

  static bool classof(const InlineAsmStmt *S) {
    return S->getStmtClass() == IntegerLiteralClass;
  }
};

/// This represents a floating point literal.
class InlineAsmFloatingLiteral : public InlineAsmExpr {

  /// Used to store the floating point value.
  llvm::APFloat Value;

  /// Used to store the user written integer literal.
  /// If this literal is an exact machine floating literal,
  /// the LiteralData will not include 0[fFdD] prefix.
  ///
  /// It's useful in sycl code generator to print the original
  /// literal.
  StringRef LiteralData;

  /// An exact machine floating literal used to specify IEEE 754
  /// double-precision floating point values, the constant begins with 0d or 0D
  /// followed by 16 hex digits. To specify IEEE 754 single-precision floating
  /// point values, the constant begins with 0f or 0F followed by 8 hex digits.
  ///
  /// 0[fF]{hexdigit}{8}      // single-precision floating point
  /// 0[dD]{hexdigit}{16}     // double-precision floating point
  bool IsExactMachineFloatingLiteral = false;

public:
  InlineAsmFloatingLiteral(InlineAsmType *Type, llvm::APFloat Value,
                           StringRef LiteralData,
                           bool IsExactMachineFloatingLiteral = false)
      : InlineAsmExpr(FloatingLiteralClass, Type), Value(Value),
        LiteralData(LiteralData),
        IsExactMachineFloatingLiteral(IsExactMachineFloatingLiteral) {}

  llvm::APFloat getValue() const { return Value; }

  bool isExactMachineFloatingLiteral() const {
    return IsExactMachineFloatingLiteral;
  }

  StringRef getLiteral() const { return LiteralData; }

  static bool classof(const InlineAsmStmt *S) {
    return S->getStmtClass() == FloatingLiteralClass;
  }
};

/// This represents a reference to a declared variable, label, etc.
class InlineAsmDeclRefExpr : public InlineAsmExpr {

  /// The referenced declaration.
  InlineAsmDecl *Decl;

  /// Used to store the parameterized name variable index.
  /// e.g.
  /// .reg .s32 %p<10>;
  /// mov.s32 %p0, 0; // %p0 has ParameterizedNameIndex equals to 0;
  /// mov.s32 %p1, 1; // %p1 has ParameterizedNameIndex equals to 1;
  unsigned ParameterizedNameIndex = 0;

public:
  InlineAsmDeclRefExpr(InlineAsmVarDecl *D)
      : InlineAsmExpr(DeclRefExprClass, D->getType()), Decl(D) {}

  InlineAsmDeclRefExpr(InlineAsmVarDecl *D, unsigned Idx)
      : InlineAsmExpr(DeclRefExprClass, D->getType()), Decl(D),
        ParameterizedNameIndex(Idx) {}

  const InlineAsmDecl &getDecl() const { return *Decl; }

  bool hasParameterizedName() const {
    if (const auto *Var = dyn_cast<InlineAsmVarDecl>(Decl))
      return Var->isParameterizedNameDecl();
    return false;
  }

  size_t getParameterizedNameIndex() const {
    assert(hasParameterizedName() &&
           "This declaration was not a Parameterized");
    return ParameterizedNameIndex;
  }

  static bool classof(const InlineAsmStmt *S) {
    return S->getStmtClass() == DeclRefExprClass;
  }
};

/// This represents a device asm vector expression.
/// Usually has 2, 4 or 8 elements. e.g.
///
/// ld.global.v4.f32  {a,b,c,d}, [addr+16];
/// mov.b64 {lo,hi}, %x;    // %x is a double; lo,hi are .u32
/// mov.b32 %r1,{x,y,z,w};  // x,y,z,w have type .b8
class InlineAsmVectorExpr : public InlineAsmExpr {
  SmallVector<InlineAsmExpr *, 4> Elements;

public:
  InlineAsmVectorExpr(InlineAsmVectorType *Type,
                      ArrayRef<InlineAsmExpr *> Elements)
      : InlineAsmExpr(VectorExprClass, Type), Elements(Elements) {}

  using element_range =
      llvm::iterator_range<SmallVector<InlineAsmExpr *, 4>::const_iterator>;

  element_range elements() const {
    return element_range(Elements.begin(), Elements.end());
  }

  const InlineAsmExpr *getElement(unsigned I) const { return Elements[I]; }
  unsigned getNumElements() const { return Elements.size(); }

  static bool classof(const InlineAsmStmt *S) {
    return S->getStmtClass() == VectorExprClass;
  }
};

/// This represents a device asm discard expression '_'.
/// e.g. mov.b64 {%r1, _}, %x; // %x is.b64, %r1 is .b32
class InlineAsmDiscardExpr : public InlineAsmExpr {
public:
  InlineAsmDiscardExpr(InlineAsmDiscardType *Any)
      : InlineAsmExpr(DiscardExprClass, Any) {}

  static bool classof(const InlineAsmStmt *S) {
    return S->getStmtClass() == DiscardExprClass;
  }
};

/// This represents a device asm memory operand(except array subscripts).
/// FIXME: Memory operand is not supported now.
/// [var] the name of an addressable variable var.
///
/// [reg] an integer or bit-size type register reg containing a byte address.
///
/// [reg+immOff] a sum of register reg containing a byte address plus a constant
/// integer byte offset (signed, 32-bit).
///
/// [var+immOff] a sum of address of addressable variable var containing a byte
/// address plus a constant integer byte offset (signed, 32-bit).
///
/// [immAddr] an immediate absolute byte address (unsigned, 32-bit).
///
/// (exclude) var[immOff] an array element as described in Arrays as Operands.
class InlineAsmAddressExpr : public InlineAsmExpr {
public:
  enum MemOpKind { Imm, Reg, Var, RegImm, VarImm };

private:
  MemOpKind OpKind;
  InlineAsmDeclRefExpr *SymbolRef;
  InlineAsmIntegerLiteral *ImmAddr;

public:
  InlineAsmAddressExpr(InlineAsmBuiltinType *Type, MemOpKind Kind,
                       InlineAsmDeclRefExpr *Symbol,
                       InlineAsmIntegerLiteral *Imm)
      : InlineAsmExpr(AddressExprClass, Type), OpKind(Kind), SymbolRef(Symbol),
        ImmAddr(Imm) {}

  MemOpKind getMemoryOpKind() const { return OpKind; }
  InlineAsmDeclRefExpr *getSymbol() const { return SymbolRef; }
  InlineAsmIntegerLiteral *getImmAddr() const { return ImmAddr; }

  static bool classof(const InlineAsmStmt *S) {
    return S->getStmtClass() == AddressExprClass;
  }
};

/// This represents a type cast expression.
/// Only allowed cast to s64 or u64.
class InlineAsmCastExpr : public InlineAsmExpr {
  const InlineAsmExpr *SubExpr;

public:
  InlineAsmCastExpr(InlineAsmBuiltinType *Type, const InlineAsmExpr *Op)
      : InlineAsmExpr(CastExprClass, Type), SubExpr(Op) {}

  const InlineAsmExpr *getSubExpr() const { return SubExpr; }

  static bool classof(const InlineAsmStmt *S) {
    return S->getStmtClass() == CastExprClass;
  }
};

/// This represents a parentheses expression, ( expr ).
class InlineAsmParenExpr : public InlineAsmExpr {
  InlineAsmExpr *SubExpr;

public:
  InlineAsmParenExpr(InlineAsmExpr *SubExpr)
      : InlineAsmExpr(ParenExprClass, SubExpr->getType()), SubExpr(SubExpr) {}

  const InlineAsmExpr *getSubExpr() const { return SubExpr; }

  static bool classof(const InlineAsmStmt *S) {
    return S->getStmtClass() == ParenExprClass;
  }
};

/// UnaryOperator - This represents the unary-expressions.
class InlineAsmUnaryOperator : public InlineAsmExpr {
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
  InlineAsmExpr *SubExpr;

public:
  InlineAsmUnaryOperator(Opcode Op, InlineAsmExpr *Expr, InlineAsmType *Type)
      : InlineAsmExpr(UnaryOperatorClass, Type), Op(Op), SubExpr(Expr) {}

public:
  Opcode getOpcode() const { return (Opcode)Op; }

  const InlineAsmExpr *getSubExpr() const { return SubExpr; }

  static bool classof(const InlineAsmStmt *S) {
    return S->getStmtClass() == UnaryOperatorClass;
  }
};

/// A builtin binary operation expression such as "x + y" or "x <= y".
///
/// This expression node kind describes a builtin binary operation,
/// such as "x + y" for integer values "x" and "y". The operands will
/// already have been converted to appropriate types (e.g., by
/// performing promotions or conversions).
class InlineAsmBinaryOperator : public InlineAsmExpr {
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
  InlineAsmExpr *LHS;
  InlineAsmExpr *RHS;

public:
  InlineAsmBinaryOperator(Opcode Op, InlineAsmExpr *LHS, InlineAsmExpr *RHS,
                          InlineAsmType *Type)
      : InlineAsmExpr(BinaryOperatorClass, Type), Op(Op), LHS(LHS), RHS(RHS) {}

public:
  Opcode getOpcode() const { return Op; }
  InlineAsmExpr *getLHS() const { return LHS; }
  InlineAsmExpr *getRHS() const { return RHS; }

  static bool classof(const InlineAsmStmt *S) {
    return S->getStmtClass() == BinaryOperatorClass;
  }
};

/// ConditionalOperator - The ?: ternary operator.
class InlineAsmConditionalOperator : public InlineAsmExpr {
  InlineAsmExpr *Cond;
  InlineAsmExpr *LHS;
  InlineAsmExpr *RHS;

public:
  InlineAsmConditionalOperator(InlineAsmExpr *C, InlineAsmExpr *L,
                               InlineAsmExpr *R, InlineAsmType *Type)
      : InlineAsmExpr(ConditionalOperatorClass, Type), Cond(C), LHS(L), RHS(R) {
  }

  const InlineAsmExpr *getCond() const { return Cond; }
  const InlineAsmExpr *getLHS() const { return LHS; }
  const InlineAsmExpr *getRHS() const { return RHS; }

  static bool classof(const InlineAsmStmt *S) {
    return S->getStmtClass() == ConditionalOperatorClass;
  }
};
} // namespace dpct
} // namespace clang

#endif // CLANG_DPCT_ASM_NODES_H
