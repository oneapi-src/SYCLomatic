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

class InlineAsmType;
class InlineAsmDecl;
class InlineAsmExpr;
class InlineAsmStmt;
class InlineAsmParser;
class InlineAsmIntegerLiteral;

/// The base class of the type hierarchy.
/// Types, once created, are immutable.
class InlineAsmType {
public:
  enum TypeClass { BuiltinClass, VectorClass, DiscardClass };

private:
  TypeClass tClass;

protected:
  InlineAsmType(TypeClass TC) : tClass(TC) {}

public:
  virtual ~InlineAsmType();
  TypeClass getTypeClass() const { return tClass; }

  void *operator new(size_t bytes) noexcept {
    llvm_unreachable("InlineAsmType cannot be allocated with regular 'new'.");
  }

  void operator delete(void *data) noexcept {
    llvm_unreachable("InlineAsmType cannot be released with regular 'delete'.");
  }

  bool isSignedInteger() const;
  bool isUnsignedInteger() const;
};

/// This class is used for builtin types like 'u32'.
class InlineAsmBuiltinType : public InlineAsmType {
public:
  enum TypeKind : uint8_t {
#define BUILTIN_TYPE(X, Y) TK_##X,
#include "AsmTokenKinds.def"
    NUM_TYPES
  };

private:
  TypeKind Kind;

public:
  InlineAsmBuiltinType(TypeKind Kind)
      : InlineAsmType(BuiltinClass), Kind(Kind) {}

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

  static bool classof(const InlineAsmType *T) {
    return T->getTypeClass() == BuiltinClass;
  }
};

// This class is used for device asm vector types.
class InlineAsmVectorType : public InlineAsmType {
public:
  enum VecKind : uint8_t {
#define VECTOR(X, Y) TK_##X,
#include "AsmTokenKinds.def"
  };

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

  /// The declaration identifier.
  InlineAsmIdentifierInfo *Name;

protected:
  InlineAsmDecl(DeclClass DC, InlineAsmIdentifierInfo *Name)
      : dClass(DC), Name(Name) {}

public:
  virtual ~InlineAsmDecl();
  DeclClass getDeclClass() const { return dClass; }
  InlineAsmIdentifierInfo *getDeclName() const { return Name; }

  void *operator new(size_t bytes) noexcept {
    llvm_unreachable("InlineAsmDecl cannot be allocated with regular 'new'.");
  }

  void operator delete(void *data) noexcept {
    llvm_unreachable("InlineAsmDecl cannot be released with regular 'delete'.");
  }
};

/// Represents a variable declaration.
class InlineAsmVariableDecl : public InlineAsmDecl {

  /// The state space of a variable, e.g. '.reg', '.global', '.local', etc.
  InlineAsmIdentifierInfo *StorageClass;

  /// The type of this variable.
  InlineAsmType *Type;

  /// Alignment of this variable, specificed by '.align' attribute.
  unsigned Align;

  /// The num of parameterized names, e.g. %p<10>
  unsigned NumParameterizedNames = 0;

  /// Has '.align' attribute in this variable declaration.
  bool HasAlign;

  /// Has parameterized names in this variable declaration.
  bool IsParameterizedNameDecl = false;

public:
  InlineAsmVariableDecl(InlineAsmIdentifierInfo *Name, InlineAsmType *Type)
      : InlineAsmDecl(VariableDeclClass, Name), Type(Type) {}

  const InlineAsmIdentifierInfo *getStorageClass() const {
    return StorageClass;
  }

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
    DeclStmtClass,
    CompoundStmtClass,
    InstructionClass,
    ConditionalInstructionClass,
    UnaryOperatorClass,
    BinaryOperatorClass,
    ConditionalOperatorClass,
    VectorExprClass,
    DiscardExprClass,
    AddressExprClass,
    CastExprClass,
    ParenExprClass,
    DeclRefExprClass,
    IntegerLiteralClass,
    FloatingLiteralClass
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
class InlineAsmInstruction : public InlineAsmStmt {

  /// The opcode of instruction, must predefined in AsmTokenKinds.def
  /// e.g. asmtok::op_mov, asmtok::op_setp, etc.
  InlineAsmIdentifierInfo *Opcode;

  /// The type attributes in this instruction.
  /// e.g. instruction mov.s32 val, 123; has a type attribute 's32'.
  SmallVector<InlineAsmType *, 4> Types;

  /// This represents arrtibutes like: comparsion operator, rounding modifiers,
  /// ... e.g. instruction setp.eq.s32 has a comparsion operator 'eq'.
  SmallVector<InlineAsmIdentifierInfo *, 4> Attributes;
  InlineAsmExpr *OutputOperand;
  SmallVector<InlineAsmExpr *, 4> InputOperands;

  // Predicate output, e.g. given shfl.sync.up.b32  Ry|p, Rx, 0x1,  0x0,
  // 0xffffffff; p is a predicate output.
  InlineAsmExpr *PredOutput = nullptr;

public:
  InlineAsmInstruction(InlineAsmIdentifierInfo *Op,
                       ArrayRef<InlineAsmType *> Types,
                       ArrayRef<InlineAsmIdentifierInfo *> Attrs,
                       InlineAsmExpr *OutputOp,
                       ArrayRef<InlineAsmExpr *> InputOps, InlineAsmExpr *Pred)
      : InlineAsmStmt(InstructionClass), Opcode(Op), Types(Types),
        Attributes(Attrs), OutputOperand(OutputOp), InputOperands(InputOps),
        PredOutput(Pred) {}

  InlineAsmInstruction(InlineAsmIdentifierInfo *Op,
                       ArrayRef<InlineAsmType *> Types,
                       ArrayRef<InlineAsmIdentifierInfo *> Attrs,
                       InlineAsmExpr *OutputOp,
                       ArrayRef<InlineAsmExpr *> InputOps)
      : InlineAsmStmt(InstructionClass), Opcode(Op), Types(Types),
        Attributes(Attrs), OutputOperand(OutputOp), InputOperands(InputOps) {}

  using type_range =
      llvm::iterator_range<SmallVector<InlineAsmType *, 4>::const_iterator>;
  using attribute_range = llvm::iterator_range<
      SmallVector<InlineAsmIdentifierInfo *, 4>::const_iterator>;
  using input_operand_range =
      llvm::iterator_range<SmallVector<InlineAsmExpr *, 4>::const_iterator>;

  InlineAsmIdentifierInfo *getOpcode() const { return Opcode; }
  const InlineAsmIdentifierInfo *getAttribute(unsigned I) const {
    return Attributes[I];
  }
  const InlineAsmExpr *getOutputOperand() const { return OutputOperand; }
  const InlineAsmExpr *getPredOutputOperand() const { return PredOutput; }
  const InlineAsmExpr *getInputOperand(unsigned I) const {
    return InputOperands[I];
  }
  size_t getNumInputOperands() const { return InputOperands.size(); }

  size_t getNumTypes() const { return Types.size(); }
  InlineAsmType *getType(unsigned I) { return Types[I]; }
  const InlineAsmType *getType(unsigned I) const { return Types[I]; }

  type_range types() const { return type_range(Types.begin(), Types.end()); }

  attribute_range attrs() const {
    return attribute_range(Attributes.begin(), Attributes.end());
  }

  input_operand_range input_operands() const {
    return input_operand_range(InputOperands.begin(), InputOperands.end());
  }

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
    return UnaryOperatorClass <= S->getStmtClass() &&
           S->getStmtClass() <= FloatingLiteralClass;
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
  InlineAsmDeclRefExpr(InlineAsmVariableDecl *D)
      : InlineAsmExpr(DeclRefExprClass, D->getType()), Decl(D) {}

  InlineAsmDeclRefExpr(InlineAsmVariableDecl *D, unsigned Idx)
      : InlineAsmExpr(DeclRefExprClass, D->getType()), Decl(D),
        ParameterizedNameIndex(Idx) {}

  const InlineAsmDecl &getDecl() const { return *Decl; }

  bool hasParameterizedName() const {
    if (const auto *Var = dyn_cast<InlineAsmVariableDecl>(Decl))
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
  InlineAsmExpr *SubExpr;

public:
  InlineAsmAddressExpr(InlineAsmBuiltinType *Type, InlineAsmExpr *SubExpr)
      : InlineAsmExpr(AddressExprClass, Type), SubExpr(SubExpr) {}

  const InlineAsmExpr *getSubExpr() const { return SubExpr; }

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
  const InlineAsmExpr *getLHS() const { return LHS; }
  const InlineAsmExpr *getRHS() const { return RHS; }

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
  InlineAsmDiscardType *DiscardType;

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
  InlineAsmBuiltinType *getBuiltinType(StringRef TypeName);
  InlineAsmBuiltinType *getBuiltinType(InlineAsmBuiltinType::TypeKind Kind);
  InlineAsmBuiltinType *getBuiltinTypeFromTokenKind(asmtok::TokenKind Kind);
  InlineAsmDiscardType *getDiscardType();

  InlineAsmBuiltinType *getS64Type() {
    return getBuiltinType(InlineAsmBuiltinType::TK_s64);
  }

  InlineAsmBuiltinType *getU64Type() {
    return getBuiltinType(InlineAsmBuiltinType::TK_u64);
  }

  InlineAsmBuiltinType *getF32Type() {
    return getBuiltinType(InlineAsmBuiltinType::TK_f32);
  }

  InlineAsmBuiltinType *getF64Type() {
    return getBuiltinType(InlineAsmBuiltinType::TK_f64);
  }
};

/// Introduces a new scope for parsing when meet compound statement.
/// e.g. { stmt stmt }
class InlineAsmScope {
  using DeclSetTy = SmallPtrSet<InlineAsmVariableDecl *, 32>;
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

  void addDecl(InlineAsmVariableDecl *D) { DeclsInScope.insert(D); }

  bool isDeclScope(const InlineAsmVariableDecl *D) const {
    return DeclsInScope.contains(D);
  }

  bool contains(const InlineAsmScope &rhs) const { return Depth < rhs.Depth; }

  /// Lookup a simple declaration in current scope and parent scopes.
  InlineAsmVariableDecl *lookupDecl(InlineAsmIdentifierInfo *II) const;

  /// Lookup a parameterized name declaration in current scope and parent
  /// scopes.
  InlineAsmVariableDecl *
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

  InlineAsmDeclResult addInlineAsmOperands(StringRef Operand,
                                           StringRef Constraint);

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
