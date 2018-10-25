//===--- ASTTraversal.cpp --------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "ASTTraversal.h"

#include "SaveNewFiles.h"
#include "Utility.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "llvm/ADT/StringSet.h"
#include <string>
#include <utility>
#include <iostream>

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::syclct;

extern std::string CudaPath;

void IncludesCallbacks::InclusionDirective(
    SourceLocation HashLoc, const Token &IncludeTok, StringRef FileName,
    bool IsAngled, CharSourceRange FilenameRange, const FileEntry *File,
    StringRef SearchPath, StringRef RelativePath, const Module *Imported,
    SrcMgr::CharacteristicKind FileType) {

  if (!SM.isWrittenInMainFile(HashLoc)) {
    return;
  }

  // Insert SYCL headers.
  if (!SyclHeaderInserted) {

    std::string Replacement = std::string("#include <CL/sycl.hpp>") +
                              getNL(FilenameRange.getEnd(), SM) +
                              "#include <syclct/syclct.hpp>" +
                              getNL(FilenameRange.getEnd(), SM);
    CharSourceRange InsertRange(SourceRange(HashLoc, HashLoc), false);
    TransformSet.emplace_back(
        new ReplaceInclude(InsertRange, std::move(Replacement)));
    SyclHeaderInserted = true;
  }

  std::string IncludePath = SearchPath;
  makeCanonical(IncludePath);

  std::string IncludingFile = SM.getFilename(HashLoc);

  // replace "#include <math.h>" with <cmath>
  if (IsAngled && FileName.compare(StringRef("math.h")) == 0) {
    TransformSet.emplace_back(new ReplaceInclude(
        CharSourceRange(SourceRange(HashLoc, FilenameRange.getEnd()),
                        /*IsTokenRange=*/false),
        "#include <cmath>"));
  }
  if (!isChildPath(CudaPath, IncludePath) &&
      IncludePath.compare(0 , 15,
          "/usr/local/cuda",
          15) /*CudaPath detection have not consider soft link, here do special
          for /usr/local/cuda*/) {
    // if <cuda_runtime.h>, no matter where it from, replace with sycl header
    if (!(IsAngled && FileName.compare(StringRef("cuda_runtime.h")) == 0))
      return;
  }

  // Multiple CUDA headers in an including file will be replaced with one
  // include of the SYCL header.
  if ((SeenFiles.find(IncludingFile) == end(SeenFiles)) &&
      (!SyclHeaderInserted)) {
    SeenFiles.insert(IncludingFile);
    std::string Replacement = std::string("<CL/sycl.hpp>") +
                              getNL(FilenameRange.getEnd(), SM) +
                              "#include <syclct/syclct.hpp>";
    TransformSet.emplace_back(
        new ReplaceInclude(FilenameRange, std::move(Replacement)));
    SyclHeaderInserted = true;
  } else {
    // Replace the complete include directive with an empty string.
    TransformSet.emplace_back(new ReplaceInclude(
        CharSourceRange(SourceRange(HashLoc, FilenameRange.getEnd()),
                        /*IsTokenRange=*/false),
        ""));
  }
}

void IterationSpaceBuiltinRule::registerMatcher(MatchFinder &MF) {
  // TODO: check that threadIdx is not a local variable.
  MF.addMatcher(
      memberExpr(hasObjectExpression(opaqueValueExpr(hasSourceExpression(
                     declRefExpr(to(varDecl(hasAnyName("threadIdx", "blockDim",
                                                       "blockIdx", "gridDim"))
                                        .bind("varDecl")))))))
          .bind("memberExpr"),
      this);
}

void IterationSpaceBuiltinRule::run(const MatchFinder::MatchResult &Result) {
  const MemberExpr *ME = getNodeAsType<MemberExpr>(Result, "memberExpr");
  if (!ME)
    return;
  const VarDecl *VD = getNodeAsType<VarDecl>(Result, "varDecl", false);
  assert(ME && VD && "Unknown result");

  ValueDecl *Field = ME->getMemberDecl();
  StringRef FieldName = Field->getName();
  unsigned Dimension;

  if (FieldName == "__fetch_builtin_x")
    Dimension = 0;
  else if (FieldName == "__fetch_builtin_y")
    Dimension = 1;
  else if (FieldName == "__fetch_builtin_z")
    Dimension = 2;
  else
    llvm_unreachable("Unknown field name");

  // TODO: do not assume the argument is named "item"
  std::string Replacement = "item";
  StringRef BuiltinName = VD->getName();

  if (BuiltinName == "threadIdx")
    Replacement += ".get_local_id(";
  else if (BuiltinName == "blockDim")
    Replacement += ".get_local_range().get(";
  else if (BuiltinName == "blockIdx")
    Replacement += ".get_group(";
  else if (BuiltinName == "gridDim")
    Replacement += ".get_group_range(";
  else
    llvm_unreachable("Unknown builtin variable");

  Replacement += std::to_string(Dimension);
  Replacement += ")";
  emplaceTransformation(new ReplaceStmt(ME, std::move(Replacement)));
}

REGISTER_RULE(IterationSpaceBuiltinRule)

void ErrorHandlingIfStmtRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      // Match if-statement that has no else and has a condition of either an
      // operator!= or a variable of type enum.
      ifStmt(unless(hasElse(anything())),
             hasCondition(
                 anyOf(binaryOperator(hasOperatorName("!=")).bind("op!="),
                       ignoringImpCasts(
                           declRefExpr(hasType(hasCanonicalType(enumType())))
                               .bind("var")))))
          .bind("errIf"),
      this);
  MF.addMatcher(
      // Match if-statement that has no else and has a condition of
      // operator==.
      ifStmt(unless(hasElse(anything())),
             hasCondition(binaryOperator(hasOperatorName("==")).bind("op==")))
          .bind("errIfSpecial"),
      this);
}

static bool isVarRef(const Expr *E) {
  if (auto D = dyn_cast<DeclRefExpr>(E))
    return isa<VarDecl>(D->getDecl());
  else
    return false;
}

static std::string getVarType(const Expr *E) {
  return E->getType().getCanonicalType().getUnqualifiedType().getAsString();
}

static bool isCudaFailureCheck(const BinaryOperator *Op, bool IsEq = false) {
  auto Lhs = Op->getLHS()->IgnoreImplicit();
  auto Rhs = Op->getRHS()->IgnoreImplicit();

  const Expr *Literal = nullptr;
  if (isVarRef(Lhs) && getVarType(Lhs) == "enum cudaError")
    Literal = Rhs;
  else if (isVarRef(Rhs) && getVarType(Rhs) == "enum cudaError")
    Literal = Lhs;
  else
    return false;

  if (auto IntLit = dyn_cast<IntegerLiteral>(Literal)) {
    if (IsEq ^ (IntLit->getValue() != 0))
      return false;
  } else if (auto D = dyn_cast<DeclRefExpr>(Literal)) {
    auto EnumDecl = dyn_cast<EnumConstantDecl>(D->getDecl());
    if (!EnumDecl)
      return false;
    // Check for cudaSuccess or CUDA_SUCCESS.
    if (IsEq ^ (EnumDecl->getInitVal() != 0))
      return false;
  } else {
    // The expression is neither an int literal nor an enum value.
    return false;
  }

  return true;
}

static bool isCudaFailureCheck(const DeclRefExpr *E) {
  return isVarRef(E) && getVarType(E) == "enum cudaError";
}

void ErrorHandlingIfStmtRule::run(const MatchFinder::MatchResult &Result) {
  static std::vector<std::string> NameList = {"errIf", "errIfSpecial"};
  const IfStmt *If = getNodeAsType<IfStmt>(Result, "errIf");
  if (!If)
    if (!(If = getNodeAsType<IfStmt>(Result, "errIfSpecial")))
      return;
  auto EmitNotRemoved = [&](SourceLocation SL, const Stmt *R) {
    report(SL, Diagnostics::STMT_NOT_REMOVED,
           getStmtSpelling(R, *Result.Context).c_str());
  };
  auto isErrorHandlingSafeToRemove = [&](const Stmt *S) {
    if (const auto *CE = dyn_cast<CallExpr>(S)) {
      if (!CE->getDirectCallee()) {
        EmitNotRemoved(S->getSourceRange().getBegin(), S);
        return false;
      }
      auto Name = CE->getDirectCallee()->getNameAsString();
      static const llvm::StringSet<> SafeCallList = {
          "printf", "puts", "exit", "cudaDeviceReset", "fprintf"};
      if (SafeCallList.find(Name) == SafeCallList.end()) {
        EmitNotRemoved(S->getSourceRange().getBegin(), S);
        return false;
      }
#if 0
    //TODO: enable argument check
    for (const auto *S : CE->arguments()) {
      if (!isErrorHandlingSafeToRemove(S->IgnoreImplicit()))
        return false;
    }
#endif
      return true;
    }
#if 0
  //TODO: enable argument check
  else if (isa <DeclRefExpr>(S))
    return true;
  else if (isa<IntegerLiteral>(S))
    return true;
  else if (isa<StringLiteral>(S))
    return true;
#endif
    EmitNotRemoved(S->getSourceRange().getBegin(), S);
    return false;
  };

  auto isErrorHandling = [&](const Stmt *Block) {
    if (!isa<CompoundStmt>(Block))
      return isErrorHandlingSafeToRemove(Block);
    const CompoundStmt *CS = cast<CompoundStmt>(Block);
    for (const auto *S : CS->children()) {
      if (!isErrorHandlingSafeToRemove(S->IgnoreImplicit())) {
        return false;
      }
    }
    return true;
  };

  if (![&] {
        if (auto Op = getNodeAsType<BinaryOperator>(Result, "op!=")) {
          if (!isCudaFailureCheck(Op))
            return false;
        } else if (auto Op = getNodeAsType<BinaryOperator>(Result, "op==")) {
          if (!isCudaFailureCheck(Op, true))
            return false;
          report(Op->getBeginLoc(), Diagnostics::IFSTMT_SPECIAL_CASE,
                 getStmtSpelling(Op, *Result.Context).c_str());
        } else {
          auto CondVar = getNodeAsType<DeclRefExpr>(Result, "var");
          if (!isCudaFailureCheck(CondVar))
            return false;
        }
        // We know that it's error checking condition, check the body
        if (!isErrorHandling(If->getThen())) {
          report(If->getSourceRange().getBegin(),
                 Diagnostics::IFSTMT_NOT_REMOVED);

          return false;
        }
        return true;
      }()) {

    return;
  }

  emplaceTransformation(new ReplaceStmt(If, ""));
}

REGISTER_RULE(ErrorHandlingIfStmtRule)

void AlignAttrsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(cxxRecordDecl(hasAttr(attr::Aligned)).bind("classDecl"), this);
}

void AlignAttrsRule::run(const MatchFinder::MatchResult &Result) {
  auto C = getNodeAsType<CXXRecordDecl>(Result, "classDecl");
  if (!C)
    return;
  auto &AV = C->getAttrs();

  for (auto A : AV) {
    if (A->getKind() == attr::Aligned) {
      auto SM = Result.SourceManager;
      auto ExpB = SM->getExpansionLoc(A->getLocation());
      if (!strncmp(SM->getCharacterData(ExpB), "__align__(", 10))
        emplaceTransformation(new ReplaceToken(ExpB, "__sycl_align__"));
    }
  }
}

REGISTER_RULE(AlignAttrsRule)

void FunctionAttrsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      functionDecl(anyOf(hasAttr(attr::CUDAGlobal), hasAttr(attr::CUDADevice),
                         hasAttr(attr::CUDAHost)))
          .bind("functionDecl"),
      this);
}

void FunctionAttrsRule::run(const MatchFinder::MatchResult &Result) {
  const FunctionDecl *FD = getNodeAsType<FunctionDecl>(Result, "functionDecl");
  if (!FD)
    return;
  const AttrVec &AV = FD->getAttrs();

  for (const Attr *A : AV) {
    attr::Kind AK = A->getKind();
    if (!A->isImplicit() && (AK == attr::CUDAGlobal || AK == attr::CUDADevice ||
                             AK == attr::CUDAHost))
      emplaceTransformation(new RemoveAttr(A));
  }
}

REGISTER_RULE(FunctionAttrsRule)

// Rule for types replacements in var. declarations.
void TypeInVarDeclRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(varDecl(anyOf(hasType(cxxRecordDecl(hasName("cudaDeviceProp"))),
                              hasType(enumDecl(hasName("cudaError"))),
                              hasType(typedefDecl(hasName("cudaError_t"))),
                              hasType(typedefDecl(hasName("dim3")))))
                    .bind("TypeInVarDecl"),
                this);
}

void TypeInVarDeclRule::run(const MatchFinder::MatchResult &Result) {
  const VarDecl *D = getNodeAsType<VarDecl>(Result, "TypeInVarDecl");
  if (!D)
    return;
  const clang::Type *Type = D->getTypeSourceInfo()->getTypeLoc().getTypePtr();

  if (dyn_cast<SubstTemplateTypeParmType>(Type)) {
    return;
  }

  std::string TypeName =
      Type->getCanonicalTypeInternal().getBaseTypeIdentifier()->getName().str();
  auto Search = MapNames::TypeNamesMap.find(TypeName);
  if (Search == MapNames::TypeNamesMap.end()) {
    // TODO report translation error
    return;
  }
  std::string Replacement = Search->second;
  emplaceTransformation(new ReplaceTypeInVarDecl(D, std::move(Replacement)));
}

REGISTER_RULE(TypeInVarDeclRule)

// Rule for types replacements in var. declarations.
void SyclStyleVectorRule::registerMatcher(MatchFinder &MF) {
  // basic:  eg. int2 xx
  MF.addMatcher(varDecl(hasType(typedefDecl(hasName("int2")))).bind("BasicVar"),
                this);

  // pointer: eg. int2 * xx
  MF.addMatcher(
      varDecl(hasType(pointsTo(typedefDecl(hasName("int2"))))).bind("PtrVar"),
      this);
  // array: eg. int2 array_[xx]
  MF.addMatcher(varDecl(hasType(arrayType(hasElementType(typedefType(
                            hasDeclaration(typedefDecl(hasName("int2"))))))))
                    .bind("ArrayVar"),
                this);

  // int2.x/y/z => int2.x()/y()/z()
  MF.addMatcher(
      memberExpr(
          hasObjectExpression(hasType(qualType(hasCanonicalType(
              recordType(hasDeclaration(cxxRecordDecl(hasName("int2")))))))))
          .bind("VecMemberExpr"),
      this);
}

void SyclStyleVectorRule::run(const MatchFinder::MatchResult &Result) {
  if (const VarDecl *D = getNodeAsType<VarDecl>(Result, "BasicVar")) {

    const clang::Type *Type = D->getTypeSourceInfo()->getTypeLoc().getTypePtr();

    if (dyn_cast<SubstTemplateTypeParmType>(Type)) {
      return;
    }

    std::string TypeName = Type->getCanonicalTypeInternal()
                               .getBaseTypeIdentifier()
                               ->getName()
                               .str();
    auto Search = MapNames::TypeNamesMap.find(TypeName);
    if (Search == MapNames::TypeNamesMap.end()) {
      // TODO report translation error
      return;
    }
    std::string Replacement = Search->second;
    emplaceTransformation(new ReplaceTypeInVarDecl(D, std::move(Replacement)));
  }
  if (const VarDecl *D = getNodeAsType<VarDecl>(Result, "PtrVar")) {
    const clang::Type *Type = D->getTypeSourceInfo()->getTypeLoc().getTypePtr();

    std::string TypeName = Type->getCanonicalTypeInternal()
                               .getBaseTypeIdentifier()
                               ->getName()
                               .str();
    auto Search = MapNames::TypeNamesMap.find(TypeName);
    if (Search == MapNames::TypeNamesMap.end()) {
      // TODO report translation error
      return;
    }
    std::string Replacement = "cl::sycl::";

    emplaceTransformation(
        new InsertNameSpaceInVarDecl(D, std::move(Replacement)));
  }
  if (const VarDecl *D = getNodeAsType<VarDecl>(Result, "ArrayVar")) {
    const clang::Type *Type = D->getTypeSourceInfo()->getTypeLoc().getTypePtr();

    std::string TypeName = Type->getCanonicalTypeInternal()
                               .getBaseTypeIdentifier()
                               ->getName()
                               .str();
    auto Search = MapNames::TypeNamesMap.find(TypeName);
    if (Search == MapNames::TypeNamesMap.end()) {
      // TODO report translation error
      return;
    }
    std::string Replacement = "cl::sycl::";
    emplaceTransformation(
        new InsertNameSpaceInVarDecl(D, std::move(Replacement)));
  }
  if (const MemberExpr *ME =
          getNodeAsType<MemberExpr>(Result, "VecMemberExpr")) {
    auto Search = MemberNamesMap.find(ME->getMemberNameInfo().getAsString());
    if (Search == MemberNamesMap.end()) {
      // TODO report translation error
      return;
    }
    emplaceTransformation(new RenameFieldInMemberExpr(ME, Search->second + ""));
  }
}

REGISTER_RULE(SyclStyleVectorRule)

void SyclStyleVectorCtorRule::registerMatcher(MatchFinder &MF) {
  // Find sycl sytle vector:eg.int2 constructors which are part of different
  // casts (representing different syntaxes). This includes copy constructors.
  // All constructors will be visited once.
  MF.addMatcher(cxxConstructExpr(hasType(typedefDecl(hasName("int2"))),
                                 hasParent(cxxFunctionalCastExpr().bind(
                                     "int2CtorFuncCast"))),
                this);
  MF.addMatcher(
      cxxConstructExpr(hasType(typedefDecl(hasName("int2"))),
                       hasParent(cStyleCastExpr().bind("int2CtorCCast"))),
      this);

  // translate utility for vector type: eg: make_int2
  MF.addMatcher(callExpr(callee(functionDecl(hasAnyName("make_int2"))))
                    .bind("VecUtilFunc"),
                this);
  // (int2 *)&xxx;
  MF.addMatcher(cStyleCastExpr(hasType(pointsTo(typedefDecl(hasName("int2")))))
                    .bind("int2PtrCast"),
                this);
  // sizeof(int2)
  MF.addMatcher(
      unaryExprOrTypeTraitExpr(allOf(hasArgumentOfType(asString("int2")),
                                     has(qualType(hasCanonicalType(type())))))
          .bind("int2Sizeof"),
      this);
}

// Determines which case of construction applies and creates replacements for
// the syntax. Returns the constructor node and a boolean indicating if a
// closed brace needs to be appended.
void SyclStyleVectorCtorRule::run(const MatchFinder::MatchResult &Result) {
  // Most commonly used syntax cases are checked first.
  if (auto Cast =
          getNodeAsType<CXXFunctionalCastExpr>(Result, "int2CtorFuncCast")) {
    // int2 a = int2(1); // function style cast
    // int2 b = int2(a); // copy constructor
    // func(int(1), int2(a));
    emplaceTransformation(
        new ReplaceToken(Cast->getBeginLoc(), "cl::sycl::int2"));
  } else if (auto Cast =
                 getNodeAsType<CStyleCastExpr>(Result, "int2CtorCCast")) {
    // int2 a = (int2)1;
    // int2 b = (int2)a; // copy constructor
    // func((int2)1, (int2)a);
    emplaceTransformation(new ReplaceCCast(Cast, "(cl::sycl::int2)"));
  } else if (const CallExpr *CE =
                 getNodeAsType<CallExpr>(Result, "VecUtilFunc")) {

    std::string FuncName =
        CE->getDirectCallee()->getNameInfo().getName().getAsString();
    if (FuncName == "make_int2") {
      emplaceTransformation(new ReplaceStmt(CE->getCallee(), "cl::sycl::int2"));
    } else {
      llvm_unreachable("Unknown function name");
    }
  } else if (const CStyleCastExpr *CPtrCast =
                 getNodeAsType<CStyleCastExpr>(Result, "int2PtrCast")) {
    emplaceTransformation(
        new InsertNameSpaceInCastExpr(CPtrCast, "cl::sycl::"));
  } else if (const UnaryExprOrTypeTraitExpr *ExprSizeof =
                 getNodeAsType<UnaryExprOrTypeTraitExpr>(Result,
                                                         "int2Sizeof")) {
    if (ExprSizeof->isArgumentType()) {
      emplaceTransformation(new InsertText(ExprSizeof->getArgumentTypeInfo()
                                               ->getTypeLoc()
                                               .getSourceRange()
                                               .getBegin(),
                                           "cl::sycl::"));
    }
  }
  return;
}

REGISTER_RULE(SyclStyleVectorCtorRule)

void ReplaceDim3CtorRule::registerMatcher(MatchFinder &MF) {
  // Find dim3 constructors which are part of different casts (representing
  // different syntaxes). This includes copy constructors. All constructors
  // will be visited once.
  MF.addMatcher(cxxConstructExpr(hasType(typedefDecl(hasName("dim3"))),
                                 hasParent(cxxFunctionalCastExpr().bind(
                                     "dim3CtorFuncCast"))),
                this);
  MF.addMatcher(
      cxxConstructExpr(hasType(typedefDecl(hasName("dim3"))),
                       hasParent(cStyleCastExpr().bind("dim3CtorCCast"))),
      this);
  MF.addMatcher(cxxConstructExpr(
                    hasType(typedefDecl(hasName("dim3"))),
                    hasParent(implicitCastExpr().bind("dim3CtorImplicitCast"))),
                this);

  // Find all other dim3 constructors with 3 parameters (not copy ctors).
  MF.addMatcher(cxxConstructExpr(hasType(typedefDecl(hasName("dim3"))),
                                 unless(hasParent(castExpr())),
                                 argumentCountIs(3))
                    .bind("dim3Ctor"),
                this);
}

// Determines which case of construction applies and creates replacements for
// the syntax. Returns the constructor node and a boolean indicating if a
// closed brace needs to be appended.
std::pair<const CXXConstructExpr *, bool>
ReplaceDim3CtorRule::rewriteSyntax(const MatchFinder::MatchResult &Result) {
  const CXXConstructExpr *Ctor = nullptr;
  bool CloseBrace = false;
  if ((Ctor = getNodeAsType<CXXConstructExpr>(Result, "dim3Ctor"))) {
    // dim3 a(1);
    // No syntax needs to be rewritten.
  } else if (auto Cast = getNodeAsType<ImplicitCastExpr>(
                 Result, "dim3CtorImplicitCast")) {
    if ((Ctor = dyn_cast<CXXConstructExpr>(Cast->getSubExpr()))) {
      if (!isa<CXXTemporaryObjectExpr>(Ctor)) {
        // dim3 a = 1;
        // func(1);
        emplaceTransformation(
            new InsertBeforeStmt(Cast, "cl::sycl::range<3>("));
        CloseBrace = true;
      } else {
        // dim3 a = dim3(1, 2); // temporary object expression
        // func(dim(1, 2));
        emplaceTransformation(
            new ReplaceToken(Cast->getBeginLoc(), "cl::sycl::range<3>"));
      }
    }
  } else if (auto Cast = getNodeAsType<CXXFunctionalCastExpr>(
                 Result, "dim3CtorFuncCast")) {
    // dim3 a = dim3(1); // function style cast
    // dim3 b = dim3(a); // copy constructor
    // func(dim(1), dim3(a));
    if ((Ctor = dyn_cast<CXXConstructExpr>(Cast->getSubExpr()))) {
      emplaceTransformation(
          new ReplaceToken(Cast->getBeginLoc(), "cl::sycl::range<3>"));
    }
  } else if (auto Cast =
                 getNodeAsType<CStyleCastExpr>(Result, "dim3CtorCCast")) {
    // dim3 a = (dim3)1;
    // dim3 b = (dim3)a; // copy constructor
    // func((dim)1, (dim3)a);
    if ((Ctor = dyn_cast<CXXConstructExpr>(Cast->getSubExpr()))) {

      emplaceTransformation(new ReplaceCCast(Cast, "cl::sycl::range<3>("));
      CloseBrace = true;
    }
  }

  return {Ctor, CloseBrace};
}

void ReplaceDim3CtorRule::rewriteArglist(
    const std::pair<const CXXConstructExpr *, bool> &CtorCase) {
  auto Ctor = CtorCase.first;
  auto CloseBrace = CtorCase.second;
  if (!Ctor)
    return;
  if (Ctor->getNumArgs() == 1) {
    // Copy Constructor
    if (CloseBrace) {
      emplaceTransformation(new InsertAfterStmt(Ctor->getArg(0), ")"));
    }
    return;
  }

  if (isa<CXXDefaultArgExpr>(Ctor->getArg(0))) {
    // If first is default arg, all three are default arguments -- this is
    // the default ctor: dim3 a;
  } else if (isa<CXXDefaultArgExpr>(Ctor->getArg(1))) {
    if (CloseBrace)
      emplaceTransformation(new InsertAfterStmt(Ctor->getArg(0), ", 1, 1)"));
    else
      emplaceTransformation(new InsertAfterStmt(Ctor->getArg(0), ", 1, 1"));
  } else if (isa<CXXDefaultArgExpr>(Ctor->getArg(2))) {
    emplaceTransformation(new InsertAfterStmt(Ctor->getArg(1), ", 1"));
  }
}

void ReplaceDim3CtorRule::run(const MatchFinder::MatchResult &Result) {
  rewriteArglist(rewriteSyntax(Result));
}

REGISTER_RULE(ReplaceDim3CtorRule)

// rule for dim3 types member fields replacements.
void Dim3MemberFieldsRule::registerMatcher(MatchFinder &MF) {
  // dim3->x/y/z => dim3->operator[](0)/(1)/(2)
  MF.addMatcher(
      memberExpr(
          hasDescendant(declRefExpr(
              hasType(pointerType()),
              to(varDecl(hasType(pointsTo(typedefDecl(hasName("dim3")))))))))
          .bind("Dim3MemberPointerExpr"),
      this);

  // dim3.x/y/z => dim3[0]/[1]/[2]
  MF.addMatcher(
      memberExpr(
          hasObjectExpression(hasType(qualType(hasCanonicalType(
              recordType(hasDeclaration(cxxRecordDecl(hasName("dim3")))))))))
          .bind("Dim3MemberDotExpr"),
      this);
}

void Dim3MemberFieldsRule::run(const MatchFinder::MatchResult &Result) {
  if (const MemberExpr *ME =
          getNodeAsType<MemberExpr>(Result, "Dim3MemberPointerExpr")) {
    auto Search = MapNames::Dim3MemberPointerNamesMap.find(
        ME->getMemberNameInfo().getAsString());
    if (Search != MapNames::Dim3MemberPointerNamesMap.end()) {
      emplaceTransformation(
          new RenameFieldInMemberExpr(ME, Search->second + ""));
    }
  }

  if (const MemberExpr *ME =
          getNodeAsType<MemberExpr>(Result, "Dim3MemberDotExpr")) {
    auto SM = Result.SourceManager;
    clang::SourceLocation Begin(ME->getBeginLoc()), Temp(ME->getEndLoc());
    clang::SourceLocation End(
        clang::Lexer::getLocForEndOfToken(Temp, 0, *SM, LangOptions()));
    std::string Ret =
        std::string(SM->getCharacterData(Begin),
                    SM->getCharacterData(End) - SM->getCharacterData(Begin));

    std::size_t PosisitonOfDot = 0;
    std::size_t Current = Ret.find('.');

    // Find the last position of dot '.'
    while (Current != std::string::npos) {
      PosisitonOfDot = Current;
      Current = Ret.find('.', PosisitonOfDot + 1);
    }

    auto Search = MapNames::Dim3MemberNamesMap.find(
        ME->getMemberNameInfo().getAsString());
    if (Search != MapNames::Dim3MemberNamesMap.end()) {
      emplaceTransformation(
          new RenameFieldInMemberExpr(ME, Search->second + "", PosisitonOfDot));
    }
  }
}

REGISTER_RULE(Dim3MemberFieldsRule)

// Rule for return types replacements.
void ReturnTypeRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      functionDecl(
          returns(hasCanonicalType(
              anyOf(recordType(hasDeclaration(
                        cxxRecordDecl(hasName("cudaDeviceProp")))),
                    enumType(hasDeclaration(enumDecl(hasName("cudaError"))))))))
          .bind("functionDecl"),
      this);
}

void ReturnTypeRule::run(const MatchFinder::MatchResult &Result) {
  const FunctionDecl *FD = getNodeAsType<FunctionDecl>(Result, "functionDecl");
  if (!FD)
    return;
  const clang::Type *Type = FD->getReturnType().getTypePtr();
  std::string TypeName =
      Type->getCanonicalTypeInternal().getBaseTypeIdentifier()->getName().str();
  auto Search = MapNames::TypeNamesMap.find(TypeName);
  if (Search == MapNames::TypeNamesMap.end()) {
    // TODO report translation error
    return;
  }
  std::string Replacement = Search->second;
  emplaceTransformation(new ReplaceReturnType(FD, std::move(Replacement)));
}

REGISTER_RULE(ReturnTypeRule)

// Rule for cudaDeviceProp variables.
void DevicePropVarRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      memberExpr(
          hasObjectExpression(hasType(qualType(hasCanonicalType(recordType(
              hasDeclaration(cxxRecordDecl(hasName("cudaDeviceProp")))))))))
          .bind("DevicePropVar"),
      this);
}

void DevicePropVarRule::run(const MatchFinder::MatchResult &Result) {
  const MemberExpr *ME = getNodeAsType<MemberExpr>(Result, "DevicePropVar");
  if (!ME)
    return;
  auto Search = PropNamesMap.find(ME->getMemberNameInfo().getAsString());
  if (Search == PropNamesMap.end()) {
    // TODO report translation error
    return;
  }
  emplaceTransformation(new RenameFieldInMemberExpr(ME, Search->second + "()"));
  if ((Search->second.compare(0, 13, "major_version") == 0) ||
      (Search->second.compare(0, 13, "minor_version") == 0)) {
    report(ME->getBeginLoc(), Comments::VERSION_COMMENT);
  }
  if (Search->second.compare(0, 14, "get_integrated") == 0) {
    report(ME->getBeginLoc(), Comments::NOT_SUPPORT_API_INTEGRATEDORNOT);
  }
}

REGISTER_RULE(DevicePropVarRule)

// Rule for enums constants.
void EnumConstantRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(declRefExpr(to(enumConstantDecl(
                                hasType(enumDecl(hasName("cudaComputeMode"))))))
                    .bind("EnumConstant"),
                this);
}

void EnumConstantRule::run(const MatchFinder::MatchResult &Result) {
  const DeclRefExpr *E = getNodeAsType<DeclRefExpr>(Result, "EnumConstant");
  if (!E)
    return;
  assert(E && "Unknown result");
  auto Search = EnumNamesMap.find(E->getNameInfo().getName().getAsString());
  if (Search == EnumNamesMap.end()) {
    // TODO report translation error
    return;
  }
  emplaceTransformation(new ReplaceStmt(E, "syclct::" + Search->second));
}

REGISTER_RULE(EnumConstantRule)

void ErrorConstantsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      declRefExpr(to(enumConstantDecl(hasType(enumDecl(hasName("cudaError"))))))
          .bind("ErrorConstants"),
      this);
}

void ErrorConstantsRule::run(const MatchFinder::MatchResult &Result) {
  const DeclRefExpr *DE = getNodeAsType<DeclRefExpr>(Result, "ErrorConstants");
  if (!DE)
    return;
  assert(DE && "Unknown result");
  auto *EC = cast<EnumConstantDecl>(DE->getDecl());
  emplaceTransformation(new ReplaceStmt(DE, EC->getInitVal().toString(10)));
}

REGISTER_RULE(ErrorConstantsRule)

void FunctionCallRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(hasAnyName(
                         "cudaGetDeviceCount", "cudaGetDeviceProperties",
                         "cudaDeviceReset", "cudaSetDevice",
                         "cudaDeviceGetAttribute", "cudaDeviceGetP2PAttribute",
                         "cudaGetDevice", "cudaGetLastError",
                         "cudaDeviceSynchronize", "cudaGetErrorString"))),
                     hasParent(compoundStmt())))
          .bind("FunctionCall"),
      this);
  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(hasAnyName(
                         "cudaGetDeviceCount", "cudaGetDeviceProperties",
                         "cudaDeviceReset", "cudaSetDevice",
                         "cudaDeviceGetAttribute", "cudaDeviceGetP2PAttribute",
                         "cudaGetDevice", "cudaGetLastError",
                         "cudaDeviceSynchronize", "cudaGetErrorString"))),
                     unless(hasParent(compoundStmt()))))
          .bind("FunctionCallUsed"),
      this);
}

void FunctionCallRule::run(const MatchFinder::MatchResult &Result) {
  bool IsAssigned = false;
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "FunctionCall");
  if (!CE) {
    if (!(CE = getNodeAsType<CallExpr>(Result, "FunctionCallUsed")))
      return;
    IsAssigned = true;
  }
  assert(CE && "Unknown result");

  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();
  if (FuncName == "cudaGetDeviceCount") {
    std::string ResultVarName = DereferenceArg(CE->getArg(0));
    emplaceTransformation(new InsertBeforeStmt(CE, ResultVarName + " = "));
    emplaceTransformation(
        new ReplaceStmt(CE, "syclct::get_device_manager().device_count()"));
  } else if (FuncName == "cudaGetDeviceProperties") {
    std::string ResultVarName = DereferenceArg(CE->getArg(0));
    emplaceTransformation(new InsertBeforeStmt(CE, ResultVarName + " = "));
    emplaceTransformation(new ReplaceStmt(
        CE->getCallee(), "syclct::get_device_manager().get_device"));
    emplaceTransformation(new RemoveArg(CE, 0));
    emplaceTransformation(new InsertAfterStmt(CE, ".get_device_info()"));
  } else if (FuncName == "cudaDeviceReset") {
    emplaceTransformation(new ReplaceStmt(
        CE, "syclct::get_device_manager().current_device().reset()"));
  } else if (FuncName == "cudaSetDevice") {
    emplaceTransformation(new ReplaceStmt(
        CE->getCallee(), "syclct::get_device_manager().select_device"));
  } else if (FuncName == "cudaDeviceGetAttribute") {
    std::string ResultVarName = DereferenceArg(CE->getArg(0));
    std::string AttributeName = ((const clang::DeclRefExpr *)CE->getArg(1))
                                    ->getNameInfo()
                                    .getName()
                                    .getAsString();
    auto Search = EnumConstantRule::EnumNamesMap.find(AttributeName);
    if (Search == EnumConstantRule::EnumNamesMap.end()) {
      // TODO report translation error
      return;
    }
    emplaceTransformation(new InsertBeforeStmt(CE, ResultVarName + " = "));
    emplaceTransformation(new ReplaceStmt(
        CE->getCallee(), "syclct::get_device_manager().get_device"));
    emplaceTransformation(new RemoveArg(CE, 0));
    emplaceTransformation(new RemoveArg(CE, 1));
    emplaceTransformation(new InsertAfterStmt(CE, "." + Search->second + "()"));
  } else if (FuncName == "cudaDeviceGetP2PAttribute") {
    std::string ResultVarName = DereferenceArg(CE->getArg(0));
    emplaceTransformation(new ReplaceStmt(CE, ResultVarName + " = 0"));
    report(CE->getBeginLoc(), Comments::NOTSUPPORTED, "P2P Access");
  } else if (FuncName == "cudaGetDevice") {
    std::string ResultVarName = DereferenceArg(CE->getArg(0));
    emplaceTransformation(new InsertBeforeStmt(CE, ResultVarName + " = "));
    emplaceTransformation(new ReplaceStmt(
        CE, "syclct::get_device_manager().current_device_id()"));
  } else if (FuncName == "cudaDeviceSynchronize") {
    std::string ReplStr = "syclct::get_device_manager()."
                          "current_device().queues_wait_"
                          "and_throw()";
    if (IsAssigned) {
      ReplStr = "(" + ReplStr + ", 0)";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP);
    }
    emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));

  } else if (FuncName == "cudaGetLastError") {
    emplaceTransformation(new ReplaceStmt(CE, "0"));
  } else if (FuncName == "cudaGetErrorString") {
    emplaceTransformation(
        new InsertBeforeStmt(CE, "\"cudaGetErrorString not supported\"/*"));
    emplaceTransformation(new InsertAfterStmt(CE, "*/"));
  } else {
    llvm_unreachable("Unknown function name");
  }
}

REGISTER_RULE(FunctionCallRule)

void KernelCallRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  MF.addMatcher(cudaKernelCallExpr().bind("kernelCall"), this);
}

void KernelCallRule::run(const ast_matchers::MatchFinder::MatchResult &Result) {
  if (auto KCall = getNodeAsType<CUDAKernelCallExpr>(Result, "kernelCall")) {
    emplaceTransformation(new ReplaceStmt(KCall, ""));
    emplaceTransformation(new ReplaceKernelCallExpr(KCall));
  }
}

REGISTER_RULE(KernelCallRule)

// Memory translation rules live here.
void MemoryTranslationRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(hasAnyName("cudaMalloc", "cudaMemcpy",
                                                    "cudaFree", "cudaMemset"))),
                     hasParent(compoundStmt())))
          .bind("call"),
      this);
  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(hasAnyName("cudaMalloc", "cudaMemcpy",
                                                    "cudaFree", "cudaMemset"))),
                     unless(hasParent(compoundStmt()))))
          .bind("callUsed"),
      this);
}

void MemoryTranslationRule::run(const MatchFinder::MatchResult &Result) {
  bool IsAssigned = false;
  const CallExpr *C = getNodeAsType<CallExpr>(Result, "call");
  if (!C) {
    if (!(C = getNodeAsType<CallExpr>(Result, "callUsed")))
      return;
    IsAssigned = true;
  }
  std::string Name = C->getCalleeDecl()->getAsFunction()->getNameAsString();
  if (IsAssigned) {
    report(C->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP);
    emplaceTransformation(new InsertAfterStmt(C, ", 0)"));
  }
  if (Name == "cudaMalloc") {
    std::string Name = "syclct::sycl_malloc";
    if (IsAssigned) {
      Name = "(" + Name;
    }
    std::vector<const Expr *> Args{C->getArg(0), C->getArg(1)};
    emplaceTransformation(
        new ReplaceCallExpr(C, std::move(Name), std::move(Args)));
  } else if (Name == "cudaMemcpy") {
    std::string Name = "syclct::sycl_memcpy";
    if (IsAssigned) {
      Name = "(" + Name;
    }
    // Input:
    // cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    // cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
    // cudaMemcpy(x_A, y_A, size, someDynamicCudaMemcpyKindValue);
    //
    // Desired output:
    // sycl_memcpy<float>(d_A, h_A, numElements);
    // sycl_memcpy_back<float>(h_A, d_A, numElements);
    // sycl_memcpy<float>(x_A, y_A, numElements,
    // someDynamicCudaMemcpyKindValue);
    //
    // Current output:
    // sycl_memcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    // sycl_memcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
    // sycl_memcpy(x_A, y_A, size, someDynamicCudaMemcpyKindValue);

    // Translate C->getArg(3) if this is enum constant.
    // TODO: this is a hack until we get pass ordering and make
    // different passes work with each other well together.
    const Expr *Direction = C->getArg(3);
    std::string DirectionName;
    const DeclRefExpr *DD = dyn_cast_or_null<DeclRefExpr>(Direction);
    if (DD && isa<EnumConstantDecl>(DD->getDecl())) {
      DirectionName = DD->getNameInfo().getName().getAsString();
      auto Search = EnumConstantRule::EnumNamesMap.find(DirectionName);
      assert(Search != EnumConstantRule::EnumNamesMap.end());
      Direction = nullptr;
      DirectionName = "syclct::" + Search->second;
    }
    std::vector<const Expr *> Args{C->getArg(0), C->getArg(1), C->getArg(2),
                                   Direction};
    std::vector<std::string> NewTypes{"(void*)", "(void*)", "", DirectionName};
    emplaceTransformation(new ReplaceCallExpr(
        C, std::move(Name), std::move(Args), std::move(NewTypes)));
  } else if (Name == "cudaMemset") {
    std::string Name = "syclct::sycl_memset";
    if (IsAssigned) {
      Name = "(" + Name;
    }
    std::vector<const Expr *> Args{C->getArg(0), C->getArg(1), C->getArg(2)};
    std::vector<std::string> NewTypes{"(void*)", "(int)", "(size_t)"};

    emplaceTransformation(
        new ReplaceCallExpr(C, std::move(Name), std::move(Args), NewTypes));

  } else if (Name == "cudaFree") {
    std::string Name = "syclct::sycl_free";
    if (IsAssigned) {
      Name = "(" + Name;
    }
    std::vector<const Expr *> Args{C->getArg(0)};
    emplaceTransformation(
        new ReplaceCallExpr(C, std::move(Name), std::move(Args)));
  }
}

REGISTER_RULE(MemoryTranslationRule)
static const CXXConstructorDecl *getIfConstructorDecl(const Decl *ND) {
  if (const auto *Tmpl = dyn_cast<FunctionTemplateDecl>(ND))
    ND = Tmpl->getTemplatedDecl();
  return dyn_cast<CXXConstructorDecl>(ND);
}

// Translation rule for Inserting try-catch around functions.
class ErrorTryCatchRule : public NamedTranslationRule<ErrorTryCatchRule> {
  std::unordered_set<unsigned> Insertions;

public:
  ErrorTryCatchRule() { SetRuleProperty(ApplyToCudaFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override {
    MF.addMatcher(functionDecl(hasBody(compoundStmt())).bind("functionDecl"),
                  this);
  }
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override {
    const FunctionDecl *FD =
        getNodeAsType<FunctionDecl>(Result, "functionDecl");
    if (!FD)
      return;
    for (const auto *Attr : FD->attrs()) {
      attr::Kind AK = Attr->getKind();
      if (AK == attr::CUDAGlobal || AK == attr::CUDADevice)
        return;
    }

    auto BodySLoc = FD->getBody()->getSourceRange().getBegin().getRawEncoding();
    if (Insertions.find(BodySLoc) != Insertions.end())
      return;
    Insertions.insert(BodySLoc);

    // First check if this is a constructor decl
    if (const CXXConstructorDecl *CDecl = getIfConstructorDecl(FD))
      emplaceTransformation(new InsertBeforeCtrInitList(CDecl, "try "));
    else
      emplaceTransformation(new InsertBeforeStmt(FD->getBody(), "try "));

    emplaceTransformation(new InsertAfterStmt(
        FD->getBody(), "\ncatch (cl::sycl::exception const &exc) {\n"
                       "  std::cerr << exc.what() << \"EOE at line \" << "
                       "__LINE__ << std::endl;\n"
                       "  std::exit(1);\n"
                       "}"));
  }
};

REGISTER_RULE(ErrorTryCatchRule)

void KernelIterationSpaceRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(functionDecl(hasAttr(attr::CUDAGlobal)).bind("functionDecl"),
                this);
}

void KernelIterationSpaceRule::run(const MatchFinder::MatchResult &Result) {
  if (auto FD = getNodeAsType<FunctionDecl>(Result, "functionDecl"))
    emplaceTransformation(new InsertArgument(FD, "cl::sycl::nd_item<3> item"));
}
REGISTER_RULE(KernelIterationSpaceRule)

void UnnamedTypesRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      cxxRecordDecl(unless(has(cxxRecordDecl(isImplicit()))), hasDefinition())
          .bind("unnamedType"),
      this);
}

void UnnamedTypesRule::run(const MatchFinder::MatchResult &Result) {
  auto D = getNodeAsType<CXXRecordDecl>(Result, "unnamedType");
  if (D && D->getName().empty())
    emplaceTransformation(new InsertClassName(D));
}

REGISTER_RULE(UnnamedTypesRule)

void MathFunctionsRule::registerMatcher(MatchFinder &MF) {
  std::vector<std::string> FunctionNames;
  for (auto Function : FunctionNamesMap)
    FunctionNames.push_back(Function.first);

  MF.addMatcher(
      callExpr(callee(functionDecl(
                   internal::Matcher<NamedDecl>(
                       new internal::HasNameMatcher(FunctionNames)),
                   unless(hasDeclContext(namespaceDecl(anything()))))))
          .bind("math"),
      this);
}

void MathFunctionsRule::run(const MatchFinder::MatchResult &Result) {
  static auto End = FunctionNamesMap.end();
  auto C = getNodeAsType<CallExpr>(Result, "math");
  if (C) {
    auto Name = FunctionNamesMap.find(C->getDirectCallee()->getName().str());
    if (Name != End)
      emplaceTransformation(
          new ReplaceToken(C->getBeginLoc(), std::string(Name->second)));
  }
}

REGISTER_RULE(MathFunctionsRule)

void SyncThreadsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(callExpr(callee(functionDecl(hasAnyName("__syncthreads"))))
                    .bind("syncthreads"),
                this);
}

void SyncThreadsRule::run(const MatchFinder::MatchResult &Result) {
  if (auto CE = getNodeAsType<CallExpr>(Result, "syncthreads")) {
    // TODO: Variable name "item" comes from KernelIterationSpaceRule, need a
    // mechanism to check/guarantee the consistency.
    std::string Replacement = "item.barrier()";
    emplaceTransformation(new ReplaceStmt(CE, std::move(Replacement)));
  }
}

REGISTER_RULE(SyncThreadsRule)

void ASTTraversalManager::matchAST(ASTContext &Context, TransformSetTy &TS) {
  this->Context = &Context;
  for (auto &I : Storage) {
    I->registerMatcher(Matchers);
    if (auto TR = dyn_cast<TranslationRule>(&*I)) {
      TR->TM = this;
      TR->setTransformSet(TS);
    }
  }
  Matchers.matchAST(Context);
}

void ASTTraversalManager::emplaceAllRules(int SourceFileFlag) {
  std::vector<std::vector<std::string>> Rules;

  for (auto &F : ASTTraversalMetaInfo::getConstructorTable()) {

    auto RuleObj = (TranslationRule *)F.second();
    CommonRuleProperty RuleProperty = RuleObj->GetRuleProperty();

    auto RType = RuleProperty.RType;
    auto RulesDependon = RuleProperty.RulesDependon;

    if (RType & SourceFileFlag) {
      std::string CurrentRuleName = ASTTraversalMetaInfo::getName(F.first);
      std::vector<std::string> Vec;
      Vec.push_back(CurrentRuleName);
      for (auto const &RuleName : RulesDependon) {
        Vec.push_back(RuleName);
      }
      Rules.push_back(Vec);
    }
  }

  std::vector<std::string> SortedRules = ruleTopoSort(Rules);

  for (std::vector<std::string>::reverse_iterator it = SortedRules.rbegin();
       it != SortedRules.rend(); it++) {
    auto *ID = ASTTraversalMetaInfo::getID(*it);
    if (!ID) {
      llvm::errs() << "[ERROR] Rule\"" << *it << "\" not found\n";
      std::exit(TranslationError);
    }
    emplaceTranslationRule(ID);
  }
}

const CompilerInstance &TranslationRule::getCompilerInstance() {
  return TM->CI;
}
