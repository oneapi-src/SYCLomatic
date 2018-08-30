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

#include "Utility.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "llvm/ADT/StringSet.h"
#include <string>
#include <utility>

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::syclct;

extern std::string CudaPath;

void IncludesCallbacks::InclusionDirective(
    SourceLocation HashLoc, const Token &IncludeTok, StringRef FileName,
    bool IsAngled, CharSourceRange FilenameRange, const FileEntry *File,
    StringRef SearchPath, StringRef RelativePath, const Module *Imported,
    SrcMgr::CharacteristicKind FileType) {

  std::string IncludePath = SearchPath;
  makeCanonical(IncludePath);

  if (!isChildPath(CudaPath, IncludePath))
    return;

  // Multiple CUDA headers in an including file will be replaced with one
  // include of the SYCL header.
  std::string IncludingFile = SM.getFilename(HashLoc);
  if (SeenFiles.find(IncludingFile) == end(SeenFiles)) {
    SeenFiles.insert(IncludingFile);
    std::string Replacement = std::string("<CL/sycl.hpp>") +
                              getNL(FilenameRange.getEnd(), SM) +
                              "#include <syclct.hpp>";
    TransformSet.emplace_back(
        new ReplaceInclude(FilenameRange, std::move(Replacement)));
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
  const MemberExpr *ME = Result.Nodes.getNodeAs<MemberExpr>("memberExpr");
  const VarDecl *VD = Result.Nodes.getNodeAs<VarDecl>("varDecl");
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
    Replacement += ".get_local(";
  else if (BuiltinName == "blockDim")
    Replacement += ".get_local_range().get(";
  else if (BuiltinName == "blockIdx")
    Replacement += ".get_group(";
  else if (BuiltinName == "gridDim")
    Replacement += ".get_num_groups(";
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
  auto If = Result.Nodes.getNodeAs<IfStmt>("errIf");
  if (!If)
    If = Result.Nodes.getNodeAs<IfStmt>("errIfSpecial");
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
        if (auto Op = Result.Nodes.getNodeAs<BinaryOperator>("op!=")) {
          if (!isCudaFailureCheck(Op))
            return false;
        } else if (auto Op = Result.Nodes.getNodeAs<BinaryOperator>("op==")) {
          if (!isCudaFailureCheck(Op, true))
            return false;
          report(Op->getLocStart(), Diagnostics::IFSTMT_SPECIAL_CASE,
                 getStmtSpelling(Op, *Result.Context).c_str());
        } else {
          auto CondVar = Result.Nodes.getNodeAs<DeclRefExpr>("var");
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

void FunctionAttrsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      functionDecl(anyOf(hasAttr(attr::CUDAGlobal), hasAttr(attr::CUDADevice),
                         hasAttr(attr::CUDAHost)))
          .bind("functionDecl"),
      this);
}

void FunctionAttrsRule::run(const MatchFinder::MatchResult &Result) {
  const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>("functionDecl");
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
  const VarDecl *D = Result.Nodes.getNodeAs<VarDecl>("TypeInVarDecl");
  const clang::Type *Type = D->getTypeSourceInfo()->getTypeLoc().getTypePtr();

  if (dyn_cast<SubstTemplateTypeParmType>(Type)) {
    return;
  }

  std::string TypeName =
      Type->getCanonicalTypeInternal().getBaseTypeIdentifier()->getName().str();
  auto Search = TypeNamesMap.find(TypeName);
  if (Search == TypeNamesMap.end()) {
    // TODO report translation error
    return;
  }
  std::string Replacement = Search->second;
  emplaceTransformation(new ReplaceTypeInVarDecl(D, std::move(Replacement)));
}

REGISTER_RULE(TypeInVarDeclRule)

void ReplaceDim3CtorRule::registerMatcher(MatchFinder &MF) {
  // Find dim3 constructors which are part of different casts (representing
  // different syntaxes). This includes copy constructors. All constructors
  // will be visited once.
  MF.addMatcher(
      cxxConstructExpr(hasType(typedefDecl(hasName("dim3"))),
                       hasParent(cxxFunctionalCastExpr().bind("dim3Cast")))
          .bind("dim3CtorFuncCast"),
      this);
  MF.addMatcher(cxxConstructExpr(hasType(typedefDecl(hasName("dim3"))),
                                 hasParent(cStyleCastExpr().bind("dim3Cast")))
                    .bind("dim3CtorCCast"),
                this);
  MF.addMatcher(cxxConstructExpr(hasType(typedefDecl(hasName("dim3"))),
                                 hasParent(implicitCastExpr().bind("dim3Cast")))
                    .bind("dim3CtorImplicitCast"),
                this);

  // Find all other dim3 constructors with 3 parameters (not copy ctors).
  MF.addMatcher(cxxConstructExpr(hasType(typedefDecl(hasName("dim3"))),
                                 unless(hasParent(castExpr())))
                    .bind("dim3Ctor"),
                this);
}

// Determines which case of construction applies and creates replacements for
// the syntax. Returns the constructor node and a boolean indicating if a
// closed brace needs to be appended.
std::pair<const CXXConstructExpr *, bool>
ReplaceDim3CtorRule::rewriteSyntax(const MatchFinder::MatchResult &Result) {
  // Most commonly used syntax cases are checked first.
  if (auto Ctor = Result.Nodes.getNodeAs<CXXConstructExpr>("dim3Ctor")) {
    // dim3 a(1);
    // No syntax needs to be rewritten.
    return {Ctor, false};
  }

  if (auto ImplCastCtor =
          Result.Nodes.getNodeAs<CXXConstructExpr>("dim3CtorImplicitCast")) {
    auto Cast = Result.Nodes.getNodeAs<ImplicitCastExpr>("dim3Cast");
    if (!isa<CXXTemporaryObjectExpr>(ImplCastCtor)) {
      // dim3 a = 1;
      // func(1);
      emplaceTransformation(new InsertBeforeStmt(Cast, "cl::sycl::range<3>("));
      return {ImplCastCtor, true};
    } else {
      // dim3 a = dim3(1, 2); // temporary object expression
      // func(dim(1, 2));
      emplaceTransformation(
          new ReplaceToken(Cast->getLocStart(), "cl::sycl::range<3>"));
      return {ImplCastCtor, false};
    }
  } else if (auto FuncCastCtor =
                 Result.Nodes.getNodeAs<CXXConstructExpr>("dim3CtorFuncCast")) {
    auto Cast = Result.Nodes.getNodeAs<CXXFunctionalCastExpr>("dim3Cast");
    // dim3 a = dim3(1); // function style cast
    // dim3 b = dim3(a); // copy constructor
    // func(dim(1), dim3(a));
    emplaceTransformation(
        new ReplaceToken(Cast->getLocStart(), "cl::sycl::range<3>"));
    return {FuncCastCtor, false};
  }

  if (auto CCastCtor =
          Result.Nodes.getNodeAs<CXXConstructExpr>("dim3CtorCCast")) {
    // dim3 a = (dim3)1;
    // dim3 b = (dim3)a; // copy constructor
    // func((dim)1, (dim3)a);
    auto Cast = Result.Nodes.getNodeAs<CStyleCastExpr>("dim3Cast");
    emplaceTransformation(new ReplaceCCast(Cast, "cl::sycl::range<3>("));
    return {CCastCtor, true};
  }

  assert(false && "This must not happen.");
}

void ReplaceDim3CtorRule::rewriteArglist(
    const std::pair<const CXXConstructExpr *, bool> &CtorCase) {
  auto Ctor = CtorCase.first;
  auto CloseBrace = CtorCase.second;

  if (CtorCase.first->getNumArgs() == 1) {
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
  const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>("functionDecl");
  const clang::Type *Type = FD->getReturnType().getTypePtr();
  std::string TypeName =
      Type->getCanonicalTypeInternal().getBaseTypeIdentifier()->getName().str();
  auto Search = TypeInVarDeclRule::TypeNamesMap.find(TypeName);
  if (Search == TypeInVarDeclRule::TypeNamesMap.end()) {
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
  const MemberExpr *ME = Result.Nodes.getNodeAs<MemberExpr>("DevicePropVar");
  auto Search = PropNamesMap.find(ME->getMemberNameInfo().getAsString());
  if (Search == PropNamesMap.end()) {
    // TODO report translation error
    return;
  }
  emplaceTransformation(new RenameFieldInMemberExpr(ME, Search->second + "()"));
  if ((Search->second.compare(0, 13, "major_version") == 0) ||
      (Search->second.compare(0, 13, "minor_version") == 0)) {
    report(ME->getLocStart(), Comments::VERSION_COMMENT);
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
  const DeclRefExpr *E = Result.Nodes.getNodeAs<DeclRefExpr>("EnumConstant");
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
  const DeclRefExpr *DE = Result.Nodes.getNodeAs<DeclRefExpr>("ErrorConstants");
  assert(DE && "Unknown result");
  auto *EC = cast<EnumConstantDecl>(DE->getDecl());
  emplaceTransformation(new ReplaceStmt(DE, EC->getInitVal().toString(10)));
}

REGISTER_RULE(ErrorConstantsRule)

void FunctionCallRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      callExpr(callee(functionDecl(hasAnyName(
                   "cudaGetDeviceCount", "cudaGetDeviceProperties",
                   "cudaDeviceReset", "cudaSetDevice", "cudaDeviceGetAttribute",
                   "cudaDeviceGetP2PAttribute", "cudaGetDevice",
                   "cudaGetLastError", "cudaDeviceSynchronize"))))
          .bind("FunctionCall"),
      this);
}

void FunctionCallRule::run(const MatchFinder::MatchResult &Result) {
  const CallExpr *CE = Result.Nodes.getNodeAs<CallExpr>("FunctionCall");
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
    report(CE->getLocStart(), Comments::NOTSUPPORTED, "P2P Access");
  } else if (FuncName == "cudaGetDevice") {
    std::string ResultVarName = DereferenceArg(CE->getArg(0));
    emplaceTransformation(new InsertBeforeStmt(CE, ResultVarName + " = "));
    emplaceTransformation(new ReplaceStmt(
        CE, "syclct::get_device_manager().current_device_id()"));
  } else if (FuncName == "cudaDeviceSynchronize") {
    emplaceTransformation(new ReplaceStmt(CE, "(syclct::get_device_manager()."
                                              "current_device().queues_wait_"
                                              "and_throw(), 0)"));
    report(CE->getLocStart(), Diagnostics::NOERROR_RETURN_COMMA_OP);
  } else if (FuncName == "cudaGetLastError") {
    emplaceTransformation(new ReplaceStmt(CE, "0"));
  } else {
    llvm_unreachable("Unknown function name");
  }
}

REGISTER_RULE(FunctionCallRule)

void KernelCallRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  MF.addMatcher(cudaKernelCallExpr().bind("kernelCall"), this);
}

void KernelCallRule::run(const ast_matchers::MatchFinder::MatchResult &Result) {
  auto KCall = Result.Nodes.getNodeAs<CUDAKernelCallExpr>("kernelCall");
  emplaceTransformation(new ReplaceKernelCallExpr(KCall));
}

REGISTER_RULE(KernelCallRule)

// Memory translation rules live here.
void MemoryTranslationRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      callExpr(callee(functionDecl(hasAnyName("cudaMalloc", "cudaMemcpy",
                                              "cudaFree", "cudaMemset"))))
          .bind("call"),
      this);
}

void MemoryTranslationRule::run(const MatchFinder::MatchResult &Result) {
  const CallExpr *C = Result.Nodes.getNodeAs<CallExpr>("call");
  std::string Name = C->getCalleeDecl()->getAsFunction()->getNameAsString();
  report(C->getLocStart(), Diagnostics::NOERROR_RETURN_COMMA_OP);
  emplaceTransformation(new InsertAfterStmt(C, ", 0)"));
  if (Name == "cudaMalloc") {
    // Input:
    // float *d_A = NULL;
    // cudaMalloc((void **)&d_A, size);
    //
    // Desired output:
    // float *d_A = NULL;
    // sycl_malloc<float>((float **)&d_A, numElements);
    //
    // Current output:
    // float *d_A = NULL;
    // sycl_malloc<char>((void **)&d_A, size);
    //
    //
    // NOTE: "ErroCodeType sycl_malloc<T>(T**, size_t)" signature is used
    // instead of "T* sycl_malloc<T>(size_t)" to make it easier to hook it
    // up with error handling. This may change.

    std::string Name = "(syclct::sycl_malloc<char>";
    std::vector<const Expr *> Args{C->getArg(0), C->getArg(1)};
    emplaceTransformation(
        new ReplaceCallExpr(C, std::move(Name), std::move(Args)));
  } else if (Name == "cudaMemcpy") {
    // Input:
    // cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    // cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
    // cudaMemcpy(x_A, y_A, size, someDynamicCudaMemcpyKindValue);
    //
    // Desired output:
    // sycl_memcpy<float>(d_A, h_A, numElements);
    // sycl_memcpy_back<float>(h_A, d_A, numElements);
    // sycl_memcpy<float>(d_A, h_A, numElements,
    // someDynamicCudaMemcpyKindValue);
    //
    // Current output:
    // sycl_memcpy<char>(d_A, h_A, size, cudaMemcpyHostToDevice);
    // sycl_memcpy<char>(d_A, h_A, size, cudaMemcpyDeviceToHost);
    // sycl_memcpy<char>(d_A, h_A, size, someDynamicCudaMemcpyKindValue);

    std::string Name = "(syclct::sycl_memcpy<char>";
    std::vector<const Expr *> Args{C->getArg(0), C->getArg(1), C->getArg(2),
                                   C->getArg(3)};
    emplaceTransformation(
        new ReplaceCallExpr(C, std::move(Name), std::move(Args)));
  } else if (Name == "cudaMemset") {
    // Input:
    // cudaMemset(d_A, 0x12, size);
    //
    // Desired output:
    // syclct::sycl_memset((void*)d_A, (unsigned char)0x12,  (unsigned
    // int)size);
    std::string Name = "(syclct::sycl_memset";
    std::vector<const Expr *> Args{C->getArg(0), C->getArg(1), C->getArg(2)};
    std::vector<std::string> NewTypes{"(void*)", "(unsigned char)",
                                      "(unsigned int)"};

    emplaceTransformation(
        new ReplaceCallExpr(C, std::move(Name), std::move(Args), NewTypes));

  } else if (Name == "cudaFree") {
    // Input:
    // cudaFree(d_A);
    //
    // Output:
    // sycl_free<char>(d_A);
    std::string Name = "(syclct::sycl_free<char>";
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
public:
  std::unordered_set<unsigned> Insertions;
  void registerMatcher(ast_matchers::MatchFinder &MF) override {
    MF.addMatcher(functionDecl(hasBody(compoundStmt())).bind("functionDecl"),
                  this);
  }
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override {
    const FunctionDecl *FD =
        Result.Nodes.getNodeAs<FunctionDecl>("functionDecl");
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
  const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>("functionDecl");
  emplaceTransformation(new InsertArgument(FD, "cl::sycl::nd_item<3> item"));
}
REGISTER_RULE(KernelIterationSpaceRule)

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

const CompilerInstance &TranslationRule::getCompilerInstance() {
  return TM->CI;
}
