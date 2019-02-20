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
#include "AnalysisInfo.h"
#include "Debug.h"
#include "SaveNewFiles.h"
#include "Utility.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Path.h"

#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::syclct;
using namespace clang::tooling;

extern std::string CudaPath;
extern std::string SyclctInstallPath; // Installation directory for this tool

std::unordered_map<std::string, std::unordered_set</* Comment ID */ int>>
    TranslationRule::ReportedComment;

void IncludesCallbacks::InclusionDirective(
    SourceLocation HashLoc, const Token &IncludeTok, StringRef FileName,
    bool IsAngled, CharSourceRange FilenameRange, const FileEntry *File,
    StringRef SearchPath, StringRef RelativePath, const Module *Imported,
    SrcMgr::CharacteristicKind FileType) {

  std::string IncludePath = SearchPath;
  makeCanonical(IncludePath);
  std::string IncludingFile = SM.getFilename(HashLoc);

  IncludingFile = getAbsolutePath(IncludingFile);
  makeCanonical(IncludingFile);

  // eg. '/home/path/util.h' -> '/home/path'
  StringRef Directory = llvm::sys::path::parent_path(IncludingFile);
  std::string InRoot = ATM.InRoot;

  bool IsIncludingFileInInRoot = !llvm::sys::fs::is_directory(IncludingFile) &&
                                 (isChildPath(InRoot, Directory.str()) ||
                                  isSamePath(InRoot, Directory.str()));

  std::string FilePath = File->getName();
  makeCanonical(FilePath);
  std::string DirPath = llvm::sys::path::parent_path(FilePath);
  bool IsFileInInRoot =
      !isChildPath(SyclctInstallPath, DirPath) &&
      (isChildPath(InRoot, DirPath) || isSamePath(InRoot, DirPath));

  if (IsFileInInRoot && !StringRef(FilePath).endswith(".cu")) {
    auto Find = IncludeFileMap.find(FilePath);
    if (Find == IncludeFileMap.end()) {
      IncludeFileMap[FilePath] = false;
    }
  }

  if (!SM.isWrittenInMainFile(HashLoc) && !IsIncludingFileInInRoot) {
    return;
  }

  // Insert SYCL headers for file inputted or file included.
  // E.g. A.cu included B.cu, both A.cu and B.cu are inserted "#include
  // <CL/sycl.hpp>\n#include <syclct/syclct.hpp>"
  if (!SyclHeaderInserted || SeenFiles.find(IncludingFile) == end(SeenFiles)) {
    SeenFiles.insert(IncludingFile);
    std::string Replacement = std::string("#include <CL/sycl.hpp>") +
                              getNL(FilenameRange.getEnd(), SM) +
                              "#include <syclct/syclct.hpp>" +
                              getNL(FilenameRange.getEnd(), SM);
    CharSourceRange InsertRange(SourceRange(HashLoc, HashLoc), false);
    TransformSet.emplace_back(
        new ReplaceInclude(InsertRange, std::move(Replacement)));
    SyclHeaderInserted = true;
  }

  // replace "#include <math.h>" with <cmath>
  if (IsAngled && FileName.compare(StringRef("math.h")) == 0) {
    TransformSet.emplace_back(new ReplaceInclude(
        CharSourceRange(SourceRange(HashLoc, FilenameRange.getEnd()),
                        /*IsTokenRange=*/false),
        "#include <cmath>"));
  }

  if (!isChildPath(CudaPath, IncludePath) &&
      // CudaPath detection have not consider soft link, here do special
      // for /usr/local/cuda
      IncludePath.compare(0, 15, "/usr/local/cuda", 15)) {

    // Replace "#include "*.cuh"" with "include "*.sycl.hpp""
    if (!IsAngled && FileName.endswith(".cuh")) {
      CharSourceRange InsertRange(SourceRange(HashLoc, FilenameRange.getEnd()),
                                  /* IsTokenRange */ false);
      std::string NewFileName = "#include \"" +
                                FileName.drop_back(strlen(".cuh")).str() +
                                ".sycl.hpp\"";
      TransformSet.emplace_back(
          new ReplaceInclude(InsertRange, std::move(NewFileName)));
      return;
    }

    // Replace "#include "*.cu"" with "include "*.sycl.cpp""
    if (!IsAngled && FileName.endswith(".cu")) {
      CharSourceRange InsertRange(SourceRange(HashLoc, FilenameRange.getEnd()),
                                  /* IsTokenRange */ false);
      std::string NewFileName = "#include \"" +
                                FileName.drop_back(strlen(".cu")).str() +
                                ".sycl.cpp\"";
      TransformSet.emplace_back(
          new ReplaceInclude(InsertRange, std::move(NewFileName)));
      return;
    }

    // if <cuda_runtime.h>, no matter where it from, replace with sycl header
    if (!(IsAngled && FileName.compare(StringRef("cuda_runtime.h")) == 0))
      return;
  }

  // Extra process thrust headers, map to PSTL mapping headers in runtime.
  // For multi thrust header files, only insert once for PSTL mapping header.
  if (IsAngled && (FileName.find("thrust/") != std::string::npos)) {
    if (!ThrustHeaderInserted) {
      std::string Replacement;
      if (!SyclHeaderInserted) {
        Replacement =
            std::string("<CL/sycl.hpp>") + getNL(FilenameRange.getEnd(), SM) +
            "#include <syclct/syclct.hpp>" + getNL(FilenameRange.getEnd(), SM) +
            "#include <syclct/syclct_thrust.hpp>";
        SyclHeaderInserted = true;
      } else {
        Replacement = std::string("<syclct/syclct_thrust.hpp>");
      }
      ThrustHeaderInserted = true;
      TransformSet.emplace_back(
          new ReplaceInclude(FilenameRange, std::move(Replacement)));
      return;
    }
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

void TranslationRule::print(llvm::raw_ostream &OS) {
  const auto &EmittedTransformations = getEmittedTransformations();
  if (EmittedTransformations.empty()) {
    return;
  }

  OS << "[" << getName() << "]\n";
  constexpr char Indent[] = "  ";
  for (const TextModification *TM : EmittedTransformations) {
    OS << Indent;
    TM->print(OS, getCompilerInstance().getASTContext(),
              /* Print parent */ false);
  }
}

void TranslationRule::printStatistics(llvm::raw_ostream &OS) {
  const auto &EmittedTransformations = getEmittedTransformations();
  if (EmittedTransformations.empty()) {
    return;
  }

  OS << "<Statistics of " << getName() << ">\n";
  std::unordered_map<std::string, size_t> TMNameCountMap;
  for (const TextModification *TM : EmittedTransformations) {
    const std::string Name = TM->getName();
    if (TMNameCountMap.count(Name) == 0) {
      TMNameCountMap.emplace(std::make_pair(Name, 1));
    } else {
      ++TMNameCountMap[Name];
    }
  }

  constexpr char Indent[] = "  ";
  for (const auto &Pair : TMNameCountMap) {
    const std::string &Name = Pair.first;
    const size_t &Numbers = Pair.second;
    OS << Indent << "Emitted # of replacement <" << Name << ">: " << Numbers
       << "\n";
  }
}

void TranslationRule::emplaceTransformation(const char *RuleID,
                                            TextModification *TM) {
  ASTTraversalMetaInfo::getEmittedTransformations()[RuleID].emplace_back(TM);
  TransformSet->emplace_back(TM);
}

void IterationSpaceBuiltinRule::registerMatcher(MatchFinder &MF) {
  // TODO: check that threadIdx is not a local variable.
  MF.addMatcher(
      memberExpr(hasObjectExpression(opaqueValueExpr(hasSourceExpression(
                     declRefExpr(to(varDecl(hasAnyName("threadIdx", "blockDim",
                                                       "blockIdx", "gridDim"))
                                        .bind("varDecl")))))),
                 hasAncestor(functionDecl().bind("func")))
          .bind("memberExpr"),
      this);
}

void IterationSpaceBuiltinRule::run(const MatchFinder::MatchResult &Result) {
  const MemberExpr *ME = getNodeAsType<MemberExpr>(Result, "memberExpr");
  if (!ME)
    return;
  if (auto FD = getAssistNodeAsType<FunctionDecl>(Result, "func"))
    SyclctGlobalInfo::getInstance().registerDeviceFunctionInfo(FD)->setItem();
  const VarDecl *VD = getAssistNodeAsType<VarDecl>(Result, "varDecl", false);
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
    syclct_unreachable("Unknown field name");

  std::string Replacement = getItemName();
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
    syclct_unreachable("Unknown builtin variable");

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
    report(SL, Diagnostics::STMT_NOT_REMOVED);
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
        bool IsIfstmtSpecialCase = false;
        SourceLocation Ip;
        if (auto Op = getNodeAsType<BinaryOperator>(Result, "op!=")) {
          if (!isCudaFailureCheck(Op))
            return false;
        } else if (auto Op = getNodeAsType<BinaryOperator>(Result, "op==")) {
          if (!isCudaFailureCheck(Op, true))
            return false;
          IsIfstmtSpecialCase = true;
          Ip = Op->getBeginLoc();

        } else {
          auto CondVar = getNodeAsType<DeclRefExpr>(Result, "var");
          if (!isCudaFailureCheck(CondVar))
            return false;
        }
        // We know that it's error checking condition, check the body
        if (!isErrorHandling(If->getThen())) {
          if (IsIfstmtSpecialCase) {
            report(Ip, Diagnostics::IFSTMT_SPECIAL_CASE);
          } else {
            report(If->getSourceRange().getBegin(),
                   Diagnostics::IFSTMT_NOT_REMOVED);
          }
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

void AtomicFunctionRule::registerMatcher(MatchFinder &MF) {
  std::vector<std::string> AtomicFuncNames(AtomicFuncNamesMap.size());
  std::transform(
      AtomicFuncNamesMap.begin(), AtomicFuncNamesMap.end(),
      AtomicFuncNames.begin(),
      [](const std::pair<std::string, std::string> &p) { return p.first; });

  auto hasAnyAtomicFuncName = [&]() {
    return internal::Matcher<NamedDecl>(
        new internal::HasNameMatcher(AtomicFuncNames));
  };

  // Support all integer type, float and double
  // Type half and half2 are not supported
  auto supportedTypes = [&]() {
    // TODO: investigate usage of __half and __half2 types and support it
    return anyOf(hasType(pointsTo(isInteger())),
                 hasType(pointsTo(asString("float"))),
                 hasType(pointsTo(asString("double"))));
  };

  auto supportedAtomicFunctions = [&]() {
    return allOf(hasAnyAtomicFuncName(), hasParameter(0, supportedTypes()));
  };

  auto unsupportedAtomicFunctions = [&]() {
    return allOf(hasAnyAtomicFuncName(),
                 unless(hasParameter(0, supportedTypes())));
  };

  MF.addMatcher(callExpr(callee(functionDecl(supportedAtomicFunctions())))
                    .bind("supportedAtomicFuncCall"),
                this);

  MF.addMatcher(callExpr(callee(functionDecl(unsupportedAtomicFunctions())))
                    .bind("unsupportedAtomicFuncCall"),
                this);
}

void AtomicFunctionRule::ReportUnsupportedAtomicFunc(const CallExpr *CE) {
  if (!CE)
    return;

  std::ostringstream OSS;
  // Atomic functions with __half and half2 are not supported.
  OSS << "half version of " << CE->getDirectCallee()->getName().str();
  report(CE->getBeginLoc(), Comments::API_NOT_TRANSLATED, OSS.str());
}

void AtomicFunctionRule::TranslateAtomicFunc(const CallExpr *CE) {
  if (!CE)
    return;

  // TODO: 1. Investigate are there usages of atomic functions on local address
  //          space
  //       2. If item 1. shows atomic functions on local address space is
  //          significant, detect whether this atomic operation operates in
  //          global space or local space (currently, all in global space,
  //          see syclct_atomic.hpp for more details)
  const std::string AtomicFuncName = CE->getDirectCallee()->getName().str();
  assert(AtomicFuncNamesMap.find(AtomicFuncName) != AtomicFuncNamesMap.end());
  std::string ReplacedAtomicFuncName = AtomicFuncNamesMap.at(AtomicFuncName);

  // Explicitly cast all arguments except first argument
  const Type *Arg0Type = CE->getArg(0)->getType().getTypePtrOrNull();
  // Atomic operation's first argument is always pointer type
  assert(Arg0Type && Arg0Type->isPointerType());
  const QualType PointeeType = Arg0Type->getPointeeType();

  std::string TypeName;
  if (auto *SubstedType = dyn_cast<SubstTemplateTypeParmType>(PointeeType)) {
    // Type is substituted in template initialization, use the template
    // parameter name
    TypeName =
        SubstedType->getReplacedParameter()->getIdentifier()->getName().str();
  } else {
    TypeName = PointeeType.getAsString();
  }
  // add exceptions for atomic tranlastion:
  // eg. source code: atomicMin(double), don't translate it, its user code.
  //     also: atomic_fetch_min<double> is not available in compute++.
  if ((TypeName == "double" && AtomicFuncName != "atomicAdd") ||
      (TypeName == "float" &&
       !(AtomicFuncName == "atomicAdd" || AtomicFuncName == "atomicExch"))) {

    return;
  }

  emplaceTransformation(
      new ReplaceCalleeName(CE, std::move(ReplacedAtomicFuncName)));

  const unsigned NumArgs = CE->getNumArgs();
  for (unsigned i = 1; i < NumArgs; ++i) {
    const Expr *Arg = CE->getArg(i);
    emplaceTransformation(new InsertBeforeStmt(Arg, "(" + TypeName + ")("));
    emplaceTransformation(new InsertAfterStmt(Arg, ")"));
  }
}

void AtomicFunctionRule::run(const MatchFinder::MatchResult &Result) {
  ReportUnsupportedAtomicFunc(
      getNodeAsType<CallExpr>(Result, "unsupportedAtomicFuncCall"));

  TranslateAtomicFunc(
      getNodeAsType<CallExpr>(Result, "supportedAtomicFuncCall"));
}

REGISTER_RULE(AtomicFunctionRule)

// Rule for types replacements in var declarations and field declarations
void TypeInDeclRule::registerMatcher(MatchFinder &MF) {
  auto HasCudaType = []() {
    return anyOf(hasType(typedefDecl(hasName("dim3"))),
                 hasType(typedefDecl(hasName("cudaError_t"))),
                 hasType(enumDecl(hasName("cudaError"))),
                 hasType(cxxRecordDecl(hasName("cudaDeviceProp"))));
  };
  auto HasCudaTypePtr = []() {
    return anyOf(hasType(pointsTo(typedefDecl(hasName("dim3")))),
                 hasType(pointsTo(typedefDecl(hasName("cudaError_t")))),
                 hasType(pointsTo(enumDecl(hasName("cudaError")))),
                 hasType(pointsTo(cxxRecordDecl(hasName("cudaDeviceProp")))));
  };
  auto HasCudaTypePtrPtr = []() {
    return anyOf(
        hasType(pointsTo(pointsTo(typedefDecl(hasName("dim3"))))),
        hasType(pointsTo(pointsTo(typedefDecl(hasName("cudaError_t"))))),
        hasType(pointsTo(pointsTo(enumDecl(hasName("cudaError"))))),
        hasType(pointsTo(pointsTo(cxxRecordDecl(hasName("cudaDeviceProp"))))));
  };
  auto HasCudaTypeRef = []() {
    return anyOf(hasType(references(typedefDecl(hasName("dim3")))),
                 hasType(references(typedefDecl(hasName("cudaError_t")))),
                 hasType(references(enumDecl(hasName("cudaError")))),
                 hasType(references(cxxRecordDecl(hasName("cudaDeviceProp")))));
  };

  MF.addMatcher(varDecl(anyOf(HasCudaType(), HasCudaTypePtr(),
                              HasCudaTypePtrPtr(), HasCudaTypeRef()),
                        unless(hasType(substTemplateTypeParmType())))
                    .bind("TypeInVarDecl"),
                this);
  MF.addMatcher(fieldDecl(anyOf(HasCudaType(), HasCudaTypePtr(),
                                HasCudaTypePtrPtr(), HasCudaTypeRef()),
                          unless(hasType(substTemplateTypeParmType())))
                    .bind("TypeInFieldDecl"),
                this);
}

void TypeInDeclRule::run(const MatchFinder::MatchResult &Result) {
  const VarDecl *D = getNodeAsType<VarDecl>(Result, "TypeInVarDecl");
  const FieldDecl *FD = getNodeAsType<FieldDecl>(Result, "TypeInFieldDecl");
  QualType QT;
  if (D) {
    QT = D->getType();
  } else if (FD) {
    QT = FD->getType();
  } else {
    return;
  }

  std::istringstream ISS(QT.getAsString());
  std::vector<std::string> Strs(std::istream_iterator<std::string>{ISS},
                                std::istream_iterator<std::string>());
  auto it = std::remove_if(Strs.begin(), Strs.end(), [](llvm::StringRef Str) {
    return (Str.contains("&") || Str.contains("*"));
  });
  if (it != Strs.end())
    Strs.erase(it);

  const std::string &TypeName = Strs.back();
  auto Search = MapNames::TypeNamesMap.find(TypeName);
  if (Search == MapNames::TypeNamesMap.end()) {
    // TODO report translation error
    return;
  }

  std::string Replacement = QT.getAsString();
  assert(Replacement.find(TypeName) != std::string::npos);
  Replacement = Replacement.substr(Replacement.find(TypeName));
  Replacement.replace(0, TypeName.length(), Search->second);
  if (D) {
    emplaceTransformation(new ReplaceTypeInDecl(D, std::move(Replacement)));
  } else {
    emplaceTransformation(new ReplaceTypeInDecl(FD, std::move(Replacement)));
  }
}

REGISTER_RULE(TypeInDeclRule)

// Supported vector types
const std::unordered_set<std::string> SupportedVectorTypes{"int2", "double2",
                                                           "uint4"};

static internal::Matcher<NamedDecl> vectorTypeName() {
  std::vector<std::string> TypeNames(SupportedVectorTypes.begin(),
                                     SupportedVectorTypes.end());
  return internal::Matcher<NamedDecl>(new internal::HasNameMatcher(TypeNames));
}

namespace clang {
namespace ast_matchers {

AST_MATCHER(QualType, vectorType) {
  return (SupportedVectorTypes.find(Node.getAsString()) !=
          SupportedVectorTypes.end());
}

AST_MATCHER(TypedefDecl, typedefVecDecl) {
  if (!Node.getUnderlyingType().getBaseTypeIdentifier())
    return false;

  const std::string BaseTypeName =
      Node.getUnderlyingType().getBaseTypeIdentifier()->getName().str();
  return (SupportedVectorTypes.find(BaseTypeName) !=
          SupportedVectorTypes.end());
}

} // namespace ast_matchers
} // namespace clang

// Rule for types replacements in var. declarations.
void VectorTypeNamespaceRule::registerMatcher(MatchFinder &MF) {
  auto unlessMemory =
      unless(anyOf(hasAttr(attr::CUDAConstant), hasAttr(attr::CUDADevice),
                   hasAttr(attr::CUDAShared)));

  // basic: eg. int2 xx
  auto basicType = [&]() {
    return allOf(hasType(typedefDecl(vectorTypeName())),
                 unless(hasType(substTemplateTypeParmType())), unlessMemory);
  };

  // pointer: eg. int2 * xx
  auto ptrType = [&]() {
    return allOf(hasType(pointsTo(typedefDecl(vectorTypeName()))),
                 unlessMemory);
  };

  // array: eg. int2 array_[xx]
  auto arrType = [&]() {
    return allOf(hasType(arrayType(hasElementType(typedefType(
                     hasDeclaration(typedefDecl(vectorTypeName())))))),
                 unlessMemory);
  };

  // reference: eg int2 & xx
  auto referenceType = [&]() {
    return allOf(hasType(references(typedefDecl(vectorTypeName()))),
                 unlessMemory);
  };

  MF.addMatcher(
      varDecl(anyOf(basicType(), ptrType(), arrType(), referenceType()))
          .bind("vecVarDecl"),
      this);

  // typedef int2 xxx
  MF.addMatcher(typedefDecl(typedefVecDecl()).bind("typeDefDecl"), this);

  auto vectorTypeAccess = [&]() {
    return anyOf(vectorType(), references(vectorType()),
                 pointsTo(vectorType()));
  };

  // int2 func() => cl::sycl::int2 func()
  MF.addMatcher(
      functionDecl(returns(vectorTypeAccess())).bind("funcReturnsVectorType"),
      this);
}

bool VectorTypeNamespaceRule::isNamespaceInserted(SourceLocation SL) {
  unsigned int Key = SL.getRawEncoding();
  if (DupFilter.find(Key) == end(DupFilter)) {
    DupFilter.insert(Key);
    return false;
  } else {
    return true;
  }
}

void VectorTypeNamespaceRule::run(const MatchFinder::MatchResult &Result) {
  // int2 => cl::sycl::int2
  if (const VarDecl *D = getNodeAsType<VarDecl>(Result, "vecVarDecl")) {
    if (!isNamespaceInserted(
            D->getTypeSourceInfo()->getTypeLoc().getBeginLoc())) {
      emplaceTransformation(new InsertNameSpaceInVarDecl(D, "cl::sycl::"));
    }
  }

  // typedef int2 xxx => typedef cl::sycl::int2 xxx
  if (const TypedefDecl *TD =
          getNodeAsType<TypedefDecl>(Result, "typeDefDecl")) {
    const SourceLocation UnderlyingTypeSL =
        TD->getTypeSourceInfo()->getTypeLoc().getBeginLoc();

    if (!isNamespaceInserted(UnderlyingTypeSL)) {
      emplaceTransformation(new InsertText(UnderlyingTypeSL, "cl::sycl::"));
    }
  }

  // int2 func() => cl::sycl::int2 func()
  if (const FunctionDecl *FD =
          getNodeAsType<FunctionDecl>(Result, "funcReturnsVectorType")) {

    if (!isNamespaceInserted(FD->getReturnTypeSourceRange().getBegin())) {
      emplaceTransformation(new InsertText(
          FD->getReturnTypeSourceRange().getBegin(), "cl::sycl::"));
    }
  }
}

REGISTER_RULE(VectorTypeNamespaceRule)

void VectorTypeMemberAccessRule::registerMatcher(MatchFinder &MF) {
  auto memberAccess = [&]() {
    return hasObjectExpression(hasType(qualType(hasCanonicalType(
        recordType(hasDeclaration(cxxRecordDecl(vectorTypeName())))))));
  };

  // int2.x => static_cast<int>(int2.x())
  MF.addMatcher(
      memberExpr(allOf(memberAccess(), unless(hasParent(binaryOperator(allOf(
                                           hasLHS(memberExpr(memberAccess())),
                                           isAssignmentOperator()))))))
          .bind("VecMemberExpr"),
      this);

  // int2.x += xxx => int2.x() += static_cast<int>(xxx)
  MF.addMatcher(
      binaryOperator(allOf(hasLHS(memberExpr(memberAccess())
                                      .bind("VecMemberExprAssignmentLHS")),
                           isAssignmentOperator()))
          .bind("VecMemberExprAssignment"),
      this);
}

void VectorTypeMemberAccessRule::run(const MatchFinder::MatchResult &Result) {
  auto GetMemberAccessName = [&](const MemberExpr *ME) {
    const std::string MemberAccessName = ME->getMemberNameInfo().getAsString();
    assert(MemberNamesMap.find(MemberAccessName) != MemberNamesMap.end());
    return MemberAccessName;
  };

  // xxx = int2.x => xxx = static_cast<int>(int2.x())
  if (const MemberExpr *ME =
          getNodeAsType<MemberExpr>(Result, "VecMemberExpr")) {
    std::string ReplacedMemberAccessName =
        MemberNamesMap.at(GetMemberAccessName(ME));

    std::ostringstream CastPrefix;
    CastPrefix << "static_cast<" << ME->getType().getAsString() << ">(";
    emplaceTransformation(new InsertBeforeStmt(ME, CastPrefix.str()));
    emplaceTransformation(
        new RenameFieldInMemberExpr(ME, std::move(ReplacedMemberAccessName)));
    emplaceTransformation(new InsertAfterStmt(ME, ")"));
  }

  // int2.x += xxx => int2.x() += xxx
  const BinaryOperator *BO =
      getAssistNodeAsType<BinaryOperator>(Result, "VecMemberExprAssignment");
  if (!BO)
    return;

  const MemberExpr *ME =
      getAssistNodeAsType<MemberExpr>(Result, "VecMemberExprAssignmentLHS");
  assert(ME != nullptr);
  std::string ReplacedMemberAccessName =
      MemberNamesMap.at(GetMemberAccessName(ME));
  emplaceTransformation(
      new RenameFieldInMemberExpr(ME, std::move(ReplacedMemberAccessName)));
}

REGISTER_RULE(VectorTypeMemberAccessRule)

namespace clang {
namespace ast_matchers {

AST_MATCHER(FunctionDecl, overloadedVectorOperator) {
  if (!SyclctGlobalInfo::isInRoot(Node.getBeginLoc()))
    return false;

  switch (Node.getOverloadedOperator()) {
  default: { return false; }
#define OVERLOADED_OPERATOR_MULTI(...)
#define OVERLOADED_OPERATOR(Name, ...)                                         \
  case OO_##Name: {                                                            \
    break;                                                                     \
  }
#include "clang/Basic/OperatorKinds.def"
#undef OVERLOADED_OPERATOR
#undef OVERLOADED_OPERATOR_MULTI
  }

  // Check parameter is vector type
  auto SupportedParamType = [&](const ParmVarDecl *PD) {
    assert(PD != nullptr);
    const IdentifierInfo *IDInfo =
        PD->getOriginalType().getBaseTypeIdentifier();
    if (!IDInfo)
      return false;

    const std::string TypeName = IDInfo->getName().str();
    return (SupportedVectorTypes.find(TypeName) != SupportedVectorTypes.end());
  };

  assert(Node.getNumParams() < 3);
  // As long as one parameter is vector type
  for (unsigned i = 0, End = Node.getNumParams(); i != End; ++i) {
    if (SupportedParamType(Node.getParamDecl(i))) {
      return true;
    }
  }

  return false;
}

} // namespace ast_matchers
} // namespace clang

void VectorTypeOperatorRule::registerMatcher(MatchFinder &MF) {
  auto vectorTypeOverLoadedOperator = [&]() {
    return functionDecl(overloadedVectorOperator());
  };

  // Matches user overloaded operator declaration
  MF.addMatcher(vectorTypeOverLoadedOperator().bind("overloadedOperatorDecl"),
                this);

  // Matches call of user overloaded operator
  MF.addMatcher(cxxOperatorCallExpr(callee(vectorTypeOverLoadedOperator()))
                    .bind("callOverloadedOperator"),
                this);
}

const char VectorTypeOperatorRule::NamespaceName[] =
    "syclct_operator_overloading";

void VectorTypeOperatorRule::TranslateOverloadedOperatorDecl(
    const MatchFinder::MatchResult &Result, const FunctionDecl *FD) {
  if (!FD)
    return;

  // Helper function to get the scope of function declartion
  // Eg:
  //
  //    void test();
  //   ^            ^
  //   |            |
  // Begin         End
  //
  //    void test() {}
  //   ^              ^
  //   |              |
  // Begin           End
  auto GetFunctionSourceRange = [&](const SourceManager &SM,
                                    const SourceLocation &StartLoc,
                                    const SourceLocation &EndLoc) {
    const std::pair<FileID, unsigned> StartLocInfo =
        SM.getDecomposedExpansionLoc(StartLoc);
    llvm::StringRef Buffer(SM.getCharacterData(EndLoc));
    size_t Offset = Buffer.find_first_of(";\r\n");
    assert(Offset != llvm::StringRef::npos);
    const std::pair<FileID, unsigned> EndLocInfo =
        SM.getDecomposedExpansionLoc(EndLoc.getLocWithOffset(Offset + 1));
    assert(StartLocInfo.first == EndLocInfo.first);

    return SourceRange(
        SM.getComposedLoc(StartLocInfo.first, StartLocInfo.second),
        SM.getComposedLoc(EndLocInfo.first, EndLocInfo.second));
  };

  // Add namespace to user overloaded operator declaration
  // double2& operator+=(double2& lhs, const double2& rhs)
  // =>
  // namespace syclct_operator_overloading {
  //
  // double2& operator+=(double2& lhs, const double2& rhs)
  //
  // }
  const auto &SM = *Result.SourceManager;
  const std::string NL = getNL(FD->getBeginLoc(), SM);

  std::ostringstream Prologue;
  // clang-format off
  Prologue << NL
           << "namespace " << NamespaceName << " {" << NL
           << NL;
  // clang-format on

  std::ostringstream Epilogue;
  // clang-format off
  Epilogue << NL
           << "}  // namespace " << NamespaceName << NL
           << NL;
  // clang-format on

  const SourceRange SR =
      GetFunctionSourceRange(SM, FD->getBeginLoc(), FD->getEndLoc());
  emplaceTransformation(new InsertText(SR.getBegin(), Prologue.str()));
  emplaceTransformation(new InsertText(SR.getEnd(), Epilogue.str()));
}

void VectorTypeOperatorRule::TranslateOverloadedOperatorCall(
    const MatchFinder::MatchResult &Result, const CXXOperatorCallExpr *CE) {
  if (!CE)
    return;

  // Explicitly call user overloaded operator
  //
  // For non-assignment operator:
  // a == b
  // =>
  // syclct_operator_overloading::operator==(a, b)
  //
  // For assignment operator:
  // a += b
  // =>
  // a = syclct_operator_overloading::operator+=(a, b)

  const std::string OperatorName = BinaryOperator::getOpcodeStr(
      BinaryOperator::getOverloadedOpcode(CE->getOperator()));

  std::ostringstream FuncCall;

  if (CE->isAssignmentOp()) {
    const auto &SM = *Result.SourceManager;
    const char *Start = SM.getCharacterData(CE->getBeginLoc());
    const char *End = SM.getCharacterData(CE->getOperatorLoc());
    const std::string LHSText(Start, End - Start);
    FuncCall << LHSText << " = ";
  }

  FuncCall << NamespaceName << "::operator" << OperatorName;

  std::string OperatorReplacement = (CE->getNumArgs() == 1)
                                        ? /* Unary operator */ ""
                                        : /* Binary operator */ ",";
  emplaceTransformation(
      new ReplaceToken(CE->getOperatorLoc(), std::move(OperatorReplacement)));
  emplaceTransformation(new InsertBeforeStmt(CE, FuncCall.str() + "("));
  emplaceTransformation(new InsertAfterStmt(CE, ")"));
}

void VectorTypeOperatorRule::run(const MatchFinder::MatchResult &Result) {
  // Add namespace to user overloaded operator declaration
  TranslateOverloadedOperatorDecl(
      Result, getNodeAsType<FunctionDecl>(Result, "overloadedOperatorDecl"));

  // Explicitly call user overloaded operator
  TranslateOverloadedOperatorCall(
      Result,
      getNodeAsType<CXXOperatorCallExpr>(Result, "callOverloadedOperator"));
}

REGISTER_RULE(VectorTypeOperatorRule)

void VectorTypeCtorRule::registerMatcher(MatchFinder &MF) {
  // Find sycl sytle vector:eg.int2 constructors which are part of different
  // casts (representing different syntaxes). This includes copy constructors.
  // All constructors will be visited once.
  MF.addMatcher(
      cxxConstructExpr(hasType(typedefDecl(vectorTypeName())),
                       hasParent(cxxFunctionalCastExpr().bind("CtorFuncCast"))),
      this);

  MF.addMatcher(cxxConstructExpr(hasType(typedefDecl(vectorTypeName())),
                                 hasParent(cStyleCastExpr().bind("CtorCCast"))),
                this);

  // (int2 *)&xxx;
  MF.addMatcher(cStyleCastExpr(hasType(pointsTo(typedefDecl(vectorTypeName()))))
                    .bind("PtrCast"),
                this);

  // make_int2
  auto makeVectorFunc = [&]() {
    std::vector<std::string> MakeVectorFuncNames;
    for (const std::string &TypeName : SupportedVectorTypes) {
      MakeVectorFuncNames.emplace_back("make_" + TypeName);
    }

    return internal::Matcher<NamedDecl>(
        new internal::HasNameMatcher(MakeVectorFuncNames));
  };

  // translate utility for vector type: eg: make_int2
  MF.addMatcher(
      callExpr(callee(functionDecl(makeVectorFunc()))).bind("VecUtilFunc"),
      this);

  // sizeof(int2)
  MF.addMatcher(
      unaryExprOrTypeTraitExpr(allOf(hasArgumentOfType(vectorType()),
                                     has(qualType(hasCanonicalType(type())))))
          .bind("Sizeof"),
      this);
}

// Determines which case of construction applies and creates replacements for
// the syntax. Returns the constructor node and a boolean indicating if a
// closed brace needs to be appended.
void VectorTypeCtorRule::run(const MatchFinder::MatchResult &Result) {
  // Most commonly used syntax cases are checked first.
  if (auto Cast =
          getNodeAsType<CXXFunctionalCastExpr>(Result, "CtorFuncCast")) {
    // int2 a = int2(1); // function style cast
    // int2 b = int2(a); // copy constructor
    // func(int(1), int2(a));
    std::string Replacement = "cl::sycl::" + Cast->getType().getAsString();
    emplaceTransformation(
        new ReplaceToken(Cast->getBeginLoc(), std::move(Replacement)));
    return;
  }

  if (auto Cast = getNodeAsType<CStyleCastExpr>(Result, "CtorCCast")) {
    // int2 a = (int2)1;
    // int2 b = (int2)a; // copy constructor
    // func((int2)1, (int2)a);
    std::string Replacement =
        "(cl::sycl::" + Cast->getType().getAsString() + ")";
    emplaceTransformation(new ReplaceCCast(Cast, std::move(Replacement)));
    return;
  }

  if (const CallExpr *CE = getNodeAsType<CallExpr>(Result, "VecUtilFunc")) {
    const llvm::StringRef FuncName = CE->getDirectCallee()->getName();
    assert(FuncName.startswith("make_") &&
           "Found non make_<vector type> function");
    llvm::StringRef TypeName = FuncName.substr(strlen("make_"));
    emplaceTransformation(
        new ReplaceStmt(CE->getCallee(), "cl::sycl::" + TypeName.str()));
    return;
  }

  if (const CStyleCastExpr *CPtrCast =
          getNodeAsType<CStyleCastExpr>(Result, "PtrCast")) {
    emplaceTransformation(
        new InsertNameSpaceInCastExpr(CPtrCast, "cl::sycl::"));
    return;
  }

  if (const UnaryExprOrTypeTraitExpr *ExprSizeof =
          getNodeAsType<UnaryExprOrTypeTraitExpr>(Result, "Sizeof")) {
    if (ExprSizeof->isArgumentType()) {
      emplaceTransformation(new InsertText(ExprSizeof->getArgumentTypeInfo()
                                               ->getTypeLoc()
                                               .getSourceRange()
                                               .getBegin(),
                                           "cl::sycl::"));
    }
    return;
  }
}

REGISTER_RULE(VectorTypeCtorRule)

void ReplaceDim3CtorRule::registerMatcher(MatchFinder &MF) {
  // Find dim3 constructors which are part of different casts (representing
  // different syntaxes). This includes copy constructors. All constructors
  // will be visited once.
  MF.addMatcher(cxxConstructExpr(hasType(typedefDecl(hasName("dim3"))),
                                 argumentCountIs(1),
                                 unless(hasAncestor(cxxConstructExpr(
                                     hasType(typedefDecl(hasName("dim3")))))))
                    .bind("dim3Top"),
                this);

  MF.addMatcher(cxxConstructExpr(hasType(typedefDecl(hasName("dim3"))),
                                 argumentCountIs(3), hasParent(varDecl()),
                                 unless(hasAncestor(cxxConstructExpr(
                                     hasType(typedefDecl(hasName("dim3")))))))
                    .bind("dim3CtorDecl"),
                this);

  MF.addMatcher(
      cxxConstructExpr(
          hasType(typedefDecl(hasName("dim3"))), argumentCountIs(3),
          // skip fields in a struct.  The source loc is
          // messed up (points to the start of the struct)
          unless(hasAncestor(cxxRecordDecl())), unless(hasParent(varDecl())),
          unless(hasAncestor(
              cxxConstructExpr(hasType(typedefDecl(hasName("dim3")))))))
          .bind("dim3CtorNoDecl"),
      this);
}

ReplaceDim3Ctor *ReplaceDim3CtorRule::getReplaceDim3Modification(
    const MatchFinder::MatchResult &Result) {
  if (auto Ctor = getNodeAsType<CXXConstructExpr>(Result, "dim3CtorDecl")) {
    // dim3 a(1);
    if (Ctor->getParenOrBraceRange().isInvalid()) {
      // dim3 a;
      // No replacements are needed
      return nullptr;
    } else {
      // dim3 a(1);
      return new ReplaceDim3Ctor(Ctor, SSM, true /*isDecl*/);
    }
  } else if (auto Ctor =
                 getNodeAsType<CXXConstructExpr>(Result, "dim3CtorNoDecl")) {
    return new ReplaceDim3Ctor(Ctor, SSM);
  } else if (auto Ctor = getNodeAsType<CXXConstructExpr>(Result, "dim3Top")) {
    if (auto A = ReplaceDim3Ctor::getConstructExpr(Ctor->getArg(0))) {
      // strip the top CXXConstructExpr, if there's a CXXConstructExpr further
      // down
      return new ReplaceDim3Ctor(Ctor, SSM, A);
    } else {
      // Copy constructor case: dim3 a(copyfrom)
      // No replacements are needed
      return nullptr;
    }
  }
  return nullptr;
}

void ReplaceDim3CtorRule::run(const MatchFinder::MatchResult &Result) {
  ReplaceDim3Ctor *R = getReplaceDim3Modification(Result);
  if (R) {
    // add a transformation that will filter out all nested transformations
    emplaceTransformation(R->getEmpty());
    // all the nested transformations will be applied when R->getReplacement()
    // is called
    emplaceTransformation(R);
  }
}

REGISTER_RULE(ReplaceDim3CtorRule)

void Dim3MemberFieldsRule::FieldsRename(const MatchFinder::MatchResult &Result,
                                        std::string Str, const MemberExpr *ME) {
  auto SM = Result.SourceManager;
  SourceLocation Begin = SM->getSpellingLoc(ME->getBeginLoc());
  SourceLocation End = SM->getSpellingLoc(ME->getEndLoc());
  std::string Ret =
      std::string(SM->getCharacterData(Begin), SM->getCharacterData(End));

  std::size_t Position = std::string::npos;
  std::size_t Current = Ret.find(Str);

  // Find the last position of dot '.'
  while (Current != std::string::npos) {
    Position = Current;
    Current = Ret.find(Str, Position + 1);
  }

  if (Position != std::string::npos) {
    auto Search = MapNames::Dim3MemberNamesMap.find(
        ME->getMemberNameInfo().getAsString());
    if (Search != MapNames::Dim3MemberNamesMap.end()) {
      emplaceTransformation(
          new RenameFieldInMemberExpr(ME, Search->second + "", Position));
      std::string NewMemberStr = Ret.substr(0, Position) + Search->second;
      StmtStringPair SSP = {ME, NewMemberStr};
      SSM->insert(SSP);
    }
  }
}

// rule for dim3 types member fields replacements.
void Dim3MemberFieldsRule::registerMatcher(MatchFinder &MF) {
  // dim3->x/y/z => dim3->operator[](0)/(1)/(2)
  MF.addMatcher(
      memberExpr(
          has(implicitCastExpr(hasType(pointsTo(typedefDecl(hasName("dim3")))))
                  .bind("ImplCast")))
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
    // E.g.
    // dim3 *pd3;
    // pd3->x;
    // will translate to:
    // cl::sycl::range<3> *pd3;
    // (*pd3)[0];
    auto Impl = getAssistNodeAsType<ImplicitCastExpr>(Result, "ImplCast");
    emplaceTransformation(new InsertBeforeStmt(Impl, "(*"));
    emplaceTransformation(new InsertAfterStmt(Impl, ")"));

    FieldsRename(Result, "->", ME);
  }

  if (const MemberExpr *ME =
          getNodeAsType<MemberExpr>(Result, "Dim3MemberDotExpr")) {
    FieldsRename(Result, ".", ME);
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
  auto functionName = [&]() {
    return hasAnyName(
        "cudaGetDeviceCount", "cudaGetDeviceProperties", "cudaDeviceReset",
        "cudaSetDevice", "cudaDeviceGetAttribute", "cudaDeviceGetP2PAttribute",
        "cudaGetDevice", "cudaGetLastError", "cudaPeekAtLastError",
        "cudaDeviceSynchronize", "cudaThreadSynchronize", "cudaGetErrorString",
        "cudaGetErrorName", "cudaDeviceSetCacheConfig",
        "cudaDeviceGetCacheConfig", "__longlong_as_double",
        "__double_as_longlong", "clock");
  };
  MF.addMatcher(callExpr(allOf(callee(functionDecl(functionName())),
                               hasParent(compoundStmt())))
                    .bind("FunctionCall"),
                this);
  MF.addMatcher(callExpr(allOf(callee(functionDecl(functionName())),
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

  std::string Prefix = "";
  std::string Poststr = "";
  if (IsAssigned) {
    Prefix = "(";
    Poststr = ", 0)";
  }

  if (FuncName == "cudaGetDeviceCount") {
    std::string ResultVarName = DereferenceArg(CE->getArg(0));
    emplaceTransformation(new InsertBeforeStmt(CE, ResultVarName + " = "));
    emplaceTransformation(
        new ReplaceStmt(CE, "syclct::get_device_manager().device_count()"));
  } else if (FuncName == "cudaGetDeviceProperties") {
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP);
    }
    std::string ResultVarName = DereferenceArg(CE->getArg(0));
    emplaceTransformation(new ReplaceStmt(
        CE->getCallee(), Prefix + "syclct::get_device_manager().get_device"));
    emplaceTransformation(new RemoveArg(CE, 0));
    emplaceTransformation(new InsertAfterStmt(
        CE, ".get_device_info(" + ResultVarName + ")" + Poststr));
  } else if (FuncName == "cudaDeviceReset") {
    emplaceTransformation(new ReplaceStmt(
        CE, "syclct::get_device_manager().current_device().reset()"));
  } else if (FuncName == "cudaSetDevice") {
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP);
    }
    emplaceTransformation(
        new ReplaceStmt(CE->getCallee(),
                        Prefix + "syclct::get_device_manager().select_device"));
    if (IsAssigned)
      emplaceTransformation(new InsertAfterStmt(CE, ", 0)"));

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
  } else if (FuncName == "cudaDeviceSynchronize" ||
             FuncName == "cudaThreadSynchronize") {
    std::string ReplStr = "syclct::get_device_manager()."
                          "current_device().queues_wait_"
                          "and_throw()";
    if (IsAssigned) {
      ReplStr = "(" + ReplStr + ", 0)";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP);
    }
    emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));

  } else if (FuncName == "cudaGetLastError" ||
             FuncName == "cudaPeekAtLastError") {
    report(CE->getBeginLoc(),
           Comments::TRNA_WARNING_ERROR_HANDLING_API_REPLACED_0, FuncName);
    emplaceTransformation(new ReplaceStmt(CE, "0"));
  } else if (FuncName == "cudaGetErrorString" ||
             FuncName == "cudaGetErrorName") {
    report(CE->getBeginLoc(),
           Comments::TRNA_WARNING_ERROR_HANDLING_API_COMMENTED, FuncName);
    emplaceTransformation(
        new InsertBeforeStmt(CE, "\"" + FuncName + " not supported\"/*"));
    emplaceTransformation(new InsertAfterStmt(CE, "*/"));
  } else if (FuncName == "cudaDeviceSetCacheConfig" ||
             FuncName == "cudaDeviceGetCacheConfig") {
    // SYCL has no corresponding implementation for
    // "cudaDeviceSetCacheConfig/cudaDeviceGetCacheConfig", so simply translate
    // "cudaDeviceSetCacheConfig/cudaDeviceGetCacheConfig" into expression "0;".
    std::string Replacement = "0";
    emplaceTransformation(new ReplaceStmt(CE, std::move(Replacement)));
  } else if (FuncName == "__double_as_longlong") {
    emplaceTransformation(new ReplaceCalleeName(CE, "syclct::d2ll"));
  } else if (FuncName == "__longlong_as_double") {
    emplaceTransformation(new ReplaceCalleeName(CE, "syclct::ll2d"));
  } else if (FuncName == "clock") {
    report(CE->getBeginLoc(), Diagnostics::API_NOT_TRANSLATED_SYCL_UNDEF);
  } else {
    syclct_unreachable("Unknown function name");
  }
}

REGISTER_RULE(FunctionCallRule)

// kernel call information collection
void KernelCallRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  MF.addMatcher(
      cudaKernelCallExpr(hasAncestor(functionDecl().bind("callContext")))
          .bind("kernelCall"),
      this);
}

void KernelCallRule::run(const ast_matchers::MatchFinder::MatchResult &Result) {
  auto FD = getAssistNodeAsType<FunctionDecl>(Result, "callContext");
  if (auto KCall =
          getAssistNodeAsType<CUDAKernelCallExpr>(Result, "kernelCall")) {
    emplaceTransformation(new ReplaceStmt(KCall, ""));
    if (!FD->isImplicitlyInstantiable())
      SyclctGlobalInfo::getInstance().registerKernelCallExpr(KCall);
  }
}

REGISTER_RULE(KernelCallRule)

// __device__ function call information collection
void DeviceFunctionCallRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  MF.addMatcher(
      callExpr(hasAncestor(functionDecl(anyOf(hasAttr(attr::CUDADevice),
                                              hasAttr(attr::CUDAGlobal)),
                                        unless(hasAttr(attr::CUDAHost)))
                               .bind("funcDecl")))
          .bind("callExpr"),
      this);
}

void DeviceFunctionCallRule::run(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  auto CE = getAssistNodeAsType<CallExpr>(Result, "callExpr");
  auto FD = getAssistNodeAsType<FunctionDecl>(Result, "funcDecl");
  if (CE && FD) {
    if (!FD->isImplicitlyInstantiable())
      SyclctGlobalInfo::getInstance().registerDeviceFunctionInfo(FD)->addCallee(
          CE);
  }
}

REGISTER_RULE(DeviceFunctionCallRule)

/// __constant__/__shared__/__device__ var information collection
void MemVarRule::registerMatcher(MatchFinder &MF) {
  auto DeclMatcher = varDecl(
      anyOf(hasAttr(attr::CUDAConstant), hasAttr(attr::CUDADevice),
            hasAttr(attr::CUDAShared)),
      unless(hasAnyName("threadIdx", "blockDim", "blockIdx", "gridDim")));
  MF.addMatcher(DeclMatcher.bind("var"), this);
  MF.addMatcher(
      declRefExpr(anyOf(hasParent(implicitCastExpr(
                                      unless(hasParent(arraySubscriptExpr())))
                                      .bind("impl")),
                        anything()),
                  to(DeclMatcher), hasAncestor(functionDecl().bind("func")))
          .bind("used"),
      this);
}

void MemVarRule::insertExplicitCast(const ImplicitCastExpr *Impl,
                                    const QualType &Type) {
  if (Impl->getCastKind() == CastKind::CK_LValueToRValue) {
    if (!Type->isArrayType()) {
      auto TypeName = Type.getAsString();
      if (Type->isPointerType()) {
        TypeName = Type->getPointeeType().getAsString();
      }
      auto Itr = MapNames::TypeNamesMap.find(TypeName);
      if (Itr != MapNames::TypeNamesMap.end())
        TypeName = Itr->second;
      if (Type->isPointerType()) {
        TypeName += "*";
      }
      emplaceTransformation(new InsertBeforeStmt(Impl, "(" + TypeName + ")"));
    }
  }
}

void MemVarRule::run(const MatchFinder::MatchResult &Result) {
  if (auto MemVar = getNodeAsType<VarDecl>(Result, "var")) {
    emplaceTransformation(ReplaceVarDecl::getVarDeclReplacement(
        MemVar,
        MemVarInfo::buildMemVarInfo(MemVar)->getDeclarationReplacement()));
  }
  auto MemVarRef = getNodeAsType<DeclRefExpr>(Result, "used");
  auto Func = getAssistNodeAsType<FunctionDecl>(Result, "func");
  SyclctGlobalInfo &Global = SyclctGlobalInfo::getInstance();
  if (MemVarRef && Func) {
    if (Func->hasAttr<CUDAGlobalAttr>() ||
        (Func->hasAttr<CUDADeviceAttr>() && !Func->hasAttr<CUDAHostAttr>())) {
      auto VD = dyn_cast<VarDecl>(MemVarRef->getDecl());
      if (auto Var = Global.findMemVarInfo(VD))
        Global.registerDeviceFunctionInfo(Func)->addVar(Var);
      if (auto Impl = getAssistNodeAsType<ImplicitCastExpr>(Result, "impl"))
        insertExplicitCast(Impl, VD->getType());
    }
  }
}

REGISTER_RULE(MemVarRule)

void MemoryTranslationRule::MallocTranslation(
    const MatchFinder::MatchResult &Result, const CallExpr *C) {
  SyclctGlobalInfo::getInstance().registerCudaMalloc(C);
  emplaceTransformation(new ReplaceCalleeName(C, "syclct::sycl_malloc"));
}

void MemoryTranslationRule::MemcpyTranslation(
    const MatchFinder::MatchResult &Result, const CallExpr *C) {
  // Input:
  //   cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  //   cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
  //   cudaMemcpy(x_A, y_A, size, someDynamicCudaMemcpyKindValue);
  //
  // Desired output:
  //   sycl_memcpy<float>(d_A, h_A, numElements);
  //   sycl_memcpy_back<float>(h_A, d_A, numElements);
  //   sycl_memcpy<float>(x_A, y_A, numElements,
  //   someDynamicCudaMemcpyKindValue);
  //
  // Current output:
  //   sycl_memcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  //   sycl_memcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
  //   sycl_memcpy(x_A, y_A, size, someDynamicCudaMemcpyKindValue);

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

  emplaceTransformation(new ReplaceCalleeName(C, "syclct::sycl_memcpy"));
  emplaceTransformation(new InsertBeforeStmt(C->getArg(0), "(void*)("));
  emplaceTransformation(new InsertAfterStmt(C->getArg(0), ")"));
  emplaceTransformation(new InsertBeforeStmt(C->getArg(1), "(void*)("));
  emplaceTransformation(new InsertAfterStmt(C->getArg(1), ")"));
  emplaceTransformation(
      new ReplaceStmt(C->getArg(3), std::move(DirectionName)));
}

void MemoryTranslationRule::MemcpyToSymbolTranslation(
    const MatchFinder::MatchResult &Result, const CallExpr *C) {
  // Input:
  //   cudaMemcpyToSymbol(d_A, h_A, size, offset, cudaMemcpyHostToDevice);
  //   cudaMemcpyToSymbol(d_B, d_C, size, offset, cudaMemcpyDeviceToDevice);
  //   cudaMemcpyToSymbol(h_A, d_B, size, offset, cudaMemcpyDefault);

  // Desired output:
  //   syclct::sycl_memcpy_to_symbol(d_A.get_ptr(), (void*)(h_A), size,
  //                                 offset, syclct::host_to_device);
  //
  //   syclct::sycl_memcpy_to_symbol(d_B.get_ptr(), d_C, size, offset,
  //                                 syclct::device_to_device);
  //
  //   syclct::sycl_memcpy_to_symbol(h_A.get_ptr(), (void*)(d_B), size,
  //                                 offset, syclct::automatic);
  const Expr *Direction = C->getArg(4);
  std::string DirectionName;
  const DeclRefExpr *DD = dyn_cast_or_null<DeclRefExpr>(Direction);
  if (DD && isa<EnumConstantDecl>(DD->getDecl())) {
    DirectionName = DD->getNameInfo().getName().getAsString();
    auto Search = EnumConstantRule::EnumNamesMap.find(DirectionName);
    assert(Search != EnumConstantRule::EnumNamesMap.end());
    Direction = nullptr;
    DirectionName = "syclct::" + Search->second;
  }

  SyclctGlobalInfo &Global = SyclctGlobalInfo::getInstance();
  auto MallocInfo = Global.findCudaMalloc(C->getArg(1));
  auto VD = CudaMallocInfo::getDecl(C->getArg(0));
  if (MallocInfo && VD) {
    if (auto Var = Global.findMemVarInfo(VD)) {
      emplaceTransformation(new ReplaceStmt(
          C, Var->getName() + ".assign(" +
                 MallocInfo->getAssignArgs(Var->getType()->getName()) + ")"));
      return;
    }
  }

  std::string VarName = getStmtSpelling(C->getArg(0), *Result.Context);
  // Translate variable name such as "&const_angle[0]", "&const_one"
  // into "const_angle.get_ptr()", "const_one.get_ptr()".
  VarName.erase(std::remove(VarName.begin(), VarName.end(), '&'),
                VarName.end());
  std::size_t pos = VarName.find("[");
  VarName = (pos != std::string::npos) ? VarName.substr(0, pos) : VarName;
  VarName += ".get_ptr()";

  emplaceTransformation(
      new ReplaceCalleeName(C, "syclct::sycl_memcpy_to_symbol"));
  emplaceTransformation(new ReplaceToken(C->getArg(0)->getBeginLoc(),
                                         C->getArg(0)->getEndLoc(),
                                         std::move(VarName)));
  emplaceTransformation(new InsertBeforeStmt(C->getArg(1), "(void*)("));
  emplaceTransformation(new InsertAfterStmt(C->getArg(1), ")"));
  emplaceTransformation(
      new ReplaceStmt(C->getArg(4), std::move(DirectionName)));
}

void MemoryTranslationRule::MemcpyFromSymbolTranslation(
    const MatchFinder::MatchResult &Result, const CallExpr *C) {
  // Input:
  //   cudaMemcpyToSymbol(h_A, d_A, size, offset, cudaMemcpyDeviceToHost);
  //   cudaMemcpyToSymbol(d_B, d_A, size, offset, cudaMemcpyDeviceToDevice);

  // Desired output:
  //   syclct::sycl_memcpy_to_symbol((void*)(h_A), d_A.get_ptr(), size, offset,
  //                                 syclct::device_to_host);
  //
  //   syclct::sycl_memcpy_to_symbol((void*)(d_B), d_A.get_ptr(), size, offset,
  //                                 syclct::device_to_device);
  const Expr *Direction = C->getArg(4);
  std::string DirectionName;
  const DeclRefExpr *DD = dyn_cast_or_null<DeclRefExpr>(Direction);
  if (DD && isa<EnumConstantDecl>(DD->getDecl())) {
    DirectionName = DD->getNameInfo().getName().getAsString();
    auto Search = EnumConstantRule::EnumNamesMap.find(DirectionName);
    assert(Search != EnumConstantRule::EnumNamesMap.end());
    Direction = nullptr;
    DirectionName = "syclct::" + Search->second;
  }

  std::string VarName = getStmtSpelling(C->getArg(1), *Result.Context);
  // Translate variable name such as "&const_angle[0]", "&const_one"
  // into "const_angle.get_ptr()", "const_one.get_ptr()".
  VarName.erase(std::remove(VarName.begin(), VarName.end(), '&'),
                VarName.end());
  std::size_t pos = VarName.find("[");
  VarName = (pos != std::string::npos) ? VarName.substr(0, pos) : VarName;
  VarName += ".get_ptr()";

  emplaceTransformation(new InsertBeforeStmt(C->getArg(0), "(void*)("));
  emplaceTransformation(new InsertAfterStmt(C->getArg(0), ")"));
  emplaceTransformation(
      new ReplaceCalleeName(C, "syclct::sycl_memcpy_from_symbol"));
  emplaceTransformation(new ReplaceToken(C->getArg(1)->getBeginLoc(),
                                         C->getArg(1)->getEndLoc(),
                                         std::move(VarName)));
  emplaceTransformation(
      new ReplaceStmt(C->getArg(4), std::move(DirectionName)));
}

void MemoryTranslationRule::FreeTranslation(
    const MatchFinder::MatchResult &Result, const CallExpr *C) {
  emplaceTransformation(new ReplaceCalleeName(C, "syclct::sycl_free"));
}

void MemoryTranslationRule::MemsetTranslation(
    const MatchFinder::MatchResult &Result, const CallExpr *C) {
  emplaceTransformation(new ReplaceCalleeName(C, "syclct::sycl_memset"));
  emplaceTransformation(new InsertBeforeStmt(C->getArg(0), "(void*)("));
  emplaceTransformation(new InsertAfterStmt(C->getArg(0), ")"));
  emplaceTransformation(new InsertBeforeStmt(C->getArg(1), "(int)("));
  emplaceTransformation(new InsertAfterStmt(C->getArg(1), ")"));
  emplaceTransformation(new InsertBeforeStmt(C->getArg(2), "(size_t)("));
  emplaceTransformation(new InsertAfterStmt(C->getArg(2), ")"));
}

// Memory translation rules live here.
void MemoryTranslationRule::registerMatcher(MatchFinder &MF) {
  auto memoryAPI = [&]() {
    return hasAnyName("cudaMalloc", "cudaMemcpy", "cudaMemcpyToSymbol",
                      "cudaMemcpyFromSymbol", "cudaFree", "cudaMemset");
  };

  MF.addMatcher(callExpr(allOf(callee(functionDecl(memoryAPI())),
                               hasParent(compoundStmt())))
                    .bind("call"),
                this);

  MF.addMatcher(callExpr(allOf(callee(functionDecl(memoryAPI())),
                               unless(hasParent(compoundStmt()))))
                    .bind("callUsed"),
                this);
}

void MemoryTranslationRule::run(const MatchFinder::MatchResult &Result) {
  auto TranslateCallExpr = [&](const CallExpr *C, const bool IsAssigned) {
    if (!C)
      return;

    if (IsAssigned) {
      report(C->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP);
      emplaceTransformation(new InsertBeforeStmt(C, "("));
      emplaceTransformation(new InsertAfterStmt(C, ", 0)"));
    }

    static const std::unordered_map<
        std::string, std::function<void(const MatchFinder::MatchResult &Result,
                                        const CallExpr *C)>>
        TranslationDispatcher = {
#define MEMTRANS_DECLFIND(Name)                                                \
  {"cuda" #Name, std::bind(&MemoryTranslationRule::Name##Translation, this,    \
                           std::placeholders::_1, std::placeholders::_2)},
            // clang-format off
            MEMTRANS_DECLFIND(Malloc)
            MEMTRANS_DECLFIND(Memcpy)
            MEMTRANS_DECLFIND(MemcpyToSymbol)
            MEMTRANS_DECLFIND(MemcpyFromSymbol)
            MEMTRANS_DECLFIND(Free)
            MEMTRANS_DECLFIND(Memset)
// clang-format on
#undef MEMTRANS_DECLFIND
        };

    const std::string Name =
        C->getCalleeDecl()->getAsFunction()->getNameAsString();
    assert(TranslationDispatcher.find(Name) != TranslationDispatcher.end());
    TranslationDispatcher.at(Name)(Result, C);
  };

  TranslateCallExpr(getNodeAsType<CallExpr>(Result, "call"),
                    /* IsAssigned */ false);
  TranslateCallExpr(getNodeAsType<CallExpr>(Result, "callUsed"),
                    /* IsAssigned */ true);
}

REGISTER_RULE(MemoryTranslationRule)

static const CXXConstructorDecl *getIfConstructorDecl(const Decl *ND) {
  if (const auto *Tmpl = dyn_cast<FunctionTemplateDecl>(ND))
    ND = Tmpl->getTemplatedDecl();
  return dyn_cast<CXXConstructorDecl>(ND);
}

void ErrorTryCatchRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  MF.addMatcher(functionDecl(hasBody(compoundStmt()),
                             unless(anyOf(hasAttr(attr::CUDAGlobal),
                                          hasAttr(attr::CUDADevice),
                                          hasAncestor(lambdaExpr(anything())))))
                    .bind("functionDecl"),
                this);
}

void ErrorTryCatchRule::run(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  const FunctionDecl *FD = getNodeAsType<FunctionDecl>(Result, "functionDecl");
  if (!FD)
    return;

  // Filter out compiler generated methods
  if (const CXXMethodDecl *CXXMDecl = dyn_cast<CXXMethodDecl>(FD)) {
    if (!CXXMDecl->isUserProvided()) {
      return;
    }
  }

  auto BodySLoc = FD->getBody()->getSourceRange().getBegin().getRawEncoding();
  if (Insertions.find(BodySLoc) != Insertions.end())
    return;

  Insertions.insert(BodySLoc);

  // First check if this is a constructor decl
  if (const CXXConstructorDecl *CDecl = getIfConstructorDecl(FD)) {
    emplaceTransformation(new InsertBeforeCtrInitList(CDecl, "try "));
  } else {
    emplaceTransformation(new InsertBeforeStmt(FD->getBody(), "try "));
  }

  emplaceTransformation(new InsertAfterStmt(
      FD->getBody(), "\ncatch (cl::sycl::exception const &exc) {\n"
                     "  std::cerr << exc.what() << \"EOE at line \" << "
                     "__LINE__ << std::endl;\n"
                     "  std::exit(1);\n"
                     "}"));
}

REGISTER_RULE(ErrorTryCatchRule)

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
  auto C = getNodeAsType<CallExpr>(Result, "math");
  if (!C) {
    return;
  }

  const std::string FuncName = C->getDirectCallee()->getNameAsString();

  if (FunctionNamesMap.count(FuncName) != 0) {
    std::string NewFuncName = FunctionNamesMap.at(FuncName);
    if (FuncName == "abs") {
      // further check the type of the args.
      if (!C->getArg(0)->getType()->isIntegerType()) {
        NewFuncName = "cl::sycl::fabs";
      }
    }
    emplaceTransformation(new ReplaceCalleeName(C, std::move(NewFuncName)));

    if (FuncName == "min") {
      const LangOptions &LO = Result.Context->getLangOpts();
      std::string FT = C->getType().getAsString(PrintingPolicy(LO));
      for (unsigned i = 0; i < C->getNumArgs(); i++) {
        std::string ArgT =
            C->getArg(i)->getType().getAsString(PrintingPolicy(LO));
        std::string ArgExpr = C->getArg(i)->getStmtClassName();
        if (ArgT != FT || ArgExpr == "BinaryOperator") {
          emplaceTransformation(
              new InsertBeforeStmt(C->getArg(i), "(" + FT + ")("));
          emplaceTransformation(new InsertAfterStmt(C->getArg(i), ")"));
        }
      }
    }
  }
}

REGISTER_RULE(MathFunctionsRule)

void SyncThreadsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(callExpr(callee(functionDecl(hasAnyName("__syncthreads"))),
                         hasAncestor(functionDecl().bind("func")))
                    .bind("syncthreads"),
                this);
}

void SyncThreadsRule::run(const MatchFinder::MatchResult &Result) {
  if (auto CE = getNodeAsType<CallExpr>(Result, "syncthreads")) {
    if (auto FD = getAssistNodeAsType<FunctionDecl>(Result, "func"))
      SyclctGlobalInfo::getInstance().registerDeviceFunctionInfo(FD)->setItem();
    std::string Replacement = getItemName() + ".barrier()";
    emplaceTransformation(new ReplaceStmt(CE, std::move(Replacement)));
  }
}

REGISTER_RULE(SyncThreadsRule)

void KernelFunctionInfoRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      varDecl(hasType(recordDecl(hasName("cudaFuncAttributes")))).bind("decl"),
      this);
  MF.addMatcher(
      callExpr(callee(functionDecl(hasAnyName("cudaFuncGetAttributes"))))
          .bind("call"),
      this);
  MF.addMatcher(memberExpr(hasObjectExpression(hasType(
                               recordDecl(hasName("cudaFuncAttributes")))))
                    .bind("member"),
                this);
}

void KernelFunctionInfoRule::run(const MatchFinder::MatchResult &Result) {
  if (auto V = getNodeAsType<VarDecl>(Result, "decl"))
    emplaceTransformation(
        new ReplaceTypeInDecl(V, "sycl_kernel_function_info"));
  else if (auto C = getNodeAsType<CallExpr>(Result, "call")) {
    emplaceTransformation(
        new ReplaceToken(C->getBeginLoc(), "(get_kernel_function_info"));
    emplaceTransformation(new InsertAfterStmt(C, ", 0)"));
    auto FuncArg = C->getArg(1);
    emplaceTransformation(new InsertBeforeStmt(FuncArg, "(const void *)"));
  } else if (auto M = getNodeAsType<MemberExpr>(Result, "member")) {
    auto MemberName = M->getMemberNameInfo();
    auto NameMap = AttributesNamesMap.find(MemberName.getAsString());
    if (NameMap != AttributesNamesMap.end())
      emplaceTransformation(new ReplaceToken(MemberName.getBeginLoc(),
                                             std::string(NameMap->second)));
  }
}

REGISTER_RULE(KernelFunctionInfoRule)

void TypeCastRule::registerMatcher(MatchFinder &MF) {

  MF.addMatcher(
      declRefExpr(hasParent(implicitCastExpr(
                      hasParent(cStyleCastExpr(unless(
                          hasType(pointsTo(typedefDecl(hasName("double2"))))))),
                      hasType(pointsTo(typedefDecl(hasName("double2")))))))

          .bind("Double2CastExpr"),
      this);
}

void TypeCastRule::run(const MatchFinder::MatchResult &Result) {

  if (const DeclRefExpr *E =
          getNodeAsType<DeclRefExpr>(Result, "Double2CastExpr")) {
    std::string Name = E->getNameInfo().getName().getAsString();

    emplaceTransformation(new InsertBeforeStmt(E, "(&"));
    emplaceTransformation(new InsertAfterStmt(E, "[0])"));
  }
}

REGISTER_RULE(TypeCastRule)

void RecognizeAPINameRule::registerMatcher(MatchFinder &MF) {
  std::vector<std::string> AllAPINames =
      TranslationStatistics::GetAllAPINames();

  MF.addMatcher(callExpr(allOf(callee(functionDecl(internal::Matcher<NamedDecl>(
                                   new internal::HasNameMatcher(AllAPINames)))),
                               unless(hasAncestor(cudaKernelCallExpr()))))
                    .bind("APINamesUsed"),
                this);
}

void RecognizeAPINameRule::run(const MatchFinder::MatchResult &Result) {
  const CallExpr *C = getNodeAsType<CallExpr>(Result, "APINamesUsed");
  if (!C) {
    return;
  }

  std::string APIName = C->getCalleeDecl()->getAsFunction()->getNameAsString();
  if (!TranslationStatistics::IsTranslated(APIName)) {

    const SourceManager &SM = (*Result.Context).getSourceManager();
    const SourceLocation FileLoc = SM.getFileLoc(C->getBeginLoc());
    std::string SLStr = FileLoc.printToString(SM);
    std::size_t Pos = SLStr.find(':');
    std::string FileName = SLStr.substr(0, Pos);
    LOCStaticsMap[FileName][2]++;

    report(C->getBeginLoc(), Comments::API_NOT_TRANSLATED, APIName.c_str());
  }
}

REGISTER_RULE(RecognizeAPINameRule)

void ASTTraversalManager::matchAST(ASTContext &Context, TransformSetTy &TS,
                                   StmtStringMap &SSM) {
  this->Context = &Context;
  for (auto &I : Storage) {
    I->registerMatcher(Matchers);
    if (auto TR = dyn_cast<TranslationRule>(&*I)) {
      TR->TM = this;
      TR->setTransformSet(TS);
      TR->setStmtStringMap(SSM);
    }
  }

  DebugInfo::printTranslationRules(Storage);

  Matchers.matchAST(Context);

  DebugInfo::printMatchedRules(Storage);
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
