//===--- Utility.cpp -----------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===---------------------------------------------------------------===//

#include "Utility.h"

#include "SaveNewFiles.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <algorithm>

using namespace llvm;
using namespace clang;
using namespace std;

namespace path = llvm::sys::path;
namespace fs = llvm::sys::fs;

bool makeCanonical(SmallVectorImpl<char> &Path) {
  if (fs::make_absolute(Path) != std::error_code()) {
    llvm::errs() << "Could not get absolute path from '" << Path << "'\n ";
    return false;
  }
  path::remove_dots(Path, /* remove_dot_dot= */ true);
  return true;
}

bool makeCanonical(string &PathPar) {
  SmallString<256> Path = StringRef(PathPar);
  if (!makeCanonical(Path))
    return false;
  PathPar.assign(begin(Path), end(Path));
  return true;
}

bool isCanonical(StringRef Path) {
  bool HasNoDots = all_of(path::begin(Path), path::end(Path),
                          [](StringRef e) { return e != "." && e != ".."; });
  return HasNoDots && path::is_absolute(Path);
}

bool isChildPath(const std::string &Root, const std::string &Child) {
  auto Diff = mismatch(path::begin(Root), path::end(Root), path::begin(Child));
  // Root is not considered prefix of Child if they are equal.
  return Diff.first == path::end(Root) && Diff.second != path::end(Child);
}

const char *getNL(SourceLocation Loc, const SourceManager &SM) {
  auto LocInfo = SM.getDecomposedLoc(Loc);
  auto Buffer = SM.getBufferData(LocInfo.first);
  Buffer = Buffer.data() + LocInfo.second;
  // Search for both to avoid searching till end of file.
  auto pos = Buffer.find_first_of("\r\n");
  if (pos == StringRef::npos || Buffer[pos] == '\n')
    return "\n";
  else
    return "\r\n";
}

StringRef getIndent(SourceLocation Loc, const SourceManager &SM) {
  auto LocInfo = SM.getDecomposedLoc(Loc);
  auto Buffer = SM.getBufferData(LocInfo.first);
  // Find start of indentation.
  auto begin = Buffer.find_last_of('\n', LocInfo.second);
  if (begin != StringRef::npos) {
    ++begin;
  } else {
    // We're at the beginning of the file.
    begin = 0;
  }
  auto end = Buffer.find_if([](char c) { return !isspace(c); }, begin);
  return Buffer.substr(begin, end - begin);
}

// Get textual representation of the Stmt.  This helper function is tricky.
// Ideally, we should use SourceLocation information to be able to access the
// actual character used for spelling in the source code (either before or
// after preprocessor). But the quality of the this information is bad.  This
// should be addressed in the clang sources in the long run, but we need a
// working solution right now, so we use another way of getting the spelling.
// Specific example, when SourceLocation information is broken - DeclRefExpr
// has valid information only about beginning of the expression, pointers to
// the end of the expression point to the beginning.
std::string getStmtSpelling(const Stmt *S, const ASTContext &Context) {
  std::string StrBuffer;
  llvm::raw_string_ostream TmpStream(StrBuffer);
  auto LangOpts = Context.getLangOpts();
  S->printPretty(TmpStream, nullptr, PrintingPolicy(LangOpts), 0, "\n",
                 &Context);
  return TmpStream.str();
}

SourceProcessType GetSourceFileType(llvm::StringRef SourcePath) {
  SmallString<256> FilePath = SourcePath;
  auto Extension = path::extension(FilePath);

  if (Extension == ".cu") {
    return TypeCudaSource;
  } else if (Extension == ".cuh") {
    return TypeCudaHeader;
  } else if (Extension == ".cpp" || Extension == ".cxx" || Extension == ".cc" ||
             Extension == ".c") {
    return TypeCppSource;
  } else if (Extension == ".hpp" || Extension == ".hxx" || Extension == ".h") {
    return TypeCppHeader;
  } else {
    llvm::errs() << "[ERROR] Not support\"" << Extension << "\" file type!\n";
    std::exit(1);
  }
}

std::vector<std::string>
ruleTopoSort(std::vector<std::vector<std::string>> &TableRules) {
  std::vector<std::string> Vec;

  std::vector<std::list<int>> AdjacencyList;
  std::vector<int> InDegree;
  std::stack<int> Stack;
  std::vector<std::string> RuleNames;

  int n = TableRules.size();
  AdjacencyList.assign(n, std::list<int>());
  InDegree.assign(n, 0);

  for (int i = 0; i < n; i++) {
    RuleNames.push_back(TableRules[i].at(0));
  }

  for (int i = 0; i < n; i++) {
    for (std::vector<std::string>::iterator it = TableRules[i].begin() + 1;
         it != TableRules[i].end(); ++it) {

      // if detect rule depend on itself,  then just ignore
      if (*it == *TableRules[i].begin()) {
        continue;
      }

      std::vector<std::string>::iterator index =
          find(RuleNames.begin(), RuleNames.end(), *it);
      if (index != RuleNames.end()) {
        AdjacencyList[i].push_back(index - RuleNames.begin());
        InDegree[index - RuleNames.begin()]++;
      }
    }
  }

  for (int i = 0; i < n; i++)
    if (InDegree[i] == 0)
      Stack.push(i);

  while (!Stack.empty()) {
    int v = Stack.top();
    Stack.pop();
    InDegree[v] = -1;

    for (std::list<int>::iterator it = AdjacencyList[v].begin();
         it != AdjacencyList[v].end(); it++) {
      InDegree[*it]--;
      if (InDegree[*it] == 0)
        Stack.push(*it);
    }
    AdjacencyList[v].clear();
    Vec.push_back(RuleNames[v]);
  }
  if (Vec.size() != InDegree.size()) {
    std::cout << "Error: Two rules have dependency on each other！\n";
    exit(TranslationError);
  }

  return Vec;
}
