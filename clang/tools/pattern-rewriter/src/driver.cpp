//===- driver.cpp -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <fstream>
#include <ios>
#include <sstream>
#include <string>
#include <vector>

#include "./pattern.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/YAMLTraits.h"

template <>
struct llvm::yaml::CustomMappingTraits<std::map<std::string, pattern::Rule>> {
  static void inputOne(IO &IO, StringRef Key,
                       std::map<std::string, pattern::Rule> &Value) {
    IO.mapRequired(Key.str().c_str(), Value[Key.str().c_str()]);
  }

  static void output(IO &IO, std::map<std::string, pattern::Rule> &V) {
    for (auto &P : V)
      IO.mapRequired(P.first.c_str(), P.second);
  }
};

template <> struct llvm::yaml::MappingTraits<pattern::Rule> {
  static void mapping(IO &IO, pattern::Rule &Value) {
    IO.mapRequired("match", Value.Match);
    IO.mapRequired("rewrite", Value.Rewrite);
    IO.mapOptional("subrules", Value.Subrules);
  }
};

static std::string readFile(const std::string &Name) {
  std::ifstream Stream(Name, std::ios::in | std::ios::binary);
  std::string Contents((std::istreambuf_iterator<char>(Stream)),
                       (std::istreambuf_iterator<char>()));
  return Contents;
}

static void writeFile(const std::string &Name, const std::string &Contents) {
  std::ofstream Stream(Name, std::ios::out | std::ios::binary);
  Stream.write(Contents.data(), Contents.size());
  Stream.close();
}

static std::string fixLineEndings(const std::string &Input) {
  std::stringstream OutputStream;
  int Index = 0;
  int Size = Input.size();
  while (Index < Size) {
    char Character = Input[Index];
    if (Character != '\r') {
      OutputStream << Character;
    }
    Index++;
  }
  return OutputStream.str();
}

int main(int argc, char *argv[]) {
  llvm::cl::opt<std::string> InputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"),
      llvm::cl::value_desc("filename"), llvm::cl::Required);

  llvm::cl::opt<std::string> OutputFilename(
      "o", llvm::cl::desc("Specify output filename"),
      llvm::cl::value_desc("filename"), llvm::cl::Required);

  llvm::cl::opt<std::string> RuleSetFilename(
      "r", llvm::cl::desc("Specify rule set filename"),
      llvm::cl::value_desc("filename"), llvm::cl::Required);

  llvm::cl::ParseCommandLineOptions(argc, argv);

  const auto RuleSetFile = fixLineEndings(readFile(RuleSetFilename.getValue()));
  llvm::yaml::Input RuleSetParser(RuleSetFile);
  std::map<std::string, pattern::Rule> RuleSet;
  RuleSetParser >> RuleSet;

  const auto Input = fixLineEndings(readFile(InputFilename.getValue()));

  std::string Output = Input;
  for (const auto &[Name, Rule] : RuleSet) {
    Output = pattern::applyRule(Rule, Output);
  }
  writeFile(OutputFilename.getValue(), Output);
  return 0;
}
