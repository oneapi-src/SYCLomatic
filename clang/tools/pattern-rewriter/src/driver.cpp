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

static constexpr auto Description = R"--(

SYCLomatic provides the pattern-rewriter tool to apply code updates and
adjustments automatically based on user-defined patterns in YAML format. It can
process any code before or after migration. For example, you can use it to
automate the manual adjustments after migration to SYCL, enabling you to re-run
migration multiple times and re-apply your adjustments.
)--";

static constexpr auto Examples = R"--(
EXAMPLES:

Rewrite SYCL source file:

  pattern-rewriter source.dp.cpp -r rules.yaml -o output.dp.cpp

Rewrite CUDA source file:

  pattern-rewriter source.cu -r rules.yaml -o output.cu

Contents for YAML rules file:

  # Remove function call expression
  - match: foo(${args})
    rewrite: (/* foo(${args}) */ 0)

  # Insert an include directive
  - match: "#include <cuda.h>"
    rewrite: |+
      #include <cuda.h>
      #include "./helper.h"

  # Rename a struct
  - match: |+
      struct point {
        ${members}
      }
    rewrite: |+
      struct Point2D {
        ${members}
      }

)--";

template <>
struct llvm::yaml::CustomMappingTraits<std::map<std::string, pattern::Rule>> {
  static void inputOne(IO &IO, StringRef Key,
                       std::map<std::string, pattern::Rule> &Value) {
    IO.mapRequired(Key.str().c_str(), Value[Key.str().c_str()]);
  }

  static void output(IO &IO, std::map<std::string, pattern::Rule> &V) {
    for (auto &P : V) {
      IO.mapRequired(P.first.c_str(), P.second);
    }
  }
};

template <> struct llvm::yaml::MappingTraits<pattern::Rule> {
  static void mapping(IO &IO, pattern::Rule &Value) {
    IO.mapRequired("match", Value.Match);
    IO.mapRequired("rewrite", Value.Rewrite);
    IO.mapOptional("subrules", Value.Subrules);
  }
};

template <> struct llvm::yaml::SequenceElementTraits<pattern::Rule> {
  static const bool flow = false;
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

llvm::cl::OptionCategory &getPatReCategory() {
  static llvm::cl::OptionCategory PatReCategory("Pattern Rewriter options");
  return PatReCategory;
}

int main(int argc, char *argv[]) {
  llvm::cl::opt<std::string> InputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"),
      llvm::cl::value_desc("filename"), llvm::cl::cat(getPatReCategory()),
      llvm::cl::Required);

  llvm::cl::opt<std::string> OutputFilename(
      "o", llvm::cl::desc("[required] Specify output filename"),
      llvm::cl::value_desc("filename"), llvm::cl::cat(getPatReCategory()),
      llvm::cl::Required);

  llvm::cl::opt<std::string> RulesFilename(
      "r", llvm::cl::desc("[required] Specify rules filename"),
      llvm::cl::value_desc("filename"), llvm::cl::cat(getPatReCategory()),
      llvm::cl::Required);

  llvm::cl::extrahelp MoreHelp(Examples);

  llvm::cl::HideUnrelatedOptions(getPatReCategory());

  llvm::cl::ParseCommandLineOptions(argc, argv, Description);

  const auto RulesFile = fixLineEndings(readFile(RulesFilename.getValue()));
  llvm::yaml::Input RulesParser(RulesFile);
  std::vector<pattern::Rule> Rules;
  RulesParser >> Rules;

  std::string Output = fixLineEndings(readFile(InputFilename.getValue()));
  for (const auto &Rule : Rules) {
    Output = pattern::applyRule(Rule, Output);
  }
  writeFile(OutputFilename.getValue(), Output);
  return 0;
}
