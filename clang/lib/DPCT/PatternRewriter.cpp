//===--------------- PatternRewriter.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AnalysisInfo.h"
#include "Rules.h"
#include "SaveNewFiles.h"
#include "llvm/ADT/StringRef.h"
#include <PatternRewriter.h>

#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <variant>
#include <vector>

std::set<std::string> MainSrcFilesHasCudaSyntex;

struct SpacingElement {};

struct LiteralElement {
  char Value;
};

struct CodeElement {
  std::string Name;
  int SuffixLength = -1;
};

using Element = std::variant<SpacingElement, LiteralElement, CodeElement>;

using MatchPattern = std::vector<Element>;

struct MatchResult {
  int Start;
  int End;
  std::unordered_map<std::string, std::string> Bindings;
};

static SourceFileType SrcFileType = SourceFileType::SFT_CAndCXXSource;

extern llvm::cl::opt<bool> MigrateBuildScriptOnly;

static bool isWhitespace(char Character) {
  return Character == ' ' || Character == '\t' || Character == '\n';
}

static bool isRightDelimiter(char Character) {
  return Character == '}' || Character == ']' || Character == ')';
}

static int detectIndentation(const std::string &Input, int Start) {
  int Indentation = 0;
  int Index = Start - 1;
  while (Index >= 0 && isWhitespace(Input[Index])) {
    if (Input[Index] == '\n' || Index == 0) {
      Indentation = Start - Index - 1;
      break;
    }
    Index--;
  }
  return Indentation;
}

static std::string join(const std::vector<std::string> Lines,
                        const std::string &Separator) {
  if (Lines.size() == 0) {
    return "";
  }
  std::stringstream OutputStream;
  const int Count = Lines.size();
  for (int i = 0; i < Count - 1; i++) {
    OutputStream << Lines[i];
    OutputStream << Separator;
  }
  OutputStream << Lines.back();
  return OutputStream.str();
}

static std::string trim(const std::string &Input) {
  const int Size = Input.size();
  int Index = 0;

  while (Index < Size && isWhitespace(Input[Index])) {
    Index++;
  }

  int End = Size - 1;
  while (End > (Index + 1) && isWhitespace(Input[End])) {
    End--;
  }

  return Input.substr(Index, End + 1);
}

static std::string indent(const std::string &Input, int Indentation) {
  std::vector<std::string> Output;
  const auto Indent = std::string(Indentation, ' ');
  const auto Lines = split(Input, '\n');
  for (const auto &Line : Lines) {
    const bool ContainsNonWhitespace = (trim(Line).size() > 0);
    Output.push_back(ContainsNonWhitespace ? (Indent + trim(Line)) : "");
  }
  std::string Str = trim(join(Output, "\n"));
  return Str;
}

/*
Determines the number of pattern elements that form the suffix of a code
element. The suffix of a code element extends up to the next code element, an
unbalanced right Delimiter, or the end of the pattern. Example:

Pattern:
  if (${a} == ${b}) ${body}

${a}:
  Suffix: [Spacing, '=', '=', Spacing]
  SuffixLength: 4

${b}:
  Suffix: [')']
  SuffixLength: 1

${body}:
  Suffix: []
  SuffixLength: 0
*/
static void adjustSuffixLengths(MatchPattern &Pattern) {
  int SuffixTerminator = Pattern.size() - 1;
  for (int i = Pattern.size() - 1; i >= 0; i--) {
    auto &Element = Pattern[i];

    if (std::holds_alternative<CodeElement>(Element)) {
      auto &Code = std::get<CodeElement>(Element);
      Code.SuffixLength = SuffixTerminator - i;
      SuffixTerminator = i - 1;
      continue;
    }

    if (std::holds_alternative<LiteralElement>(Element)) {
      auto &Literal = std::get<LiteralElement>(Element);
      if (isRightDelimiter(Literal.Value)) {
        SuffixTerminator = i;
      }
      continue;
    }
  }
}

static void removeTrailingSpacingElement(MatchPattern &Pattern) {
  if (std::holds_alternative<SpacingElement>(Pattern.back())) {
    Pattern.pop_back();
  }
}

static MatchPattern parseMatchPattern(std::string Pattern) {
  MatchPattern Result;

  const size_t Size = Pattern.size();
  size_t Index = 0;

  if (Size == 0) {
    return Result;
  }
  while (Index < Size) {
    const char Character = Pattern[Index];

    if (isWhitespace(Character)) {
      if (Result.size() > 0) {
        Result.push_back(SpacingElement{});
      }
      while (Index < Size && isWhitespace(Pattern[Index])) {
        Index++;
      }
      continue;
    }

    // Treat variable name with escape character(like "\${var_name}") as a
    // normal string
    if (Index < (Size - 2) && Pattern[Index] == '\\' &&
        Pattern[Index + 1] == '$' && Pattern[Index + 2] == '{') {
      Index += 1; // Skip '\\'
      auto RightCurly = Pattern.find('}', Index);
      if (RightCurly == std::string::npos) {
        throw std::runtime_error("Invalid rewrite pattern expression");
      }
      RightCurly += 1; // Skip '}'

      while (Index < RightCurly) {
        Result.push_back(LiteralElement{Pattern[Index]});
        Index++;
      }
      continue;
    }

    if (Index < (Size - 1) && Character == '$' && Pattern[Index + 1] == '{') {
      Index += 2;

      const auto RightCurly = Pattern.find('}', Index);
      if (RightCurly == std::string::npos) {
        throw std::runtime_error("Invalid match pattern expression");
      }
      std::string Name = Pattern.substr(Index, RightCurly - Index);
      Index = RightCurly + 1;

      Result.push_back(CodeElement{Name});
      continue;
    }

    Result.push_back(LiteralElement{Character});
    Index++;
  }

  removeTrailingSpacingElement(Result);
  adjustSuffixLengths(Result);
  return Result;
}

static std::optional<MatchResult> findMatch(const MatchPattern &Pattern,
                                            const std::string &Input,
                                            const int Start);

static std::optional<MatchResult> findFullMatch(const MatchPattern &Pattern,
                                                const std::string &Input,
                                                const int Start);

static int parseCodeElement(const MatchPattern &Suffix,
                            const std::string &Input, const int Start,
                            bool IsPartialMatch = true);

static int parseBlock(char LeftDelimiter, char RightDelimiter,
                      const std::string &Input, const int Start,
                      bool IsPartialMatch) {
  const int Size = Input.size();
  int Index = Start;

  if (Index >= Size || Input[Index] != LeftDelimiter) {
    return -1;
  }
  Index++;

  Index = parseCodeElement({}, Input, Index, IsPartialMatch);
  if (Index == -1) {
    return -1;
  }

  if (Index >= Size || Input[Index] != RightDelimiter) {
    return -1;
  }
  Index++;
  return Index;
}

static int parseCodeElement(const MatchPattern &Suffix,
                            const std::string &Input, const int Start,
                            bool IsPartialMatch) {
  int Index = Start;
  const int Size = Input.size();
  while (Index >= 0 && Index < Size) {

    if (SrcFileType == SourceFileType::SFT_CMakeScript) {
      if (Input[Index] == '#') {
        for (; Index < Size && Input[Index] != '\n'; Index++) {
        }
        continue;
      }
    }

    const auto Character = Input[Index];

    if (Suffix.size() > 0) {
      std::optional<MatchResult> SuffixMatch;

      if (IsPartialMatch) {
        SuffixMatch = findMatch(Suffix, Input, Index);
      } else {
        SuffixMatch = findFullMatch(Suffix, Input, Index);
      }

      if (SuffixMatch.has_value()) {
        return Index;
      }

      if (isRightDelimiter(Character) || Index == Size - 1) {
        return -1;
      }
    }

    if (Character == '{') {
      Index = parseBlock('{', '}', Input, Index, IsPartialMatch);
      continue;
    }

    if (Character == '[') {
      Index = parseBlock('[', ']', Input, Index, IsPartialMatch);
      continue;
    }

    if (Character == '(') {
      Index = parseBlock('(', ')', Input, Index, IsPartialMatch);
      continue;
    }

    if (isRightDelimiter(Input[Index])) {
      break;
    }

    /*
    The following parsers skip character literals, string literals, and
    comments. These tokens are skipped since they may contain unbalanced
    delimiters.
    */

    if (Character == '\'') {
      Index++;
      while (Index < Size &&
             !(Input[Index - 1] != '\\' && Input[Index] == '\'')) {
        Index++;
      }
      if (Index >= Size) {
        return -1;
      }
      Index++;
      continue;
    }

    if (Character == '"') {
      Index++;
      while (Index < Size &&
             !(Input[Index - 1] != '\\' && Input[Index] == '"')) {
        Index++;
      }
      if (Index >= Size) {
        return -1;
      }
      Index++;
      continue;
    }

    if (Character == '/' && Index < (Size - 1) && Input[Index + 1] == '/') {
      Index += 2;
      while (Index < Size && Input[Index] != '\n') {
        Index++;
      }
      if (Index >= Size) {
        return -1;
      }
      Index++;
      continue;
    }

    if (Character == '/' && Index < (Size - 1) && Input[Index + 1] == '*') {
      Index += 2;
      while (Index < Size &&
             !(Input[Index - 1] == '*' && Input[Index] == '/')) {
        Index++;
      }
      if (Index >= Size) {
        return -1;
      }
      Index++;
      continue;
    }

    Index++;
  }
  return Suffix.size() == 0 ? Index : -1;
}

// Add '-' as a valid identified char, as cmake target name including '-' is
// valid
static bool isIdentifiedChar(char Char) {

  if ((Char >= 'a' && Char <= 'z') || (Char >= 'A' && Char <= 'Z') ||
      (Char >= '0' && Char <= '9') || (Char == '_') || (Char == '-')) {
    return true;
  }

  return false;
}

static void
updateExtentionName(const std::string &Input, size_t Next,
                    std::unordered_map<std::string, std::string> &Bindings) {
  auto Extension = clang::dpct::DpctGlobalInfo::getSYCLSourceExtension();
  if (Input.compare(Next, strlen(".cpp"), ".cpp") == 0) {
    size_t Pos = Next - 1;
    for (; Pos > 0 && isIdentifiedChar(Input[Pos]); Pos--) {
    }
    Pos = Pos == 0 ? 0 : Pos + 1;
    std::string FileName = Input.substr(Pos, Next + strlen(".cpp") - 1 - Pos);

    std::string SyclFileName;
    rewriteFileName(SyclFileName, FileName);
    bool HasCudaSyntax = false;
    for (const auto &File : MainSrcFilesHasCudaSyntex) {
      if (File.find(FileName) != std::string::npos) {
        HasCudaSyntax = true;
      }
    }

    if (HasCudaSyntax)
      Bindings["rewrite_extention_name"] = "cpp" + Extension;
    else
      Bindings["rewrite_extention_name"] = "cpp";
  } else {
    Bindings["rewrite_extention_name"] = Extension.erase(0, 1);
  }
}

static std::optional<MatchResult> findFullMatch(const MatchPattern &Pattern,
                                                const std::string &Input,
                                                const int Start) {

  MatchResult Result;

  int Index = Start;
  int PatternIndex = 0;
  const int PatternSize = Pattern.size();
  const int Size = Input.size();
  bool CodeElementExist = false;
  for (const auto &Element : Pattern) {
    if (std::holds_alternative<CodeElement>(Element)) {
      CodeElementExist = true;
    }
  }

  while (PatternIndex < PatternSize && Index < Size) {

    if (SrcFileType == SourceFileType::SFT_CMakeScript) {
      if (Input[Index] == '#') {
        for (; Index < Size && Input[Index] != '\n'; Index++) {
        }
      }
    }

    const auto &Element = Pattern[PatternIndex];

    if (std::holds_alternative<SpacingElement>(Element)) {
      if (!isWhitespace(Input[Index])) {
        return {};
      }
      while (Index < Size && isWhitespace(Input[Index])) {
        Index++;
      }
      PatternIndex++;
      continue;
    }

    if (std::holds_alternative<LiteralElement>(Element)) {
      const auto &Literal = std::get<LiteralElement>(Element);
      if (Input[Index] != Literal.Value) {
        return {};
      }

      if (!CodeElementExist && Index - PatternSize >= 0 && Index < Size - 1 &&
          PatternIndex == PatternSize - 1) {
        if (!isIdentifiedChar(Input[Index - PatternSize]) &&
            !isIdentifiedChar(Input[Index + 1])) {

          if (Input[Index - PatternSize] != '{' && Input[Index + 1] != '}' &&
              !isWhitespace(Input[Index - PatternSize]) &&
              !isWhitespace(Input[Index + 1]) &&
              Input[Index - PatternSize] != '*') {
            return {};
          }
        }

        if (isIdentifiedChar(Input[Index - PatternSize]) &&
            Input[Index - PatternSize + 1] != '.') {
          return {};
        }
      }

      // If input value has been matched to the end but match pattern still has
      // value, it is considered not matched case.
      if (Index == Size - 1 && PatternIndex < PatternSize - 1) {
        return {};
      }

      // To make sure first character after the matched word isn't an
      // identified character or suffix match '('.
      if (Index < Size - 1 && isIdentifiedChar(Input[Index + 1]) &&
          PatternIndex + 1 == PatternSize && Literal.Value != '(') {
        return {};
      }

      Index++;
      PatternIndex++;
      continue;
    }

    if (std::holds_alternative<CodeElement>(Element)) {
      const auto &Code = std::get<CodeElement>(Element);
      MatchPattern Suffix(Pattern.begin() + PatternIndex + 1,
                          Pattern.begin() + PatternIndex + 1 +
                              Code.SuffixLength);

      int Next = parseCodeElement(Suffix, Input, Index, false);
      if (Next == -1) {
        return {};
      }

      std::string ElementContents = Input.substr(Index, Next - Index);

      if (SrcFileType == SourceFileType::SFT_CMakeScript) {
        updateExtentionName(Input, Next, Result.Bindings);
      }

      if (Result.Bindings.count(Code.Name)) {
        if (Result.Bindings[Code.Name] != ElementContents) {
          return {};
        }
      } else {
        Result.Bindings[Code.Name] = std::move(ElementContents);
      }
      Index = Next;
      PatternIndex++;
      continue;
    }

    throw std::runtime_error("Internal error: invalid pattern element");
  }

  Result.Start = Start;
  Result.End = Index;
  return Result;
}

static std::optional<MatchResult> findMatch(const MatchPattern &Pattern,
                                            const std::string &Input,
                                            const int Start) {
  MatchResult Result;

  int Index = Start;
  int PatternIndex = 0;
  const int PatternSize = Pattern.size();
  const int Size = Input.size();
  while (PatternIndex < PatternSize && Index < Size) {

    if (SrcFileType == SourceFileType::SFT_CMakeScript) {
      if (Input[Index] == '#') {
        for (; Index < Size && Input[Index] != '\n'; Index++) {
        }
      }
    }

    const auto &Element = Pattern[PatternIndex];

    if (std::holds_alternative<SpacingElement>(Element)) {
      if (!isWhitespace(Input[Index])) {
        return {};
      }
      while (Index < Size && isWhitespace(Input[Index])) {
        Index++;
      }
      PatternIndex++;
      continue;
    }

    if (std::holds_alternative<LiteralElement>(Element)) {
      const auto &Literal = std::get<LiteralElement>(Element);
      if (Input[Index] != Literal.Value) {
        return {};
      }

      Index++;
      PatternIndex++;
      continue;
    }

    if (std::holds_alternative<CodeElement>(Element)) {
      const auto &Code = std::get<CodeElement>(Element);
      MatchPattern Suffix(Pattern.begin() + PatternIndex + 1,
                          Pattern.begin() + PatternIndex + 1 +
                              Code.SuffixLength);

      int Next = parseCodeElement(Suffix, Input, Index);
      if (Next == -1) {
        return {};
      }
      std::string ElementContents = Input.substr(Index, Next - Index);
      if (Result.Bindings.count(Code.Name)) {
        if (Result.Bindings[Code.Name] != ElementContents) {
          return {};
        }
      } else {
        Result.Bindings[Code.Name] = std::move(ElementContents);
      }
      Index = Next;
      PatternIndex++;
      continue;
    }

    throw std::runtime_error("Internal error: invalid pattern element");
  }

  Result.Start = Start;
  Result.End = Index;
  return Result;
}

static void instantiateTemplate(
    const std::string &Template,
    const std::unordered_map<std::string, std::string> &Bindings,
    const int Indentation, std::ostream &OutputStream) {
  const auto LeadingSpace = std::string(Indentation, ' ');
  const int Size = Template.size();
  int Index = 0;

  while (Index < Size && isWhitespace(Template[Index])) {
    Index++;
  }

  int End = Size - 1;
  while (End > (Index + 1) && isWhitespace(Template[End])) {
    End--;
  }

  while (Index <= End) {

    // Skip variable name with escape character, like "\${var_name}"
    if (Index < (Size - 2) && Template[Index] == '\\' &&
        Template[Index + 1] == '$' && Template[Index + 2] == '{') {
      Index += 1; // Skip '\\'
      auto RightCurly = Template.find('}', Index);
      if (RightCurly == std::string::npos) {
        throw std::runtime_error("Invalid rewrite pattern expression");
      }
      RightCurly += 1; // Skip '}'
      std::string Name = Template.substr(Index, RightCurly - Index);
      OutputStream << Name;
    }

    auto Character = Template[Index];
    if (Index < (Size - 1) && Character == '$' && Template[Index + 1] == '{') {
      Index += 2;

      const auto RightCurly = Template.find('}', Index);
      if (RightCurly == std::string::npos) {
        throw std::runtime_error("Invalid rewrite pattern expression");
      }
      std::string Name = Template.substr(Index, RightCurly - Index);
      Index = RightCurly + 1;

      const auto &BindingIterator = Bindings.find(Name);
      if (BindingIterator != Bindings.end()) {
        const std::string Contents = BindingIterator->second;
        OutputStream << Contents;
      }
      continue;
    }

    OutputStream << Character;
    if (Character == '\n') {
      OutputStream << LeadingSpace;
    }

    Index++;
  }
}

bool fixLineEndings(const std::string &Input, std::string &Output) {
  std::stringstream OutputStream;
  bool isCRLF = false;
  int Index = 0;
  int Size = Input.size();
  while (Index < Size) {
    char Character = Input[Index];
    if (Character != '\r') {
      OutputStream << Character;
    } else {
      isCRLF = true;
    }
    Index++;
  }
  Output = OutputStream.str();
  return isCRLF;
}

bool skipCmakeComments(std::ostream &OutputStream, const std::string &Input,
                       int &Index) {
  const int Size = Input.size();
  bool CommentFound = false;
  if (Input[Index] == '#') {
    CommentFound = true;
    for (; Index < Size && Input[Index] != '\n'; Index++) {
      OutputStream << Input[Index];
    }
    if (Index != Size) {
      OutputStream << "\n";
    }
    Index++;
    if (Index < Size && isWhitespace(Input[Index])) {
      for (; Index < Size && isWhitespace(Input[Index]); Index++) {
        OutputStream << Input[Index];
      }
    }
  }
  return CommentFound;
}

void setFileTypeProcessed(enum SourceFileType FileType) {
  SrcFileType = FileType;
}

std::string applyPatternRewriter(const MetaRuleObject::PatternRewriter &PP,
                                 const std::string &Input) {
  std::stringstream OutputStream;

  if (PP.In.size() == 0) {
    return Input;
  }

  const auto Pattern = parseMatchPattern(PP.In);
  const int Size = Input.size();
  int Index = 0;
  while (Index < Size) {

    if (SrcFileType == SourceFileType::SFT_CMakeScript) {
      if (skipCmakeComments(OutputStream, Input, Index)) {
        continue;
      }
    }

    std::optional<MatchResult> Result;
    if (PP.MatchMode) {
      Result = findFullMatch(Pattern, Input, Index);
    } else {
      Result = findMatch(Pattern, Input, Index);
    }

    if (Result.has_value()) {
      auto &Match = Result.value();
      for (const auto &[Name, Value] : Match.Bindings) {
        const auto &SubruleIterator = PP.Subrules.find(Name);
        if (SubruleIterator != PP.Subrules.end()) {
          Match.Bindings[Name] =
              applyPatternRewriter(SubruleIterator->second, Value);
        }
      }

      const int Indentation = detectIndentation(Input, Index);

      instantiateTemplate(PP.Out, Match.Bindings, Indentation, OutputStream);
      Index = Match.End;
      while (Input[Index] == '\n') {
        OutputStream << Input[Index];
        Index++;
      }
      continue;
    }

    OutputStream << Input[Index];
    Index++;
  }
  return OutputStream.str();
}
