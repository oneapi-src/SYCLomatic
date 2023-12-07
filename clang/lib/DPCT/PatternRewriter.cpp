//===--------------- PatternRewriter.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <PatternRewriter.h>

#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <variant>
#include <vector>

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

extern llvm::cl::opt<bool> MigrateCmakeScriptOnly;
extern llvm::cl::opt<bool> MigrateCmakeScript;

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
    Output.push_back(ContainsNonWhitespace ? (Indent + Line) : "");
  }
  std::string Str = trim(join(Output, "\n"));
  if (isWhitespace(Input[0])) {
    Str = " " + Str;
  }
  return Str;
}

static std::string dedent(const std::string &Input, int Indentation) {
  std::stringstream OutputStream;
  const int Size = Input.size();
  int Index = 0;
  int Skip = 0;
  while (Index < Size) {
    char Character = Input[Index];
    if (Skip > 0 && Character == ' ') {
      Skip--;
      Index++;
      continue;
    }
    if (Character == '\n') {
      Skip = Indentation;
    }
    OutputStream << Character;
    Index++;
  }
  return OutputStream.str();
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

  const int Size = Pattern.size();
  int Index = 0;

  if(Size == 0) {
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

static int parseCodeElement(const MatchPattern &Suffix,
                            const std::string &Input, const int Start);

static int parseBlock(char LeftDelimiter, char RightDelimiter,
                      const std::string &Input, const int Start) {
  const int Size = Input.size();
  int Index = Start;

  if (Index >= Size || Input[Index] != LeftDelimiter) {
    return -1;
  }
  Index++;

  Index = parseCodeElement({}, Input, Index);
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
                            const std::string &Input, const int Start) {
  int Index = Start;
  const int Size = Input.size();
  while (Index >= 0 && Index < Size) {
    const auto Character = Input[Index];

    if (Suffix.size() > 0) {
      const auto SuffixMatch = findMatch(Suffix, Input, Index);
      if (SuffixMatch.has_value()) {
        return Index;
      }

      if (isRightDelimiter(Character) || Index == Size - 1) {
        return -1;
      }
    }

    if (Character == '{') {
      Index = parseBlock('{', '}', Input, Index);
      continue;
    }

    if (Character == '[') {
      Index = parseBlock('[', ']', Input, Index);
      continue;
    }

    if (Character == '(') {
      Index = parseBlock('(', ')', Input, Index);
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

static bool isIdentifiedChar(char Char) {

  if ((Char >= 'a' && Char <= 'z') || (Char >= 'A' && Char <= 'Z') ||
      (Char >= '0' && Char <= '9') || (Char == '_')) {
    return true;
  }

  return false;
}

static std::optional<MatchResult> findFullMatch(const MatchPattern &Pattern,
                                                const std::string &Input,
                                                const int Start) {
  MatchResult Result;

  int Index = Start;
  int PatternIndex = 0;
  const int PatternSize = Pattern.size();
  const int Size = Input.size();

  while (PatternIndex < PatternSize && Index < Size) {
    const auto &Element = Pattern[PatternIndex];

    if (std::holds_alternative<LiteralElement>(Element)) {
      const auto &Literal = std::get<LiteralElement>(Element);
      if (Input[Index] != Literal.Value) {
        return {};
      }

      Index++;
      PatternIndex++;
      continue;
    }

    throw std::runtime_error("Internal error: invalid pattern element");
  }

  if (!isIdentifiedChar(Input[Start - 1]) && !isIdentifiedChar(Input[Index])) {
    Result.Start = Start;
    Result.End = Index;
  } else {
    return {};
  }

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
      const int Indentation = detectIndentation(Input, Index);
      std::string ElementContents =
          dedent(Input.substr(Index, Next - Index), Indentation);
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
    const auto Character = Template[Index];

    if (Index < (Size - 1) && Character == '$' && Template[Index + 1] == '{') {
      const int BindingStart = Index;
      Index += 2;

      const auto RightCurly = Template.find('}', Index);
      if (RightCurly == std::string::npos) {
        throw std::runtime_error("Invalid rewrite pattern expression");
      }
      std::string Name = Template.substr(Index, RightCurly - Index);
      Index = RightCurly + 1;

      const auto &BindingIterator = Bindings.find(Name);
      if (BindingIterator != Bindings.end()) {
        const int BindingIndentation =
            detectIndentation(Template, BindingStart) + Indentation;
        const std::string Contents =
            indent(BindingIterator->second, BindingIndentation);
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

int skipCmakeComments(std::ostream &OutputStream, const std::string &Input,
                      int Index) {
  const int Size = Input.size();
  if (Input[Index] == '#') {
    for (; Index < Size && Input[Index] != '\n'; Index++) {
      OutputStream << Input[Index];
    }
  }
  return Index;
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

    if (MigrateCmakeScript || MigrateCmakeScriptOnly) {
      Index = skipCmakeComments(OutputStream, Input, Index);
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
      continue;
    }

    OutputStream << Input[Index];
    Index++;
  }
  return OutputStream.str();
}
