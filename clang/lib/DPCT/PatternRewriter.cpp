//===--------------- PatternRewriter.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PatternRewriter.h"
#include "AnalysisInfo.h"
#include "Diagnostics.h"
#include "MigrateCmakeScript.h"
#include "MigratePythonBuildScript.h"
#include "Rules.h"
#include "SaveNewFiles.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"

#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <variant>
#include <vector>

std::set<std::string> MainSrcFilesHasCudaSyntex;
bool LANG_Cplusplus_20_Used = false;

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
  bool FullMatchFound = false;
};

static SourceFileType SrcFileType = SourceFileType::SFT_CAndCXXSource;

static bool isWhitespace(char Character) {
  return Character == ' ' || Character == '\t' || Character == '\n';
}

static bool isNotWhitespace(char Character) { return !isWhitespace(Character); }

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

static std::optional<MatchResult>
findMatch(const MatchPattern &Pattern, const std::string &Input,
          const int Start, RuleMatchMode Mode, std::string FileName = "",
          const clang::tooling::UnifiedPath OutRoot = "");

static int parseCodeElement(const MatchPattern &Suffix,
                            const std::string &Input, const int Start,
                            RuleMatchMode Mode);

static int parseBlock(char LeftDelimiter, char RightDelimiter,
                      const std::string &Input, const int Start,
                      RuleMatchMode Mode) {
  const int Size = Input.size();
  int Index = Start;

  if (Index >= Size || Input[Index] != LeftDelimiter) {
    return -1;
  }
  Index++;

  Index = parseCodeElement({}, Input, Index, Mode);
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
                            RuleMatchMode Mode) {
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
    if(Suffix.size() == 0 && Character =='"') {
      return Index;
    }
    if (Suffix.size() > 0) {
      std::optional<MatchResult> SuffixMatch;

      SuffixMatch = findMatch(Suffix, Input, Index, Mode);

      if (SuffixMatch.has_value()) {
        return Index;
      }

      if (isRightDelimiter(Character) || Index == Size - 1) {
        return -1;
      }
    }

    if (Character == '{') {
      Index = parseBlock('{', '}', Input, Index, Mode);
      continue;
    }

    if (Character == '[') {
      Index = parseBlock('[', ']', Input, Index, Mode);
      continue;
    }

    if (Character == '(') {
      Index = parseBlock('(', ')', Input, Index, Mode);
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

    if (SrcFileType == SourceFileType::SFT_CMakeScript) {
      if (Index - 1 >= 0 && Character == '"' && Input[Index - 1] == '\\') {
        Index++;
        while (Index < Size &&
               !(Input[Index - 1] == '\\' && Input[Index] == '"')) {
          Index++;
        }
        if (Index >= Size) {
          return -1;
        }
        Index++;
        continue;
      }
    }

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

static void applyExtenstionNameChange(
    const std::string &Input, size_t Next,
    std::unordered_map<std::string, std::string> &Bindings,
    std::string FileName, const clang::tooling::UnifiedPath OutRoot,
    std::string ExtensionType) {
  size_t Pos = Next - 1;
  for (; Pos > 0 && !isWhitespace(Input[Pos]); Pos--) {
  }
  Pos = Pos == 0 ? 0 : Pos + 1;
  std::string SrcFile = Input.substr(Pos, Next + ExtensionType.length() +
                                              1 /*strlen of "."*/ - Pos);
  bool HasCudaSyntax = false;

  for (const auto &_File : MainSrcFilesHasCudaSyntex) {

    llvm::SmallString<512> File(_File);
    llvm::sys::path::native(File);

#ifdef _WIN32
    if (llvm::sys::path::filename(FileName).lower() == "cmakelists.txt") {
#else
    if (llvm::sys::path::filename(FileName) == "CMakeLists.txt") {
#endif
      // In a CMakeLists.txt file, the relative directory path for a source
      // file is the location of the CMakeLists.txt file itself

      llvm::SmallString<512> CMakeFilePath;
      if (llvm::sys::path::filename(SrcFile) == SrcFile) {
        // To get the directory path where CMake script is located
        SmallString<512> CMakeDirectory(FileName);
        llvm::sys::path::replace_path_prefix(
            CMakeDirectory, OutRoot.getCanonicalPath().str(), ".");
        llvm::sys::path::remove_dots(CMakeDirectory,
                                     /* remove_dot_dot= */ true);
        llvm::sys::path::remove_filename(CMakeDirectory);

        std::string TempFile = CMakeDirectory.c_str();
        TempFile = TempFile + "/" + SrcFile;
        CMakeFilePath = TempFile.c_str();

        llvm::sys::path::native(CMakeFilePath);
      } else {
        std::string FileName = llvm::sys::path::filename(SrcFile).str();
        SmallString<512> _SrcFile(SrcFile);
        llvm::sys::path::remove_dots(_SrcFile, /* remove_dot_dot= */ true);
        llvm::sys::path::replace_path_prefix(_SrcFile, "${CMAKE_SOURCE_DIR}",
                                             ".");
        llvm::sys::path::remove_dots(_SrcFile, /* remove_dot_dot= */ true);
        llvm::sys::path::remove_filename(_SrcFile);
        std::string ParentPath = _SrcFile.c_str();

        auto LastDotPos = ParentPath.find_last_of('.');
        if (LastDotPos != std::string::npos) {
#ifdef _WIN32
          auto PrexPos = ParentPath.find('\\', LastDotPos);
#else
          auto PrexPos = ParentPath.find('/', LastDotPos);
#endif
          if (PrexPos != std::string::npos)
            ParentPath = ParentPath.substr(PrexPos + 1);
        }

        CMakeFilePath = ParentPath + "/" + FileName;
        llvm::sys::path::native(CMakeFilePath);
      }
      if (llvm::StringRef(File).ends_with(CMakeFilePath)) {
        HasCudaSyntax = true;
        break;
      }
    } else {
      // For other module files (e.g., .cmake files), just check the
      // file names.
      if (llvm::sys::path::filename(File) ==
          llvm::sys::path::filename(SrcFile)) {
        HasCudaSyntax = true;
        break;
      }
    }
  }

  if (HasCudaSyntax)
    Bindings["rewrite_extention_name"] =
        ExtensionType + clang::dpct::DpctGlobalInfo::getSYCLSourceExtension();
  else
    Bindings["rewrite_extention_name"] = ExtensionType;
}

static void
updateExtentionName(const std::string &Input, size_t Next,
                    std::unordered_map<std::string, std::string> &Bindings,
                    std::string FileName,
                    const clang::tooling::UnifiedPath OutRoot) {
  if (Input.compare(Next, strlen(".cpp"), ".cpp") == 0 &&
      !isIdentifiedChar(Input[Next + strlen(".cpp")])) {
    applyExtenstionNameChange(Input, Next, Bindings, FileName, OutRoot, "cpp");
  } else if (Input.compare(Next, strlen(".c"), ".c") == 0 &&
             !isIdentifiedChar(Input[Next + strlen(".c")])) {
    applyExtenstionNameChange(Input, Next, Bindings, FileName, OutRoot, "c");
  } else if (Input.compare(Next, strlen(".cc"), ".cc") == 0 &&
             !isIdentifiedChar(Input[Next + strlen(".cc")])) {
    applyExtenstionNameChange(Input, Next, Bindings, FileName, OutRoot, "cc");
  } else if (Input.compare(Next, strlen(".cxx"), ".cxx") == 0 &&
             !isIdentifiedChar(Input[Next + strlen(".cxx")])) {
    applyExtenstionNameChange(Input, Next, Bindings, FileName, OutRoot, "cxx");
  } else {
    auto Extension = clang::dpct::DpctGlobalInfo::getSYCLSourceExtension();
    Bindings["rewrite_extention_name"] = Extension.erase(0, 1);
  }
}

static void updateCplusplusStandard(
    std::unordered_map<std::string, std::string> &Bindings) {
  if (LANG_Cplusplus_20_Used) {
    Bindings["rewrite_cplusplus_version"] = "20";
  } else {
    Bindings["rewrite_cplusplus_version"] = "17";
  }
}

static bool checkMatchContition(const int Size, const int Index,
                                const int PatternSize, const int PatternIndex,
                                const std::string &Input, MatchResult &Result,
                                bool (*FuncPtr)(char)) {
  if (PatternIndex == 0 && Index - 1 >= 0 && FuncPtr(Input[Index - 1]) &&
      FuncPtr(Input[Index])) {
    return false;
  }

  // If input value has been matched to the end but match pattern template
  // still has value, it is considered not matched case.
  if (Index == Size - 1 && PatternIndex < PatternSize - 1) {
    return false;
  }

  // If match pattern template has been matched to the end but input value
  // still not the end, it is considered not matched case.
  if (PatternIndex == PatternSize - 1 && FuncPtr(Input[Index + 1])) {
    return false;
  }

  if (PatternIndex == PatternSize - 1 && !FuncPtr(Input[Index + 1])) {
    Result.FullMatchFound = true;
  }
  return true;
}

static std::optional<MatchResult>
findMatch(const MatchPattern &Pattern, const std::string &Input,
          const int Start, RuleMatchMode Mode, std::string FileName,
          const clang::tooling::UnifiedPath OutRoot) {
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

      if (Mode == RuleMatchMode::Full &&
          !checkMatchContition(Size, Index, PatternSize, PatternIndex, Input,
                               Result, isIdentifiedChar)) {

        return {};
      }

      if (Mode == RuleMatchMode::StrictFull &&
          !checkMatchContition(Size, Index, PatternSize, PatternIndex, Input,
                               Result, isNotWhitespace)) {

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

      int Next = parseCodeElement(Suffix, Input, Index, Mode);
      if (Next == -1) {
        return {};
      }
      std::string ElementContents = Input.substr(Index, Next - Index);

      if (SrcFileType == SourceFileType::SFT_CMakeScript) {
        if (Code.Name == "empty" && !ElementContents.empty() &&
            ElementContents.find_first_not_of(' ') != std::string::npos) {
          // For reversed variable ${empty}, it should be empty string or string
          // only including spaces.
          return {};
        }
        updateCplusplusStandard(Result.Bindings);
        updateExtentionName(Input, Next, Result.Bindings, FileName, OutRoot);
      }

      Result.Bindings[Code.Name] = std::move(ElementContents);
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

static void constructWaringMsg(const std::string &Input, size_t index,
                               const std::string &FileName,
                               const std::string &FrontPart,
                               std::string Warning, std::string &OutStr) {
  std::string Buffer;
  if (!FrontPart.empty())
    Buffer = FrontPart + Input;

  size_t LineNumber = 1;
  size_t Count = 0;
  const auto Lines = split(Buffer, '\n');
  for (const auto &Line : Lines) {
    if (index + FrontPart.size() > Count &&
        index + FrontPart.size() <= Count + Line.size()) {
      break;
    }

    Count += Line.size() + 1;
    LineNumber++;
  }

  std::string WarningMsg =
      FileName + ":" + std::to_string(LineNumber) + ":warning:";

  if (llvm::StringRef(FileName).ends_with(".txt") ||
      llvm::StringRef(FileName).ends_with(".cmake")) {
    WarningMsg += clang::dpct::DiagnosticsUtils::getMsgText(
        clang::dpct::CMakeScriptMigrationMsgs::WARNING_FOR_SYNTAX_REMOVED,
        Warning);
    WarningMsg += "\n";

    addCmakeWarningMsg(WarningMsg, FileName);

    OutStr =
        "\n# " +
        clang::dpct::DiagnosticsUtils::getMsgText(
            clang::dpct::CMakeScriptMigrationMsgs::WARNING_FOR_SYNTAX_REMOVED,
            Warning) +
        "\n" + OutStr;
  } else if (llvm::StringRef(FileName).ends_with(".py")) {
    WarningMsg += clang::dpct::DiagnosticsUtils::getMsgText(
        clang::dpct::PythonBuildScriptMigrationMsgs::WARNING_FOR_SYNTAX_REMOVED,
        Warning);
    WarningMsg += "\n";

    addPythonWarningMsg(WarningMsg, FileName);

    OutStr = "\n# " +
             clang::dpct::DiagnosticsUtils::getMsgText(
                 clang::dpct::PythonBuildScriptMigrationMsgs::
                     WARNING_FOR_SYNTAX_REMOVED,
                 Warning) +
             "\n" + OutStr;
  }
}

std::string applyPatternRewriter(const MetaRuleObject::PatternRewriter &PP,
                                 const std::string &Input, std::string FileName,
                                 std::string FrontPart,
                                 const clang::tooling::UnifiedPath OutRoot) {
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
    Result = findMatch(Pattern, Input, Index, PP.MatchMode, FileName, OutRoot);

    if (Result.has_value()) {
      auto &Match = Result.value();
      for (const auto &[Name, Value] : Match.Bindings) {
        const auto &SubruleIterator = PP.Subrules.find(Name);
        if (SubruleIterator != PP.Subrules.end()) {

          if (SrcFileType == SourceFileType::SFT_CMakeScript) {
            auto Pos = Input.find(Value);
            if (Pos != std::string::npos) {
              FrontPart = Input.substr(0, Pos);
            }
          }

          Match.Bindings[Name] = applyPatternRewriter(
              SubruleIterator->second, Value, FileName, FrontPart, OutRoot);
        }
      }

      std::string OutStr = PP.Out;

      if (SrcFileType == SourceFileType::SFT_CMakeScript &&
          !PP.Warning.empty() && Result->FullMatchFound) {
        constructWaringMsg(Input, Match.End, FileName, FrontPart, PP.Warning,
                           OutStr);
      }

      const int Indentation = detectIndentation(Input, Index);
      instantiateTemplate(OutStr, Match.Bindings, Indentation, OutputStream);
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
