#include "../../lib/DPCT/IncMigration/ReMigration.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Core/UnifiedPath.h"
#include "gtest/gtest.h"
#include <unordered_map>

using namespace llvm;
using namespace clang::tooling;
using namespace clang::dpct;

namespace {
  // clang-format off
/*
aaa bb ccc
aaa bb ccc
aaa bb ccc
aaa bb ccc
aaa bb ccc
aaa bb ccc
aaa bb ccc
*/
  // clang-format on
StringRef getLineStringUnittest(clang::tooling::UnifiedPath FilePath,
                                unsigned LineNumber) {
  static std::string S = "aaa bb ccc\n";
  return StringRef(S);
}
unsigned getLineNumberUnittest(clang::tooling::UnifiedPath FilePath,
                               unsigned Offset) {
  static std::vector<unsigned> LineOffsets = {0, 11, 22, 33, 44, 55, 66, 77, 88, 99};
  auto Iter = std::upper_bound(LineOffsets.begin(), LineOffsets.end(), Offset);
  if (Iter == LineOffsets.end())
    return LineOffsets.size();
  return std::distance(LineOffsets.begin(), Iter);
}
unsigned getLineBeginOffsetUnittest(clang::tooling::UnifiedPath FilePath,
                                    unsigned LineNumber) {
  static std::unordered_map<unsigned, unsigned> LineOffsets = {
      {1, 0}, {2, 11}, {3, 22}, {4, 33}, {5, 44}, {6, 55}, {7, 66}, {8, 77}, {9, 88}, {10, 99}};
  return LineOffsets[LineNumber];
}
} // namespace

class ReMigrationTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(ReMigrationTest, calculateUpdatedRanges) {
  // clang-format off
  // Old base:
/*
aaa bb ccc
aaa bb ccc
aaa bb ccc
aaa bb ccc
aaa bb ccc
aaa bb ccc
aaa bb ccc
*/
  // Old migrated:
/*
aaa zzz ccc
ppp aaa bb ccc
aaa yyy ccc
aaa bb ccc
aaxxx bb ccc
aaa bb ccc
qqq aaa bb ccc
*/
  // New base:
/*
aaa bb ccc
Hi bb Everyone
aaa bb ccc
aaa bb Everyone
Hi Hello Everyone
aaa Hi bb ccc
aaa bb ccc
*/
  // clang-format on

  Replacements NewRepl;
  Replacements Repls;
  Replacements Expected;

  llvm::cantFail(NewRepl.add(Replacement("file1.cpp", 4, 2, "zzz")));
  llvm::cantFail(NewRepl.add(Replacement("file1.cpp", 11, 0, "ppp ")));
  llvm::cantFail(NewRepl.add(Replacement("file1.cpp", 26, 2, "yyy")));
  llvm::cantFail(NewRepl.add(Replacement("file1.cpp", 46, 1, "xxx")));
  llvm::cantFail(NewRepl.add(Replacement("file1.cpp", 65, 1, "\nqqq")));

  llvm::cantFail(
      Repls.add(Replacement("file1.cpp", 11, 11, "Hi bb Everyone\n")));
  llvm::cantFail(
      Repls.add(Replacement("file1.cpp", 33, 11, "aaa bb Everyone\n")));
  llvm::cantFail(
      Repls.add(Replacement("file1.cpp", 44, 11, "Hi Hello Everyone\n")));
  llvm::cantFail(
      Repls.add(Replacement("file1.cpp", 55, 11, "aaa Hi bb ccc\n")));

  llvm::cantFail(Expected.add(Replacement("file1.cpp", 4, 2, "zzz")));
  llvm::cantFail(Expected.add(Replacement("file1.cpp", 30, 2, "yyy")));

  Replacements Result = calculateUpdatedRanges(Repls, NewRepl);
  ASSERT_EQ(Expected.size(), Result.size());
  size_t Num = Expected.size();
  auto ExpectedIt = Expected.begin();
  auto ResultIt = Result.begin();
  for (size_t i = 0; i < Num; ++i) {
    EXPECT_EQ(ExpectedIt->getFilePath(), ResultIt->getFilePath());
    EXPECT_EQ(ExpectedIt->getOffset(), ResultIt->getOffset());
    EXPECT_EQ(ExpectedIt->getLength(), ResultIt->getLength());
    EXPECT_EQ(ExpectedIt->getReplacementText(), ResultIt->getReplacementText());
    ExpectedIt++;
    ResultIt++;
  }
}

TEST_F(ReMigrationTest, groupReplcementsByFile) {
  std::string FilePath1 =
      clang::tooling::UnifiedPath("file1.cpp").getCanonicalPath().str();
  std::string FilePath2 =
      clang::tooling::UnifiedPath("file2.cpp").getCanonicalPath().str();
  std::string FilePath3 =
      clang::tooling::UnifiedPath("file3.cpp").getCanonicalPath().str();

  std::vector<Replacement> Repls = {{FilePath1, 0, 0, ""},
                                    {FilePath2, 3, 0, ""},
                                    {FilePath1, 1, 0, ""},
                                    {FilePath3, 4, 0, ""},
                                    {FilePath2, 2, 0, ""}};
  std::map<std::string, std::vector<Replacement>> Expected = {
      {FilePath1,
       {
           {FilePath1, 0, 0, ""},
           {FilePath1, 1, 0, ""},
       }},
      {FilePath2,
       {
           {FilePath2, 2, 0, ""},
           {FilePath2, 3, 0, ""},
       }},
      {FilePath3, {{FilePath3, 4, 0, ""}}}};

  auto Result = groupReplcementsByFile(Repls);
  EXPECT_EQ(Expected.size(), Result.size());

  size_t Num = Expected.size();
  auto ExpectedIt = Expected.begin();
  auto ResultIt = Result.begin();
  for (size_t i = 0; i < Num; ++i) {
    EXPECT_EQ(ExpectedIt->first, ResultIt->first);
    EXPECT_EQ(ExpectedIt->second.size(), ResultIt->second.size());
    size_t SubNum = ExpectedIt->second.size();
    std::sort(ExpectedIt->second.begin(), ExpectedIt->second.end());
    std::sort(ResultIt->second.begin(), ResultIt->second.end());
    for (size_t j = 0; j < SubNum; ++j) {
      EXPECT_EQ(ExpectedIt->second[j].getFilePath(),
                ResultIt->second[j].getFilePath());
      EXPECT_EQ(ExpectedIt->second[j].getOffset(),
                ResultIt->second[j].getOffset());
      EXPECT_EQ(ExpectedIt->second[j].getLength(),
                ResultIt->second[j].getLength());
      EXPECT_EQ(ExpectedIt->second[j].getReplacementText(),
                ResultIt->second[j].getReplacementText());
    }
    ExpectedIt++;
    ResultIt++;
  }
}

TEST_F(ReMigrationTest, splitReplInOrderToNotCrossLines) {
  // clang-format off
  // Old base:
/*
Example:
aaabbbbccc
dddeeeefff
ggghhhhiii
zzzzzzzzzz
yyyyyyyyyy
//
// Original repl:
// (ccc\ndddeeeefff\nggg) =>（jjj\nkkk）
// (zz\nyy) =>（xx)
//
// Splitted repls:
// (ccc\n) =>（jjj\nkkk）
// (dddeeeefff\n) => ""
// (ggg) => ""
// (zz\n) => (xx)
// (yy) => ""
eee bb ccc
*/
  // clang-format on
  getLineStringHook = getLineStringUnittest;
  getLineNumberHook = getLineNumberUnittest;
  getLineBeginOffsetHook = getLineBeginOffsetUnittest;
  std::vector<Replacement> Repls = {Replacement("file1.cpp", 7, 18, "jjj\nkkk"),
                                    Replacement("file1.cpp", 41, 5, "xx")};
  std::vector<Replacement> Expected = {
      Replacement("file1.cpp", 7, 4, "jjj\nkkk"),
      Replacement("file1.cpp", 11, 11, ""), Replacement("file1.cpp", 22, 3, ""),
      Replacement("file1.cpp", 41, 3, "xx"),
      Replacement("file1.cpp", 44, 2, "")};
  auto Result = splitReplInOrderToNotCrossLines(Repls);
  std::sort(Result.begin(), Result.end());
  EXPECT_EQ(Expected, Result);
}

TEST_F(ReMigrationTest, convertReplcementsLineString) {
  // clang-format off
  // Old base:
/*
aaa bb ccc
aaa bb ccc
aaa bb ccc
aaa bb ccc
aaa bb ccc
aaa bb ccc
aaa bb ccc
aaa bb ccc
aaa bb ccc
aaa bb ccc
*/
  // After appling repls:
/*
aaa zzz ccc
ppp aaa bb ccc
aaa bb ccc
aaa yyy ccc
aaa bb ccc
aaa bb ccq
qqqaa bb ccc
aaa bb ddd
<empty without \n>
eee bb ccc
*/
  // clang-format on
  getLineStringHook = getLineStringUnittest;
  getLineNumberHook = getLineNumberUnittest;
  getLineBeginOffsetHook = getLineBeginOffsetUnittest;
  std::vector<Replacement> Repls = {
      Replacement("file1.cpp", 4, 2, "zzz"),
      Replacement("file1.cpp", 64, 3, "q\nqqq"),
      Replacement("file1.cpp", 33, 11, "aaa yyy ccc\n"),
      Replacement("file1.cpp", 11, 0, "ppp "),
      Replacement("file1.cpp", 84, 18, "ddd\neee")
    };
  std::map<unsigned, std::string> Expected = {{1, "aaa zzz ccc\n"},
                                              {2, "ppp aaa bb ccc\n"},
                                              {4, "aaa yyy ccc\n"},
                                              {6, "aaa bb ccq\nqqq"},
                                              {7, "aa bb ccc\n"},
                                              {8, "aaa bb ddd\neee"},
                                              {9, ""},
                                              {10, " bb ccc\n"}
                                            };
  auto Result = convertReplcementsLineString(Repls);
  EXPECT_EQ(Expected, Result);
}

TEST_F(ReMigrationTest, mergeMapsByLine) {
  getLineBeginOffsetHook = getLineBeginOffsetUnittest;
  getLineStringHook = getLineStringUnittest;
  std::map<unsigned, std::string> MapA = {{1, "zzzzz\n"}, {3, "xxxx1\n"},
                                          {4, "wwww1\n"}, {5, "vvvvv\n"},
                                          {7, "ppppp\n"}, {9, "iiijjjkkk\n"}};
  std::map<unsigned, std::string> MapB = {
      {2, "yyyyy\n"}, {3, "xxxx2\n"}, {4, "wwww2\n"}, {7, ""}, {8, "qqqqq"}};
  UnifiedPath FilePath("test.cu");
  auto Result = mergeMapsByLine(MapA, MapB, FilePath);
  std::sort(Result.begin(), Result.end());

  std::vector<Replacement> Expected = {
      Replacement("test.cu", 0, 11, "zzzzz\n"),
      Replacement("test.cu", 11, 11, "yyyyy\n"),
      Replacement("test.cu", 22, 22,
                  "<<<<<<<\nxxxx1\nwwww1\n=======\nxxxx2\nwwww2\n>>>>>>>\n"),
      Replacement("test.cu", 44, 11, "vvvvv\n"),
      Replacement("test.cu", 66, 11, "<<<<<<<\nppppp\n=======\n>>>>>>>\n"),
      Replacement("test.cu", 77, 11, "qqqqq"),
      Replacement("test.cu", 88, 11, "iiijjjkkk\n"),
  };

  ASSERT_EQ(Expected.size(), Result.size());
  size_t Num = Expected.size();
  auto ExpectedIt = Expected.begin();
  auto ResultIt = Result.begin();
  for (size_t i = 0; i < Num; ++i) {
    EXPECT_EQ(ExpectedIt->getFilePath(), ResultIt->getFilePath());
    EXPECT_EQ(ExpectedIt->getOffset(), ResultIt->getOffset());
    EXPECT_EQ(ExpectedIt->getLength(), ResultIt->getLength());
    EXPECT_EQ(ExpectedIt->getReplacementText(), ResultIt->getReplacementText());
    ExpectedIt++;
    ResultIt++;
  }

  EXPECT_EQ(Expected, Result);
}
