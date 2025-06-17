#include "../../lib/DPCT/IncMigration/ReMigration.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang::tooling;
using namespace clang::dpct;

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
