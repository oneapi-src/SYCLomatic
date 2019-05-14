#include "../../lib/SyclCT/AnalysisInfo.h"
#include "gtest/gtest.h"

#include "clang/Tooling/Tooling.h"

using namespace llvm;
using namespace clang::syclct;

class AnalysisInfoTest : public ::testing::Test {
protected:
  std::string TempDir;
  std::string TempDirAbsolute;

  void SetUp() override {
    SmallString<512> CurrentDir;
    llvm::sys::fs::current_path(CurrentDir);
    SmallString<512> UniqueDir;
    llvm::sys::fs::createUniqueDirectory(CurrentDir + "/temp", UniqueDir);

    TempDirAbsolute = UniqueDir.str();

    // .e.g /foo/bar => bar
    TempDir = llvm::sys::path::stem(TempDirAbsolute).str();

    llvm::sys::fs::create_directories(TempDirAbsolute + "/a/b/in");
  }

  void TearDown() override {
    llvm::sys::fs::remove_directories(TempDirAbsolute);
  }
};

TEST_F(AnalysisInfoTest, isInRoot) {
  const std::string InRootPath = TempDirAbsolute + "/a/b";
  const std::string FilePath = TempDir + "/../" + TempDir + "/a/b/in/File.cpp";
  SyclctGlobalInfo::setInRoot(InRootPath);
  bool ret = SyclctGlobalInfo::isInRoot(FilePath);
  ASSERT_EQ(true, ret);
}
