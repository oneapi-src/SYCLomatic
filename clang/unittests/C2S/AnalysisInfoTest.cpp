#include "../../lib/C2S/AnalysisInfo.h"
#include "gtest/gtest.h"

#include "clang/Tooling/Tooling.h"
#include "llvm/Support/Process.h"

using namespace llvm;
using namespace clang::c2s;

class AnalysisInfoTest : public ::testing::Test {
protected:
  std::string TempDir;
  std::string TempDirAbsolute;

  void SetUp() override {
    SmallString<512> CurrentDir;
    llvm::sys::fs::current_path(CurrentDir);
    SmallString<512> UniqueDir;
    llvm::sys::fs::createUniqueDirectory(CurrentDir + "/temp", UniqueDir);

    TempDirAbsolute = UniqueDir.str().str();

    // .e.g /foo/bar => bar
    TempDir = llvm::sys::path::stem(TempDirAbsolute).str();

    llvm::sys::fs::create_directories(TempDirAbsolute + "/a/b/in");
    llvm::SmallString<256> TempDirAbsoluteReal;
    llvm::sys::fs::real_path(TempDirAbsolute + "/a/b/in", TempDirAbsoluteReal);
    TempDirAbsolute = TempDirAbsoluteReal.str().str();

    int FD = 0;
    llvm::sys::fs::openFileForWrite(
            TempDir + "/../" + TempDir + "/a/b/in/File.cpp", FD);
    llvm::sys::Process::SafelyCloseFileDescriptor(FD);
  }

  void TearDown() override {
    llvm::sys::fs::remove_directories(TempDirAbsolute);
  }
};

TEST_F(AnalysisInfoTest, isInRoot) {
  const std::string InRootPath = TempDirAbsolute;
  const std::string FilePath = TempDir + "/../" + TempDir + "/a/b/in/File.cpp";
  C2SGlobalInfo::setInRoot(InRootPath);
  bool ret = C2SGlobalInfo::isInRoot(FilePath);
  ASSERT_EQ(true, ret);
}
