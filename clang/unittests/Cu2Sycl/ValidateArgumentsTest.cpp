#include "../../lib/Cu2Sycl/ValidateArguments.cpp"
#include "gtest/gtest.h"

#include "clang/Tooling/Tooling.h"
#include <string>

using namespace std;
using clang::tooling::getAbsolutePath;

class MakeCanonicalOrSetDefaults : public ::testing::Test {
protected:
  std::string TestRunPath;
  std::string TempDir;
  std::string TempDirAbsolute;

  void SetUp() override {
    SmallString<256> CurrentDir;
    fs::current_path(CurrentDir);
    SmallString<256> UniqueDir;
    fs::createUniqueDirectory(CurrentDir + "/temp", UniqueDir);

    TestRunPath = StringRef(CurrentDir);
    TempDirAbsolute = StringRef(UniqueDir);
    TempDir = (path::stem(TempDirAbsolute) + "/").str();

    fs::create_directories(TempDirAbsolute + "/a/b/in");
  }

  void TearDown() override { fs::remove_directories(TempDirAbsolute); }
};

TEST_F(MakeCanonicalOrSetDefaults, getDefaultOutRootEmpty) {
  string OutRoot;
  getDefaultOutRoot(OutRoot);
  ASSERT_EQ(OutRoot, TestRunPath);
}

TEST_F(MakeCanonicalOrSetDefaults, getDefaultOutRoot) {
  string OutRoot = "is not empty";
  getDefaultOutRoot(OutRoot);
  ASSERT_EQ(OutRoot, TestRunPath);
}

TEST(getDefaultInRoot, onlyOneInputAllowed) {
  string InRoot = "";
  ASSERT_EQ(false, getDefaultInRoot(
                       InRoot, {"/a/b/in/file.cpp", "/a/b/in/c/file.cpp"}));
}

TEST(getDefaultInRoot, noInroot) {
  string inroot = "";
  ASSERT_EQ(true, getDefaultInRoot(inroot, {"/a/b/in/file.cpp"}));
  ASSERT_EQ(inroot, "/a/b/in");
}

TEST_F(MakeCanonicalOrSetDefaults, empty) {
  string InRoot;
  string OutRoot;
  ASSERT_EQ(true, makeCanonicalOrSetDefaults(
                      InRoot, OutRoot, {TempDirAbsolute + "/a/b/in/file.cpp"}));
  ASSERT_EQ(InRoot, TempDirAbsolute + "/a/b/in");
  ASSERT_EQ(OutRoot, TestRunPath);
}

TEST_F(MakeCanonicalOrSetDefaults, emptyOnlyOneFileAllowed) {
  string InRoot;
  string OutRoot;
  ASSERT_EQ(false,
            makeCanonicalOrSetDefaults(
                InRoot, OutRoot, {"/a/b/in/file.cpp", "/a/b/in/file.cpp"}));
}

TEST_F(MakeCanonicalOrSetDefaults, dotAtTheEnd) {
  string InRoot = TempDirAbsolute + "/a/b/in/.";
  string OutRoot = TempDirAbsolute + "/a/b/.";
  ASSERT_EQ(true, makeCanonicalOrSetDefaults(
                      InRoot, OutRoot,
                      {TempDirAbsolute + "/a/b/in/file.cpp",
                       TempDirAbsolute + "/a/b/in/c/file.cpp"}));
  ASSERT_EQ(InRoot, TempDirAbsolute + "/a/b/in");
  ASSERT_EQ(OutRoot, TempDirAbsolute + "/a/b");
}

TEST_F(MakeCanonicalOrSetDefaults, dotInTheMiddle) {
  string InRoot = TempDirAbsolute + "/a/b/./in/.";
  string OutRoot = TempDirAbsolute + "/a/b/.";
  ASSERT_EQ(true, makeCanonicalOrSetDefaults(
                      InRoot, OutRoot,
                      {TempDirAbsolute + "/a/b/./in/file.cpp",
                       TempDirAbsolute + "/a/b/in/c/file.cpp"}));
  ASSERT_EQ(InRoot, TempDirAbsolute + "/a/b/in");
  ASSERT_EQ(OutRoot, TempDirAbsolute + "/a/b");
}

TEST_F(MakeCanonicalOrSetDefaults, dotDotAtTheEnd) {
  string InRoot = TempDirAbsolute + "/a/b/in/..";
  string OutRoot = TempDirAbsolute + "/a/b/";
  ASSERT_EQ(true, makeCanonicalOrSetDefaults(
                      InRoot, OutRoot,
                      {TempDirAbsolute + "/a/b/in/file.cpp",
                       TempDirAbsolute + "/a/b/in/c/file.cpp"}));
  ASSERT_EQ(InRoot, TempDirAbsolute + "/a/b");
  ASSERT_EQ(OutRoot, TempDirAbsolute + "/a/b");
}

TEST_F(MakeCanonicalOrSetDefaults, dotDotInTheMiddle) {
  string InRoot = TempDirAbsolute + "/a/b/../b/in";
  string OutRoot = TempDirAbsolute + "/a/b/";
  ASSERT_EQ(true, makeCanonicalOrSetDefaults(
                      InRoot, OutRoot,
                      {TempDirAbsolute + "/a/b/in/file.cpp",
                       TempDirAbsolute + "/a/b/in/c/file.cpp"}));
  ASSERT_EQ(InRoot, TempDirAbsolute + "/a/b/in");
  ASSERT_EQ(OutRoot, TempDirAbsolute + "/a/b");
}

TEST_F(MakeCanonicalOrSetDefaults, relativePaths) {
  string InRoot = TempDir + "a/b/../b/in";
  string OutRoot = TempDir + "a/b/";
  ASSERT_EQ(true, makeCanonicalOrSetDefaults(InRoot, OutRoot,
                                             {TempDir + "a/b/in/file.cpp",
                                              TempDir + "a/b/in/c/file.cpp"}));
  ASSERT_EQ(InRoot, TempDirAbsolute + "/a/b/in");
  ASSERT_EQ(OutRoot, TempDirAbsolute + "/a/b");
}

TEST_F(MakeCanonicalOrSetDefaults, relativePathsNoRoots) {
  string InRoot;
  string OutRoot;
  ASSERT_EQ(true, makeCanonicalOrSetDefaults(InRoot, OutRoot, {"file.cpp"}));
  ASSERT_EQ(InRoot, TestRunPath);
  ASSERT_EQ(OutRoot, TestRunPath);
}

TEST_F(MakeCanonicalOrSetDefaults, relativePathsDots) {
  string InRoot = ".";
  string OutRoot = "..";
  ASSERT_EQ(true, makeCanonicalOrSetDefaults(InRoot, OutRoot, {"./file.cpp"}));
  ASSERT_EQ(InRoot, TestRunPath);
  ASSERT_EQ(OutRoot, path::parent_path(TestRunPath));
}

TEST_F(MakeCanonicalOrSetDefaults, relativeOutRoot) {
  string InRoot = TempDirAbsolute + "/a/b/in";
  string OutRoot = "..";
  ASSERT_EQ(true, makeCanonicalOrSetDefaults(InRoot, OutRoot, {"./file.cpp"}));
  ASSERT_EQ(InRoot, TempDirAbsolute + "/a/b/in");
  ASSERT_EQ(OutRoot, path::parent_path(TestRunPath));
}

TEST(validatePaths, validCase) {
  string InRoot = "/a/b/in";
  ASSERT_EQ(true,
            validatePaths(InRoot, {"/a/b/in/file.cpp", "/a/b/in/c/file.cpp"}));
}

TEST(validatePaths, inrootIsStringPrefix) {
  string InRoot = "/a/b/in";
  ASSERT_EQ(false, validatePaths(InRoot, {"/a/b/infalse/file.cpp"}));
  ASSERT_EQ(InRoot, "/a/b/in");
}

TEST(validatePaths, relativePaths) {
  string InRoot = "/a/b/in";
  ASSERT_EQ(false, validatePaths(InRoot, {"./file.cpp"}));
  ASSERT_EQ(InRoot, "/a/b/in");
}

TEST(validatePaths, secondFileNotInInroot) {
  string InRoot = "/a/b/in";
  ASSERT_EQ(false,
            validatePaths(InRoot, {"/a/b/in/file.cpp", "/a/b/c/in/file.cpp"}));
  ASSERT_EQ(InRoot, "/a/b/in");
}

TEST(validatePaths, noExtension) {
  string InRoot = "/a/b/in";
  ASSERT_EQ(false,
            validatePaths(InRoot, {"/a/b/in/file", "/a/b/c/in/file.cpp"}));
  ASSERT_EQ(InRoot, "/a/b/in");
}

TEST(validatePaths, noExtensionSecondNotInInroot) {
  string InRoot = "/a/b/in";
  ASSERT_EQ(false, validatePaths(InRoot, {"invalid1", "/a/b/c/in/file.cpp"}));
  ASSERT_EQ(InRoot, "/a/b/in");
}

TEST(isCanonical, hidden) {
  string s = "/a/b/.a/.c";
  ASSERT_EQ(true, isCanonical(s));
}

TEST(isCanonical, dots) {
  string s = "/a/b/./.c";
  ASSERT_EQ(false, isCanonical(s));
}

TEST(isCanonical, dotDots) {
  string s = "/a/b/../.c";
  ASSERT_EQ(false, isCanonical(s));
}

