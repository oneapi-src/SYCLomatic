#include "../../lib/C2S/ValidateArguments.cpp"
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

    TestRunPath = StringRef(CurrentDir).str();
    TempDirAbsolute = StringRef(UniqueDir).str();
    TempDir = (path::stem(TempDirAbsolute) + "/").str();

    fs::create_directories(TempDirAbsolute + "/a/b/in");
#if _WIN32
    std::replace(TempDirAbsolute.begin(), TempDirAbsolute.end(), '\\', '/');
    std::replace(TestRunPath.begin(), TestRunPath.end(), '\\', '/');
#endif
  }

  void TearDown() override { fs::remove_directories(TempDirAbsolute); }
};

TEST_F(MakeCanonicalOrSetDefaults, getDefaultOutRootEmpty) {
  string OutRoot;
  getDefaultOutRoot(OutRoot);
#if _WIN32
  std::replace(OutRoot.begin(), OutRoot.end(), '\\', '/');
#endif
  ASSERT_EQ(OutRoot, TestRunPath + "/dpct_output");
}

TEST_F(MakeCanonicalOrSetDefaults, getDefaultOutRoot) {
  string OutRoot = "is not empty";
  getDefaultOutRoot(OutRoot);
#if _WIN32
  std::replace(OutRoot.begin(), OutRoot.end(), '\\', '/');
#endif
  ASSERT_EQ(OutRoot, TestRunPath + "/dpct_output");
}

TEST(getDefaultInRoot, noInroot) {
  string inroot = "";
#if _WIN32
  ASSERT_EQ(true, getDefaultInRoot(inroot, {"p:/a/b/in/file.cpp"}));
  std::replace(inroot.begin(), inroot.end(), '\\', '/');
  ASSERT_EQ(inroot, "p:/a/b/in");

  ASSERT_EQ(true, getDefaultInRoot(inroot,
                                   {"p:/a/b/in/file.cpp", "p:/a/b/in/c/file.cpp"}));
  std::replace(inroot.begin(), inroot.end(), '\\', '/');
  ASSERT_EQ(inroot, "p:/a/b/in");
#else
  ASSERT_EQ(true, getDefaultInRoot(inroot, {"/a/b/in/file.cpp"}));
  ASSERT_EQ(inroot, "/a/b/in");

  ASSERT_EQ(true, getDefaultInRoot(inroot,
                                   {"/a/b/in/file.cpp", "/a/b/in/c/file.cpp"}));
  ASSERT_EQ(inroot, "/a/b/in");
#endif
  ASSERT_EQ(true, getDefaultInRoot(inroot, {}));
  ASSERT_EQ(inroot, ".");
}

TEST_F(MakeCanonicalOrSetDefaults, empty) {
  string InRoot;
  string OutRoot;
  ASSERT_EQ(true, makeInRootCanonicalOrSetDefaults(
                      InRoot, {TempDirAbsolute + "/a/b/in/file.cpp"}));
  ASSERT_EQ(true, makeOutRootCanonicalOrSetDefaults(OutRoot));
#if _WIN32
  std::replace(InRoot.begin(), InRoot.end(), '\\', '/');
  std::replace(OutRoot.begin(), OutRoot.end(), '\\', '/');
#endif
  ASSERT_EQ(InRoot, TempDirAbsolute + "/a/b/in");
  ASSERT_EQ(OutRoot, TestRunPath + "/dpct_output");
}

TEST_F(MakeCanonicalOrSetDefaults, emptyOnlyOneFileAllowed) {
  string InRoot;
  string OutRoot;
  ASSERT_EQ(false,
            makeInRootCanonicalOrSetDefaults(
                InRoot,{"/a/b/in/file.cpp", "/a/b/in/file.cpp"}));
  ASSERT_EQ(true,
            makeOutRootCanonicalOrSetDefaults(OutRoot));
}

TEST_F(MakeCanonicalOrSetDefaults, dotAtTheEnd) {
  string InRoot = TempDirAbsolute + "/a/b/in/.";
  string OutRoot = TempDirAbsolute + "/a/b/.";
  ASSERT_EQ(true, makeInRootCanonicalOrSetDefaults(
                      InRoot,
                      {TempDirAbsolute + "/a/b/in/file.cpp",
                       TempDirAbsolute + "/a/b/in/c/file.cpp"}));
  ASSERT_EQ(true, makeOutRootCanonicalOrSetDefaults(OutRoot));
#if _WIN32
  std::replace(InRoot.begin(), InRoot.end(), '\\', '/');
  std::replace(OutRoot.begin(), OutRoot.end(), '\\', '/');
#endif
  ASSERT_EQ(InRoot, TempDirAbsolute + "/a/b/in");
  ASSERT_EQ(OutRoot, TempDirAbsolute + "/a/b");
}

TEST_F(MakeCanonicalOrSetDefaults, dotInTheMiddle) {
  string InRoot = TempDirAbsolute + "/a/b/./in/.";
  string OutRoot = TempDirAbsolute + "/a/b/.";
  ASSERT_EQ(true, makeInRootCanonicalOrSetDefaults(
                      InRoot,
                      {TempDirAbsolute + "/a/b/./in/file.cpp",
                       TempDirAbsolute + "/a/b/in/c/file.cpp"}));
  ASSERT_EQ(true, makeOutRootCanonicalOrSetDefaults(OutRoot));
#if _WIN32
  std::replace(InRoot.begin(), InRoot.end(), '\\', '/');
  std::replace(OutRoot.begin(), OutRoot.end(), '\\', '/');
#endif
  ASSERT_EQ(InRoot, TempDirAbsolute + "/a/b/in");
  ASSERT_EQ(OutRoot, TempDirAbsolute + "/a/b");
}

TEST_F(MakeCanonicalOrSetDefaults, dotDotAtTheEnd) {
  string InRoot = TempDirAbsolute + "/a/b/in/..";
  string OutRoot = TempDirAbsolute + "/a/b/";
  ASSERT_EQ(true, makeInRootCanonicalOrSetDefaults(
                      InRoot,
                      {TempDirAbsolute + "/a/b/in/file.cpp",
                       TempDirAbsolute + "/a/b/in/c/file.cpp"}));
  ASSERT_EQ(true, makeOutRootCanonicalOrSetDefaults(OutRoot));
#if _WIN32
  std::replace(InRoot.begin(), InRoot.end(), '\\', '/');
  std::replace(OutRoot.begin(), OutRoot.end(), '\\', '/');
#endif
  ASSERT_EQ(InRoot, TempDirAbsolute + "/a/b");
  ASSERT_EQ(OutRoot, TempDirAbsolute + "/a/b");
}

TEST_F(MakeCanonicalOrSetDefaults, dotDotInTheMiddle) {
  string InRoot = TempDirAbsolute + "/a/b/../b/in";
  string OutRoot = TempDirAbsolute + "/a/b/";
  ASSERT_EQ(true, makeInRootCanonicalOrSetDefaults(
                      InRoot,
                      {TempDirAbsolute + "/a/b/in/file.cpp",
                       TempDirAbsolute + "/a/b/in/c/file.cpp"}));
  ASSERT_EQ(true, makeOutRootCanonicalOrSetDefaults(OutRoot));
#if _WIN32
  std::replace(InRoot.begin(), InRoot.end(), '\\', '/');
  std::replace(OutRoot.begin(), OutRoot.end(), '\\', '/');
#endif
  ASSERT_EQ(InRoot, TempDirAbsolute + "/a/b/in");
  ASSERT_EQ(OutRoot, TempDirAbsolute + "/a/b");
}

TEST_F(MakeCanonicalOrSetDefaults, relativePaths) {
  string InRoot = TempDir + "a/b/../b/in";
  string OutRoot = TempDir + "a/b/";
  ASSERT_EQ(true, makeInRootCanonicalOrSetDefaults(InRoot,
                                             {TempDir + "a/b/in/file.cpp",
                                              TempDir + "a/b/in/c/file.cpp"}));
  ASSERT_EQ(true, makeOutRootCanonicalOrSetDefaults(OutRoot));
#if _WIN32
  std::replace(InRoot.begin(), InRoot.end(), '\\', '/');
  std::replace(OutRoot.begin(), OutRoot.end(), '\\', '/');
#endif
  ASSERT_EQ(InRoot, TempDirAbsolute + "/a/b/in");
  ASSERT_EQ(OutRoot, TempDirAbsolute + "/a/b");
}

TEST_F(MakeCanonicalOrSetDefaults, relativePathsNoRoots) {
  string InRoot;
  string OutRoot;
  ASSERT_EQ(true, makeInRootCanonicalOrSetDefaults(InRoot, {"file.cpp"}));
  ASSERT_EQ(true, makeOutRootCanonicalOrSetDefaults(OutRoot));
#if _WIN32
  std::replace(InRoot.begin(), InRoot.end(), '\\', '/');
  std::replace(OutRoot.begin(), OutRoot.end(), '\\', '/');
#endif
  ASSERT_EQ(InRoot, TestRunPath);
  ASSERT_EQ(OutRoot, TestRunPath + "/dpct_output" );
}

TEST_F(MakeCanonicalOrSetDefaults, relativePathsDots) {
  string InRoot = ".";
  string OutRoot = "..";
  ASSERT_EQ(true, makeInRootCanonicalOrSetDefaults(InRoot, {"./file.cpp"}));
  ASSERT_EQ(true, makeOutRootCanonicalOrSetDefaults(OutRoot));
#if _WIN32
  std::replace(InRoot.begin(), InRoot.end(), '\\', '/');
  std::replace(OutRoot.begin(), OutRoot.end(), '\\', '/');
#endif
  ASSERT_EQ(InRoot, TestRunPath);
  ASSERT_EQ(OutRoot, path::parent_path(TestRunPath));
}

TEST_F(MakeCanonicalOrSetDefaults, relativeOutRoot) {
  string InRoot = TempDirAbsolute + "/a/b/in";
  string OutRoot = "..";
  ASSERT_EQ(true, makeInRootCanonicalOrSetDefaults(InRoot, {"./file.cpp"}));
  ASSERT_EQ(true, makeOutRootCanonicalOrSetDefaults(OutRoot));
#if _WIN32
  std::replace(InRoot.begin(), InRoot.end(), '\\', '/');
  std::replace(OutRoot.begin(), OutRoot.end(), '\\', '/');
#endif
  ASSERT_EQ(InRoot, TempDirAbsolute + "/a/b/in");
  ASSERT_EQ(OutRoot, path::parent_path(TestRunPath));
}

TEST(validatePaths, validCase) {
#if _WIN32
  string InRoot = "p:/a/b/in";
  ASSERT_EQ(true,
            validatePaths(InRoot, {"p:/a/b/in/file.cpp", "p:/a/b/in/c/file.cpp"}));
#else
  string InRoot = "/a/b/in";
  ASSERT_EQ(true,
            validatePaths(InRoot, {"/a/b/in/file.cpp", "/a/b/in/c/file.cpp"}));
#endif
}

TEST(validatePaths, inrootIsStringPrefix) {
#if _WIN32
  string InRoot = "p:/a/b/in";
  ASSERT_EQ(false, validatePaths(InRoot, {"p:/a/b/infalse/file.cpp"}));
  std::replace(InRoot.begin(), InRoot.end(), '\\', '/');
  ASSERT_EQ(InRoot, "p:/a/b/in");
#else
  string InRoot = "/a/b/in";
  ASSERT_EQ(false, validatePaths(InRoot, {"/a/b/infalse/file.cpp"}));
  ASSERT_EQ(InRoot, "/a/b/in");
#endif
}

TEST(validatePaths, relativePaths) {
#if _WIN32
  string InRoot = "p:/a/b/in";
  ASSERT_EQ(false, validatePaths(InRoot, {"./file.cpp"}));
  std::replace(InRoot.begin(), InRoot.end(), '\\', '/');
  ASSERT_EQ(InRoot, "p:/a/b/in");
#else
  string InRoot = "/a/b/in";
  ASSERT_EQ(false, validatePaths(InRoot, {"./file.cpp"}));
  ASSERT_EQ(InRoot, "/a/b/in");
#endif
}

TEST(validatePaths, secondFileNotInInroot) {
#if _WIN32
  string InRoot = "p:/a/b/in";
  ASSERT_EQ(false,
            validatePaths(InRoot, {"p:/a/b/in/file.cpp", "p:/a/b/c/in/file.cpp"}));
  std::replace(InRoot.begin(), InRoot.end(), '\\', '/');
  ASSERT_EQ(InRoot, "p:/a/b/in");
#else
  string InRoot = "/a/b/in";
  ASSERT_EQ(false,
            validatePaths(InRoot, {"/a/b/in/file.cpp", "/a/b/c/in/file.cpp"}));
  ASSERT_EQ(InRoot, "/a/b/in");
#endif
}

TEST(validatePaths, noExtension) {
#if _WIN32
  string InRoot = "p:/a/b/in";
  ASSERT_EQ(false,
            validatePaths(InRoot, {"p:/a/b/in/file", "p:/a/b/c/in/file.cpp"}));
  std::replace(InRoot.begin(), InRoot.end(), '\\', '/');
  ASSERT_EQ(InRoot, "p:/a/b/in");
#else
  string InRoot = "/a/b/in";
  ASSERT_EQ(false,
            validatePaths(InRoot, {"/a/b/in/file", "/a/b/c/in/file.cpp"}));
  ASSERT_EQ(InRoot, "/a/b/in");
#endif
}

TEST(validatePaths, noExtensionSecondNotInInroot) {
#if _WIN32
  string InRoot = "p:/a/b/in";
  ASSERT_EQ(false, validatePaths(InRoot, {"invalid1", "p:/a/b/c/in/file.cpp"}));
  std::replace(InRoot.begin(), InRoot.end(), '\\', '/');
  ASSERT_EQ(InRoot, "p:/a/b/in");
#else
  string InRoot = "/a/b/in";
  ASSERT_EQ(false, validatePaths(InRoot, {"invalid1", "/a/b/c/in/file.cpp"}));
  ASSERT_EQ(InRoot, "/a/b/in");
#endif
}

TEST(isCanonical, hidden) {
#if _WIN32
  string s = "p:/a/b/.a/.c";
#else
  string s = "/a/b/.a/.c";
#endif
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

