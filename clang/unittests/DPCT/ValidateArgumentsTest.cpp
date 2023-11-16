#include "../../lib/DPCT/ValidateArguments.cpp"
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
  clang::tooling::DpctPath OutRoot;
  getDefaultOutRoot(OutRoot);
#if _WIN32
  std::replace(OutRoot.begin(), OutRoot.end(), '\\', '/');
#endif
  ASSERT_EQ(OutRoot, TestRunPath + "/dpct_output");
}

TEST_F(MakeCanonicalOrSetDefaults, getDefaultOutRoot) {
  clang::tooling::DpctPath OutRoot = std::string("is not empty");
  getDefaultOutRoot(OutRoot);
#if _WIN32
  std::replace(OutRoot.begin(), OutRoot.end(), '\\', '/');
#endif
  ASSERT_EQ(OutRoot, TestRunPath + "/dpct_output");
}

TEST(getDefaultInRoot, noInroot) {
  clang::tooling::DpctPath inroot;
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
  clang::tooling::DpctPath InRoot;
  clang::tooling::DpctPath OutRoot;
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
  clang::tooling::DpctPath InRoot;
  clang::tooling::DpctPath OutRoot;
  ASSERT_EQ(false,
            makeInRootCanonicalOrSetDefaults(
                InRoot,{"/a/b/in/file.cpp", "/a/b/in/file.cpp"}));
  ASSERT_EQ(true,
            makeOutRootCanonicalOrSetDefaults(OutRoot));
}

TEST_F(MakeCanonicalOrSetDefaults, dotAtTheEnd) {
  clang::tooling::DpctPath InRoot = TempDirAbsolute + "/a/b/in/.";
  clang::tooling::DpctPath OutRoot = TempDirAbsolute + "/a/b/.";
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
  clang::tooling::DpctPath InRoot = TempDirAbsolute + "/a/b/./in/.";
  clang::tooling::DpctPath OutRoot = TempDirAbsolute + "/a/b/.";
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
  clang::tooling::DpctPath InRoot = TempDirAbsolute + "/a/b/in/..";
  clang::tooling::DpctPath OutRoot = TempDirAbsolute + "/a/b/";
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
  clang::tooling::DpctPath InRoot = TempDirAbsolute + "/a/b/../b/in";
  clang::tooling::DpctPath OutRoot = TempDirAbsolute + "/a/b/";
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
  clang::tooling::DpctPath InRoot = TempDir + "a/b/../b/in";
  clang::tooling::DpctPath OutRoot = TempDir + "a/b/";
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
  clang::tooling::DpctPath InRoot;
  clang::tooling::DpctPath OutRoot;
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
  clang::tooling::DpctPath InRoot = std::string(".");
  clang::tooling::DpctPath OutRoot = std::string("..");
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
  clang::tooling::DpctPath InRoot = TempDirAbsolute + "/a/b/in";
  clang::tooling::DpctPath OutRoot = std::string("..");
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
