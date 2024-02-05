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
  }

  void TearDown() override { fs::remove_directories(TempDirAbsolute); }
};

TEST_F(MakeCanonicalOrSetDefaults, getDefaultOutRootEmpty) {
  clang::tooling::UnifiedPath OutRoot;
  getDefaultOutRoot(OutRoot);
  ASSERT_EQ(OutRoot, TestRunPath + "/dpct_output");
}

TEST_F(MakeCanonicalOrSetDefaults, getDefaultOutRoot) {
  clang::tooling::UnifiedPath OutRoot = std::string("is not empty");
  getDefaultOutRoot(OutRoot);
  ASSERT_EQ(OutRoot, TestRunPath + "/dpct_output");
}

TEST(getDefaultInRoot, noInroot) {
  clang::tooling::UnifiedPath inroot;
#if _WIN32
  ASSERT_EQ(true, getDefaultInRoot(inroot, {"p:/a/b/in/file.cpp"}));
  ASSERT_EQ(inroot, "p:/a/b/in");

  ASSERT_EQ(true, getDefaultInRoot(inroot,
                                   {"p:/a/b/in/file.cpp", "p:/a/b/in/c/file.cpp"}));
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
  clang::tooling::UnifiedPath InRoot;
  clang::tooling::UnifiedPath OutRoot;
  ASSERT_EQ(true, makeInRootCanonicalOrSetDefaults(
                      InRoot, {TempDirAbsolute + "/a/b/in/file.cpp"}));
  ASSERT_EQ(InRoot, TempDirAbsolute + "/a/b/in");
  ASSERT_EQ(OutRoot, TestRunPath + "/dpct_output");
}

TEST_F(MakeCanonicalOrSetDefaults, emptyOnlyOneFileAllowed) {
  clang::tooling::UnifiedPath InRoot;
  clang::tooling::UnifiedPath OutRoot;
  ASSERT_EQ(false,
            makeInRootCanonicalOrSetDefaults(
                InRoot,{"/a/b/in/file.cpp", "/a/b/in/file.cpp"}));
}

TEST_F(MakeCanonicalOrSetDefaults, dotAtTheEnd) {
  clang::tooling::UnifiedPath InRoot = TempDirAbsolute + "/a/b/in/.";
  clang::tooling::UnifiedPath OutRoot = TempDirAbsolute + "/a/b/.";
  ASSERT_EQ(true, makeInRootCanonicalOrSetDefaults(
                      InRoot,
                      {TempDirAbsolute + "/a/b/in/file.cpp",
                       TempDirAbsolute + "/a/b/in/c/file.cpp"}));
  ASSERT_EQ(InRoot, TempDirAbsolute + "/a/b/in");
  ASSERT_EQ(OutRoot, TempDirAbsolute + "/a/b");
}

TEST_F(MakeCanonicalOrSetDefaults, dotInTheMiddle) {
  clang::tooling::UnifiedPath InRoot = TempDirAbsolute + "/a/b/./in/.";
  clang::tooling::UnifiedPath OutRoot = TempDirAbsolute + "/a/b/.";
  ASSERT_EQ(true, makeInRootCanonicalOrSetDefaults(
                      InRoot,
                      {TempDirAbsolute + "/a/b/./in/file.cpp",
                       TempDirAbsolute + "/a/b/in/c/file.cpp"}));
  ASSERT_EQ(InRoot, TempDirAbsolute + "/a/b/in");
  ASSERT_EQ(OutRoot, TempDirAbsolute + "/a/b");
}

TEST_F(MakeCanonicalOrSetDefaults, dotDotAtTheEnd) {
  clang::tooling::UnifiedPath InRoot = TempDirAbsolute + "/a/b/in/..";
  clang::tooling::UnifiedPath OutRoot = TempDirAbsolute + "/a/b/";
  ASSERT_EQ(true, makeInRootCanonicalOrSetDefaults(
                      InRoot,
                      {TempDirAbsolute + "/a/b/in/file.cpp",
                       TempDirAbsolute + "/a/b/in/c/file.cpp"}));
  ASSERT_EQ(InRoot, TempDirAbsolute + "/a/b");
  ASSERT_EQ(OutRoot, TempDirAbsolute + "/a/b");
}

TEST_F(MakeCanonicalOrSetDefaults, dotDotInTheMiddle) {
  clang::tooling::UnifiedPath InRoot = TempDirAbsolute + "/a/b/../b/in";
  clang::tooling::UnifiedPath OutRoot = TempDirAbsolute + "/a/b/";
  ASSERT_EQ(true, makeInRootCanonicalOrSetDefaults(
                      InRoot,
                      {TempDirAbsolute + "/a/b/in/file.cpp",
                       TempDirAbsolute + "/a/b/in/c/file.cpp"}));
  ASSERT_EQ(InRoot, TempDirAbsolute + "/a/b/in");
  ASSERT_EQ(OutRoot, TempDirAbsolute + "/a/b");
}

TEST_F(MakeCanonicalOrSetDefaults, relativePaths) {
  clang::tooling::UnifiedPath InRoot = TempDir + "a/b/../b/in";
  clang::tooling::UnifiedPath OutRoot = TempDir + "a/b/";
  ASSERT_EQ(true, makeInRootCanonicalOrSetDefaults(InRoot,
                                             {TempDir + "a/b/in/file.cpp",
                                              TempDir + "a/b/in/c/file.cpp"}));
  ASSERT_EQ(InRoot, TempDirAbsolute + "/a/b/in");
  ASSERT_EQ(OutRoot, TempDirAbsolute + "/a/b");
}

TEST_F(MakeCanonicalOrSetDefaults, relativePathsNoRoots) {
  clang::tooling::UnifiedPath InRoot;
  clang::tooling::UnifiedPath OutRoot;
  ASSERT_EQ(true, makeInRootCanonicalOrSetDefaults(InRoot, {"file.cpp"}));
  ASSERT_EQ(InRoot, TestRunPath);
  ASSERT_EQ(OutRoot, TestRunPath + "/dpct_output" );
}

TEST_F(MakeCanonicalOrSetDefaults, relativePathsDots) {
  clang::tooling::UnifiedPath InRoot = std::string(".");
  clang::tooling::UnifiedPath OutRoot = std::string("..");
  ASSERT_EQ(true, makeInRootCanonicalOrSetDefaults(InRoot, {"./file.cpp"}));
  ASSERT_EQ(InRoot, TestRunPath);
  ASSERT_EQ(OutRoot, path::parent_path(TestRunPath));
}

TEST_F(MakeCanonicalOrSetDefaults, relativeOutRoot) {
  clang::tooling::UnifiedPath InRoot = TempDirAbsolute + "/a/b/in";
  clang::tooling::UnifiedPath OutRoot = std::string("..");
  ASSERT_EQ(true, makeInRootCanonicalOrSetDefaults(InRoot, {"./file.cpp"}));
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
  ASSERT_EQ(InRoot, "p:/a/b/in");
#else
  string InRoot = "/a/b/in";
  ASSERT_EQ(false, validatePaths(InRoot, {"invalid1", "/a/b/c/in/file.cpp"}));
  ASSERT_EQ(InRoot, "/a/b/in");
#endif
}
