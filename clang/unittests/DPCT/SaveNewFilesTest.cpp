#include "../../lib/DPCT/SaveNewFiles.cpp"
#include "gtest/gtest.h"

TEST(rewriteDir, fileUnderInRoot) {
#if _WIN32
  clang::tooling::DpctPath AbsPath = StringRef{"p:/a/b/in/file.cpp"};
  rewriteDir(AbsPath, "p:/a/b/in", "p:/a/c");
  std::replace(AbsPath.begin(), AbsPath.end(), '\\', '/');
  EXPECT_EQ(AbsPath, "p:/a/c/file.cpp");
#else
  clang::tooling::DpctPath AbsPath = StringRef{"/a/b/in/file.cpp"};
  rewriteDir(AbsPath, clang::tooling::DpctPath("/a/b/in"), clang::tooling::DpctPath("/a/c"));
  EXPECT_EQ(AbsPath, "/a/c/file.cpp");
#endif
}

TEST(rewriteDir, fileInDirUnderInRoot) {
#if _WIN32
  clang::tooling::DpctPath AbsPath = StringRef{"p:/a/b/in/d/file.cpp"};
  rewriteDir(AbsPath, "p:/a/b/in", "p:/a/c");
  std::replace(AbsPath.begin(), AbsPath.end(), '\\', '/');
  EXPECT_EQ(AbsPath, "p:/a/c/d/file.cpp");
#else
  clang::tooling::DpctPath AbsPath = StringRef{"/a/b/in/d/file.cpp"};
  rewriteDir(AbsPath, clang::tooling::DpctPath("/a/b/in"), clang::tooling::DpctPath("/a/c"));
  EXPECT_EQ(AbsPath, "/a/c/d/file.cpp");
#endif
}

TEST(rewriteFileName, renameCU) {
  clang::tooling::DpctPath AbsPath = StringRef("/a/b/in/file.cu");
  rewriteFileName(AbsPath);
  EXPECT_EQ(AbsPath, "/a/b/in/file.dp.cpp");
}

TEST(rewriteFileName, renameCUH) {
  clang::tooling::DpctPath AbsPath = StringRef("/a/b/in/d/file.cuh");
  rewriteFileName(AbsPath);
  EXPECT_EQ(AbsPath, "/a/b/in/d/file.dp.hpp");
}

TEST(rewriteFileName, dontRenameH) {
  clang::tooling::DpctPath AbsPath = StringRef("/a/b/in/file.h");
  rewriteFileName(AbsPath);
  EXPECT_EQ(AbsPath, "/a/b/in/file.h");
}

TEST(rewriteFileName, renameCppfile) {
  clang::tooling::DpctPath AbsPath = StringRef("/a/b/in/file.cpp");
  rewriteFileName(AbsPath);
  EXPECT_EQ(AbsPath, "/a/b/in/file.cpp.dp.cpp");
}

TEST(rewriteFileName, renameCxxfile) {
  clang::tooling::DpctPath AbsPath = StringRef("/a/b/in/file.cxx");
  rewriteFileName(AbsPath);
  EXPECT_EQ(AbsPath, "/a/b/in/file.cxx.dp.cpp");
}

TEST(rewriteFileName, renameCCfile) {
  clang::tooling::DpctPath AbsPath = StringRef("/a/b/in/file.cc");
  rewriteFileName(AbsPath);
  EXPECT_EQ(AbsPath, "/a/b/in/file.cc.dp.cpp");
}

TEST(rewriteFileName, dontRenameHpp) {
  clang::tooling::DpctPath AbsPath = StringRef("/a/b/in/file.hpp");
  rewriteFileName(AbsPath);
  EXPECT_EQ(AbsPath, "/a/b/in/file.hpp");
}

TEST(rewriteFileName, dontRenameHxx) {
  clang::tooling::DpctPath AbsPath = StringRef("/a/b/in/file.hxx");
  rewriteFileName(AbsPath);
  EXPECT_EQ(AbsPath, "/a/b/in/file.hxx");
}
