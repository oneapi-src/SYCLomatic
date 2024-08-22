#include "../../lib/DPCT/SaveNewFiles.cpp"
#include "gtest/gtest.h"

TEST(rewriteCanonicalDir, fileUnderInRoot) {
#if _WIN32
  clang::tooling::UnifiedPath AbsPath = StringRef{"p:/a/b/in/file.cpp"};
  rewriteCanonicalDir(AbsPath, "p:/a/b/in", "p:/a/c");
  EXPECT_EQ(AbsPath, "p:/a/c/file.cpp");
#else
  clang::tooling::UnifiedPath AbsPath = StringRef{"/a/b/in/file.cpp"};
  rewriteCanonicalDir(AbsPath, clang::tooling::UnifiedPath("/a/b/in"), clang::tooling::UnifiedPath("/a/c"));
  EXPECT_EQ(AbsPath, "/a/c/file.cpp");
#endif
}

TEST(rewriteCanonicalDir, fileInDirUnderInRoot) {
#if _WIN32
  clang::tooling::UnifiedPath AbsPath = StringRef{"p:/a/b/in/d/file.cpp"};
  rewriteCanonicalDir(AbsPath, "p:/a/b/in", "p:/a/c");
  EXPECT_EQ(AbsPath, "p:/a/c/d/file.cpp");
#else
  clang::tooling::UnifiedPath AbsPath = StringRef{"/a/b/in/d/file.cpp"};
  rewriteCanonicalDir(AbsPath, clang::tooling::UnifiedPath("/a/b/in"), clang::tooling::UnifiedPath("/a/c"));
  EXPECT_EQ(AbsPath, "/a/c/d/file.cpp");
#endif
}

TEST(rewriteFileName, renameCU) {
  clang::tooling::UnifiedPath AbsPath = StringRef("/a/b/in/file.cu");
  rewriteFileName(AbsPath);
  EXPECT_EQ(AbsPath, "/a/b/in/file.dp.cpp");
}

TEST(rewriteFileName, renameCUH) {
  clang::tooling::UnifiedPath AbsPath = StringRef("/a/b/in/d/file.cuh");
  rewriteFileName(AbsPath);
  EXPECT_EQ(AbsPath, "/a/b/in/d/file.dp.hpp");
}

TEST(rewriteFileName, dontRenameH) {
  clang::tooling::UnifiedPath AbsPath = StringRef("/a/b/in/file.h");
  rewriteFileName(AbsPath);
  EXPECT_EQ(AbsPath, "/a/b/in/file.h");
}

TEST(rewriteFileName, renameCppfile) {
  clang::tooling::UnifiedPath AbsPath = StringRef("/a/b/in/file.cpp");
  rewriteFileName(AbsPath);
  EXPECT_EQ(AbsPath, "/a/b/in/file.cpp.dp.cpp");
}

TEST(rewriteFileName, renameCxxfile) {
  clang::tooling::UnifiedPath AbsPath = StringRef("/a/b/in/file.cxx");
  rewriteFileName(AbsPath);
  EXPECT_EQ(AbsPath, "/a/b/in/file.cxx.dp.cpp");
}

TEST(rewriteFileName, renameCCfile) {
  clang::tooling::UnifiedPath AbsPath = StringRef("/a/b/in/file.cc");
  rewriteFileName(AbsPath);
  EXPECT_EQ(AbsPath, "/a/b/in/file.cc.dp.cpp");
}

TEST(rewriteFileName, dontRenameHpp) {
  clang::tooling::UnifiedPath AbsPath = StringRef("/a/b/in/file.hpp");
  rewriteFileName(AbsPath);
  EXPECT_EQ(AbsPath, "/a/b/in/file.hpp");
}

TEST(rewriteFileName, dontRenameHxx) {
  clang::tooling::UnifiedPath AbsPath = StringRef("/a/b/in/file.hxx");
  rewriteFileName(AbsPath);
  EXPECT_EQ(AbsPath, "/a/b/in/file.hxx");
}
