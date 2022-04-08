#include "../../lib/C2S/SaveNewFiles.cpp"
#include "gtest/gtest.h"

TEST(rewriteDir, fileUnderInRoot) {
#if _WIN32
  SmallString<512> AbsPath = StringRef{"p:/a/b/in/file.cpp"};
  rewriteDir(AbsPath, "p:/a/b/in", "p:/a/c");
  std::replace(AbsPath.begin(), AbsPath.end(), '\\', '/');
  EXPECT_EQ(AbsPath, "p:/a/c/file.cpp");
#else
  SmallString<512> AbsPath = StringRef{"/a/b/in/file.cpp"};
  rewriteDir(AbsPath, "/a/b/in", "/a/c");
  EXPECT_EQ(AbsPath, "/a/c/file.cpp");
#endif
}

TEST(rewriteDir, fileInDirUnderInRoot) {
#if _WIN32
  SmallString<512> AbsPath = StringRef{"p:/a/b/in/d/file.cpp"};
  rewriteDir(AbsPath, "p:/a/b/in", "p:/a/c");
  std::replace(AbsPath.begin(), AbsPath.end(), '\\', '/');
  EXPECT_EQ(AbsPath, "p:/a/c/d/file.cpp");
#else
  SmallString<512> AbsPath = StringRef{"/a/b/in/d/file.cpp"};
  rewriteDir(AbsPath, "/a/b/in", "/a/c");
  EXPECT_EQ(AbsPath, "/a/c/d/file.cpp");
#endif
}

TEST(rewriteFileName, renameCU) {
  SmallString<512> AbsPath = StringRef("/a/b/in/file.cu");
  rewriteFileName(AbsPath);
  EXPECT_EQ(AbsPath, "/a/b/in/file.dp.cpp");
}

TEST(rewriteFileName, renameCUH) {
  SmallString<512> AbsPath = StringRef("/a/b/in/d/file.cuh");
  rewriteFileName(AbsPath);
  EXPECT_EQ(AbsPath, "/a/b/in/d/file.dp.hpp");
}

TEST(rewriteFileName, dontRenameH) {
  SmallString<512> AbsPath = StringRef("/a/b/in/file.h");
  rewriteFileName(AbsPath);
  EXPECT_EQ(AbsPath, "/a/b/in/file.h");
}

TEST(rewriteFileName, renameCppfile) {
  SmallString<512> AbsPath = StringRef("/a/b/in/file.cpp");
  rewriteFileName(AbsPath);
  EXPECT_EQ(AbsPath, "/a/b/in/file.cpp.dp.cpp");
}

TEST(rewriteFileName, renameCxxfile) {
  SmallString<512> AbsPath = StringRef("/a/b/in/file.cxx");
  rewriteFileName(AbsPath);
  EXPECT_EQ(AbsPath, "/a/b/in/file.cxx.dp.cpp");
}

TEST(rewriteFileName, renameCCfile) {
  SmallString<512> AbsPath = StringRef("/a/b/in/file.cc");
  rewriteFileName(AbsPath);
  EXPECT_EQ(AbsPath, "/a/b/in/file.cc.dp.cpp");
}

TEST(rewriteFileName, dontRenameHpp) {
  SmallString<512> AbsPath = StringRef("/a/b/in/file.hpp");
  rewriteFileName(AbsPath);
  EXPECT_EQ(AbsPath, "/a/b/in/file.hpp");
}

TEST(rewriteFileName, dontRenameHxx) {
  SmallString<512> AbsPath = StringRef("/a/b/in/file.hxx");
  rewriteFileName(AbsPath);
  EXPECT_EQ(AbsPath, "/a/b/in/file.hxx");
}
