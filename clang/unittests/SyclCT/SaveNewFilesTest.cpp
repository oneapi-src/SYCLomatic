#include "../../lib/SyclCT/SaveNewFiles.cpp"
#include "gtest/gtest.h"

TEST(rewriteDir, fileUnderInRoot) {
  SmallString<256> AbsPath = StringRef{"/a/b/in/file.cpp"};
  rewriteDir(AbsPath, "/a/b/in", "/a/c");
  EXPECT_EQ(AbsPath, "/a/c/file.cpp");
}

TEST(rewriteDir, fileInDirUnderInRoot) {
  SmallString<256> AbsPath = StringRef{"/a/b/in/d/file.cpp"};
  rewriteDir(AbsPath, "/a/b/in", "/a/c");
  EXPECT_EQ(AbsPath, "/a/c/d/file.cpp");
}

TEST(rewriteFileName, renameCU) {
  SmallString<256> AbsPath = StringRef("/a/b/in/file.cu");
  rewriteFileName(AbsPath);
  EXPECT_EQ(AbsPath, "/a/b/in/file.sycl.cpp");
}

TEST(rewriteFileName, renameCUH) {
  SmallString<256> AbsPath = StringRef("/a/b/in/d/file.cuh");
  rewriteFileName(AbsPath);
  EXPECT_EQ(AbsPath, "/a/b/in/d/file.sycl.hpp");
}

TEST(rewriteFileName, dontRenameH) {
  SmallString<256> AbsPath = StringRef("/a/b/in/file.h");
  rewriteFileName(AbsPath);
  EXPECT_EQ(AbsPath, "/a/b/in/file.h");
}

TEST(rewriteFileName, dontRenameCPP) {
  SmallString<256> AbsPath = StringRef("/a/b/in/file.cpp");
  rewriteFileName(AbsPath);
  EXPECT_EQ(AbsPath, "/a/b/in/file.cpp");
}
