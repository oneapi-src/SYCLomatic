#include "../../lib/DPCT/IncMigration/ReMigration.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Core/UnifiedPath.h"
#include "gtest/gtest.h"
#include <unordered_map>

using namespace llvm;
using namespace clang::tooling;
using namespace clang::dpct;

class ReMigrationTest1 : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(ReMigrationTest1, calculateUpdatedRanges) {
  // clang-format off
  // Old base:
/*
aaa bb ccc
aaa bb ccc
aaa bb ccc
aaa bb ccc
aaa bb ccc
aaa bb ccc
aaa bb ccc
*/
  // Old migrated:
/*
aaa zzz ccc
ppp aaa bb ccc
aaa yyy ccc
aaa bb ccc
aaxxx bb ccc
aaa bb ccc
qqq aaa bb ccc
*/
  // New base:
/*
aaa bb ccc
Hi bb Everyone
aaa bb ccc
aaa bb Everyone
Hi Hello Everyone
aaa Hi bb ccc
aaa bb ccc
*/
  // clang-format on

  Replacements NewRepl;
  Replacements Repls;
  Replacements Expected;

  llvm::cantFail(NewRepl.add(Replacement("file1.cpp", 4, 2, "zzz")));
  llvm::cantFail(NewRepl.add(Replacement("file1.cpp", 11, 0, "ppp ")));
  llvm::cantFail(NewRepl.add(Replacement("file1.cpp", 26, 2, "yyy")));
  llvm::cantFail(NewRepl.add(Replacement("file1.cpp", 46, 1, "xxx")));
  llvm::cantFail(NewRepl.add(Replacement("file1.cpp", 65, 1, "\nqqq")));

  llvm::cantFail(
      Repls.add(Replacement("file1.cpp", 11, 11, "Hi bb Everyone\n")));
  llvm::cantFail(
      Repls.add(Replacement("file1.cpp", 33, 11, "aaa bb Everyone\n")));
  llvm::cantFail(
      Repls.add(Replacement("file1.cpp", 44, 11, "Hi Hello Everyone\n")));
  llvm::cantFail(
      Repls.add(Replacement("file1.cpp", 55, 11, "aaa Hi bb ccc\n")));

  llvm::cantFail(Expected.add(Replacement("file1.cpp", 4, 2, "zzz")));
  llvm::cantFail(Expected.add(Replacement("file1.cpp", 30, 2, "yyy")));

  Replacements Result = calculateUpdatedRanges(Repls, NewRepl);
  ASSERT_EQ(Expected.size(), Result.size());
  size_t Num = Expected.size();
  auto ExpectedIt = Expected.begin();
  auto ResultIt = Result.begin();
  for (size_t i = 0; i < Num; ++i) {
    EXPECT_EQ(ExpectedIt->getFilePath(), ResultIt->getFilePath());
    EXPECT_EQ(ExpectedIt->getOffset(), ResultIt->getOffset());
    EXPECT_EQ(ExpectedIt->getLength(), ResultIt->getLength());
    EXPECT_EQ(ExpectedIt->getReplacementText(), ResultIt->getReplacementText());
    ExpectedIt++;
    ResultIt++;
  }
}

TEST_F(ReMigrationTest1, groupReplcementsByFile) {
  std::string FilePath1 =
      clang::tooling::UnifiedPath("file1.cpp").getCanonicalPath().str();
  std::string FilePath2 =
      clang::tooling::UnifiedPath("file2.cpp").getCanonicalPath().str();
  std::string FilePath3 =
      clang::tooling::UnifiedPath("file3.cpp").getCanonicalPath().str();

  std::vector<Replacement> Repls = {{FilePath1, 0, 0, ""},
                                    {FilePath2, 3, 0, ""},
                                    {FilePath1, 1, 0, ""},
                                    {FilePath3, 4, 0, ""},
                                    {FilePath2, 2, 0, ""}};
  std::map<std::string, std::vector<Replacement>> Expected = {
      {FilePath1,
       {
           {FilePath1, 0, 0, ""},
           {FilePath1, 1, 0, ""},
       }},
      {FilePath2,
       {
           {FilePath2, 2, 0, ""},
           {FilePath2, 3, 0, ""},
       }},
      {FilePath3, {{FilePath3, 4, 0, ""}}}};

  auto Result = groupReplcementsByFile(Repls);
  EXPECT_EQ(Expected.size(), Result.size());

  size_t Num = Expected.size();
  auto ExpectedIt = Expected.begin();
  auto ResultIt = Result.begin();
  for (size_t i = 0; i < Num; ++i) {
    EXPECT_EQ(ExpectedIt->first, ResultIt->first);
    EXPECT_EQ(ExpectedIt->second.size(), ResultIt->second.size());
    size_t SubNum = ExpectedIt->second.size();
    std::sort(ExpectedIt->second.begin(), ExpectedIt->second.end());
    std::sort(ResultIt->second.begin(), ResultIt->second.end());
    for (size_t j = 0; j < SubNum; ++j) {
      EXPECT_EQ(ExpectedIt->second[j].getFilePath(),
                ResultIt->second[j].getFilePath());
      EXPECT_EQ(ExpectedIt->second[j].getOffset(),
                ResultIt->second[j].getOffset());
      EXPECT_EQ(ExpectedIt->second[j].getLength(),
                ResultIt->second[j].getLength());
      EXPECT_EQ(ExpectedIt->second[j].getReplacementText(),
                ResultIt->second[j].getReplacementText());
    }
    ExpectedIt++;
    ResultIt++;
  }
}

class ReMigrationTest2 : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
  // clang-format off
/*
aaabbbbccc
dddeeeefff
ggghhhhiii
zzzzzzzzzz
yyyyyyyyyy
*/
  // clang-format on
  static StringRef getLineStringUnittest(clang::tooling::UnifiedPath FilePath,
                                         unsigned LineNumber) {
    static std::vector<std::string> LineString = {
        "aaabbbbccc\n", "dddeeeefff\n", "ggghhhhiii\n", "zzzzzzzzzz\n",
        "yyyyyyyyyy\n"};
    return StringRef(LineString[LineNumber - 1]);
  }
  static unsigned getLineNumberUnittest(clang::tooling::UnifiedPath FilePath,
                                        unsigned Offset) {
    static std::vector<unsigned> LineOffsets = {0, 11, 22, 33, 44};
    auto Iter =
        std::upper_bound(LineOffsets.begin(), LineOffsets.end(), Offset);
    if (Iter == LineOffsets.end())
      return LineOffsets.size();
    return std::distance(LineOffsets.begin(), Iter);
  }
  static unsigned
  getLineBeginOffsetUnittest(clang::tooling::UnifiedPath FilePath,
                             unsigned LineNumber) {
    static std::unordered_map<unsigned, unsigned> LineOffsets = {
        {1, 0}, {2, 11}, {3, 22}, {4, 33}, {5, 44}};
    return LineOffsets[LineNumber];
  }
};

TEST_F(ReMigrationTest2, splitReplInOrderToNotCrossLines) {
  // Example:
  //
  // Original repl:
  // (ccc\ndddeeeefff\nggg) =>（jjj\nkkk）
  // (zz\nyy) =>（xx)
  //
  // Splitted repls:
  // (ccc\n) =>（jjj\nkkkhhhhiii\n）
  // (dddeeeefff\n) => ""
  // (ggghhhhiii\n) => ""
  // (zz\n) => (xxyyyyyyyy\n)
  // (yyyyyyyyyy) => ""

  getLineStringHook = this->getLineStringUnittest;
  getLineNumberHook = this->getLineNumberUnittest;
  getLineBeginOffsetHook = this->getLineBeginOffsetUnittest;
  std::vector<Replacement> Repls = {Replacement("file1.cpp", 7, 18, "jjj\nkkk"),
                                    Replacement("file1.cpp", 41, 5, "xx")};
  std::vector<Replacement> Expected = {
      Replacement("file1.cpp", 7, 4, "jjj\nkkkhhhhiii\n"),
      Replacement("file1.cpp", 11, 11, ""),
      Replacement("file1.cpp", 22, 11, ""),
      Replacement("file1.cpp", 41, 3, "xxyyyyyyyy\n"),
      Replacement("file1.cpp", 44, 11, "")};
  auto Result = splitReplInOrderToNotCrossLines(Repls);
  std::sort(Result.begin(), Result.end());
  EXPECT_EQ(Expected, Result);
}

class ReMigrationTest3 : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
  // clang-format off
/*
aaa bb ccc
aaa bb ccc
aaa bb ccc
aaa bb ccc
aaa bb ccc
aaa bb ccc
aaa bb ccc
aaa bb ccc
aaa bb ccc
aaa bb ccc
*/
  // clang-format on
  static StringRef getLineStringUnittest(clang::tooling::UnifiedPath FilePath,
                                         unsigned LineNumber) {
    static std::string S = "aaa bb ccc\n";
    return StringRef(S);
  }
  static unsigned getLineNumberUnittest(clang::tooling::UnifiedPath FilePath,
                                        unsigned Offset) {
    static std::vector<unsigned> LineOffsets = {0,  11, 22, 33, 44,
                                                55, 66, 77, 88, 99};
    auto Iter =
        std::upper_bound(LineOffsets.begin(), LineOffsets.end(), Offset);
    if (Iter == LineOffsets.end())
      return LineOffsets.size();
    return std::distance(LineOffsets.begin(), Iter);
  }
  static unsigned
  getLineBeginOffsetUnittest(clang::tooling::UnifiedPath FilePath,
                             unsigned LineNumber) {
    static std::unordered_map<unsigned, unsigned> LineOffsets = {
        {1, 0},  {2, 11}, {3, 22}, {4, 33}, {5, 44},
        {6, 55}, {7, 66}, {8, 77}, {9, 88}, {10, 99}};
    return LineOffsets[LineNumber];
  }
};

TEST_F(ReMigrationTest3, convertReplcementsLineString) {
  // clang-format off
  // After appling repls:
/*
aaa zzz ccc
ppp aaa bb ccc
aaa bb ccc
aaa yyy ccc
aaa bb ccc
aaa bb ccq
qqqaa bb ccc
aaa bb ddd
<empty without \n>
eee bb ccc
*/
  // clang-format on
  getLineStringHook = this->getLineStringUnittest;
  getLineNumberHook = this->getLineNumberUnittest;
  getLineBeginOffsetHook = this->getLineBeginOffsetUnittest;
  std::vector<Replacement> Repls = {
      Replacement("file1.cpp", 4, 2, "zzz"),
      Replacement("file1.cpp", 64, 3, "q\nqqq"),
      Replacement("file1.cpp", 33, 11, "aaa yyy ccc\n"),
      Replacement("file1.cpp", 11, 0, "ppp "),
      Replacement("file1.cpp", 84, 18, "ddd\neee")};
  std::map<unsigned, std::string> Expected = {{1, "aaa zzz ccc\n"},
                                              {2, "ppp aaa bb ccc\n"},
                                              {4, "aaa yyy ccc\n"},
                                              {6, "aaa bb ccq\nqqqaa bb ccc\n"},
                                              {7, ""},
                                              {8, "aaa bb ddd\neee bb ccc\n"},
                                              {9, ""},
                                              {10, ""}};
  auto Result = convertReplcementsLineString(Repls);
  EXPECT_EQ(Expected, Result);
}

TEST_F(ReMigrationTest3, mergeMapsByLine) {
  getLineBeginOffsetHook = this->getLineBeginOffsetUnittest;
  getLineStringHook = this->getLineStringUnittest;
  std::map<unsigned, std::string> MapA = {{1, "zzzzz\n"}, {3, "xxxx1\n"},
                                          {4, "wwww1\n"}, {5, "vvvvv\n"},
                                          {7, "ppppp\n"}, {9, "iiijjjkkk\n"}};
  std::map<unsigned, std::string> MapB = {
      {2, "yyyyy\n"}, {3, "xxxx2\n"}, {4, "wwww2\n"}, {7, ""}, {8, "qqqqq"}};
  UnifiedPath FilePath("test.cu");
  auto Result = mergeMapsByLine(MapA, MapB, FilePath);
  std::sort(Result.begin(), Result.end());

  std::vector<Replacement> Expected = {
      Replacement("test.cu", 0, 11, "zzzzz\n"),
      Replacement("test.cu", 11, 11, "yyyyy\n"),
      Replacement("test.cu", 22, 22,
                  "<<<<<<<\nxxxx1\nwwww1\n=======\nxxxx2\nwwww2\n>>>>>>>\n"),
      Replacement("test.cu", 44, 11, "vvvvv\n"),
      Replacement("test.cu", 66, 11, "<<<<<<<\nppppp\n=======\n>>>>>>>\n"),
      Replacement("test.cu", 77, 11, "qqqqq"),
      Replacement("test.cu", 88, 11, "iiijjjkkk\n"),
  };

  ASSERT_EQ(Expected.size(), Result.size());
  size_t Num = Expected.size();
  auto ExpectedIt = Expected.begin();
  auto ResultIt = Result.begin();
  for (size_t i = 0; i < Num; ++i) {
    EXPECT_EQ(ExpectedIt->getFilePath(), ResultIt->getFilePath());
    EXPECT_EQ(ExpectedIt->getOffset(), ResultIt->getOffset());
    EXPECT_EQ(ExpectedIt->getLength(), ResultIt->getLength());
    EXPECT_EQ(ExpectedIt->getReplacementText(), ResultIt->getReplacementText());
    ExpectedIt++;
    ResultIt++;
  }

  EXPECT_EQ(Expected, Result);
}

TEST_F(ReMigrationTest1, mergeC1AndC2) {
  // clang-format off
  // original file:
/*
0123456789
0123456789
0123456789
0123456789
0123456789
*/
  // migrated file:
/*
0123456789aaa
bbb12345678ccc789
0123456789
*/
  // updated file:
/*
0123zzz456789aaa
bbyyy2345678xxxc78www0123456789
*/
  // clang-format on
  std::vector<Replacement> Repl_C1 = {
      Replacement("file1.cu", 10, 0, "aaa"),
      Replacement("file1.cu", 11, 1, "bbb"),
      Replacement("file1.cu", 20, 20, "ccc"),
  };
  GitDiffChanges Repl_C2;
  Repl_C2.ModifyFileHunks = {
      Replacement("file1.dp.cpp", 30, 2, "www"),
      Replacement("file1.dp.cpp", 25, 2, "xxx"),
      Replacement("file1.dp.cpp", 16, 2, "yyy"),
      Replacement("file1.dp.cpp", 4, 0, "zzz"),
  };
  const std::map<UnifiedPath, UnifiedPath> FileNameMap = {
      {"file1.dp.cpp", "file1.cu"}};

  std::vector<Replacement> Result = mergeC1AndC2(Repl_C1, Repl_C2, FileNameMap);
  std::vector<Replacement> Expected = {Replacement("file1.cu", 4, 0, "zzz"),
                                       Replacement("file1.cu", 10, 0, "aaa"),
                                       Replacement("file1.cu", 11, 2, "bbyyy"),
                                       Replacement("file1.cu", 20, 20, "xxxc"),
                                       Replacement("file1.cu", 42, 2, "www")};
  std::sort(Result.begin(), Result.end());
  std::sort(Expected.begin(), Expected.end());

  ASSERT_EQ(Expected.size(), Result.size());
  size_t Num = Expected.size();
  auto ExpectedIt = Expected.begin();
  auto ResultIt = Result.begin();
  for (size_t i = 0; i < Num; ++i) {
    EXPECT_EQ(ExpectedIt->getFilePath(), ResultIt->getFilePath());
    EXPECT_EQ(ExpectedIt->getOffset(), ResultIt->getOffset());
    EXPECT_EQ(ExpectedIt->getLength(), ResultIt->getLength());
    EXPECT_EQ(ExpectedIt->getReplacementText(), ResultIt->getReplacementText());
    ExpectedIt++;
    ResultIt++;
  }
}

class ReMigrationTest4 : public ::testing::Test {
protected:
  inline static std::vector<std::string> CUDACodeV2Vec = {};
  inline static std::vector<unsigned> LineOffsets = {};
  void SetUp() override {
    // clang-format off
    const std::string CUDACodeV2 = R"(#include <stdio.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA error in %s at line %d: %s\n", __FILE__, __LINE__,          \
             cudaGetErrorString(err));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

void foo() {
  float *g;
  CUDA_CHECK(cudaMalloc(&g, 100 * sizeof(float)));
  float *h;
  CUDA_CHECK(cudaMalloc(&h, 100 * sizeof(float)));
  cudaDeviceSynchronize();
  cudaFree(g);
  cudaFree(h);
}
)";
    // clang-format on
    std::istringstream ISS(CUDACodeV2);
    std::string Line;
    // TODO: This code only considers the last line is empty (file ending by a \n).
    //       What if the last line is not empty?
    while (true) {
      bool LastLine = false;
      if (!std::getline(ISS, Line))
        LastLine = true;
      else
        Line += '\n';
      LineOffsets.push_back(LineOffsets.empty()
                                ? 0
                                : LineOffsets.back() +
                                      CUDACodeV2Vec.back().size());
      CUDACodeV2Vec.push_back(Line);
      if (LastLine)
        break;
    }
    LineOffsets.insert(LineOffsets.begin(), 0);
  }
  void TearDown() override { CUDACodeV2Vec.clear(); }
  static StringRef getLineStringUnittest(clang::tooling::UnifiedPath FilePath,
                                         unsigned LineNumber) {
    return StringRef(CUDACodeV2Vec[LineNumber - 1]);
  }
  static unsigned getLineNumberUnittest(clang::tooling::UnifiedPath FilePath,
                                        unsigned Offset) {
    auto Iter =
        std::upper_bound(LineOffsets.begin() + 1, LineOffsets.end(), Offset);
    if (Iter == LineOffsets.end())
      return LineOffsets.size();
    return std::distance(LineOffsets.begin() + 1, Iter);
  }
  static unsigned
  getLineBeginOffsetUnittest(clang::tooling::UnifiedPath FilePath,
                             unsigned LineNumber) {
    return LineOffsets[LineNumber];
  }
};

TEST_F(ReMigrationTest4, reMigrationMerge) {
  getLineStringHook = this->getLineStringUnittest;
  getLineNumberHook = this->getLineNumberUnittest;
  getLineBeginOffsetHook = this->getLineBeginOffsetUnittest;

  const std::map<UnifiedPath, UnifiedPath> FileNameMap = {
      {"test.dp.cpp", "test.cu"}};

  std::vector<Replacement> Repl_C1 = {
      Replacement("test.cu", 0, 0,
                  "#include <sycl/sycl.hpp>\n#include <dpct/dpct.hpp>\n"),
      Replacement(
          "test.cu", 20, 0,
          "/*\nDPCT1009:0: SYCL reports errors using exceptions and does not "
          "use error codes. Please replace the \"get_error_string_dummy(...)\" "
          "with a real error-handling function.\n*/\n"),
      Replacement("test.cu", 186, 11, "dpct::err0"),
      Replacement("test.cu", 267, 325, ""),
      Replacement("test.cu", 694, 0, " try "),
      Replacement(
          "test.cu", 695, 0,
          "\n  dpct::device_ext &dev_ct1 = dpct::get_current_device();\n  "
          "sycl::queue &q_ct1 = dev_ct1.in_order_queue();"),
      Replacement(
          "test.cu", 721, 35,
          "DPCT_CHECK_ERROR(f = sycl::malloc_device<float>(100, q_ct1))"),
      Replacement(
          "test.cu", 784, 35,
          "DPCT_CHECK_ERROR(g = sycl::malloc_device<float>(100, q_ct1))"),
      Replacement("test.cu", 824, 63,
                  "q_ct1.memcpy(f, g, 100 * sizeof(float))"),
      Replacement("test.cu", 891, 23, "dev_ct1.queues_wait_and_throw()"),
      Replacement("test.cu", 918, 11, "dpct::dpct_free(f, q_ct1)"),
      Replacement("test.cu", 933, 11, "dpct::dpct_free(g, q_ct1)"),
      Replacement("test.cu", 947, 0,
                  "\ncatch (sycl::exception const &exc) {\n  std::cerr << "
                  "exc.what() << \"Exception caught at file:\" << __FILE__ << "
                  "\", line:\" << __LINE__ << std::endl;\n  std::exit(1);\n}")};

  GitDiffChanges Repl_C2;
  Repl_C2.ModifyFileHunks = {
      Replacement("test.dp.cpp", 70, 526, "void foo() {\n"),
      Replacement("test.dp.cpp", 715, 76,
                  "  f = sycl::malloc_device<float>(100, q_ct1);\n"),
      Replacement("test.dp.cpp", 803, 76,
                  "  g = sycl::malloc_device<float>(100, q_ct1);\n"),
      Replacement("test.dp.cpp", 1017, 163, "")};

  GitDiffChanges Repl_A;
  Repl_A.ModifyFileHunks = {
      Replacement("test.cu", 696, 63, ""),
      Replacement(
          "test.cu", 822, 67,
          "  float *h;\n  CUDA_CHECK(cudaMalloc(&h, 100 * sizeof(float)));\n"),
      Replacement("test.cu", 916, 15, ""),
      Replacement("test.cu", 946, 0, "  cudaFree(h);\n")};

  std::vector<Replacement> Repl_B = {
      Replacement("test.cu", 0, 0,
                  "#include <sycl/sycl.hpp>\n#include <dpct/dpct.hpp>\n"),
      Replacement(
          "test.cu", 20, 0,
          "/*\nDPCT1009:0: SYCL reports errors using exceptions and does not "
          "use error codes. Please replace the \"get_error_string_dummy(...)\" "
          "with a real error-handling function.\n*/\n"),
      Replacement("test.cu", 186, 11, "dpct::err0"),
      Replacement("test.cu", 267, 325, ""),
      Replacement("test.cu", 694, 0, " try "),
      Replacement(
          "test.cu", 695, 0,
          "\n  dpct::device_ext &dev_ct1 = dpct::get_current_device();\n  "
          "sycl::queue &q_ct1 = dev_ct1.in_order_queue();"),
      Replacement(
          "test.cu", 721, 35,
          "DPCT_CHECK_ERROR(g = sycl::malloc_device<float>(100, q_ct1))"),
      Replacement(
          "test.cu", 784, 35,
          "DPCT_CHECK_ERROR(h = sycl::malloc_device<float>(100, q_ct1))"),
      Replacement("test.cu", 824, 23, "dev_ct1.queues_wait_and_throw()"),
      Replacement("test.cu", 851, 11, "dpct::dpct_free(g, q_ct1)"),
      Replacement("test.cu", 866, 11, "dpct::dpct_free(h, q_ct1)"),
      Replacement("test.cu", 880, 0,
                  "\ncatch (sycl::exception const &exc) {\n  std::cerr << "
                  "exc.what() << \"Exception caught at file:\" << __FILE__ << "
                  "\", line:\" << __LINE__ << std::endl;\n  std::exit(1);\n}")};

  std::map<std::string, std::vector<Replacement>> ResultMap =
      reMigrationMerge(Repl_A, Repl_B, Repl_C1, Repl_C2, FileNameMap);

  std::vector<Replacement> Result = ResultMap.begin()->second;
  std::vector<Replacement> Expected = {
      Replacement("test.cu", 0, 19, R"(#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
)"),
      Replacement("test.cu", 20, 81, R"xxx(<<<<<<<
/*
DPCT1009:0: SYCL reports errors using exceptions and does not use error codes. Please replace the "get_error_string_dummy(...)" with a real error-handling function.
*/
#define CUDA_CHECK(call)                                                       \
=======
void foo() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
>>>>>>>
)xxx"),
      Replacement("test.cu", 101, 81, R"()"),
      Replacement("test.cu", 182, 486, R"(<<<<<<<
    dpct::err0 err = call;                                                    \
                                                                              \
=======
>>>>>>>
)"),
      Replacement("test.cu", 668, 14, R"()"),
      Replacement("test.cu", 682, 1, R"()"),
      Replacement("test.cu", 683, 13, R"(<<<<<<<
void foo()  try {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
=======
>>>>>>>
)"),
      Replacement("test.cu", 708, 51, R"(<<<<<<<
  CUDA_CHECK(DPCT_CHECK_ERROR(g = sycl::malloc_device<float>(100, q_ct1)));
=======
  g = sycl::malloc_device<float>(100, q_ct1);
>>>>>>>
)"),
      Replacement(
          "test.cu", 771, 51,
          R"(  CUDA_CHECK(DPCT_CHECK_ERROR(h = sycl::malloc_device<float>(100, q_ct1)));
)"),
      Replacement("test.cu", 822, 27, R"(  dev_ct1.queues_wait_and_throw();
)"),
      Replacement("test.cu", 849, 15, R"(  dpct::dpct_free(g, q_ct1);
)"),
      Replacement("test.cu", 864, 15, R"(  dpct::dpct_free(h, q_ct1);
)"),
      Replacement("test.cu", 879, 2, R"(}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
)")};
  std::sort(Result.begin(), Result.end());
  std::sort(Expected.begin(), Expected.end());

  ASSERT_EQ(Expected.size(), Result.size());
  size_t Num = Expected.size();
  auto ExpectedIt = Expected.begin();
  auto ResultIt = Result.begin();
  for (size_t i = 0; i < Num; ++i) {
    EXPECT_EQ(ExpectedIt->getFilePath(), ResultIt->getFilePath());
    EXPECT_EQ(ExpectedIt->getOffset(), ResultIt->getOffset());
    EXPECT_EQ(ExpectedIt->getLength(), ResultIt->getLength());
    EXPECT_EQ(ExpectedIt->getReplacementText(), ResultIt->getReplacementText());
    ExpectedIt++;
    ResultIt++;
  }
}
