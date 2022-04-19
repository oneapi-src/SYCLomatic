macro(build_lib_c2s)
  include_directories(
    # For the CUDA-Toolchain and its CudaInstallationDetector
    ../../../clang/lib/Driver/
    )

  add_subdirectory(ConfusableTable)

  add_clang_library(C2S
    ASTTraversal.cpp
    AnalysisInfo.cpp
    CallExprRewriter.cpp
    Diagnostics.cpp
    Error.cpp
    Statics.cpp
    ExprAnalysis.cpp
    ExtReplacements.cpp
    MapNames.cpp
    SaveNewFiles.cpp
    C2S.cpp
    TextModification.cpp
    Utility.cpp
    ValidateArguments.cpp
    ExternalReplacement.cpp
    SignalProcess.cpp
    VcxprojParser.cpp
    LibraryAPIMigration.cpp
    CustomHelperFiles.cpp
    GenMakefile.cpp
    IncrementalMigrationUtility.cpp
    Rules.cpp
    Homoglyph.cpp
    MisleadingBidirectional.cpp
    BarrierFenceSpaceAnalyzer.cpp

    DEPENDS
    ClangDriverOptions
    c2s_helper_headers_and_inc
    genconfusable

    LINK_LIBS
    clangBasic
    clangLex
    clangAnalysis
    clangAST
    clangASTMatchers
    clangDriver
    clangEdit
    clangFormat
    clangFrontend
    clangParse
    clangRewrite
    clangSema
    clangSerialization
    clangTooling
    clangToolingCore
    )
endmacro()

