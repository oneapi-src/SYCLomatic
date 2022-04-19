macro(build_lib_c2s)
  include_directories(
    # For the CUDA-Toolchain and its CudaInstallationDetector
    ../../../clang/lib/Driver/
    GaHelper/
    libcurl/include/
    )
  if(UNIX)
    set(LIBCURL ${CMAKE_SOURCE_DIR}/../clang/lib/C2S/libcurl/lib/linux/libcurl.a)
  else()
    set(LIBCURL ${CMAKE_SOURCE_DIR}/../clang/lib/C2S/libcurl/lib/win/libcurl_a.lib)
  endif()

  add_subdirectory(docs)
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
    GaHelper/active_user_detector.cpp
    GaHelper/filelock.cpp
    GaHelper/filesystem_util.cpp
    GaHelper/gahelper_impl.cpp
    GaHelper/http_connector.cpp
    GaHelper/network_util.cpp
    GaHelper/os_specific.cpp
    GaHelper/uuid.cpp
    GAnalytics.cpp
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
    ${LIBCURL}
    )
endmacro()

