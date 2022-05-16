if (DOXYGEN_FOUND)
if (LLVM_ENABLE_DOXYGEN)
  set(abs_srcdir ${CMAKE_CURRENT_SOURCE_DIR})
  set(abs_builddir ${CMAKE_CURRENT_BINARY_DIR})

  if (HAVE_DOT)
    set(DOT ${LLVM_PATH_DOT})
  endif()

  if (LLVM_DOXYGEN_EXTERNAL_SEARCH)
    set(enable_searchengine "YES")
    set(searchengine_url "${LLVM_DOXYGEN_SEARCHENGINE_URL}")
    set(enable_server_based_search "YES")
    set(enable_external_search "YES")
    set(extra_search_mappings "${LLVM_DOXYGEN_SEARCH_MAPPINGS}")
  else()
    set(enable_searchengine "NO")
    set(searchengine_url "")
    set(enable_server_based_search "NO")
    set(enable_external_search "NO")
    set(extra_search_mappings "")
  endif()

  # If asked, configure doxygen for the creation of a Qt Compressed Help file.
  if (LLVM_ENABLE_DOXYGEN_QT_HELP)
    set(DPCT_RUNTIME_DOXYGEN_QCH_FILENAME "org.llvm.clang.dpctruntime.qch" CACHE STRING
      "Filename of the Qt Compressed help file")
    set(DPCT_RUNTIME_DOXYGEN_QHP_NAMESPACE "org.llvm.clang.dpctruntime" CACHE STRING
      "Namespace under which the intermediate Qt Help Project file lives")
    set(DPCT_RUNTIME_DOXYGEN_QHP_CUST_FILTER_NAME "Clang ${DPCT_RUNTIME_VERSION}" CACHE STRING
      "See http://qt-project.org/doc/qt-4.8/qthelpproject.html#custom-filters")
    set(DPCT_RUNTIME_DOXYGEN_QHP_CUST_FILTER_ATTRS "Clang,${DPCT_RUNTIME_VERSION}" CACHE STRING
      "See http://qt-project.org/doc/qt-4.8/qthelpproject.html#filter-attributes")
    set(dpct_runtime_doxygen_generate_qhp "YES")
    set(dpct_runtime_doxygen_qch_filename "${DPCT_RUNTIME_DOXYGEN_QCH_FILENAME}")
    set(dpct_runtime_doxygen_qhp_namespace "${DPCT_RUNTIME_DOXYGEN_QHP_NAMESPACE}")
    set(dpct_runtime_doxygen_qhelpgenerator_path "${LLVM_DOXYGEN_QHELPGENERATOR_PATH}")
    set(dpct_runtime_doxygen_qhp_cust_filter_name "${DPCT_RUNTIME_DOXYGEN_QHP_CUST_FILTER_NAME}")
    set(dpct_runtime_doxygen_qhp_cust_filter_attrs "${DPCT_RUNTIME_DOXYGEN_QHP_CUST_FILTER_ATTRS}")
  else()
    set(dpct_runtime_doxygen_generate_qhp "NO")
    set(dpct_runtime_doxygen_qch_filename "")
    set(dpct_runtime_doxygen_qhp_namespace "")
    set(dpct_runtime_doxygen_qhelpgenerator_path "")
    set(dpct_runtime_doxygen_qhp_cust_filter_name "")
    set(dpct_runtime_doxygen_qhp_cust_filter_attrs "")
  endif()

  option(LLVM_DOXYGEN_SVG
    "Use svg instead of png files for doxygen graphs." OFF)
  if (LLVM_DOXYGEN_SVG)
    set(DOT_IMAGE_FORMAT "svg")
  else()
    set(DOT_IMAGE_FORMAT "png")
  endif()

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doxygen.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/doxygen.cfg @ONLY)

  set(abs_top_srcdir)
  set(abs_top_builddir)
  set(DOT)
  set(enable_searchengine)
  set(searchengine_url)
  set(enable_server_based_search)
  set(enable_external_search)
  set(extra_search_mappings)
  set(dpct_runtime_doxygen_generate_qhp)
  set(dpct_runtime_doxygen_qch_filename)
  set(dpct_runtime_doxygen_qhp_namespace)
  set(dpct_runtime_doxygen_qhelpgenerator_path)
  set(dpct_runtime_doxygen_qhp_cust_filter_name)
  set(dpct_runtime_doxygen_qhp_cust_filter_attrs)
  set(DOT_IMAGE_FORMAT)

  add_custom_target(doxygen-dpct-runtime
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/doxygen.cfg
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating dpct runtime doxygen documentation." VERBATIM)

  if (LLVM_BUILD_DOCS)
    add_dependencies(doxygen doxygen-dpct-runtime)
  endif()

  if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doxygen/html
      DESTINATION docs/html)
  endif()
endif()
endif()

if (LLVM_ENABLE_SPHINX)
  include(AddSphinxTarget)
  if (SPHINX_FOUND)
    if (${SPHINX_OUTPUT_HTML})
      add_sphinx_target(html dpct-runtime)
    endif()
    if (${SPHINX_OUTPUT_MAN})
      add_sphinx_target(man dpct-runtime)
    endif()
  endif()
endif()

