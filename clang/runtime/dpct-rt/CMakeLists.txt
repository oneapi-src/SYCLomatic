set(dpct_rt_files
  include/dpct/dpct.hpp
  include/dpct/atomic.hpp
  include/dpct/bindless_images.hpp
  include/dpct/device.hpp
  include/dpct/math.hpp
  include/dpct/memory.hpp
  include/dpct/kernel.hpp
  include/dpct/image.hpp
  include/dpct/util.hpp
  include/dpct/blas_utils.hpp
  include/dpct/dnnl_utils.hpp
  include/dpct/dpl_utils.hpp
  include/dpct/rng_utils.hpp
  include/dpct/lib_common_utils.hpp
  include/dpct/ccl_utils.hpp
  include/dpct/sparse_utils.hpp
  include/dpct/fft_utils.hpp
  include/dpct/lapack_utils.hpp
  include/dpct/group_utils.hpp
  include/dpct/blas_gemm_utils.hpp
  include/dpct/graph.hpp
  include/dpct/compat_service.hpp
)

set(dpct_rt_dpstd_files
  include/dpct/dpl_extras/functional.h
  include/dpct/dpl_extras/memory.h
  include/dpct/dpl_extras/iterator_adaptor.h
  include/dpct/dpl_extras/iterators.h
  include/dpct/dpl_extras/algorithm.h
  include/dpct/dpl_extras/numeric.h
  include/dpct/dpl_extras/vector.h
  include/dpct/dpl_extras/dpcpp_extensions.h
)

set(dpct_rt_codepin_files
  include/dpct/codepin/codepin.hpp
  include/dpct/codepin/serialization/basic.hpp
)

set(output_dir ${LLVM_LIBRARY_OUTPUT_INTDIR}/clang/${CLANG_VERSION}/include/dpct)
set(output_dpstd_dir ${LLVM_LIBRARY_OUTPUT_INTDIR}/clang/${CLANG_VERSION}/include/dpct/dpl_extras)
set(output_codepin_dir ${LLVM_LIBRARY_OUTPUT_INTDIR}/clang/${CLANG_VERSION}/include/dpct/codepin)

set(out_files)
foreach( f ${dpct_rt_files})
  set( src ${CMAKE_CURRENT_SOURCE_DIR}/${f} )
  set( dst ${output_dir}/${f} )
  add_custom_command(OUTPUT ${dst}
    DEPENDS ${src}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
    COMMENT "Copying clang's ${f}...")
  list(APPEND out_files ${dst})
endforeach( f )

foreach( f ${dpct_rt_dpstd_files})
  set( src ${CMAKE_CURRENT_SOURCE_DIR}/${f} )
  set( dst ${output_dpstd_dir}/${f} )
  add_custom_command(OUTPUT ${dst}
    DEPENDS ${src}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
    COMMENT "Copying clang's ${f}...")
  list(APPEND out_files ${dst})
endforeach( f )

foreach( f ${dpct_rt_codepin_files})
  set( src ${CMAKE_CURRENT_SOURCE_DIR}/${f} )
  set( dst ${output_codepin_dir}/${f} )
  add_custom_command(OUTPUT ${dst}
    DEPENDS ${src}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
    COMMENT "Copying clang's ${f}...")
  list(APPEND out_files ${dst})
endforeach( f )
add_custom_target(dpct-headers ALL DEPENDS ${out_files})

install(
  FILES ${dpct_rt_files}
  COMPONENT dpct-headers
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION include/dpct)

install(
  FILES ${dpct_rt_dpstd_files}
  COMPONENT dpct-headers
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION include/dpct/dpl_extras)

install(
  DIRECTORY include/dpct/codepin DESTINATION include/dpct
  COMPONENT dpct-headers
  USE_SOURCE_PERMISSIONS
)
if (NOT CMAKE_CONFIGURATION_TYPES) # don't add this for IDE's.
  add_llvm_install_targets(install-dpct-headers
                           DEPENDS dpct-headers
                           COMPONENT dpct-headers)
endif()
