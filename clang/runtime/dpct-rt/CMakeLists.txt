set(dpct_rt_files
  include/dpct.hpp
  include/atomic.hpp
  include/device.hpp
  include/memory.hpp
  include/kernel.hpp
  include/image.hpp
  include/util.hpp
  include/blas_utils.hpp
  include/dpstd_utils.hpp
)

set(output_dir ${LLVM_LIBRARY_OUTPUT_INTDIR}/clang/${CLANG_VERSION}/include/dpct)

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

add_custom_target(dpct-headers ALL DEPENDS ${out_files})

install(
  FILES ${dpct_rt_files}
  COMPONENT dpct-headers
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION include/dpct)

if (NOT CMAKE_CONFIGURATION_TYPES) # don't add this for IDE's.
  add_llvm_install_targets(install-dpct-headers
                           DEPENDS dpct-headers
                           COMPONENT dpct-headers)
endif()
