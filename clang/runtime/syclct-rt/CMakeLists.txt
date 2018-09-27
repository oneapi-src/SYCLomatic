set(syclct_rt_files
  include/syclct.hpp
  include/syclct_device.hpp
  include/syclct_memory.hpp
)

set(output_dir ${LLVM_LIBRARY_OUTPUT_INTDIR}/clang/${CLANG_VERSION}/include/syclct)

set(out_files)
foreach( f ${syclct_rt_files})
  set( src ${CMAKE_CURRENT_SOURCE_DIR}/${f} )
  set( dst ${output_dir}/${f} )
  add_custom_command(OUTPUT ${dst}
    DEPENDS ${src}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
    COMMENT "Copying clang's ${f}...")
  list(APPEND out_files ${dst})
endforeach( f )

add_custom_target(syclct-headers ALL DEPENDS ${out_files})

install(
  FILES ${syclct_rt_files}
  COMPONENT syclct-headers
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION include/syclct)

if (NOT CMAKE_CONFIGURATION_TYPES) # don't add this for IDE's.
  add_llvm_install_targets(install-syclct-headers
                           DEPENDS syclct-headers
                           COMPONENT syclct-headers)
endif()
