# We need to execute this script at installation time because the
# DESTDIR environment variable may be unset at configuration time.
# See PR8397.

# SYCLomatic_CUSTOMIZATION begin
# Set to an arbitrary directory to silence GNUInstallDirs warnings
# regarding being unable to determine libdir.
set(CMAKE_INSTALL_LIBDIR "lib")
include(GNUInstallDirs)
# SYCLomatic_CUSTOMIZATION end

function(install_symlink name target outdir)
  set(DESTDIR $ENV{DESTDIR})
  if(NOT IS_ABSOLUTE "${outdir}")
    set(outdir "${CMAKE_INSTALL_PREFIX}/${outdir}")
  endif()
  set(outdir "${DESTDIR}${outdir}")

  message(STATUS "Creating ${name}")

# SYCLomatic_CUSTOMIZATION begin
  if(CMAKE_HOST_UNIX)
    set(LLVM_LINK_OR_COPY create_symlink)
  else()
    set(LLVM_LINK_OR_COPY copy)
  endif()
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E ${LLVM_LINK_OR_COPY} "${target}" "${name}"
    WORKING_DIRECTORY "${outdir}" ERROR_VARIABLE has_err)
# SYCLomatic_CUSTOMIZATION else
  # execute_process(
  #   COMMAND "${CMAKE_COMMAND}" -E create_symlink "${target}" "${name}"
  #   WORKING_DIRECTORY "${outdir}" ERROR_VARIABLE has_err)
  # if(CMAKE_HOST_WIN32 AND has_err)
  #   execute_process(
  #     COMMAND "${CMAKE_COMMAND}" -E copy "${target}" "${name}"
  #     WORKING_DIRECTORY "${outdir}")
  # endif()
# SYCLomatic_CUSTOMIZATION end

endfunction()
