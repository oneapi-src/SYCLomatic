//===--------------- CMakeCommands.inc ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/****************************************************************************
*
* ENTRY_TYPE(Command, ProcessedOrNot, MigrationDesc, CUDASpecificOrNot)
*  Command: CMake Command Name
*
*  ProcessedOrNot:  false|true
*    false: this Command has not been processed.
*    true:  this Command has been processed, need check MigrationDesc field to
*           see if it has be migrated Commently.
*
*  MigrationDesc: Comment[: Comment] | Unsupported: Comment
*    Comment:   this Command has been migrated.
*    Unsupported:  this Command can not be migrated currently.
*                  Must add Comment to explain why this API is Unsupported.
*    Comment:      extra comment on the migration.
*
*  CUDASpecificOrNot:  false|true
*    false: this Command is not CUDA specific command.
*    true:  this Command is CUDA specific command.
****************************************************************************/
ENTRY_TYPE(cmake_minimum_required, true, "comment", false)
ENTRY_TYPE(cmake_parse_arguments, true, "comment", false)
ENTRY_TYPE(cmake_path, true, "comment", false)
ENTRY_TYPE(cmake_policy, true, "comment", false)
ENTRY_TYPE(file, true, "comment", false)
ENTRY_TYPE(find_file, true, "comment", false)
ENTRY_TYPE(find_library, true, "comment", false)
ENTRY_TYPE(find_package, true, "comment", false)
ENTRY_TYPE(find_path, true, "comment", false)
ENTRY_TYPE(find_program, true, "comment", false)
ENTRY_TYPE(foreach, true, "comment", false)
ENTRY_TYPE(function, true, "comment", false)
ENTRY_TYPE(get_cmake_property, true, "comment", false)
ENTRY_TYPE(get_directory_property, true, "comment", false)
ENTRY_TYPE(get_filename_component, true, "comment", false)
ENTRY_TYPE(get_property, true, "comment", false)
ENTRY_TYPE(list, true, "comment", false)
ENTRY_TYPE(macro, true, "comment", false)
ENTRY_TYPE(mark_as_advanced, true, "comment", false)
ENTRY_TYPE(message, true, "comment", false)
ENTRY_TYPE(separate_arguments, true, "comment", false)
ENTRY_TYPE(set, true, "comment", false)
ENTRY_TYPE(set_directory_properties, true, "comment", false)
ENTRY_TYPE(set_property, true, "comment", false)
ENTRY_TYPE(string, true, "comment", false)
ENTRY_TYPE(unset, true, "comment", false)
ENTRY_TYPE(add_compile_definitions, true, "comment", false)
ENTRY_TYPE(add_compile_options, true, "comment", false)
ENTRY_TYPE(add_custom_command, true, "comment", false)
ENTRY_TYPE(add_custom_target, true, "comment", false)
ENTRY_TYPE(add_definitions, true, "comment", false)
ENTRY_TYPE(add_dependencies, true, "comment", false)
ENTRY_TYPE(add_executable, true, "comment", false)
ENTRY_TYPE(add_library, true, "comment", false)
ENTRY_TYPE(add_link_options, true, "comment", false)
ENTRY_TYPE(add_subdirectory, true, "comment", false)
ENTRY_TYPE(add_test, true, "comment", false)
ENTRY_TYPE(build_command, true, "comment", false)
ENTRY_TYPE(define_property, true, "comment", false)
ENTRY_TYPE(include_directories, true, "comment", false)
ENTRY_TYPE(install, true, "comment", false)
ENTRY_TYPE(link_directories, true, "comment", false)
ENTRY_TYPE(link_libraries, true, "comment", false)
ENTRY_TYPE(project, true, "comment", false)
ENTRY_TYPE(set_source_files_properties, true, "comment", false)
ENTRY_TYPE(set_target_properties, true, "comment", false)
ENTRY_TYPE(set_tests_properties, true, "comment", false)
ENTRY_TYPE(source_group, true, "comment", false)
ENTRY_TYPE(target_compile_definitions, true, "comment", false)
ENTRY_TYPE(target_compile_features, true, "comment", false)
ENTRY_TYPE(target_compile_options, true, "comment", false)
ENTRY_TYPE(target_include_directories, true, "comment", false)
ENTRY_TYPE(target_link_directories, true, "comment", false)
ENTRY_TYPE(target_link_libraries, true, "comment", false)
ENTRY_TYPE(target_link_options, true, "comment", false)
ENTRY_TYPE(target_sources, true, "comment", false)
ENTRY_TYPE(try_compile, true, "comment", false)
ENTRY_TYPE(try_run, true, "comment", false)
ENTRY_TYPE(build_name, true, "comment", false)
ENTRY_TYPE(exec_program, true, "comment", false)
ENTRY_TYPE(export_library_dependencies, true, "comment", false)
ENTRY_TYPE(make_directory, true, "comment", false)
ENTRY_TYPE(remove, true, "comment", false)
ENTRY_TYPE(subdir_depends, true, "comment", false)
ENTRY_TYPE(subdirs, true, "comment", false)
ENTRY_TYPE(use_mangled_mesa, true, "comment", false)
ENTRY_TYPE(utility_source, true, "comment", false)
ENTRY_TYPE(variable_requires, true, "comment", false)
ENTRY_TYPE(write_file, true, "comment", false)
ENTRY_TYPE(cuda_add_cufft_to_target, true, "comment", true)
ENTRY_TYPE(cuda_add_cublas_to_target, true, "comment", true)
ENTRY_TYPE(cuda_add_executable, true, "comment", true)
ENTRY_TYPE(cuda_add_library, true, "comment", true)
ENTRY_TYPE(cuda_build_clean_target, false, "comment", true)
ENTRY_TYPE(cuda_compile, true, "comment", true)
ENTRY_TYPE(cuda_compile_ptx, true, "comment", true)
ENTRY_TYPE(cuda_compile_fatbin, true, "comment", true)
ENTRY_TYPE(cuda_compile_cubin, true, "comment", true)
ENTRY_TYPE(cuda_compute_separable_compilation_object_file_name, false, "comment", true)
ENTRY_TYPE(cuda_include_directories, true, "comment", true)
ENTRY_TYPE(cuda_link_separable_compilation_objects, false, "comment", true)
ENTRY_TYPE(cuda_select_nvcc_arch_flags, false, "comment", true)
ENTRY_TYPE(cuda_unset_include_and_libraries, false, "comment", true)
ENTRY_TYPE(cuda_include_nvcc_dependencies, false, "comment", true)
ENTRY_TYPE(cuda_language_flag, false, "comment", true)
ENTRY_TYPE(cuda_wrap_srcs, false, "comment", true)
ENTRY_TYPE(configure_file, false, "comment", false)
