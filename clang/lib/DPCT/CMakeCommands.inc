//===--------------- CMakeCommands.inc ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/****************************************************************************
*
* ENTRY_TYPE(Command, ProcessedOrNot, MigrationDesc)
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
****************************************************************************/
ENTRY_TYPE(cmake_minimum_required, true, "comment")
ENTRY_TYPE(cmake_parse_arguments, true, "comment")
ENTRY_TYPE(cmake_path, true, "comment")
ENTRY_TYPE(cmake_policy, true, "comment")
ENTRY_TYPE(file, true, "comment")
ENTRY_TYPE(find_file, true, "comment")
ENTRY_TYPE(find_library, true, "comment")
ENTRY_TYPE(find_package, true, "comment")
ENTRY_TYPE(find_path, true, "comment")
ENTRY_TYPE(find_program, true, "comment")
ENTRY_TYPE(foreach, true, "comment")
ENTRY_TYPE(function, true, "comment")
ENTRY_TYPE(get_cmake_property, true, "comment")
ENTRY_TYPE(get_directory_property, true, "comment")
ENTRY_TYPE(get_filename_component, true, "comment")
ENTRY_TYPE(get_property, true, "comment")
ENTRY_TYPE(list, true, "comment")
ENTRY_TYPE(macro, true, "comment")
ENTRY_TYPE(mark_as_advanced, true, "comment")
ENTRY_TYPE(message, true, "comment")
ENTRY_TYPE(separate_arguments, true, "comment")
ENTRY_TYPE(set, true, "comment")
ENTRY_TYPE(set_directory_properties, true, "comment")
ENTRY_TYPE(set_property, true, "comment")
ENTRY_TYPE(string, true, "comment")
ENTRY_TYPE(unset, true, "comment")
ENTRY_TYPE(add_compile_definitions, true, "comment")
ENTRY_TYPE(add_compile_options, true, "comment")
ENTRY_TYPE(add_custom_command, true, "comment")
ENTRY_TYPE(add_custom_target, true, "comment")
ENTRY_TYPE(add_definitions, true, "comment")
ENTRY_TYPE(add_dependencies, true, "comment")
ENTRY_TYPE(add_executable, true, "comment")
ENTRY_TYPE(add_library, true, "comment")
ENTRY_TYPE(add_link_options, true, "comment")
ENTRY_TYPE(add_subdirectory, true, "comment")
ENTRY_TYPE(add_test, true, "comment")
ENTRY_TYPE(build_command, true, "comment")
ENTRY_TYPE(define_property, true, "comment")
ENTRY_TYPE(include_directories, true, "comment")
ENTRY_TYPE(install, true, "comment")
ENTRY_TYPE(link_directories, true, "comment")
ENTRY_TYPE(link_libraries, true, "comment")
ENTRY_TYPE(project, true, "comment")
ENTRY_TYPE(set_source_files_properties, true, "comment")
ENTRY_TYPE(set_target_properties, true, "comment")
ENTRY_TYPE(set_tests_properties, true, "comment")
ENTRY_TYPE(source_group, true, "comment")
ENTRY_TYPE(target_compile_definitions, true, "comment")
ENTRY_TYPE(target_compile_features, true, "comment")
ENTRY_TYPE(target_compile_options, true, "comment")
ENTRY_TYPE(target_include_directories, true, "comment")
ENTRY_TYPE(target_link_directories, true, "comment")
ENTRY_TYPE(target_link_libraries, true, "comment")
ENTRY_TYPE(target_link_options, true, "comment")
ENTRY_TYPE(target_sources, true, "comment")
ENTRY_TYPE(try_compile, true, "comment")
ENTRY_TYPE(try_run, true, "comment")
ENTRY_TYPE(build_name, true, "comment")
ENTRY_TYPE(exec_program, true, "comment")
ENTRY_TYPE(export_library_dependencies, true, "comment")
ENTRY_TYPE(make_directory, true, "comment")
ENTRY_TYPE(remove, true, "comment")
ENTRY_TYPE(subdir_depends, true, "comment")
ENTRY_TYPE(subdirs, true, "comment")
ENTRY_TYPE(use_mangled_mesa, true, "comment")
ENTRY_TYPE(utility_source, true, "comment")
ENTRY_TYPE(variable_requires, true, "comment")
ENTRY_TYPE(write_file, true, "comment")
ENTRY_TYPE(cuda_add_cufft_to_target, true, "comment")
ENTRY_TYPE(cuda_add_cublas_to_target, true, "comment")
ENTRY_TYPE(cuda_add_executable, true, "comment")
ENTRY_TYPE(cuda_add_library, true, "comment")
ENTRY_TYPE(cuda_build_clean_target, false, "comment")
ENTRY_TYPE(cuda_compile, true, "comment")
ENTRY_TYPE(cuda_compile_ptx, true, "comment")
ENTRY_TYPE(cuda_compile_fatbin, true, "comment")
ENTRY_TYPE(cuda_compile_cubin, true, "comment")
ENTRY_TYPE(cuda_compute_separable_compilation_object_file_name, false, "comment")
ENTRY_TYPE(cuda_include_directories, true, "comment")
ENTRY_TYPE(cuda_link_separable_compilation_objects, false, "comment")
ENTRY_TYPE(cuda_select_nvcc_arch_flags, false, "comment")
ENTRY_TYPE(cuda_unset_include_and_libraries, false, "comment")
ENTRY_TYPE(cuda_include_nvcc_dependencies, false, "comment")
ENTRY_TYPE(cuda_language_flag, false, "comment")
ENTRY_TYPE(cuda_wrap_srcs, false, "comment")
