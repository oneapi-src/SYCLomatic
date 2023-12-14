find_package(IntelSYCL REQUIRED)

macro(FIND_DPCT_PATH dpct_path)
    # Use find_program to locate the executable
    find_program(_dpct_path
        NAMES dpct
        PATHS)

    if(_dpct_path)
        message("Found foo executable at: ${_dpct_path}")
    else()
        message(FATAL_ERROR "Could not find dpct executable.")
    endif()
    
    get_filename_component(dir_path ${_dpct_path} DIRECTORY)
    message("dir_path: ${dir_path}")
    set(cmake_yaml_file_path "${dir_path}/../extensions/opt_rules/cmake_rules/cmake_script_migration_rule.yaml")
    message("dir_path: ${cmake_yaml_file_path}")
     
    set( ${dpct_path} ${_dpct_path})
endmacro()


