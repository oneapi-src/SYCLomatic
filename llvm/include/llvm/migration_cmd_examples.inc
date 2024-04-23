const char *const DPCTExamplesMsg = R"(EXAMPLES:

Prepare for Migration
=====================

    Generate a compilation database of the project files used to guide migration
    Example command:)"
#ifndef _WIN32
R"(
            $ dpct --intercept-build make)"
#endif
R"(
            $ dpct --intercept-build --parse-build-log <logfile>
    
    Estimate the migration effect to complete a migration
    Example command:
            $ dpct --analysis-mode source.cu --analysis-mode-output-file

Migrate your code
=================

    Migrate a single source file
    Example commands:
            $ dpct source.cu
            $ dpct --keep-original-code source.cu                           # Keep original code
            $ dpct --cuda-include-path=/path/to/cuda/include source.cu      # Specify path to CUDA headers
            $ dpct --extra-arg="-std=c++11" source.cu                       # With extra options
            $ dpct --extra-arg="-I /path/to/extra/include" source.cu        # With additional app args
            $ dpct -p=/path/to/compilation-database source.cu               # With compilation database
            $ dpct --gen-build-script source.cu                             # Migrate src code and generate Makefile script
    
    Migrate an entire project
    Example commands:
            $ dpct -p=/path/to/compilation-database --in-root=/path/to/project --out-root=/path/to/migrated-project     # With compilation database
            $ dpct --process-all --in-root=/path/to/project --out-root=/path/to/migrated-project                        # Without compilation database)"
#ifdef _WIN32
R"(
            $ dpct -vcxprojfile=/path/to/vcproject/file --in-root=/path/to/project --out-root=/path/to/migrated-project # With VS project file)"
#endif
R"(
            $ dpct -in-root=/path/to/project --in-root-exclude=/path/to/project/dir1                                    # Exclude migration of a dir or file
    
    Migrate/generate build scripts for migrated SYCL code
    Example commands:
            $ dpct --migrate-build-script=cmake -p=/path/to/compilation-database --in-root=/path/to/project --out-root=/path/to/migrated-project        # Migrate src code and CMake scripts
            $ dpct --migrate-build-script-only -p=/path/to/compilation-database --in-root=/path/to/project --out-root=/path/to/migrated-project         # Migrate build script only
            $ dpct --gen-build-script -p=/path/to/compilation-database --in-root=/path/to/project --out-root=/path/to/migrated-project                  # Migrate src code and generate Makefile script
    
Utils to assist migration
=========================

    Query functionally compatible SYCL API for a CUDA API
    Example command:
            $ dpct --query-api-mapping=cudaMalloc
    
    Generate instrumented CUDA and SYCL code for debugging
    Example command:
            $ dpct --enable-codepin source.cu
    
    Generate helper function files in the out-root directory
    Example command:
            $ dpct --gen-helper-function
    
    Show the folder of helper function files
    Example command:
            $ dpct --helper-function-dir
    
)";
