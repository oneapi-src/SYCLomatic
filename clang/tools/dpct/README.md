//===----------------------------------------------------------------------===//
// DPCT installation layout
//===----------------------------------------------------------------------===//


# DPCT 2023-style layout is enabled when option "-DINTEL_DEPLOY_UNIFIED_LAYOUT=OFF" of cmake is set 

Here is DPCT 2023-style layout:

./dpct_installation_directory
├── bin
│   ├── c2s -> dpct
│   ├── dpct
│   ├── intercept-build
│   └── pattern-rewriter
├── etc
│   └── dpct
│       ├── bash-autocomplete.sh
│       ├── modulefiles
│       │   └── dpct
│       ├── sys_check.sh
│       └── vars.sh
├── include
│   └── dpct
│       ├── atomic.hpp
│       ├── blas_utils.hpp
│       ├── ccl_utils.hpp
│       ├── device.hpp
│       ├── dnnl_utils.hpp
│       ├── dpct.hpp
│       ├── dpl_extras
│       │   ├── algorithm.h
│       │   ├── dpcpp_extensions.h
│       │   ├── functional.h
│       │   ├── iterators.h
│       │   ├── memory.h
│       │   ├── numeric.h
│       │   └── vector.h
│       ├── dpl_utils.hpp
│       ├── fft_utils.hpp
│       ├── image.hpp
│       ├── kernel.hpp
│       ├── lapack_utils.hpp
│       ├── lib_common_utils.hpp
│       ├── memory.hpp
│       ├── rng_utils.hpp
│       ├── sparse_utils.hpp
│       └── util.hpp
└── share
    └── dpct
        ├── extensions
        │   └── opt_rules
        └── lib
            ├── clang
            ├── libear
            └── libscanbuild
And "vars.sh_2023_style" and "vars.bat_2023_style" work for dpct 2023-style layout for Linux platform and Windows platform.


# DPCT 2024-style layout is enabled when option "-DINTEL_DEPLOY_UNIFIED_LAYOUT=ON" of cmake is set (this option is default on) 
Here is 2024-style layout:

./dpct_installation_directory
├── bin
│   ├── c2s -> dpct
│   ├── dpct
│   └── intercept-build
├── etc
│   ├── dpct
│   │   ├── bash-autocomplete.sh
│   │   ├── sys_check
│   │   │   └── sys_check.sh
│   │   └── vars.sh
│   └── modulefiles
│       └── dpct
├── include
│   └── dpct
│       ├── atomic.hpp
│       ├── blas_utils.hpp
│       ├── ccl_utils.hpp
│       ├── device.hpp
│       ├── dnnl_utils.hpp
│       ├── dpct.hpp
│       ├── dpl_extras
│       │   ├── algorithm.h
│       │   ├── dpcpp_extensions.h
│       │   ├── functional.h
│       │   ├── iterators.h
│       │   ├── memory.h
│       │   ├── numeric.h
│       │   └── vector.h
│       ├── dpl_utils.hpp
│       ├── fft_utils.hpp
│       ├── image.hpp
│       ├── kernel.hpp
│       ├── lapack_utils.hpp
│       ├── lib_common_utils.hpp
│       ├── math.hpp
│       ├── memory.hpp
│       ├── rng_utils.hpp
│       ├── sparse_utils.hpp
│       └── util.hpp
└── opt
    └── dpct
        ├── extensions
        │   └── opt_rules
        └── lib
            ├── clang
            ├── libear
            └── libscanbuild
And "vars.sh" and "vars.bat" are DPCT 2024-style scripts for Linux platform and Windows platform.


# Side-by-side standalone support in 2024-style layout

For simplicity and reliability, two types of vars scripts are maintained, one is 2023-style script and one is a 2024-style script. To implement this, a new env/vars.sh script is created as part of 2024-style component to support side-by-side standalone.

Here is 2024-style layout to support side-by-side standalone

./dpct_installation_directory
├── bin
│   ├── c2s -> dpct
│   ├── dpct
│   └── intercept-build
├── env  ## this directory is used to store 2023-style stand-alone script
│   └── vars.sh
├── etc
│   ├── dpct
│   │   ├── bash-autocomplete.sh
│   │   ├── sys_check
│   │   │   └── sys_check.sh
│   │   └── vars.sh
│   └── modulefiles
│       └── dpct
├── include
│   └── dpct
│       ├── atomic.hpp
│       ├── blas_utils.hpp
│       ├── ccl_utils.hpp
│       ├── device.hpp
│       ├── dnnl_utils.hpp
│       ├── dpct.hpp
│       ├── dpl_extras
│       │   ├── algorithm.h
│       │   ├── dpcpp_extensions.h
│       │   ├── functional.h
│       │   ├── iterators.h
│       │   ├── memory.h
│       │   ├── numeric.h
│       │   └── vector.h
│       ├── dpl_utils.hpp
│       ├── fft_utils.hpp
│       ├── image.hpp
│       ├── kernel.hpp
│       ├── lapack_utils.hpp
│       ├── lib_common_utils.hpp
│       ├── math.hpp
│       ├── memory.hpp
│       ├── rng_utils.hpp
│       ├── sparse_utils.hpp
│       └── util.hpp
└── opt
    └── dpct
        ├── extensions
        │   └── opt_rules
        └── lib
            ├── clang
            ├── libear
            └── libscanbuild


# More detail about 2023-style layout, 2024-style layout, side-by-side standalone support, please refer to: https://github.com/intel-innersource/applications.infrastructure.oneapi.setvars-scripts/wiki/v2-Component-vars-Scripts
