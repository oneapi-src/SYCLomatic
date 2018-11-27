################################### SYCL CT ####################################
# SYCL CT bundle root
export SYCLCT_BUNDLE_ROOT=$(realpath $(dirname "${BASH_SOURCE[0]}"))

# Binary check for SYCL CT
if [[ ! -e $SYCLCT_BUNDLE_ROOT/bin/syclct || \
      ! -e $SYCLCT_BUNDLE_ROOT/bin/intercept-build ]]; then
    printf "[\033[0;31mERROR\033[0m] Cannot find neccessary syclct binary\n"
    return 1
fi
################################################################################

################################ SYCL Compiler #################################
# User can set the root directory of SYCL compiler
# Currently we use ComputeCpp by default, installed in $HOME
[ -z "$SYCL_COMPILER_ROOT" ] && SYCL_COMPILER_ROOT=$HOME/computecpp

# Binary chech for SYCL compiler
if [[ ! -e $SYCL_COMPILER_ROOT/bin/compute++ && \
      ! -e $SYCL_COMPILER_ROOT/bin/clang++ ]]; then
    unset SYCL_COMPILER_ROOT
    printf "[\033[0;33mWARNING\033[0m] Cannot find SYCL compiler\n"
fi
################################################################################

################################## OpenCL ICD ##################################
[[ -z "$OPENCL_ICD_ROOT" ]] && OPENCL_ICD_ROOT=$HOME/opencl_icd

if [[ ! -e $OPENCL_ICD_ROOT/include || ! -e $OPENCL_ICD_ROOT/lib ]]; then
    unset OPENCL_ICD_ROOT
    printf "[\033[0;33mWARNING\033[0m] Cannot find OpenCL ICD loader\n"
fi
################################################################################

############################ Environment Variables #############################
# SYCL CT bundle
export PATH=$SYCLCT_BUNDLE_ROOT/bin:$PATH
export CPLUS_INCLUDE_PATH=$SYCLCT_BUNDLE_ROOT/include:$CPLUS_INCLUDE_PATH

# SYCL compiler
if [ -n "$SYCL_COMPILER_ROOT" ]; then
    export PATH=$SYCL_COMPILER_ROOT/bin:$PATH
    export CPLUS_INCLUDE_PATH=$SYCL_COMPILER_ROOT/include:$CPLUS_INCLUDE_PATH
    export LD_LIBRARY_PATH=$SYCL_COMPILER_ROOT/lib:$LD_LIBRARY_PATH
fi

# OpenCL ICD
if [ -n "$OPENCL_ICD_ROOT" ]; then
    export CPLUS_INCLUDE_PATH=$OPENCL_ICD_ROOT/include:$CPLUS_INCLUDE_PATH
    export LD_LIBRARY_PATH=$OPENCL_ICD_ROOT/lib:$LD_LIBRARY_PATH
fi

export C_INCLUDE_PATH=$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=$LD_LIBRARY_PATH
################################################################################

################################### CUDA SDK ###################################
# Default CUDA SDK found by syclct
printf "[\033[0;32mINFO\033[0m] Default CUDA SDK selected by syclct location: \
$(syclct -- -v 2>&1 | grep -oP '(?<=Found CUDA installation: ).*(?=,)') \n"
################################################################################

printf "[\033[0;32mINFO\033[0m] SYCL CT toolchain location: \
$SYCLCT_BUNDLE_ROOT/bin \n"
printf "[\033[0;32mINFO\033[0m] SYCL compiler location: $SYCL_COMPILER_ROOT\n"
printf "[\033[0;32mINFO\033[0m] OpenCL ICD location: $OPENCL_ICD_ROOT\n"
