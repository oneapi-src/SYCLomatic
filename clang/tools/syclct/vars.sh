#!/bin/bash
################################################################################
#
# Copyright (C) 2018 Intel Corporation. All rights reserved.
#
# The information and source code contained herein is the exclusive property of
# Intel Corporation and may not be disclosed, examined or reproduced in whole or
# in part without explicit written authorization from the company.
#
################################################################################

export SYCLCT_BUNDLE_ROOT=$(dirname $(dirname $(realpath "${BASH_SOURCE[0]}")))

# Binary check for SYCL CT
if [[ ! -e $SYCLCT_BUNDLE_ROOT/bin/syclct || \
      ! -e $SYCLCT_BUNDLE_ROOT/bin/intercept-build ]]; then
    printf "[\033[0;31mERROR\033[0m] Cannot find neccessary syclct binary\n"
    return 1
fi

# SYCL CT bundle
export PATH=$SYCLCT_BUNDLE_ROOT/bin:$PATH
export CPATH=$SYCLCT_BUNDLE_ROOT/include:$CPATH
