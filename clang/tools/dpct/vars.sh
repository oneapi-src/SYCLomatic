###############################################################################
#
#Copyright 2018 - 2020 Intel Corporation.
#
#This software and the related documents are Intel copyrighted materials,
#and your use of them is governed by the express license under which they
#were provided to you ("License"). Unless the License provides otherwise,
#you may not use, modify, copy, publish, distribute, disclose or transmit
#this software or the related documents without Intel's prior written
#permission.
#
#This software and the related documents are provided as is, with no express
#or implied warranties, other than those that are expressly stated in the
#License.
#
###############################################################################

export DPCT_BUNDLE_ROOT=$(dirname $(dirname $(realpath "${BASH_SOURCE[0]}")))

if [[ ! -e $DPCT_BUNDLE_ROOT/bin/dpct || \
      ! -e $DPCT_BUNDLE_ROOT/bin/intercept-build ]]; then
    printf "[\033[0;31mERROR\033[0m] Cannot find neccessary dpct binary\n"
    return 1
fi

export PATH=$DPCT_BUNDLE_ROOT/bin:$PATH
export CPATH=$DPCT_BUNDLE_ROOT/include:$CPATH
