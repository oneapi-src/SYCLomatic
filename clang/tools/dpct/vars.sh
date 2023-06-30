#!/bin/sh
# shellcheck shell=sh
# shellcheck source=/dev/null
# shellcheck disable=SC2312

# Copyright © Intel Corporation
# SPDX-License-Identifier: MIT

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# ############################################################################

# This etc/dpct/vars.sh script is intended to be _sourced_ directly
# by the top-level setvars.sh script. This vars.sh script is not a stand-alone
# script. A component-specific vars.sh script is only required if the top-level
# global environment variables defined by setvars.sh are insufficient for the
# component that is providing this vars.sh script. For example, if a special
# ${CMPLR_OPT_ROOT} variable needs to be defined or some other
# component-unique env vars are needed. It is possible that a component needs to
# augment the top-level global env vars because some of the component's files
# are in unusual locations that are not referenced by the global environment
# variables defined by setvars.sh.

# NOTE: See the setvars.sh script for a list of the top-level environment
# variables that it is providing. Also, if a comoponent vars.sh script must
# augment an existing environment variable it should use the prepend_path() and
# prepend_manpath() functions that are provided by setvars.sh. If it is not
# sourced by setvars.sh this test should generate an error message and may
# generate unkown function

# ############################################################################


if [ -z "${SETVARS_CALL:-}" ] ; then
  >&2 echo " "
  >&2 echo ":: ERROR: This script must be sourced by setvars.sh."
  >&2 echo "   Try 'source <install-dir>/setvars.sh --help' for help."
  >&2 echo " "
  return 255
fi

if [ -z "${ONEAPI_ROOT:-}" ] ; then
  >&2 echo " "
  >&2 echo ":: ERROR: This script requires that the ONEAPI_ROOT env variable is set."
  >&2 echo "   Try 'source <install-dir>\setvars.sh --help' for help."
  >&2 echo " "
  return 254
fi


#############################################################################

DPCT_ETC_ROOT="${ONEAPI_ROOT}/etc/dpct"
BASH_AUTOCOMPLETE_SCRIPT=${DPCT_ETC_ROOT}/bash-autocomplete.sh

if [ -f "$BASH_AUTOCOMPLETE_SCRIPT" ]; then
. "${BASH_AUTOCOMPLETE_SCRIPT}"
fi
