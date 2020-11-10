#!/bin/sh "shebang"
# shellcheck shell=sh

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

# ############################################################################
# Get absolute path to script, when sourced from bash, zsh and ksh shells.
# Usage:
#   script_dir=$(get_script_path "$script_rel_path")
# Inputs:
#   script/relative/pathname/scriptname
# Outputs:
#   /script/absolute/pathname
# executing function in a *subshell* to localize vars and effects on `cd`
get_script_path() (
  script="$1"
  while [ -L "$script" ] ; do
    script_dir=$(command dirname -- "$script")
    script_dir=$(cd "$script_dir" && command pwd -P)
    script="$(readlink "$script")"
    case $script in
      (/*) ;;
       (*) script="$script_dir/$script" ;;
    esac
  done
  script_dir=$(command dirname -- "$script")
  script_dir=$(cd "$script_dir" && command pwd -P)
  echo "$script_dir"
)
# ############################################################################
usage() {
  printf "%s\n"   "ERROR: This script must be sourced."
  printf "%s\n"   "Usage: source $1"
  return 2 2>/dev/null || exit 2
}

vars_script_name=""

if [ -n "${ZSH_VERSION:-}" ] ; then     # only executed in zsh
  case $ZSH_EVAL_CONTEXT in (*:file*) vars_script_name="${(%):-%x}" ;; esac ;
  if [ -n "$ZSH_EVAL_CONTEXT" ] && [ "" = "$vars_script_name" ] ; then
    usage "${(%):-%x}" ;
  fi
elif [ -n "${KSH_VERSION:-}" ] ; then   # only executed in ksh or mksh or lksh
  if [ "$(set | grep KSH_VERSION)" = "KSH_VERSION=.sh.version" ] ; then # ksh
    if [ "$(cd "$(dirname -- "$0")" && pwd -P)/$(basename -- "$0")" \
      != "$(cd "$(dirname -- "${.sh.file}")" && pwd -P)/$(basename -- "${.sh.file}")" ] ; then
      vars_script_name="${.sh.file}" || usage "$0" ;
    fi
  else # mksh or lksh detected (also check for [lm]ksh renamed as ksh)
    _lmksh="$(basename -- "$0")" ;
    if [ "mksh" = "$_lmksh" ] || [ "lksh" = "$_lmksh" ] || [ "ksh" = "$_lmksh" ] ; then
      # force [lm]ksh to issue error msg; contains this script's rel/path/filename
      vars_script_name="$( (echo "${.sh.file}") 2>&1 )" || : ;
      vars_script_name="$(expr "$vars_script_name" : '^.*ksh: \(.*\)\[[0-9]*\]:')" ;
    fi
  fi
elif [ -n "${BASH_VERSION:-}" ] ; then  # only executed in bash
  (return 0 2>/dev/null) && vars_script_name="${BASH_SOURCE}" || usage "${BASH_SOURCE}"
else
  case ${0##*/} in (sh|dash) vars_script_name="" ;; esac
fi

if [ "" = "$vars_script_name" ] ; then
  >&2 echo ":: ERROR: Unable to proceed: no support for sourcing from '[dash|sh]' shell." ;
  >&2 echo "   This script must be sourced. Did you execute or source this script?" ;
  >&2 echo "   Can be caused by sourcing from inside a \"shebang-less\" script." ;
  >&2 echo "   Can also be caused by sourcing from ZSH version 4.x or older." ;
  return 1 2>/dev/null || exit 1
fi


# ############################################################################

# Export required env vars for dpct package.
export DPCT_BUNDLE_ROOT=$(dirname -- "$(get_script_path "$vars_script_name")")

if [[ ! -e $DPCT_BUNDLE_ROOT/bin/dpct || \
      ! -e $DPCT_BUNDLE_ROOT/bin/intercept-build ]]; then
    printf "[\033[0;31mERROR\033[0m] Cannot find neccessary dpct binary\n"
    return 1
fi

export PATH=$DPCT_BUNDLE_ROOT/bin:$PATH
export CPATH=$DPCT_BUNDLE_ROOT/include:$CPATH
