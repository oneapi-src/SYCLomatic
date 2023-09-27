# ===----- bash-autocomplete.sh -------------------------------------------=== #
#
# Copyright (C) Intel Corporation
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------=== #

# Please add "source /path/to/bash-autocomplete.sh" to your .bashrc to use this.

_c2s_dpct_filedir()
{
  # _filedir function provided by recent versions of bash-completion package is
  # better than "compgen -f" because the former honors spaces in pathnames while
  # the latter doesn't. So we use compgen only when _filedir is not provided.
  _filedir 2> /dev/null || COMPREPLY=( $( compgen -f ) )
}

_c2s_dpct()
{
  local cur prev words cword arg flags w1 w2 opt
  # If latest bash-completion is not supported just initialize COMPREPLY and
  # initialize variables by setting manually.
  _init_completion -n 2> /dev/null
  if [[ "$?" != 0 ]]; then
    COMPREPLY=()
    cword=$COMP_CWORD
    cur="${COMP_WORDS[$cword]}"
  fi

  w1="${COMP_WORDS[$cword - 1]}"
  if [[ $cword > 1 ]]; then
    w2="${COMP_WORDS[$cword - 2]}"
  fi

  # Pass all the current command-line flags to c2s/dpct, so that c2s/dpct can handle
  # these internally.
  # '=' is separated differently by bash, so we have to concat them without '#'
  for i in `seq 1 $cword`; do
    if [[ $i == $cword || "${COMP_WORDS[$(($i+1))]}" == '=' ]]; then
      arg="$arg${COMP_WORDS[$i]}"
    else
      arg="$arg${COMP_WORDS[$i]}#"
    fi
  done

  # expand ~ to $HOME
  eval local path=${COMP_WORDS[0]}

  # Get the last option.
  if [[ $w1 == '=' ]]; then
    opt=$w2
  else
    opt=$w1
  fi
  # Handle --query-api-mapping value autocompletion.
  if [[ $opt == "--query-api-mapping" || $opt == "-query-api-mapping" ]]; then
    flags=$( "$path" --query-api-mapping=- 2>/dev/null | sed -e $'s/\t.*//' )
    if [[ "$flags" == "$(echo -e '\n')" ]]; then
      [[ "$cur" == '=' || "$cur" == -*= ]] && cur=""
      _c2s_dpct_filedir
    elif [[ "$cur" == '=' ]]; then
      COMPREPLY=( $( compgen -W "$flags" -- "") )
    else
      [[ "${flags: -1}" == '=' ]] && compopt -o nospace 2> /dev/null
      COMPREPLY=( $( compgen -W "$flags" -- "$cur" ) )
    fi
    return
  fi

  # Use $'\t' so that bash expands the \t for older versions of sed.
  flags=$( "$path" --autocomplete="$arg" 2>/dev/null | sed -e $'s/\t.*//' )
  # If c2s/dpct is old that it does not support --autocomplete,
  # fall back to the filename completion.
  if [[ "$?" != 0 ]]; then
    _c2s_dpct_filedir
    return
  fi

  # When c2s/dpct does not emit any possible autocompletion, or user pushed tab after " ",
  # just autocomplete files.
  if [[ "$flags" == "$(echo -e '\n')" ]]; then
    # If -foo=<tab> and there was no possible values, autocomplete files.
    [[ "$cur" == '=' || "$cur" == -*= ]] && cur=""
    _c2s_dpct_filedir
  elif [[ "$cur" == '=' ]]; then
    COMPREPLY=( $( compgen -W "$flags" -- "") )
  else
    # Bash automatically appends a space after '=' by default.
    # Disable it so that it works nicely for options in the form of -foo=bar.
    [[ "${flags: -1}" == '=' ]] && compopt -o nospace 2> /dev/null
    COMPREPLY=( $( compgen -W "$flags" -- "$cur" ) )
  fi
}
complete -F _c2s_dpct c2s
complete -F _c2s_dpct dpct
