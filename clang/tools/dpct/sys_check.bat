::#############################################################################
::#
::#Copyright 2019 - 2020 Intel Corporation.
::#
::#This software and the related documents are Intel copyrighted materials,
::#and your use of them is governed by the express license under which they
::#were provided to you ("License"). Unless the License provides otherwise,
::#you may not use, modify, copy, publish, distribute, disclose or transmit
::#this software or the related documents without Intel's prior written
::#permission.
::#
::#This software and the related documents are provided as is, with no express
::#or implied warranties, other than those that are expressly stated in the
::#License.
::#
::#############################################################################

@echo off

:: every syscheck script should set up an ERRORSTATE variable and return it on completion.
setlocal
set /A ERRORSTATE=0

::exit with the %ERRORSTATE%
:: use /B flag, or the exit will prevent other sys_checks from running.
exit /B %ERRORSTATE%
