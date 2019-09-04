::#############################################################################
::#
::#Copyright 2019 Intel Corporation.
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

set /A ERRORSTATE=0

IF NOT EXIST "%~dp0..\bin\dpct.exe" set ERRORSTATE=1

if /I "%ERRORSTATE%" NEQ "0" (
    echo  Error: Cannot find neccessary dpct binary.
)

SET PATH=%~dp0..\bin;%PATH%
SET INCLUDE=%~dp0..\include;%INCLUDE%
SET CPATH=%~dp0..\include;%CPATH%

::always return ERRORSTATE ( which is 0 if no error )
exit /B %ERRORSTATE%
