::################################################################################
::#
::# Copyright (C) 2019 Intel Corporation. All rights reserved.
::#
::# The information and source code contained herein is the exclusive property of
::# Intel Corporation and may not be disclosed, examined or reproduced in whole or
::# in part without explicit written authorization from the company.
::#
::################################################################################

@echo off

setlocal
set /A ERRORSTATE=0


IF NOT EXIST "%~dp0..\bin\dpct.exe" set ERRORSTATE=1
IF NOT EXIST "%~dp0..\bin\intercept-build" set ERRORSTATE=1

if /I "%ERRORSTATE%" NEQ "0" (
    echo  Error: Cannot find neccessary dpct binary.
)

SET PATH="%~dp0..\bin";%PATH%
SET INCLUDE="%~dp0..\include";%INCLUDE%

::always return ERRORSTATE ( which is 0 if no error )
exit /B %ERRORSTATE%
