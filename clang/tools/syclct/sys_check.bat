::################################################################################
::#
::# Copyright (C) 2019 Intel Corporation. All rights reserved.
::#
::# The information and source code contained herein is the exclusive property of
::# Intel Corporation and may not be disclosed, examined or reproduced in whole or
::# in part without explicit written authorization from the company.
::#
::################################################################################
:: this syscheck script is provided as a model for every OneAPI component
:: each syscheck script should do four things:
::

@echo off

call common.bat :speak   This Is A Message You Will See In Verbose Mode

:: every syscheck script should set up an ERRORSTATE variable and return it on completion.
setlocal
set /A ERRORSTATE=1
nvcc --version > nul 2>nul
set ret=%ERRORLEVEL%
if "%ret%" NEQ "0" (
    echo  Warning: The oneAPI DPC++ Compatibility Tool requires the Nvidia CUDA Toolkit to be installed.
    set ERRORSTATE=1
) else (
    set ERRORSTATE=0
)
if /I "%ERRORSTATE%" NEQ "0" (
    echo  Warning: The oneAPI DPC++ Compatibility Tool component unsafe to operate.
)

::exit with the %ERRORSTATE%
:: use /B flag, or the exit will prevent other sys_checks from running.
exit /B %ERRORSTATE%
