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

set /A ERRORSTATE=0

IF NOT EXIST "%~dp0..\bin\c2s.exe" set ERRORSTATE=1

if /I "%ERRORSTATE%" NEQ "0" (
    echo  Error: Cannot find neccessary c2s binary.
)

SET PATH=%~dp0..\bin;%PATH%
SET INCLUDE=%~dp0..\include;%INCLUDE%
SET CPATH=%~dp0..\include;%CPATH%

:ParseArgs
:: Parse the incoming arguments
if /i "%1"==""              goto CheckArgs
if /i "%1"=="ia32"          (set TARGET_VS_ARCH=x86)     & shift & goto ParseArgs
if /i "%1"=="intel64"       (set TARGET_VS_ARCH=amd64)   & shift & goto ParseArgs
if /i "%1"=="vs2017"        (set TARGET_VS=vs2017)       & shift & goto ParseArgs
if /i "%1"=="vs2019"        (set TARGET_VS=vs2019)       & shift & goto ParseArgs
shift & goto ParseArgs

:CheckArgs
:: set correct defaults
if /i "%TARGET_VS_ARCH%"==""   (set TARGET_VS_ARCH=amd64)

:: Skip to End if Visual Studio environment is ready.
if defined VSCMD_VER (
    goto End
)

::detect installed VS
set MSVS_VAR_SCRIPT=

:: If there is standard installation, Priority: VS2019 > VS2017,
:: if there is no standard installation, ask user to set VS2019INSTALLDIR/VS2017INSTALLDIR, then re-run this script.
:: The exact installation directory depends on both the version and offering of Visual Studio,
:: according to the following pattern: C:\Program Files (x86)\Microsoft Visual Studio\<version>\<offering>.
if /i "%TARGET_VS%"=="" (
    call :SetVS2019INSTALLDIR
    if not defined VS2019INSTALLDIR (
        call :SetVS2017INSTALLDIR
    )
    goto SetVCVars
)

if /i "%TARGET_VS%"=="vs2019" (
    if not defined VS2019INSTALLDIR (
        call :SetVS2019INSTALLDIR
    )
    goto SetVCVars
)

if /i "%TARGET_VS%"=="vs2017" (
    if not defined VS2017INSTALLDIR (
        call :SetVS2017INSTALLDIR
    )
    goto SetVCVars
)

::default, set the latest VS in global environment
:SetVCVars
if /i "%TARGET_VS%"=="" (
    ::vs2019
    if defined VS2019INSTALLDIR (
        if exist "%VS2019INSTALLDIR%\VC\Auxiliary\Build\vcvarsall.bat" (
            goto SetVS2019
        )
    )
    ::vs2017
    if defined VS2017INSTALLDIR (
        if exist "%VS2017INSTALLDIR%\VC\Auxiliary\Build\vcvarsall.bat" (
            goto SetVS2017
        )
    )
    call :NO_VS 2017 or 2019
    goto EndWithError
)

::VS2019
if /i "%TARGET_VS%"=="vs2019" (
    if defined VS2019INSTALLDIR (
        if exist "%VS2019INSTALLDIR%\VC\Auxiliary\Build\vcvarsall.bat" (
            goto SetVS2019
        )
    )
    call :NO_VS 2019
    goto EndWithError
)

::VS2017
if /i "%TARGET_VS%"=="vs2017" (
    if defined VS2017INSTALLDIR (
        if exist "%VS2017INSTALLDIR%\VC\Auxiliary\Build\vcvarsall.bat" (
            goto SetVS2017
        )
    )
    call :NO_VS 2017
    goto EndWithError
)

:SetVS2019
set TARGET_VS=vs2019
set MSVS_VAR_SCRIPT="%VS2019INSTALLDIR%\VC\Auxiliary\Build\vcvarsall.bat"
goto Setup

:SetVS2017
set TARGET_VS=vs2017
set MSVS_VAR_SCRIPT="%VS2017INSTALLDIR%\VC\Auxiliary\Build\vcvarsall.bat"
goto Setup

:Setup
:: print product info
echo Copyright (C) 2018 - 2020 Intel Corporation. All rights reserved.
echo Intel(R) DPC++ Compatibility Tool.
echo.

@call %MSVS_VAR_SCRIPT% %TARGET_VS_ARCH%

goto End

:End
::always return ERRORSTATE ( which is 0 if no error )
exit /B %ERRORSTATE%

:: ============================================================================
:NO_VS
echo.
if /i "%*"=="2017 or 2019" (
    echo ERROR: Visual Studio %* is not found in "C:\Program Files (x86)\Microsoft Visual Studio\<2017 or 2019>\<Edition>", please set VS2017INSTALLDIR or VS2019INSTALLDIR, then re-run this script.
    goto :EOF
)
if /i "%*"=="2019" (
    echo ERROR: Visual Studio %* is not found in "C:\Program Files (x86)\Microsoft Visual Studio\2019\<Edition>", please set VS2019INSTALLDIR, then re-run this script.
    goto :EOF
)
if /i "%*"=="2017" (
    echo ERROR: Visual Studio %* is not found in "C:\Program Files (x86)\Microsoft Visual Studio\2019\<Edition>", please set VS2017INSTALLDIR, then re-run this script.
    goto :EOF
)
:EndWithError
exit /B 1

:SetVS2019INSTALLDIR
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional" (
    set "VS2019INSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional"
    goto :EOF
)
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise" (
    set "VS2019INSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise"
    goto :EOF
)
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community" (
    set "VS2019INSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community"
    goto :EOF
)
goto :EOF

:SetVS2017INSTALLDIR
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional" (
    set "VS2017INSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional"
    goto :EOF
)
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise" (
    set "VS2017INSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise"
    goto :EOF
)
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community" (
    set "VS2017INSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio\2017\Community"
    goto :EOF
)
goto :EOF
