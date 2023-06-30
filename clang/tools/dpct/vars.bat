@echo off

rem Copyright © Intel Corporation
rem SPDX-License-Identifier: MIT

rem Permission is hereby granted, free of charge, to any person obtaining a copy
rem of this software and associated documentation files (the "Software"), to deal
rem in the Software without restriction, including without limitation the rights
rem to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
rem copies of the Software, and to permit persons to whom the Software is
rem furnished to do so, subject to the following conditions:
rem
rem The above copyright notice and this permission notice shall be included in all
rem copies or substantial portions of the Software.
rem
rem THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
rem IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
rem FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
rem AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
rem LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
rem OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
rem SOFTWARE.


rem ############################################################################

rem This etc\dpct\vars.bat script is intended to be _sourced_ directly
rem by the top-level setvars.bat script. This vars.sh script is not a stand-alone
rem script. A component-specific vars.bat script is only required if the top-level
rem global environment variables defined by setvars.bat are insufficient for the
rem component that is providing this vars.bat script. For example, if a special
rem %CMPLR_OPT_ROOT% variable needs to be defined or some other
rem component-unique env vars are needed. It is possible that a component needs to
rem augment the top-level global env vars because some of the component's files
rem are in unusual locations that are not referenced by the global environment
rem variables defined by setvars.bat.

rem NOTE: See the setvars.bat script for a list of the top-level environment
rem variables that it is providing.

if not defined SETVARS_CALL (
  echo:
  echo :: ERROR: This script must be executed by setvars.bat.
  echo:   Try '[install-dir]\setvars.bat --help' for help.
  echo:
  exit /b 255
)

if not defined ONEAPI_ROOT (
  echo
  echo :: ERROR: This script requires that the ONEAPI_ROOT env variable is set.
  echo:   Try '[install-dir]\setvars.bat --help' for help.
  echo
  exit /b 254
)

if not defined VSCMD_VER (
  echo
  echo :: ERROR: This script requires Visual Studio environment.
  echo:   Try '[install-dir]\setvars.bat --help' for help.
  echo
  exit /b 254
)

rem ############################################################################
