# Getting Started with SYCLomatic


## Table of contents
  - [Prerequisites](#prerequisites)
    - [Create SYCLomatic workspace](#create-SYCLomatic-workspace)
  - [Build SYCLomatic](#build-SYCLomatic)
    - [Deployment](#deployment)
  - [Test SYCLomatic](#test-SYCLomatic)
      - [Run in-tree LIT tests](#run-in-tree-lit-tests)
      - [Run E2E test suite](#run-SYCLomatic-e2e-test-suite)
  - [Run c2s command](#Run-c2s-command)
  - [Known Issues and Limitations](#known-issues-and-limitations)
  - [Find More](#find-more)

## Prerequisites

* `git` - [Download](https://git-scm.com/downloads)
* `cmake` version 3.14 or later - [Download](http://www.cmake.org/download/)
* `python` - [Download](https://www.python.org/downloads/release/python-2716/)
* `ninja` -
[Download](https://github.com/ninja-build/ninja/wiki/Pre-built-Ninja-packages)
* C++ compiler
  * Linux: `GCC` version 7.1.0 or later (including libstdc++) -
    [Download](https://gcc.gnu.org/install/)
  * Windows: `Visual Studio` version 15.7 preview 4 or later -
    [Download](https://visualstudio.microsoft.com/downloads/)

### Create SYCLomatic workspace

Throughout this document `SYCLOMATIC_HOME` denotes the path to the local directory
created as SYCLomatic workspace. It might be useful to
create an environment variable with the same name.

**Linux**:

```bash
export SYCLOMATIC_HOME=~/workspace
mkdir $SYCLOMATIC_HOME
cd $SYCLOMATIC_HOME

git clone https://github.com/oneapi-src/SYCLomatic.git
git checkout origin/SYCLomatic -b SYCLomatic
```

**Windows (64-bit)**:

Open a developer command prompt using one of two methods:

* Click start menu and search for "**x64** Native Tools Command Prompt for VS
  XXXX", where XXXX is a version of installed Visual Studio.
* Ctrl-R, write "cmd", click enter, then run
  `"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" x64`

```bat
set SYCLOMATIC_HOME=%USERPROFILE%\workspace
mkdir %SYCLOMATIC_HOME%
cd %SYCLOMATIC_HOME%

git clone https://github.com/oneapi-src/SYCLomatic.git
git checkout origin/SYCLomatic -b SYCLomatic
```

## Build SYCLomatic


**Linux**:

```bash
cd $SYCLOMATIC_HOME
mkdir build
cd build
cmake -G Ninja -DCMAKE_INSTALL_PREFIX=$PATH_TO_C2S_INSTALL_FOLDER  -DCMAKE_BUILD_TYPE=Release  -DLLVM_ENABLE_PROJECTS="clang"  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" ../SYCLomatic/llvm
ninja install-c2s
```

**Windows (64-bit)**:

```bat
cd %SYCLOMATIC_HOME%
mkdir build
cd build
cmake -G Ninja -DCMAKE_INSTALL_PREFIX=%PATH_TO_C2S_INSTALL_FOLDER%  -DCMAKE_BUILD_TYPE=Release  -DLLVM_ENABLE_PROJECTS="clang"  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" ..\SYCLomatic\llvm
ninja install-c2s
```

### Deployment

**Linux**:
```bash
export PATH=$PATH_TO_C2S_INSTALL_FOLDER/bin:$PATH
export CPATH=$PATH_TO_C2S_INSTALL_FOLDER/include:$CPATH
```

**Windows (64-bit)**:
```bat
SET PATH=%PATH_TO_C2S_INSTALL_FOLDER%\bin;%PATH%
SET INCLUDE=%PATH_TO_C2S_INSTALL_FOLDER%\include;%INCLUDE%
SET CPATH=%PATH_TO_C2S_INSTALL_FOLDER%\include;%CPATH%
```

## Test SYCLomatic
### Run in-tree LIT tests

Note: Certain CUDA header files may need to be accessible to the tool.
After build the SYCLomatic, you can run the list test by: 

ninja check-clang-c2s


#### Run E2E test suite

Follow instructions from the link below to build and run tests:
[README](https://github.com/oneapi-src/SYCLomatic-test)


## Run c2s command
Get c2s help information by running "c2s --help".
dpct is an alias command for c2s.

## Known Issues and Limitations

* SYCL\* 2020 support work is in progress.
* 32-bit host/target is not supported.

## Find More

* DPC++ specification:
[https://spec.oneapi.com/versions/latest/elements/dpcpp/source/index.html](https://spec.oneapi.com/versions/latest/elements/dpcpp/source/index.html)
* SYCL\* 2020 specification:
[https://www.khronos.org/registry/SYCL/](https://www.khronos.org/registry/SYCL/)

## Trademarks information
Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries.
*Other names and brands may be claimed as the property of others. SYCL is a trademark of the Khronos Group Inc.
