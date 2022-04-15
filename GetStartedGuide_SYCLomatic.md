# Getting Started with SYCLomatic


## Table of contents
  - [Prerequisites](#prerequisites)
    - [Create SYCLomatic workspace](#create-SYCLomatic-workspace)
  - [Build SYCLomatic](#build-SYCLomatic)
    - [Build Doxygen documentation](#build-doxygen-documentation)
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

git clone https://fixme/URL/to/main
git checkout origin/main -b main
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

git clone https//fixme/URL/to/main 
git checkout origin/main -b main
```

## Build SYCLomatic


**Linux**:

```bash
cd $SYCLOMATIC_HOME
mkdir build
cmake -G Ninja -DCMAKE_INSTALL_PREFIX=/path/to/install/folder  -DCMAKE_BUILD_TYPE=Release  -DLLVM_ENABLE_PROJECTS="clang"  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" ../llvm
ninja install-c2s-tool
```

**Windows (64-bit)**:

```bat
cd %SYCLOMATIC_HOME%
mkdir build
cmake -G Ninja -DCMAKE_INSTALL_PREFIX=/path/to/install/folder  -DCMAKE_BUILD_TYPE=Release  -DLLVM_ENABLE_PROJECTS="clang"  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" ../llvm
ninja install-c2s-tool
```
### Build Doxygen documentation

Building Doxygen documentation is similar to building the product itself. First,
the following tools need to be installed:

* doxygen
* graphviz

Then you'll need to add the following options to your CMake configuration
command:

```
-DLLVM_ENABLE_DOXYGEN=ON
```

After CMake cache is generated, build the documentation with `doxygen-c2s and doxygen-c2s-runtime`
target. It will be put to `$SYCLOMATIC_HOME/build/tools/sycl/doc/html`
directory.

### Deployment

**Linux**:
source /path/to/c2s/install-folder/env/var.sh

**Windows (64-bit)**:
/path/to/c2s/install-folder/env/vars.bat


## Test SYCLomatic
### Run in-tree LIT tests

Note: Certain CUDA header files may need to be accessible to the tool.
After build the SYCLomatic, you can run the list test by: 

ninja check-clang-c2s


#### Run E2E test suite

Follow instructions from the link below to build and run tests:
[README](https://github.com/intel/SYCLomatic-test.git)


## Run c2s command
Get c2s help information by running "c2s --help".
Note that dpct is an old name of the executable and is alias to c2s, use c2s going forward. 

## Known Issues and Limitations

* SYCL 2020 support work is in progress.
* 32-bit host/target is not supported.

## Find More

* DPC++ specification:
[https://spec.oneapi.com/versions/latest/elements/dpcpp/source/index.html](https://spec.oneapi.com/versions/latest/elements/dpcpp/source/index.html)
* SYCL\* 2020 specification:
[https://www.khronos.org/registry/SYCL/](https://www.khronos.org/registry/SYCL/)

## [Legal information](legal_information.md)

