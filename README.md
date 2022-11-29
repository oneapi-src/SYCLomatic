
# SYCLomatic

- [SYCLomatic](#syclomatic)
  - [Introduction](#introduction)
  - [Releases](#releases)
  - [Build from source code](#build-from-source-code)
    - [Prerequisites](#prerequisites)
    - [Create SYCLomatic workspace](#create-syclomatic-workspace)
    - [Build SYCLomatic](#build-syclomatic)
    - [Deployment](#deployment)
    - [Test SYCLomatic](#test-syclomatic)
      - [Run in-tree LIT tests](#run-in-tree-lit-tests)
      - [Run E2E test suite](#run-e2e-test-suite)
  - [Run SYCLomatic](#run-syclomatic)
    - [Run c2s command](#run-c2s-command)
  - [Known Issues and Limitations](#known-issues-and-limitations)
  - [Useful Links](#useful-links)
  - [License](#license)
  - [Contributing](#contributing)
  - [Trademarks information](#trademarks-information)

## Introduction

SYCLomatic is a project to assist developers in migrating their existing code written in different programming languages to the SYCL\* C++ heterogeneous programming model. Final code editing and verification is a manual process done by the developer.

Use c2s command to make it as easy as possible to migrate existing CUDA codebases to SYCL, which is an industry standard. Once code is migrated to SYCL, it can be compiled and executed by any compiler that implements the SYCL specification as shown here:  https://www.khronos.org/sycl/

The SYCLomatic development branch is the SYCLomatic branch.

## Releases

Daily builds of the SYCLomatic branch on Linux and Windows* are available at
[releases](/../../releases).
A few times a year, we publish Release Notes to
highlight all important changes made in the project: features implemented and
issues addressed. The corresponding builds can be found using
[search](https://github.com/oneapi-src/SYCLomatic/releases)
in daily releases. None of the branches in the project are stable or rigorously
tested for production quality control, so the quality of these releases is
expected to be similar to the daily releases.

## Build from source code
### Prerequisites

* `git` - [Download](https://git-scm.com/downloads)
* `cmake` version 3.14 or later - [Download](http://www.cmake.org/download/)
* `python 3` - [Download](https://www.python.org/downloads/)
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
export PATH_TO_C2S_INSTALL_FOLDER=~/workspace/c2s_install
mkdir $SYCLOMATIC_HOME
cd $SYCLOMATIC_HOME

git clone https://github.com/oneapi-src/SYCLomatic.git
```

**Windows (64-bit)**:

Open a developer command prompt using one of two methods:

* Click start menu and search for "**x64** Native Tools Command Prompt for VS
  XXXX", where XXXX is a version of installed Visual Studio.
* Ctrl-R, write "cmd", click enter, then run
  `"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" x64`

```bat
set SYCLOMATIC_HOME=%USERPROFILE%\workspace
set PATH_TO_C2S_INSTALL_FOLDER=%USERPROFILE%\workspace\c2s_install
mkdir %SYCLOMATIC_HOME%
cd %SYCLOMATIC_HOME%

git clone https://github.com/oneapi-src/SYCLomatic.git
```

### Build SYCLomatic


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

### Test SYCLomatic
#### Run in-tree LIT tests

Note: Certain CUDA SDK, CCL and DNN header files need to be accessible to the tool. And the tool supports version 11.8 SDK header files.
After build the SYCLomatic, you can run the list test by:

ninja check-clang-c2s


#### Run E2E test suite

Follow instructions from the link below to build and run tests:
[README](https://github.com/oneapi-src/SYCLomatic-test)


## Run SYCLomatic
### Run c2s command
Get c2s help information by running "c2s --help".
dpct is an alias command for c2s.

## Known Issues and Limitations

* SYCL\* 2020 support work is in progress.
* 32-bit host/target is not supported.


## Useful Links
* More information about how to use SYCLomatic can be found in Intel(R) DPC++ Compatibility Tool documentation, which is the Intel product version of SYCLomatic:
    * [Get Started](https://software.intel.com/content/www/us/en/develop/documentation/get-started-with-intel-dpcpp-compatibility-tool/top.html)
    * [Developer Guide and Reference](https://software.intel.com/content/www/us/en/develop/documentation/intel-dpcpp-compatibility-tool-user-guide/top.html)
* [oneAPI DPC++ Compiler documentation](https://intel.github.io/llvm-docs/)
* [Book: Mastering Programming of Heterogeneous Systems using C++ & SYCL](https://protect-eu.mimecast.com/s/P9FyCjvlRipPPWgT5ya8e?domain=link.springer.com)
* [Essentials of SYCL training](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/dpc-essentials.html)
* oneAPI specification:
[https://spec.oneapi.io/versions/latest/index.html](https://spec.oneapi.io/versions/latest/index.html)
* SYCL\* 2020 specification:
[https://www.khronos.org/registry/SYCL/](https://www.khronos.org/registry/SYCL/)
* More information on oneAPI and DPC++ is available at [https://www.oneapi.com/](https://www.oneapi.com/)

## License

See [LICENSE](LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Trademarks information
Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries.<br>
\*Other names and brands may be claimed as the property of others. SYCL is a trademark of the Khronos Group Inc.
