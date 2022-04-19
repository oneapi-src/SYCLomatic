# How to mantain the helper functions
## Add new feature in a file

Example1:
```
// C2S_LABEL_BEGIN|FeatureNameDef|[Namespace]
// C2S_DEPENDENCY_EMPTY
// C2S_CODE
some code
// C2S_LABEL_END
```
Example2:
```
// C2S_LABEL_BEGIN|FeatureNameDef|[Namespace]
// C2S_PARENT_FEATURE|ParentFeatureNameRef
// C2S_DEPENDENCY_BEGIN
// FileID|ParentFeatureNameRef
// FileID|FeatureNameRef
[// FileID|FeatureNameRef]
...
// C2S_DEPENDENCY_END
// C2S_CODE
some code
// C2S_LABEL_END
```

Note: If a feature has a parent feature, the parent feature should be added into dependency list.

## 3 predefined feature name
For header file including dependency, please use predefined feature name:
* `local_include_dependency`: include local helper files
* `non_local_include_dependency`: include other files: mkl, dpl, std, ...
* `License`: save the license text
They must occur and only used once in each file.
If there is no `#include "..."` statement in the file, need add an empty `local_include_dependency` feature like:
```
// C2S_LABEL_BEGIN|local_include_dependency|
// C2S_DEPENDENCY_EMPTY
// C2S_CODE
// C2S_LABEL_END
```

## Helper file enum name definition
`clang/runtime/c2s-rt/include/HelperFilesName.inc` defines enum name for helper files:
```
HELPERFILE(c2s.hpp.inc, C2S)
HELPERFILE(device.hpp.inc, Device)
HELPERFILE(dpl_extras/algorithm.h.inc, DplExtrasAlgorithm)
HELPERFILE(dpl_extras/functional.h.inc, DplExtrasFunctional)
...
```

## Build process
When building c2s, the processFiles.py processes the input files and generates the output files:
Input files:
```
Src/clang/runtime/c2s-rt/*.hpp.inc
Src/clang/runtime/c2s-rt/dpl_extras/*.h.inc
```
Output files:
* `*.hpp.inc` and `*.h.inc`, which contains dependency information, will be included by dpcr source files
* `*.hpp` and `*.h`, final full set header files in install/include folder

## For c2s developers
Generally, request the feature before use it:
```
void requestFeature(clang::dpct::HelperFeatureEnum Feature, std::string CallerSrcFilePath);
```

Feature used in following maps are request implicitly:
* TypeNamesHelperFeaturesMap
* EnumNamesHelperFeaturesMap
* ThrustFuncNamesHelperFeaturesMap
* TextureAPIHelperFeaturesMap

E.g,
```
TypeNamesHelperFeaturesMap = {
  {"cudaDeviceProp", HelperFeatureEnum::Device_device_info}, // feature device_info in device.hpp will be requested implicitly 
}
```

Please update files in `test/dpct/helper_files_ref/include` used in LIT test when you modify the impl of the helper files

## The usage of the processFiles.py script
```
processFiles.py [-h] [--build-dir BUILD_DIR]
                [--inc-output-dir INC_OUTPUT_DIR]
                [--helper-header-output-dir HELPER_HEADER_OUTPUT_DIR]
                [--input-inc-dir INPUT_INC_DIR]
optional arguments:
  -h, --help            show this help message and exit
  --build-dir BUILD_DIR
                        The build directory.
  --inc-output-dir INC_OUTPUT_DIR
                        The path of the output inc files. These inc files will
                        be included by src code. Ignored when --build-dir
                        specified.
  --helper-header-output-dir HELPER_HEADER_OUTPUT_DIR
                        The path of the output runtime files. These files are
                        the final full set helper header files. Ignored when
                        --build-dir specified.
  --input-inc-dir INPUT_INC_DIR
                        The path of the input *.inc files. Default value is
                        dir where this script at.
```