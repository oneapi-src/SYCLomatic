# Pre-defined rules
In this folder, the tool provides some yaml files which can support the migration in some special senarios.

## cmake_rules
`cmake_script_migration_rule.yaml`: Contains the general migration rules for CMake script migration. This file will be loaded by default if user uses dpct to migrate CMake script.

`cmake_script_migration_rule_optional.yaml`: Contains some rules for CMake script migration which are not suitable for all senarios.

## opt_rules
`forceinline.yaml`: Contains a rule to migrate `__forceinline__` to `inline` instead of `__dpct_inline__`.

`intel_specific_math.yaml`: Contains some rules to migrate some cuda math API to Intel(R) hardware specific API.

`macro_checks.yaml`: Contains a rule to migrate the error-checking macro `CUDA_CHECK` to `DPCT_CHECK_ERROR`.

## python_rules
`python_build_script_migration_rule_ipex.yaml`: Contains some rules to migrate python build script for Pytorch-based projects. The migrated python build script uses the Intel(R) Extension for PyTorch (IPEX) API.

`python_build_script_migration_rule_pytorch.yaml`: Contains some rules to migrate python build script for Pytorch-based projects. The migrated python build script uses the PyTorch XPU backend API.

## pytorch_api_rule_rules
`pytorch_api.yaml`: Contains some rules to migrate CUDA backend Pytorch API for Pytorch-based projects. The migrated code uses the PyTorch XPU backend API.

## rule_templates
This folder contains exmaples for each `Kind` of rule.
