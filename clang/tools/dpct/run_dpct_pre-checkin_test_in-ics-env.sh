################################################################################
#
# Copyright (C) 2019 - 2020 Intel Corporation. All rights reserved.
#
# The information and source code contained herein is the exclusive property of
# Intel Corporation and may not be disclosed, examined or reproduced in whole or
# in part without explicit written authorization from the company.
#
################################################################################
#!/usr/bin/env bash

echo ""
echo "-------------------------------------------------------------------------------------"
echo " Please setup ICS env for the test!"
echo "  Ref to: https://securewiki.ith.intel.com/display/DPCPP/How+to+configure%2C+build+and+test+DPCT"
echo "  Detail in [Pre-checkin test] section."
echo "-------------------------------------------------------------------------------------"
echo ""

tc_result_folder=`date +%G%m%H%M%S`
tc_result_folder=tc_$tc_result_folder
mkdir -p $tc_result_folder
cd $tc_result_folder

#old test command
#tc -r none -x gcc_efi2 -t dpct_sdk_samples/vectorAdd,topologyQuery,cudaOpenMP,cppIntegration,simpleAssert,alignedTypes,simpleMPI,simplePrintf

dpct_sdk_8_samples=(
  clock
  vectorAdd
  topologyQuery
  cudaOpenMP
  cppIntegration
  simpleAssert
  alignedTypes
  simpleMPI
  simplePrintf
  template
  convolutionSeparable
  FDTD3d
  simpleTemplates
  simpleAtomicIntrinsics
  fastWalshTransform
  dwtHaar1D
  scalarProd
)

dpct_sdk_8_samples_list=$(IFS=, ; echo "syclct_sdk_8_samples_primary/${dpct_sdk_8_samples[*]}")

set -x
#Run test case from dpct_sdk_8_samples
tc -r none -x efi2_linux64_syclct -t $dpct_sdk_8_samples_list

echo "---------------------------------------------------"
#Run custom test set: dpct_internal_samples
tc -r none -x efi2_linux64_syclct -s syclct_internal_samples

echo "---------------------------------------------------"
#Run custom behavior test set: dpct_behavior_tests
tc -r none -x efi2_linux64_syclct -s syclct_behavior_tests

echo "---------------------------------------------------"
#Run dpct benchamrks.
tc -r none -x efi2_linux64_syclct -s syclct_benchmarks

echo "---------------------------------------------------"
