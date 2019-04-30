#!/usr/bin/env bash

echo ""
echo "-------------------------------------------------------------------------------------"
echo " Please setup ICS env for the test!"
echo "  Ref to: https://securewiki.ith.intel.com/display/DPCPP/How+to+configure%2C+build+and+test+SYCLCT"
echo "  Detail in [Pre-checkin test] section."
echo "-------------------------------------------------------------------------------------"
echo ""

tc_result_folder=`date +%G%m%H%M%S`
tc_result_folder=tc_$tc_result_folder
mkdir -p $tc_result_folder
cd $tc_result_folder

#old test command
#tc -r none -x gcc_efi2 -t syclct_sdk_samples/vectorAdd,topologyQuery,cudaOpenMP,cppIntegration,simpleAssert,alignedTypes,simpleMPI,simplePrintf

syclct_sdk_8_samples=(
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

syclct_sdk_8_samples_list=$(IFS=, ; echo "syclct_sdk_8_samples/${syclct_sdk_8_samples[*]}")

set -x
#Run test case from syclct_sdk_8_samples
tc -r none -x gcc_efi2 -t $syclct_sdk_8_samples_list

echo "---------------------------------------------------"
#Run custom test set: syclct_internal_samples
tc -r none -x efi2_linux64_syclct -s syclct_internal_samples

echo "---------------------------------------------------"
#Run custom behavior test set: syclct_behavior_tests
tc -r none -x efi2_linux64_syclct -s syclct_behavior_tests

echo "---------------------------------------------------"
#Run syclct benchamrks.
tc -r none -x efi2_linux64_syclct -s syclct_benchmarks

echo "---------------------------------------------------"
