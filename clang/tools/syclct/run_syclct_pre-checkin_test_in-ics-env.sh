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
mkdir  $tc_result_folder
cd  $tc_result_folder

#old test command
#tc -r none -x gcc_efi2 -t syclct_sdk_samples/vectorAdd,topologyQuery,cudaOpenMP,cppIntegration,simpleAssert,alignedTypes,simpleMPI,simplePrintf

#Run test case from syclct_sdk_8_samples
tc -r none -x gcc_efi2 -t syclct_sdk_8_samples/vectorAdd,topologyQuery,cudaOpenMP,cppIntegration,simpleAssert,alignedTypes,simpleMPI,simplePrintf

#Run test case from syclct_sdk_92_samples
#tc -r none -x gcc_efi2 -t syclct_sdk_92_samples/vectorAdd,topologyQuery,cudaOpenMP,cppIntegration,simpleAssert,alignedTypes,simpleMPI,simplePrintf

#Run test case from syclct_sdk_10_samples
#tc -r none -x gcc_efi2 -t syclct_sdk_10_samples/vectorAdd,topologyQuery,cudaOpenMP,cppIntegration,simpleAssert,alignedTypes,simpleMPI,simplePrintf

