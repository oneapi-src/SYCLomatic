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

#Test the passed case: make sure all can pass with your code change.
tc -r none -x gcc_efi2 -t syclct_sdk_samples/vectorAdd,topologyQuery,cudaOpenMP,cppIntegration,simpleAssert,alignedTypes,simpleMPI,simplePrintf

#If previous commnd failed, run following command as test suite is changing.
#tc -r none -x gcc_efi2 -t syclct_sdk_8_samples/vectorAdd,topologyQuery,cudaOpenMP,cppIntegration,simpleAssert,alignedTypes,simpleMPI,simplePrintf


