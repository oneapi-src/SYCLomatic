//===--- APINames_cuRAND.inc -----------------------------------------===//
//
// Copyright (C) 2018 - 2020 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//
/****************************************************************************
*
* ENTRY(Interface APIName, APINAME, ProcessedOrNot, Flag, Priority, MigrationDesc)
*  Interface APIName: name of library Interface API available in official document
*  APIName: name of library API
*
*  ProcessedOrNot:  false|true
*    false: this API has not been processed.
*    true:  this API has been processed, need check MigrationDesc field to
            see if it has be migrated successfully.
*
*  Flag: Bit map for all kind of flag for this API.
*    default(all bits are 0): NO_FLAG
*    bit 0: API_CALL_REMOVED
*    bit 1: API_CALL_UNSUPPORTED
*  Priority: migration priority, eg. P0 - P4
*    P0:  should support in Beta release.
*    P4:  should support in Gold.
*
*  MigrationDesc: Successful[: Comment] | Unsupported: Comment
*    Successful:   this API has been migrated.
*    Unsupported:  this API can not be migrated currently.
*                  Must add Comment to explain why this API is Unsupported.
*    Comment:      extra comment on the migration.
*
****************************************************************************/
ENTRY(curandCreateGenerator, curandCreateGenerator, true, NO_FLAG, P4, "Successful")
ENTRY(curandCreateGeneratorHost, curandCreateGeneratorHost, false, NO_FLAG, P4, "comment")
ENTRY(curandCreatePoissonDistribution, curandCreatePoissonDistribution, false, NO_FLAG, P4, "comment")
ENTRY(curandDestroyDistribution, curandDestroyDistribution, false, NO_FLAG, P4, "comment")
ENTRY(curandDestroyGenerator, curandDestroyGenerator, true, NO_FLAG, P4, "Successful")
ENTRY(curandGenerate, curandGenerate, true, NO_FLAG, P4, "Successful")
ENTRY(curandGenerateLogNormal, curandGenerateLogNormal, true, NO_FLAG, P4, "Successful")
ENTRY(curandGenerateLogNormalDouble, curandGenerateLogNormalDouble, true, NO_FLAG, P4, "Successful")
ENTRY(curandGenerateLongLong, curandGenerateLongLong, true, NO_FLAG, P4, "Successful")
ENTRY(curandGenerateNormal, curandGenerateNormal, true, NO_FLAG, P4, "Successful")
ENTRY(curandGenerateNormalDouble, curandGenerateNormalDouble, true, NO_FLAG, P4, "Successful")
ENTRY(curandGeneratePoisson, curandGeneratePoisson, true, NO_FLAG, P4, "Successful")
ENTRY(curandGenerateSeeds, curandGenerateSeeds, false, NO_FLAG, P4, "comment")
ENTRY(curandGenerateUniform, curandGenerateUniform, true, NO_FLAG, P4, "Successful")
ENTRY(curandGenerateUniformDouble, curandGenerateUniformDouble, true, NO_FLAG, P4, "Successful")
ENTRY(curandGetDirectionVectors32, curandGetDirectionVectors32, false, NO_FLAG, P4, "comment")
ENTRY(curandGetDirectionVectors64, curandGetDirectionVectors64, false, NO_FLAG, P4, "comment")
ENTRY(curandGetProperty, curandGetProperty, false, NO_FLAG, P4, "comment")
ENTRY(curandGetScrambleConstants32, curandGetScrambleConstants32, false, NO_FLAG, P4, "comment")
ENTRY(curandGetScrambleConstants64, curandGetScrambleConstants64, false, NO_FLAG, P4, "comment")
ENTRY(curandGetVersion, curandGetVersion, false, NO_FLAG, P4, "comment")
ENTRY(curandSetGeneratorOffset, curandSetGeneratorOffset, true, NO_FLAG, P4, "Successful")
ENTRY(curandSetGeneratorOrdering, curandSetGeneratorOrdering, false, NO_FLAG, P4, "comment")
ENTRY(curandSetPseudoRandomGeneratorSeed, curandSetPseudoRandomGeneratorSeed, true, NO_FLAG, P4, "Successful")
ENTRY(curandSetQuasiRandomGeneratorDimensions, curandSetQuasiRandomGeneratorDimensions, true, NO_FLAG, P4, "Successful")
ENTRY(curandSetStream, curandSetStream, false, NO_FLAG, P4, "comment")
ENTRY(curandMakeMTGP32KernelState, curandMakeMTGP32KernelState, false, NO_FLAG, P4, "comment")
ENTRY(curand, curand, false, NO_FLAG, P4, "comment")
ENTRY(curand4, curand4, false, NO_FLAG, P4, "comment")
ENTRY(curand_init, curand_init, true, NO_FLAG, P4, "Successful")
ENTRY(curand_log_normal, curand_log_normal, false, NO_FLAG, P4, "comment")
ENTRY(curand_log_normal2, curand_log_normal2, false, NO_FLAG, P4, "comment")
ENTRY(curand_log_normal2_double, curand_log_normal2_double, false, NO_FLAG, P4, "comment")
ENTRY(curand_log_normal4, curand_log_normal4, false, NO_FLAG, P4, "comment")
ENTRY(curand_log_normal_double, curand_log_normal_double, false, NO_FLAG, P4, "comment")
ENTRY(curand_normal, curand_normal, false, NO_FLAG, P4, "comment")
ENTRY(curand_normal2, curand_normal2, true, NO_FLAG, P4, "Successful")
ENTRY(curand_normal2_double, curand_normal2_double, true, NO_FLAG, P4, "Successful")
ENTRY(curand_normal4, curand_normal4, false, NO_FLAG, P4, "comment")
ENTRY(curand_normal_double, curand_normal_double, true, NO_FLAG, P4, "Successful")
ENTRY(curand_poisson, curand_poisson, false, NO_FLAG, P4, "comment")
ENTRY(curand_poisson4, curand_poisson4, false, NO_FLAG, P4, "comment")
ENTRY(curand_uniform, curand_uniform, true, NO_FLAG, P4, "Successful")
ENTRY(curand_uniform2_double, curand_uniform2_double, false, NO_FLAG, P4, "comment")
ENTRY(curand_uniform4, curand_uniform4, false, NO_FLAG, P4, "comment")
ENTRY(curand_uniform_double, curand_uniform_double, false, NO_FLAG, P4, "comment")
ENTRY(skipahead, skipahead, false, NO_FLAG, P4, "comment")
ENTRY(skipahead_sequence, skipahead_sequence, false, NO_FLAG, P4, "comment")
ENTRY(skipahead_subsequence, skipahead_subsequence, false, NO_FLAG, P4, "comment")
