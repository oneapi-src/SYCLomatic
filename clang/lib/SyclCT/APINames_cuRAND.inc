/****************************************************************************
*
* ENTRY(APINAME,  ProcessedOrNot, Priority, MigrationDesc)
*  APIName: name of CUDA API
*
*  ProcessedOrNot:  false|true
*    false: this API has not been processed.
*    true:  this API has been processed, need check MigrationDesc field to
            see if it has be migrated successfully.
*
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
//cuRAND API call
ENTRY(curandCreateGenerator, false, P4, "comment")
ENTRY(curandCreateGeneratorHost, false, P4, "comment")
ENTRY(curandCreatePoissonDistribution, false, P4, "comment")
ENTRY(curandDestroyDistribution, false, P4, "comment")
ENTRY(curandDestroyGenerator, false, P4, "comment")
ENTRY(curandGenerate, false, P4, "comment")
ENTRY(curandGenerateLogNormal, false, P4, "comment")
ENTRY(curandGenerateLogNormalDouble, false, P4, "comment")
ENTRY(curandGenerateLongLong, false, P4, "comment")
ENTRY(curandGenerateNormal, false, P4, "comment")
ENTRY(curandGenerateNormalDouble, false, P4, "comment")
ENTRY(curandGeneratePoisson, false, P4, "comment")
ENTRY(curandGenerateSeeds, false, P4, "comment")
ENTRY(curandGenerateUniform, false, P4, "comment")
ENTRY(curandGenerateUniformDouble, false, P4, "comment")
ENTRY(curandGetDirectionVectors32, false, P4, "comment")
ENTRY(curandGetDirectionVectors64, false, P4, "comment")
ENTRY(curandGetProperty, false, P4, "comment")
ENTRY(curandGetScrambleConstants32, false, P4, "comment")
ENTRY(curandGetScrambleConstants64, false, P4, "comment")
ENTRY(curandGetVersion, false, P4, "comment")
ENTRY(curandSetGeneratorOffset, false, P4, "comment")
ENTRY(curandSetGeneratorOrdering, false, P4, "comment")
ENTRY(curandSetPseudoRandomGeneratorSeed, false, P4, "comment")
ENTRY(curandSetQuasiRandomGeneratorDimensions, false, P4, "comment")
ENTRY(curandSetStream, false, P4, "comment")
ENTRY(curandMakeMTGP32KernelState, false, P4, "comment")
ENTRY(curand, false, P4, "comment")
ENTRY(curand4, false, P4, "comment")
ENTRY(curand_init, false, P4, "comment")
ENTRY(curand_log_normal, false, P4, "comment")
ENTRY(curand_log_normal2, false, P4, "comment")
ENTRY(curand_log_normal2_double, false, P4, "comment")
ENTRY(curand_log_normal4, false, P4, "comment")
ENTRY(curand_log_normal_double, false, P4, "comment")
ENTRY(curand_normal, false, P4, "comment")
ENTRY(curand_normal2, false, P4, "comment")
ENTRY(curand_normal2_double, false, P4, "comment")
ENTRY(curand_normal4, false, P4, "comment")
ENTRY(curand_normal_double, false, P4, "comment")
ENTRY(curand_poisson, false, P4, "comment")
ENTRY(curand_poisson4, false, P4, "comment")
ENTRY(curand_uniform, false, P4, "comment")
ENTRY(curand_uniform2_double, false, P4, "comment")
ENTRY(curand_uniform4, false, P4, "comment")
ENTRY(curand_uniform_double, false, P4, "comment")
ENTRY(skipahead, false, P4, "comment")
ENTRY(skipahead_sequence, false, P4, "comment")
ENTRY(skipahead_subsequence, false, P4, "comment")
