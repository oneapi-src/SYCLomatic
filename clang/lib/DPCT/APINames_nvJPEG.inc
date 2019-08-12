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
//nvJPEG call api
ENTRY(nvjpegGetProperty, false, P4, "comment")
ENTRY(nvjpegCreate, false, P4, "comment")
ENTRY(nvjpegCreateSimple, false, P4, "comment")
ENTRY(nvjpegCreateEx, false, P4, "comment")
ENTRY(nvjpegDestroy, false, P4, "comment")
ENTRY(nvjpegSetDeviceMemoryPadding, false, P4, "comment")
ENTRY(nvjpegGetDeviceMemoryPadding, false, P4, "comment")
ENTRY(nvjpegSetPinnedMemoryPadding, false, P4, "comment")
ENTRY(nvjpegGetPinnedMemoryPadding, false, P4, "comment")
ENTRY(nvjpegJpegStateCreate, false, P4, "comment")
ENTRY(nvjpegJpegStateDestroy, false, P4, "comment")
ENTRY(nvjpegGetImageInfo, false, P4, "comment")
ENTRY(nvjpegDecode, false, P4, "comment")
ENTRY(nvjpegDecodeBatchedInitialize, false, P4, "comment")
ENTRY(nvjpegDecodeBatched, false, P4, "comment")
ENTRY(nvjpegDecodePhaseOne, false, P4, "comment")
ENTRY(nvjpegDecodePhaseTwo, false, P4, "comment")
ENTRY(nvjpegDecodePhaseThree, false, P4, "comment")
ENTRY(nvjpegDecodeBatchedPhaseOne, false, P4, "comment")
ENTRY(nvjpegDecodeBatchedPhaseTwo, false, P4, "comment")
ENTRY(nvjpegDecodeBatchedPhaseThree, false, P4, "comment")
