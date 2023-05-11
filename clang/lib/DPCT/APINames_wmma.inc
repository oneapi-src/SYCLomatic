//===--------------- APINames_wmma.inc ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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
// wmma API
ENTRY(nvcuda::wmma::fill_fragment, nvcuda::wmma::fill_fragment, false, NO_FLAG, P4, "comment")
ENTRY(nvcuda::wmma::load_matrix_sync, nvcuda::wmma::load_matrix_sync, false, NO_FLAG, P4, "comment")
ENTRY(nvcuda::wmma::store_matrix_sync, nvcuda::wmma::store_matrix_sync, false, NO_FLAG, P4, "comment")
ENTRY(nvcuda::wmma::mma_sync, nvcuda::wmma::mma_sync, false, NO_FLAG, P4, "comment")
ENTRY(nvcuda::wmma::bmma_sync, nvcuda::wmma::bmma_sync, false, NO_FLAG, P4, "comment")
