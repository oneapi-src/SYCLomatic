//===--------------- APINames_ASM.inc -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/****************************************************************************
*
* ENTRY(PTX Instruction, APINAME, ProcessedOrNot, Flag, Priority, MigrationDesc)
*  PTX Instruction: name of PTX instruction available in official document
*  APIName: name of PTX instruction
*
*  ProcessedOrNot:  false|true
*    false: this instruction has not been processed.
*    true:  this instruction has been processed, need check MigrationDesc field to
            see if it has be migrated commently.
*
*  Flag: Bit map for all kind of flag for this instruction.
*    default(all bits are 0): NO_FLAG
*    bit 0: API_CALL_REMOVED
*    bit 1: API_CALL_UNSUPPORTED
*  Priority: migration priority, eg. P0 - P4
*    P0:  should support in Beta release.
*    P4:  should support in Gold.
*
*  MigrationDesc: comment[: Comment] | Unsupported: Comment
*    comment:   this API has been migrated.
*    Unsupported:  this instruction can not be migrated currently.
*                  Must add Comment to explain why this instruction is Unsupported.
*    Comment:      extra comment on the migration.
*
****************************************************************************/

// cvt
ENTRY("cvt.rni.sat.s16.f32", "cvt.rni.sat.s16.f32", false, NO_FLAG, P1, "Comment")
ENTRY("cvt.rni.sat.s16.f64", "cvt.rni.sat.s16.f64", false, NO_FLAG, P1, "Comment")
ENTRY("cvt.rni.sat.s8.f32", "cvt.rni.sat.s8.f32", false, NO_FLAG, P1, "Comment")
ENTRY("cvt.rni.sat.s8.f64", "cvt.rni.sat.s8.f64", false, NO_FLAG, P1, "Comment")
ENTRY("cvt.rni.sat.u16.f32", "cvt.rni.sat.u16.f32", false, NO_FLAG, P1, "Comment")
ENTRY("cvt.rni.sat.u16.f64", "cvt.rni.sat.u16.f64", false, NO_FLAG, P1, "Comment")
ENTRY("cvt.rni.sat.u8.f32", "cvt.rni.sat.u8.f32", false, NO_FLAG, P1, "Comment")
ENTRY("cvt.rni.sat.u8.f64", "cvt.rni.sat.u8.f64", false, NO_FLAG, P1, "Comment")
ENTRY("cvt.sat.s16.s32", "cvt.sat.s16.s32", false, NO_FLAG, P1, "Comment")
ENTRY("cvt.sat.s16.u16", "cvt.sat.s16.u16", false, NO_FLAG, P1, "Comment")
ENTRY("cvt.sat.s16.u32", "cvt.sat.s16.u32", false, NO_FLAG, P1, "Comment")
ENTRY("cvt.sat.s32.u32", "cvt.sat.s32.u32", false, NO_FLAG, P1, "Comment")
ENTRY("cvt.sat.s8.s16", "cvt.sat.s8.s16", false, NO_FLAG, P1, "Comment")
ENTRY("cvt.sat.s8.s32", "cvt.sat.s8.s32", false, NO_FLAG, P1, "Comment")
ENTRY("cvt.sat.s8.u16", "cvt.sat.s8.u16", false, NO_FLAG, P1, "Comment")
ENTRY("cvt.sat.s8.u32", "cvt.sat.s8.u32", false, NO_FLAG, P1, "Comment")
ENTRY("cvt.sat.s8.u8", "cvt.sat.s8.u8", false, NO_FLAG, P1, "Comment")
ENTRY("cvt.sat.u16.s16", "cvt.sat.u16.s16", false, NO_FLAG, P1, "Comment")
ENTRY("cvt.sat.u16.s32", "cvt.sat.u16.s32", false, NO_FLAG, P1, "Comment")
ENTRY("cvt.sat.u16.s8", "cvt.sat.u16.s8", false, NO_FLAG, P1, "Comment")
ENTRY("cvt.sat.u16.u32", "cvt.sat.u16.u32", false, NO_FLAG, P1, "Comment")
ENTRY("cvt.sat.u32.s16", "cvt.sat.u32.s16", false, NO_FLAG, P1, "Comment")
ENTRY("cvt.sat.u32.s32", "cvt.sat.u32.s32", false, NO_FLAG, P1, "Comment")
ENTRY("cvt.sat.u32.s8", "cvt.sat.u32.s8", false, NO_FLAG, P1, "Comment")
ENTRY("cvt.sat.u8.s16", "cvt.sat.u8.s16", false, NO_FLAG, P1, "Comment")
ENTRY("cvt.sat.u8.s32", "cvt.sat.u8.s32", false, NO_FLAG, P1, "Comment")
ENTRY("cvt.sat.u8.s8", "cvt.sat.u8.s8", false, NO_FLAG, P1, "Comment")
ENTRY("cvt.sat.u8.u16", "cvt.sat.u8.u16", false, NO_FLAG, P1, "Comment")
ENTRY("cvt.sat.u8.u32", "cvt.sat.u8.u32", false, NO_FLAG, P1, "Comment")

// ld
ENTRY("ld.global.u8", "ld.global.u8", false, NO_FLAG, P1, "Comment")
ENTRY("ld.global.s8", "ld.global.s8", false, NO_FLAG, P1, "Comment")
ENTRY("ld.global.b8", "ld.global.b8", false, NO_FLAG, P1, "Comment")
ENTRY("ld.global.u16", "ld.global.u16", false, NO_FLAG, P1, "Comment")
ENTRY("ld.global.s16", "ld.global.s16", false, NO_FLAG, P1, "Comment")
ENTRY("ld.global.u32", "ld.global.u32", false, NO_FLAG, P1, "Comment")
ENTRY("ld.global.s32", "ld.global.s32", false, NO_FLAG, P1, "Comment")
ENTRY("ld.global.f32", "ld.global.f32", false, NO_FLAG, P1, "Comment")
ENTRY("ld.global.f64", "ld.global.f64", false, NO_FLAG, P1, "Comment")

// mov
ENTRY("mov.u32", "mov.u32", false, NO_FLAG, P1, "Comment")
ENTRY("mov.b32", "mov.b32", false, NO_FLAG, P1, "Comment")

// add
ENTRY("add.sat.s32", "add.sat.s32", false, NO_FLAG, P1, "Comment")

// sub
ENTRY("sub.u32", "sub.u32", false, NO_FLAG, P1, "Comment")

// mul
ENTRY("mul.ftz.f32", "mul.ftz.f32", false, NO_FLAG, P1, "Comment")

// div
ENTRY("div.approx.ftz.f32", "div.approx.ftz.f32", false, NO_FLAG, P1, "Comment")

// not
ENTRY("not.b32", "not.b32", false, NO_FLAG, P1, "Comment")

// and
ENTRY("and.b32", "and.b32", false, NO_FLAG, P1, "Comment")

// or
ENTRY("or.b32", "or.b32", false, NO_FLAG, P1, "Comment")

// xor
ENTRY("xor.b32", "xor.b32", false, NO_FLAG, P1, "Comment")

// shr
ENTRY("shr.u32", "shr.u32", false, NO_FLAG, P1, "Comment")

// shfl
ENTRY("shfl.sync.up.b32", "shfl.sync.up.b32", false, NO_FLAG, P1, "Comment")

// shf
ENTRY("shf.l.clamp.b32", "shf.l.clamp.b32", false, NO_FLAG, P1, "Comment")

// bfe
ENTRY("bfe.u32", "bfe.u32", false, NO_FLAG, P1, "Comment")

// bfi
ENTRY("bfi.b32", "bfi.b32", false, NO_FLAG, P1, "Comment")

// lop3
ENTRY("lop3.b32", "lop3.b32", true, NO_FLAG, P1, "Successful")

// prmt
ENTRY("prmt.b32", "prmt.b32", false, NO_FLAG, P1, "Comment")

// vabsdiff
ENTRY("vabsdiff.u32.u32.u32.sat", "vabsdiff.u32.u32.u32.sat", false, NO_FLAG, P1, "Comment")

// vabsdiff2
ENTRY("vabsdiff2.u32.u32.u32.sat", "vabsdiff2.u32.u32.u32.sat", false, NO_FLAG, P1, "Comment")
ENTRY("vabsdiff2.u32.u32.u32", "vabsdiff2.u32.u32.u32", false, NO_FLAG, P1, "Comment")

// vabsdiff4
ENTRY("vabsdiff4.u32.u32.u32.sat", "vabsdiff4.u32.u32.u32.sat", false, NO_FLAG, P1, "Comment")
ENTRY("vabsdiff4.u32.u32.u32", "vabsdiff4.u32.u32.u32", false, NO_FLAG, P1, "Comment")

// vadd
ENTRY("vadd.u32.u32.u32.sat", "vadd.u32.u32.u32.sat", false, NO_FLAG, P1, "Comment")

// vadd2
ENTRY("vadd2.u32.u32.u32.sat", "vadd2.u32.u32.u32.sat", false, NO_FLAG, P1, "Comment")
ENTRY("vadd2.u32.u32.u32", "vadd2.u32.u32.u32", false, NO_FLAG, P1, "Comment")
ENTRY("vadd2.s32.s32.s32", "vadd2.s32.s32.s32", false, NO_FLAG, P1, "Comment")

// vadd4
ENTRY("vadd4.u32.u32.u32.sat", "vadd4.u32.u32.u32.sat", false, NO_FLAG, P1, "Comment")
ENTRY("vadd4.u32.u32.u32", "vadd4.u32.u32.u32", false, NO_FLAG, P1, "Comment")
ENTRY("vadd4.sat.s32.s32.s32", "vadd4.sat.s32.s32.s32", false, NO_FLAG, P1, "Comment")

// vavrg2
ENTRY("vavrg2.u32.u32.u32", "vavrg2.u32.u32.u32", false, NO_FLAG, P1, "Comment")
ENTRY("vavrg2.s32.s32.s32", "vavrg2.s32.s32.s32", false, NO_FLAG, P1, "Comment")

// vavrg4
ENTRY("vavrg4.u32.u32.u32", "vavrg4.u32.u32.u32", false, NO_FLAG, P1, "Comment")
ENTRY("vavrg4.s32.s32.s32", "vavrg4.s32.s32.s32", false, NO_FLAG, P1, "Comment")

// vmax
ENTRY("vmax.u32.u32.u32", "vmax.u32.u32.u32", false, NO_FLAG, P1, "Comment")
ENTRY("vmax.s32.s32.s32", "vmax.s32.s32.s32", false, NO_FLAG, P1, "Comment")

// vmax2
ENTRY("vmax2.u32.u32.u32", "vmax2.u32.u32.u32", false, NO_FLAG, P1, "Comment")

// vmax4
ENTRY("vmax4.u32.u32.u32", "vmax4.u32.u32.u32", false, NO_FLAG, P1, "Comment")

// vmin
ENTRY("vmin.u32.u32.u32", "vmin.u32.u32.u32", false, NO_FLAG, P1, "Comment")
ENTRY("vmin.s32.s32.s32", "vmin.s32.s32.s32", false, NO_FLAG, P1, "Comment")

// vmin2
ENTRY("vmin2.u32.u32.u32", "vmin2.u32.u32.u32", false, NO_FLAG, P1, "Comment")

// vmin4
ENTRY("vmin4.u32.u32.u32", "vmin4.u32.u32.u32", false, NO_FLAG, P1, "Comment")

// vset2
ENTRY("vset2.u32.u32.eq", "vset2.u32.u32.eq", false, NO_FLAG, P1, "Comment")
ENTRY("vset2.u32.u32.ge", "vset2.u32.u32.ge", false, NO_FLAG, P1, "Comment")
ENTRY("vset2.u32.u32.gt", "vset2.u32.u32.gt", false, NO_FLAG, P1, "Comment")
ENTRY("vset2.u32.u32.le", "vset2.u32.u32.le", false, NO_FLAG, P1, "Comment")
ENTRY("vset2.u32.u32.lt", "vset2.u32.u32.lt", false, NO_FLAG, P1, "Comment")
ENTRY("vset2.u32.u32.ne", "vset2.u32.u32.ne", false, NO_FLAG, P1, "Comment")
ENTRY("vset2.u32.u32", "vset2.u32.u32", false, NO_FLAG, P1, "Comment")

// vset4
ENTRY("vset4.u32.u32.eq", "vset4.u32.u32.eq", false, NO_FLAG, P1, "Comment")
ENTRY("vset4.u32.u32.ge", "vset4.u32.u32.ge", false, NO_FLAG, P1, "Comment")
ENTRY("vset4.u32.u32.gt", "vset4.u32.u32.gt", false, NO_FLAG, P1, "Comment")
ENTRY("vset4.u32.u32.le", "vset4.u32.u32.le", false, NO_FLAG, P1, "Comment")
ENTRY("vset4.u32.u32.lt", "vset4.u32.u32.lt", false, NO_FLAG, P1, "Comment")
ENTRY("vset4.u32.u32.ne", "vset4.u32.u32.ne", false, NO_FLAG, P1, "Comment")

// vsub
ENTRY("vsub.u32.u32.u32.sat", "vsub.u32.u32.u32.sat", false, NO_FLAG, P1, "Comment")

// vsub2
ENTRY("vsub2.u32.u32.u32.sat", "vsub2.u32.u32.u32.sat", false, NO_FLAG, P1, "Comment")
ENTRY("vsub2.u32.u32.u32", "vsub2.u32.u32.u32", false, NO_FLAG, P1, "Comment")

// vsub4
ENTRY("vsub4.u32.u32.u32.sat", "vsub4.u32.u32.u32.sat", false, NO_FLAG, P1, "Comment")

