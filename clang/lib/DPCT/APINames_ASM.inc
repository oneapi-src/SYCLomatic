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

ENTRY("abs", "abs", true, NO_FLAG, P1, "Successful")
ENTRY("activemask", "activemask", false, NO_FLAG, P1, "Comment")
ENTRY("add", "add", true, NO_FLAG, P1, "Partial")
ENTRY("addc", "addc", false, NO_FLAG, P1, "Comment")
ENTRY("alloca", "alloca", false, NO_FLAG, P1, "Comment")
ENTRY("and", "and", true, NO_FLAG, P1, "Successful")
ENTRY("applypriority", "applypriority", false, NO_FLAG, P1, "Comment")
ENTRY("atom", "atom", false, NO_FLAG, P1, "Comment")
ENTRY("bar", "bar", false, NO_FLAG, P1, "Comment")
ENTRY("barrier", "barrier", false, NO_FLAG, P1, "Comment")
ENTRY("bfe", "bfe", false, NO_FLAG, P1, "Comment")
ENTRY("bfi", "bfi", false, NO_FLAG, P1, "Comment")
ENTRY("bfind", "bfind", false, NO_FLAG, P1, "Comment")
ENTRY("bmsk", "bmsk", false, NO_FLAG, P1, "Comment")
ENTRY("bra", "bra", false, NO_FLAG, P1, "Comment")
ENTRY("brev", "brev", false, NO_FLAG, P1, "Comment")
ENTRY("brkpt", "brkpt", false, NO_FLAG, P1, "Comment")
ENTRY("brx", "brx", false, NO_FLAG, P1, "Comment")
ENTRY("call", "call", false, NO_FLAG, P1, "Comment")
ENTRY("clz", "clz", true, NO_FLAG, P1, "Successful")
ENTRY("cnot", "cnot", true, NO_FLAG, P1, "Successful")
ENTRY("copysign", "copysign", false, NO_FLAG, P1, "Comment")
ENTRY("cos", "cos", false, NO_FLAG, P1, "Comment")
ENTRY("cp", "cp", false, NO_FLAG, P1, "Comment")
ENTRY("createpolicy", "createpolicy", false, NO_FLAG, P1, "Comment")
ENTRY("cvt", "cvt", false, NO_FLAG, P1, "Comment")
ENTRY("cvta", "cvta", false, NO_FLAG, P1, "Comment")
ENTRY("discard", "discard", false, NO_FLAG, P1, "Comment")
ENTRY("div", "div", true, NO_FLAG, P1, "Partial")
ENTRY("dp2a", "dp2a", false, NO_FLAG, P1, "Comment")
ENTRY("dp4a", "dp4a", false, NO_FLAG, P1, "Comment")
ENTRY("elect", "elect", false, NO_FLAG, P1, "Comment")
ENTRY("ex2", "ex2", false, NO_FLAG, P1, "Comment")
ENTRY("exit", "exit", false, NO_FLAG, P1, "Comment")
ENTRY("fence", "fence", false, NO_FLAG, P1, "Comment")
ENTRY("fma", "fma", false, NO_FLAG, P1, "Comment")
ENTRY("fns", "fns", false, NO_FLAG, P1, "Comment")
ENTRY("getctarank", "getctarank", false, NO_FLAG, P1, "Comment")
ENTRY("griddepcontrol", "griddepcontrol", false, NO_FLAG, P1, "Comment")
ENTRY("isspacep", "isspacep", false, NO_FLAG, P1, "Comment")
ENTRY("istypep", "istypep", false, NO_FLAG, P1, "Comment")
ENTRY("ld", "ld", false, NO_FLAG, P1, "Comment")
ENTRY("ldmatrix", "ldmatrix", false, NO_FLAG, P1, "Comment")
ENTRY("ldu", "ldu", false, NO_FLAG, P1, "Comment")
ENTRY("lg2", "lg2", false, NO_FLAG, P1, "Comment")
ENTRY("lop3", "lop3", true, NO_FLAG, P1, "Successful")
ENTRY("mad", "mad", true, NO_FLAG, P1, "Partial")
ENTRY("mad24", "mad24", true, NO_FLAG, P1, "Partial")
ENTRY("madc", "madc", false, NO_FLAG, P1, "Comment")
ENTRY("mapa", "mapa", false, NO_FLAG, P1, "Comment")
ENTRY("match", "match", false, NO_FLAG, P1, "Comment")
ENTRY("max", "max", true, NO_FLAG, P1, "Successful")
ENTRY("mbarrier", "mbarrier", false, NO_FLAG, P1, "Comment")
ENTRY("membar", "membar", false, NO_FLAG, P1, "Comment")
ENTRY("min", "min", true, NO_FLAG, P1, "Successful")
ENTRY("mma", "mma", false, NO_FLAG, P1, "Comment")
ENTRY("mov", "mov", true, NO_FLAG, P1, "Successful")
ENTRY("movmatrix", "movmatrix", false, NO_FLAG, P1, "Comment")
ENTRY("mul", "mul", true, NO_FLAG, P1, "Partial")
ENTRY("mul24", "mul24", true, NO_FLAG, P1, "Partial")
ENTRY("multimem", "multimem", false, NO_FLAG, P1, "Comment")
ENTRY("nanosleep", "nanosleep", false, NO_FLAG, P1, "Comment")
ENTRY("neg", "neg", true, NO_FLAG, P1, "Successful")
ENTRY("not", "not", true, NO_FLAG, P1, "Successful")
ENTRY("or", "or", true, NO_FLAG, P1, "Successful")
ENTRY("pmevent", "pmevent", false, NO_FLAG, P1, "Comment")
ENTRY("popc", "popc", true, NO_FLAG, P1, "Successful")
ENTRY("prefetch", "prefetch", false, NO_FLAG, P1, "Comment")
ENTRY("prefetchu", "prefetchu", false, NO_FLAG, P1, "Comment")
ENTRY("prmt", "prmt", false, NO_FLAG, P1, "Comment")
ENTRY("rcp", "rcp", false, NO_FLAG, P1, "Comment")
ENTRY("red", "red", false, NO_FLAG, P1, "Comment")
ENTRY("redux", "redux", false, NO_FLAG, P1, "Comment")
ENTRY("rem", "rem", true, NO_FLAG, P1, "Successful")
ENTRY("ret", "ret", false, NO_FLAG, P1, "Comment")
ENTRY("rsqrt", "rsqrt", false, NO_FLAG, P1, "Comment")
ENTRY("sad", "sad", false, NO_FLAG, P1, "Comment")
ENTRY("selp", "selp", false, NO_FLAG, P1, "Comment")
ENTRY("set", "set", false, NO_FLAG, P1, "Comment")
ENTRY("setmaxnreg", "setmaxnreg", false, NO_FLAG, P1, "Comment")
ENTRY("setp", "setp", true, NO_FLAG, P1, "Successful")
ENTRY("shf", "shf", false, NO_FLAG, P1, "Comment")
ENTRY("shfl", "shfl", false, NO_FLAG, P1, "Comment")
ENTRY("shl", "shl", true, NO_FLAG, P1, "Successful")
ENTRY("shr", "shr", true, NO_FLAG, P1, "Successful")
ENTRY("sin", "sin", false, NO_FLAG, P1, "Comment")
ENTRY("slct", "slct", false, NO_FLAG, P1, "Comment")
ENTRY("sqrt", "sqrt", false, NO_FLAG, P1, "Comment")
ENTRY("st", "st", false, NO_FLAG, P1, "Comment")
ENTRY("stackrestore", "stackrestore", false, NO_FLAG, P1, "Comment")
ENTRY("stacksave", "stacksave", false, NO_FLAG, P1, "Comment")
ENTRY("stmatrix", "stmatrix", false, NO_FLAG, P1, "Comment")
ENTRY("sub", "sub", true, NO_FLAG, P1, "Partial")
ENTRY("subc", "subc", false, NO_FLAG, P1, "Comment")
ENTRY("suld", "suld", false, NO_FLAG, P1, "Comment")
ENTRY("suq", "suq", false, NO_FLAG, P1, "Comment")
ENTRY("sured", "sured", false, NO_FLAG, P1, "Comment")
ENTRY("sust", "sust", false, NO_FLAG, P1, "Comment")
ENTRY("szext", "szext", false, NO_FLAG, P1, "Comment")
ENTRY("tanh", "tanh", false, NO_FLAG, P1, "Comment")
ENTRY("testp", "testp", false, NO_FLAG, P1, "Comment")
ENTRY("tex", "tex", false, NO_FLAG, P1, "Comment")
ENTRY("tld4", "tld4", false, NO_FLAG, P1, "Comment")
ENTRY("trap", "trap", false, NO_FLAG, P1, "Comment")
ENTRY("txq", "txq", false, NO_FLAG, P1, "Comment")
ENTRY("vabsdiff", "vabsdiff", true, NO_FLAG, P1, "Partial")
ENTRY("vabsdiff2", "vabsdiff2", false, NO_FLAG, P1, "Comment")
ENTRY("vabsdiff4", "vabsdiff4", false, NO_FLAG, P1, "Comment")
ENTRY("vadd", "vadd", true, NO_FLAG, P1, "Partial")
ENTRY("vadd2", "vadd2", false, NO_FLAG, P1, "Comment")
ENTRY("vadd4", "vadd4", false, NO_FLAG, P1, "Comment")
ENTRY("vavrg2", "vavrg2", false, NO_FLAG, P1, "Comment")
ENTRY("vavrg4", "vavrg4", false, NO_FLAG, P1, "Comment")
ENTRY("vmad", "vmad", false, NO_FLAG, P1, "Comment")
ENTRY("vmax", "vmax", true, NO_FLAG, P1, "Partial")
ENTRY("vmax2", "vmax2", false, NO_FLAG, P1, "Comment")
ENTRY("vmax4", "vmax4", false, NO_FLAG, P1, "Comment")
ENTRY("vmin", "vmin", true, NO_FLAG, P1, "Partial")
ENTRY("vmin2", "vmin2", false, NO_FLAG, P1, "Comment")
ENTRY("vmin4", "vmin4", false, NO_FLAG, P1, "Comment")
ENTRY("vote", "vote", false, NO_FLAG, P1, "Comment")
ENTRY("vset", "vset", false, NO_FLAG, P1, "Comment")
ENTRY("vset2", "vset2", false, NO_FLAG, P1, "Comment")
ENTRY("vset4", "vset4", false, NO_FLAG, P1, "Comment")
ENTRY("vshl", "vshl", false, NO_FLAG, P1, "Comment")
ENTRY("vshr", "vshr", false, NO_FLAG, P1, "Comment")
ENTRY("vsub", "vsub", true, NO_FLAG, P1, "Partial")
ENTRY("vsub2", "vsub2", false, NO_FLAG, P1, "Comment")
ENTRY("vsub4", "vsub4", false, NO_FLAG, P1, "Comment")
ENTRY("wgmma", "wgmma", false, NO_FLAG, P1, "Comment")
ENTRY("wmma", "wmma", false, NO_FLAG, P1, "Comment")
ENTRY("xor", "xor", true, NO_FLAG, P1, "Partial")
