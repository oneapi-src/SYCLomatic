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
ENTRY("atom", "atom", true, NO_FLAG, P1, "Partial")
ENTRY("bar", "bar", true, NO_FLAG, P1, "Partial")
ENTRY("barrier", "barrier", false, NO_FLAG, P1, "Comment")
ENTRY("bfe", "bfe", true, NO_FLAG, P1, "Successful")
ENTRY("bfi", "bfi", true, NO_FLAG, P1, "Successful")
ENTRY("bfind", "bfind", false, NO_FLAG, P1, "Comment")
ENTRY("bmsk", "bmsk", false, NO_FLAG, P1, "Comment")
ENTRY("bra", "bra", false, NO_FLAG, P1, "Comment")
ENTRY("brev", "brev", true, NO_FLAG, P1, "Successful")
ENTRY("brkpt", "brkpt", false, NO_FLAG, P1, "Comment")
ENTRY("brx", "brx", false, NO_FLAG, P1, "Comment")
ENTRY("call", "call", false, NO_FLAG, P1, "Comment")
ENTRY("clz", "clz", true, NO_FLAG, P1, "Successful")
ENTRY("cnot", "cnot", true, NO_FLAG, P1, "Successful")
ENTRY("copysign", "copysign", true, NO_FLAG, P1, "Successful")
ENTRY("cos", "cos", true, NO_FLAG, P1, "Successful")
ENTRY("cp", "cp", false, NO_FLAG, P1, "Comment")
ENTRY("createpolicy", "createpolicy", false, NO_FLAG, P1, "Comment")
ENTRY("cvt", "cvt", true, NO_FLAG, P1, "Partial")
ENTRY("cvta", "cvta", false, NO_FLAG, P1, "Comment")
ENTRY("discard", "discard", false, NO_FLAG, P1, "Comment")
ENTRY("div", "div", true, NO_FLAG, P1, "Partial")
ENTRY("dp2a", "dp2a", true, NO_FLAG, P1, "Successful")
ENTRY("dp4a", "dp4a", true, NO_FLAG, P1, "Successful")
ENTRY("elect", "elect", false, NO_FLAG, P1, "Comment")
ENTRY("ex2", "ex2", true, NO_FLAG, P1, "Successful")
ENTRY("exit", "exit", false, NO_FLAG, P1, "Comment")
ENTRY("fence", "fence", false, NO_FLAG, P1, "Comment")
ENTRY("fma", "fma", true, NO_FLAG, P1, "Partial")
ENTRY("fns", "fns", false, NO_FLAG, P1, "Comment")
ENTRY("getctarank", "getctarank", false, NO_FLAG, P1, "Comment")
ENTRY("griddepcontrol", "griddepcontrol", false, NO_FLAG, P1, "Comment")
ENTRY("isspacep", "isspacep", false, NO_FLAG, P1, "Comment")
ENTRY("istypep", "istypep", false, NO_FLAG, P1, "Comment")
ENTRY("ld", "ld", true, NO_FLAG, P1, "Partial")
ENTRY("ldmatrix", "ldmatrix", false, NO_FLAG, P1, "Comment")
ENTRY("ldu", "ldu", false, NO_FLAG, P1, "Comment")
ENTRY("lg2", "lg2", true, NO_FLAG, P1, "Successful")
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
ENTRY("rcp", "rcp", true, NO_FLAG, P1, "Successful")
ENTRY("red", "red", false, NO_FLAG, P1, "Comment")
ENTRY("redux", "redux", false, NO_FLAG, P1, "Comment")
ENTRY("rem", "rem", true, NO_FLAG, P1, "Successful")
ENTRY("ret", "ret", true, NO_FLAG, P1, "Successful")
ENTRY("rsqrt", "rsqrt", true, NO_FLAG, P1, "Successful")
ENTRY("sad", "sad", true, NO_FLAG, P1, "Successful")
ENTRY("selp", "selp", true, NO_FLAG, P1, "Successful")
ENTRY("set", "set", false, NO_FLAG, P1, "Comment")
ENTRY("setmaxnreg", "setmaxnreg", false, NO_FLAG, P1, "Comment")
ENTRY("setp", "setp", true, NO_FLAG, P1, "Successful")
ENTRY("shf", "shf", false, NO_FLAG, P1, "Comment")
ENTRY("shfl", "shfl", true, NO_FLAG, P1, "Comment")
ENTRY("shl", "shl", true, NO_FLAG, P1, "Successful")
ENTRY("shr", "shr", true, NO_FLAG, P1, "Successful")
ENTRY("sin", "sin", true, NO_FLAG, P1, "Successful")
ENTRY("slct", "slct", true, NO_FLAG, P1, "Partial")
ENTRY("sqrt", "sqrt", true, NO_FLAG, P1, "Successful")
ENTRY("st", "st", true, NO_FLAG, P1, "Partial")
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
ENTRY("tanh", "tanh", true, NO_FLAG, P1, "Successful")
ENTRY("testp", "testp", true, NO_FLAG, P1, "Successful")
ENTRY("tex", "tex", false, NO_FLAG, P1, "Comment")
ENTRY("tld4", "tld4", false, NO_FLAG, P1, "Comment")
ENTRY("trap", "trap", false, NO_FLAG, P1, "Comment")
ENTRY("txq", "txq", false, NO_FLAG, P1, "Comment")
ENTRY("vabsdiff", "vabsdiff", true, NO_FLAG, P1, "Partial")
ENTRY("vabsdiff2", "vabsdiff2", true, NO_FLAG, P1, "Partial")
ENTRY("vabsdiff4", "vabsdiff4", true, NO_FLAG, P1, "Partial")
ENTRY("vadd", "vadd", true, NO_FLAG, P1, "Partial")
ENTRY("vadd2", "vadd2", true, NO_FLAG, P1, "Partial")
ENTRY("vadd4", "vadd4", true, NO_FLAG, P1, "Partial")
ENTRY("vavrg2", "vavrg2", true, NO_FLAG, P1, "Partial")
ENTRY("vavrg4", "vavrg4", true, NO_FLAG, P1, "Partial")
ENTRY("vmad", "vmad", false, NO_FLAG, P1, "Comment")
ENTRY("vmax", "vmax", true, NO_FLAG, P1, "Partial")
ENTRY("vmax2", "vmax2", true, NO_FLAG, P1, "Partial")
ENTRY("vmax4", "vmax4", true, NO_FLAG, P1, "Partial")
ENTRY("vmin", "vmin", true, NO_FLAG, P1, "Partial")
ENTRY("vmin2", "vmin2", true, NO_FLAG, P1, "Partial")
ENTRY("vmin4", "vmin4", true, NO_FLAG, P1, "Partial")
ENTRY("vote", "vote", false, NO_FLAG, P1, "Comment")
ENTRY("vset", "vset", true, NO_FLAG, P1, "Partial")
ENTRY("vset2", "vset2", true, NO_FLAG, P1, "Partial")
ENTRY("vset4", "vset4", true, NO_FLAG, P1, "Partial")
ENTRY("vshl", "vshl", true, NO_FLAG, P1, "Partial")
ENTRY("vshr", "vshr", true, NO_FLAG, P1, "Partial")
ENTRY("vsub", "vsub", true, NO_FLAG, P1, "Partial")
ENTRY("vsub2", "vsub2", true, NO_FLAG, P1, "Partial")
ENTRY("vsub4", "vsub4", true, NO_FLAG, P1, "Partial")
ENTRY("wgmma", "wgmma", false, NO_FLAG, P1, "Comment")
ENTRY("wmma", "wmma", false, NO_FLAG, P1, "Comment")
ENTRY("xor", "xor", true, NO_FLAG, P1, "Partial")
