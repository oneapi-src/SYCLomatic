; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mcpu=pwr7 -ppc-asm-full-reg-names -verify-machineinstrs \
; RUN:   -mtriple=powerpc64-unknown-unknown < %s | FileCheck %s \
; RUN:   -check-prefix=P7BE
; RUN: llc -mcpu=pwr8 -ppc-asm-full-reg-names -verify-machineinstrs \
; RUN:   -mtriple=powerpc64le-unknown-unknown < %s | FileCheck %s \
; RUN:   -check-prefix=P8LE
; RUN: llc -mcpu=pwr9 -ppc-asm-full-reg-names -verify-machineinstrs \
; RUN:   -mtriple=powerpc64le-unknown-unknown < %s | FileCheck %s \
; RUN:   -check-prefix=P9LE

; FIXME: P7BE for i128 looks wrong.
define <1 x i128> @One1i128() {
; P7BE-LABEL: One1i128:
; P7BE:       # %bb.0: # %entry
; P7BE-NEXT:    li r3, -1
; P7BE-NEXT:    li r4, -1
; P7BE-NEXT:    blr
;
; P8LE-LABEL: One1i128:
; P8LE:       # %bb.0: # %entry
; P8LE-NEXT:    xxleqv vs34, vs34, vs34
; P8LE-NEXT:    blr
;
; P9LE-LABEL: One1i128:
; P9LE:       # %bb.0: # %entry
; P9LE-NEXT:    xxleqv vs34, vs34, vs34
; P9LE-NEXT:    blr
entry:
  ret <1 x i128> <i128 -1>
}

define <2 x i64> @One2i64() {
; P7BE-LABEL: One2i64:
; P7BE:       # %bb.0: # %entry
; P7BE-NEXT:    vspltisb v2, -1
; P7BE-NEXT:    blr
;
; P8LE-LABEL: One2i64:
; P8LE:       # %bb.0: # %entry
; P8LE-NEXT:    xxleqv vs34, vs34, vs34
; P8LE-NEXT:    blr
;
; P9LE-LABEL: One2i64:
; P9LE:       # %bb.0: # %entry
; P9LE-NEXT:    xxleqv vs34, vs34, vs34
; P9LE-NEXT:    blr
entry:
  ret <2 x i64> <i64 -1, i64 -1>
}

define <4 x i32> @One4i32() {
; P7BE-LABEL: One4i32:
; P7BE:       # %bb.0: # %entry
; P7BE-NEXT:    vspltisb v2, -1
; P7BE-NEXT:    blr
;
; P8LE-LABEL: One4i32:
; P8LE:       # %bb.0: # %entry
; P8LE-NEXT:    xxleqv vs34, vs34, vs34
; P8LE-NEXT:    blr
;
; P9LE-LABEL: One4i32:
; P9LE:       # %bb.0: # %entry
; P9LE-NEXT:    xxleqv vs34, vs34, vs34
; P9LE-NEXT:    blr
entry:
  ret <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>
}

define <8 x i16> @One8i16() {
; P7BE-LABEL: One8i16:
; P7BE:       # %bb.0: # %entry
; P7BE-NEXT:    vspltisb v2, -1
; P7BE-NEXT:    blr
;
; P8LE-LABEL: One8i16:
; P8LE:       # %bb.0: # %entry
; P8LE-NEXT:    xxleqv vs34, vs34, vs34
; P8LE-NEXT:    blr
;
; P9LE-LABEL: One8i16:
; P9LE:       # %bb.0: # %entry
; P9LE-NEXT:    xxleqv vs34, vs34, vs34
; P9LE-NEXT:    blr
entry:
  ret <8 x i16> <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
}

define <16 x i8> @One16i8() {
; P7BE-LABEL: One16i8:
; P7BE:       # %bb.0: # %entry
; P7BE-NEXT:    vspltisb v2, -1
; P7BE-NEXT:    blr
;
; P8LE-LABEL: One16i8:
; P8LE:       # %bb.0: # %entry
; P8LE-NEXT:    xxleqv vs34, vs34, vs34
; P8LE-NEXT:    blr
;
; P9LE-LABEL: One16i8:
; P9LE:       # %bb.0: # %entry
; P9LE-NEXT:    xxleqv vs34, vs34, vs34
; P9LE-NEXT:    blr
entry:
  ret <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
}
