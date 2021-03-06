; RUN: opt < %s -passes=globalopt -S | FileCheck %s
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

;; Heap SROA has been removed. This tests we don't perform heap SROA.
; CHECK: @X =
	%struct.foo = type { i32, i32 }
@X = internal global %struct.foo* null		; <%struct.foo**> [#uses=2]

define void @bar(i32 %Size) nounwind noinline {
entry:
	%malloccall = tail call i8* @malloc(i64 8000000) ; <i8*> [#uses=1]
	%0 = bitcast i8* %malloccall to [1000000 x %struct.foo]* ; <[1000000 x %struct.foo]*> [#uses=1]
	%.sub = getelementptr [1000000 x %struct.foo], [1000000 x %struct.foo]* %0, i32 0, i32 0		; <%struct.foo*> [#uses=1]
	store %struct.foo* %.sub, %struct.foo** @X, align 4
	ret void
}

declare noalias i8* @malloc(i64)

define i32 @baz() nounwind readonly noinline {
bb1.thread:
	%0 = load %struct.foo*, %struct.foo** @X, align 4		; <%struct.foo*> [#uses=1]
	br label %bb1

bb1:		; preds = %bb1, %bb1.thread
	%i.0.reg2mem.0 = phi i32 [ 0, %bb1.thread ], [ %indvar.next, %bb1 ]		; <i32> [#uses=2]
	%sum.0.reg2mem.0 = phi i32 [ 0, %bb1.thread ], [ %3, %bb1 ]		; <i32> [#uses=1]
	%1 = getelementptr %struct.foo, %struct.foo* %0, i32 %i.0.reg2mem.0, i32 0		; <i32*> [#uses=1]
	%2 = load i32, i32* %1, align 4		; <i32> [#uses=1]
	%3 = add i32 %2, %sum.0.reg2mem.0		; <i32> [#uses=2]
	%indvar.next = add i32 %i.0.reg2mem.0, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %indvar.next, 1200		; <i1> [#uses=1]
	br i1 %exitcond, label %bb2, label %bb1

bb2:		; preds = %bb1
	ret i32 %3
}

define void @bam(i64 %Size) nounwind noinline #0 {
entry:
        %0 = load %struct.foo*, %struct.foo** @X, align 4
        ret void
}

attributes #0 = { null_pointer_is_valid }
