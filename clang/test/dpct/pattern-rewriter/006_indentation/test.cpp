// RUN: rm -rf %t && mkdir -p %t
// RUN: pattern-rewriter %S/input.hpp -r %S/rules.yaml -o %t/output.hpp
// RUN: echo "begin" > %t/diff.txt
// RUN: diff %S/expected.hpp %t/output.hpp >> %t/diff.txt
// RUN: echo "end" >> %t/diff.txt

// CHECK: begin
// CHECK-NEXT: end
