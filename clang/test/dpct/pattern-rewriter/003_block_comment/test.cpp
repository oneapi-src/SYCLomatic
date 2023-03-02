// RUN: rm -rf %t && mkdir -p %t
// R-U-N: pattern-rewriter %S/input.hpp -r %S/rules.yaml -o %t/output.hpp
// R-U-N: echo "begin" > %t/diff.txt
// R-U-N: diff --strip-trailing-cr %S/expected.hpp %t/output.hpp >> %t/diff.txt
// R-U-N: echo "end" >> %t/diff.txt

// C-H-E-C-K: begin
// C-H-E-C-K-NEXT: end
