// RUN: rm -rf %t && mkdir -p %t
// RUN: pattern-rewriter %S/input.hpp -r %S/rules.yaml -o %t/output.hpp
// RUN: diff --strip-trailing-cr %S/expected.hpp %t/output.hpp >> %t/diff.txt
