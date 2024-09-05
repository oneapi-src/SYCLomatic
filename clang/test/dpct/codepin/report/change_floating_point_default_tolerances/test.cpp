// UNSUPPORTED: system-windows
// RUN: cat %S/cuda.json > %T/cuda.json
// RUN: cat %S/sycl.json > %T/sycl.json
// RUN: cat %S/tolerances.json> %T/tolerances.json
// RUN: cd %T
// RUN: dpct --codepin-report --instrumented-cuda-log cuda.json --instrumented-sycl-log sycl.json --floating-point-comparison-epsilon=tolerances.json  || true

// RUN: cat %S/CodePin_Report_ref.csv > %T/CodePin_Report_Expected.csv
// RUN: cat %T/CodePin_Report.csv >> %T/CodePin_Report_Expected.csv

// RUN: FileCheck --match-full-lines --input-file %T/CodePin_Report_Expected.csv %T/CodePin_Report_Expected.csv
