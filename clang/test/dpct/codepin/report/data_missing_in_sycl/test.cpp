// RUN: cat %S/cuda.json > %T/cuda.json
// RUN: cat %S/sycl.json > %T/sycl.json
// RUN: cd %T
// RUN: codepin-report --cuda-log cuda.json --sycl-log sycl.json ||true

// RUN: cat %S/CodePin_Report_ref.csv > %T/CodePin_Report_Expected.csv
// RUN: cat %T/CodePin_Report.csv >> %T/CodePin_Report_Expected.csv

// RUN: FileCheck --match-full-lines --input-file %T/CodePin_Report_Expected.csv %T/CodePin_Report_Expected.csv
