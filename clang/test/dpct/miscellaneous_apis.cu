// RUN: dpct -out-root %T/miscellaneous_apis %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --match-full-lines --input-file %T/miscellaneous_apis/miscellaneous_apis.dp.cpp %s
// RUN: %if build_lit %{icpx -c -fsycl %T/miscellaneous_apis/miscellaneous_apis.dp.cpp -o %T/miscellaneous_apis/miscellaneous_apis.dp.o %}

void foo(const void **table, CUuuid *tableId) {
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cuGetExportTable was removed because the functionality is unknown for the undocumented API.
  // CHECK-NEXT: */
  cuGetExportTable(table, tableId);
}
