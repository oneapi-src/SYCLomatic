// RUN: dpct -out-root %T/miscellaneus_apis %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --match-full-lines --input-file %T/miscellaneus_apis/miscellaneus_apis.dp.cpp %s
// RUN: %if build_lit %{icpx -c -fsycl %T/miscellaneus_apis/miscellaneus_apis.dp.cpp -o %T/miscellaneus_apis/miscellaneus_apis.dp.o %}

void foo(const void **table, CUuuid *tableId) {
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cuGetExportTable was removed because the functionality is unknown for the undocumented API.
  // CHECK-NEXT: */
  cuGetExportTable(table, tableId);
}
