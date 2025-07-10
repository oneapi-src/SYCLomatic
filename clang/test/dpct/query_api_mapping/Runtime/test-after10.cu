// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaStreamEndCapture | FileCheck %s -check-prefix=CUDASTREAMENDCAPTURE
// CUDASTREAMENDCAPTURE: CUDA API:
// CUDASTREAMENDCAPTURE-NEXT:   cudaStreamEndCapture(s /*cudaStream_t*/, pg /*cudaGraph_t **/);
// CUDASTREAMENDCAPTURE-NEXT: Is migrated to (with the option --use-experimental-features=graph):
// CUDASTREAMENDCAPTURE-NEXT: dpct::experimental::end_recording(s, pg);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaStreamIsCapturing | FileCheck %s -check-prefix=CUDASTREAMISCAPTURING
// CUDASTREAMISCAPTURING: CUDA API:
// CUDASTREAMISCAPTURING-NEXT:   cudaStreamIsCapturing(s /*cudaStream_t*/,
// CUDASTREAMISCAPTURING-NEXT:                         ps /* enum cudaStreamCaptureStatus **/);
// CUDASTREAMISCAPTURING-NEXT: Is migrated to (with the option --use-experimental-features=graph):
// CUDASTREAMISCAPTURING-NEXT: *ps = s->ext_oneapi_get_state();

// RUN: dpct --cuda-include-path="%cuda-path/include" -query-api-mapping=cudaLaunchHostFunc | FileCheck %s -check-prefix=CUDALAUNCHHOSTFUNC
// CUDALAUNCHHOSTFUNC: CUDA API:
// CUDALAUNCHHOSTFUNC-NEXT:   cudaLaunchHostFunc(stream/*cudaStream_t*/, fn/*cudaHostFn_t*/, userData/*void**/);
// CUDALAUNCHHOSTFUNC-NEXT: Is migrated to: 
// CUDALAUNCHHOSTFUNC-NEXT:   stream->submit([&](sycl::handler &cgh) {
// CUDALAUNCHHOSTFUNC-NEXT:     cgh.host_task([=](){
// CUDALAUNCHHOSTFUNC-NEXT:       fn(userData);
// CUDALAUNCHHOSTFUNC-NEXT:     });
// CUDALAUNCHHOSTFUNC-NEXT:   });


// RUN: dpct --cuda-include-path="%cuda-path/include" -query-api-mapping=cudaGraphExecDestroy | FileCheck %s -check-prefix=cudaGraphExecDestroy
// cudaGraphExecDestroy: CUDA API:
// cudaGraphExecDestroy-NEXT:   cudaGraphExecDestroy(graph_exec /*cudaGraphExec_t*/);
// cudaGraphExecDestroy-NEXT: Is migrated to (with the option --use-experimental-features=graph):
// cudaGraphExecDestroy-NEXT:   delete (graph_exec);

// RUN: dpct --cuda-include-path="%cuda-path/include" -query-api-mapping=cudaGraphGetNodes | FileCheck %s -check-prefix=cudaGraphGetNodes
// cudaGraphGetNodes: CUDA API:
// cudaGraphGetNodes-NEXT:   cudaGraphGetNodes(graph /*cudaGraph_t*/, nodes /*cudaGraphNode_t **/,
// cudaGraphGetNodes-NEXT:                     num_nodes /*size_t **/);
// cudaGraphGetNodes-NEXT: Is migrated to (with the option --use-experimental-features=graph):
// cudaGraphGetNodes-NEXT:   dpct::experimental::get_nodes(graph, nodes, num_nodes);

// RUN: dpct --cuda-include-path="%cuda-path/include" -query-api-mapping=cudaGraphGetRootNodes | FileCheck %s -check-prefix=cudaGraphGetRootNodes
// cudaGraphGetRootNodes: CUDA API:
// cudaGraphGetRootNodes-NEXT:   cudaGraphGetRootNodes(graph /*cudaGraph_t*/, nodes /*cudaGraphNode_t **/,
// cudaGraphGetRootNodes-NEXT:                         num_nodes /*size_t **/);
// cudaGraphGetRootNodes-NEXT: Is migrated to (with the option --use-experimental-features=graph):
// cudaGraphGetRootNodes-NEXT:   dpct::experimental::get_root_nodes(graph, nodes, num_nodes);

// RUN: dpct --cuda-include-path="%cuda-path/include" -query-api-mapping=cudaGraphInstantiate | FileCheck %s -check-prefix=cudaGraphInstantiate
// cudaGraphInstantiate: CUDA API:
// cudaGraphInstantiate-NEXT:   cudaGraphInstantiate(graph_exec /*cudaGraphExec_t **/, graph /*cudaGraph_t*/,
// cudaGraphInstantiate-NEXT:                        node /*cudaGraphNode_t **/, buffer /*char **/,
// cudaGraphInstantiate-NEXT:                        buffer_size /*size_t*/);
// cudaGraphInstantiate-NEXT: Is migrated to (with the option --use-experimental-features=graph):
// cudaGraphInstantiate-NEXT:   *graph_exec = new sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>(graph->finalize());

// RUN: dpct --cuda-include-path="%cuda-path/include" -query-api-mapping=cudaGraphLaunch | FileCheck %s -check-prefix=cudaGraphLaunch
// cudaGraphLaunch: CUDA API:
// cudaGraphLaunch-NEXT:   cudaGraphLaunch(graph_exec /*cudaGraphExec_t*/, stream /*cudaStream_t*/);
// cudaGraphLaunch-NEXT: Is migrated to (with the option --use-experimental-features=graph):
// cudaGraphLaunch-NEXT:   stream->ext_oneapi_graph(*graph_exec);

// RUN: dpct --cuda-include-path="%cuda-path/include" -query-api-mapping=cudaGraphNodeGetType | FileCheck %s -check-prefix=cudaGraphNodeGetType
// cudaGraphNodeGetType: CUDA API:
// cudaGraphNodeGetType-NEXT:   cudaGraphNodeGetType(node /*cudaGraphNode_t*/,
// cudaGraphNodeGetType-NEXT:                        node_type /*cudaGraphNodeType **/);
// cudaGraphNodeGetType-NEXT: Is migrated to (with the option --use-experimental-features=graph):
// cudaGraphNodeGetType-NEXT:   *node_type = node->get_type();

// RUN: dpct --cuda-include-path="%cuda-path/include" -query-api-mapping=surf1Dread | FileCheck %s -check-prefix=surf1Dread
// surf1Dread: CUDA API:
// surf1Dread-NEXT:   surf1Dread<T>(obj /*cudaSurfaceObject_t*/, x /*int*/,
// surf1Dread-NEXT:                 boundary_mode /*cudaSurfaceBoundaryMode*/);
// surf1Dread-NEXT:   surf1Dread(ptr /*T **/, obj /*cudaSurfaceObject_t*/, x /*int*/,
// surf1Dread-NEXT:              boundary_mode /*cudaSurfaceBoundaryMode*/);
// surf1Dread-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// surf1Dread-NEXT:   dpct::experimental::fetch_image_by_byte<T>(obj, int(x));
// surf1Dread-NEXT:   *ptr = dpct::experimental::fetch_image_by_byte<T>(obj, int(x));

// RUN: dpct --cuda-include-path="%cuda-path/include" -query-api-mapping=surf1Dwrite | FileCheck %s -check-prefix=surf1Dwrite
// surf1Dwrite: CUDA API:
// surf1Dwrite-NEXT:   surf1Dwrite(val /*T*/, obj /*cudaSurfaceObject_t*/, x /*int*/,
// surf1Dwrite-NEXT:               boundary_mode /*cudaSurfaceBoundaryMode*/);
// surf1Dwrite-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// surf1Dwrite-NEXT:   sycl::ext::oneapi::experimental::write_image(obj, int(x / sizeof(val)), val);

// RUN: dpct --cuda-include-path="%cuda-path/include" -query-api-mapping=tex1DLayered | FileCheck %s -check-prefix=tex1DLayered
// tex1DLayered: CUDA API:
// tex1DLayered-NEXT:   tex1DLayered<T>(tex /*cudaTextureObject_t*/, x /*float*/, layer /*int*/);
// tex1DLayered-NEXT:   tex1DLayered(ptr /*T **/, tex /*cudaTextureObject_t*/, x /*float*/,
// tex1DLayered-NEXT:                layer /*int*/);
// tex1DLayered-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// tex1DLayered-NEXT:   sycl::ext::oneapi::experimental::sample_image_array<T>(tex, float(x), layer);
// tex1DLayered-NEXT:   *ptr = sycl::ext::oneapi::experimental::sample_image_array<T>(tex, float(x), layer);

// RUN: dpct --cuda-include-path="%cuda-path/include" -query-api-mapping=tex2DLayered | FileCheck %s -check-prefix=tex2DLayered
// tex2DLayered: CUDA API:
// tex2DLayered-NEXT:   tex2DLayered<T>(tex /*cudaTextureObject_t*/, x /*float*/, y /*float*/,
// tex2DLayered-NEXT:                   layer /*int*/);
// tex2DLayered-NEXT:   tex2DLayered(ptr /*T **/, tex /*cudaTextureObject_t*/, x /*float*/,
// tex2DLayered-NEXT:                y /*float*/, layer /*int*/);
// tex2DLayered-NEXT:   tex2DLayered<T>(tex /*cudaTextureObject_t*/, x /*float*/, y /*float*/,
// tex2DLayered-NEXT:                   layer /*int*/, is_resident /*bool **/);
// tex2DLayered-NEXT:   tex2DLayered(ptr /*T **/, tex /*cudaTextureObject_t*/, x /*float*/,
// tex2DLayered-NEXT:                y /*float*/, layer /*int*/, is_resident /*bool **/);
// tex2DLayered-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// tex2DLayered-NEXT:   sycl::ext::oneapi::experimental::sample_image_array<T>(tex, sycl::float2(x, y), layer);
// tex2DLayered-NEXT:   *ptr = sycl::ext::oneapi::experimental::sample_image_array<T>(tex, sycl::float2(x, y), layer);
// tex2DLayered-NEXT:   sycl::ext::oneapi::experimental::sample_image_array<T>(tex, sycl::float2(x, y), is_resident);
// tex2DLayered-NEXT:   *ptr = sycl::ext::oneapi::experimental::sample_image_array<T>(tex, sycl::float2(x, y), is_resident);
