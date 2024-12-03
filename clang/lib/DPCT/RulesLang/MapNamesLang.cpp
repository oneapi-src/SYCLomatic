//===--------------- MapNamesLang.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MapNamesLang.h"
#include "RuleInfra/MapNames.h"
#include "RulesLang/RulesLang.h"

using namespace clang;
using namespace clang::dpct;

namespace clang {
namespace dpct {

std::unordered_map<std::string, std::string> MapNamesLang::AtomicFuncNamesMap;
std::unordered_map<std::string, std::pair<std::string, std::string>>
    MapNamesLang::MathTypeCastingMap;

void MapNamesLang::setExplicitNamespaceMap(
    const std::set<ExplicitNamespace> &ExplicitNamespaces) {
  MathTypeCastingMap = {
      {"__half_as_short",
       {"short", MapNames::getClNamespace(false, true) + "half"}},
      {"__half_as_ushort",
       {"unsigned short", MapNames::getClNamespace(false, true) + "half"}},
      {"__short_as_half",
       {MapNames::getClNamespace(false, true) + "half", "short"}},
      {"__ushort_as_half",
       {MapNames::getClNamespace(false, true) + "half", "unsigned short"}},
      {"__double_as_longlong", {"long long", "double"}},
      {"__float_as_int", {"int", "float"}},
      {"__float_as_uint", {"unsigned int", "float"}},
      {"__int_as_float", {"float", "int"}},
      {"__longlong_as_double", {"double", "long long"}},
      {"__uint_as_float", {"float", "unsigned int"}}};

  // Atomic function names mapping
  AtomicFuncNamesMap = {
      {"atomicAdd", MapNames::getDpctNamespace() + "atomic_fetch_add"},
      {"atomicAdd_system", MapNames::getDpctNamespace() + "atomic_fetch_add"},
      {"atomicSub", MapNames::getDpctNamespace() + "atomic_fetch_sub"},
      {"atomicSub_system", MapNames::getDpctNamespace() + "atomic_fetch_sub"},
      {"atomicAnd", MapNames::getDpctNamespace() + "atomic_fetch_and"},
      {"atomicAnd_system", MapNames::getDpctNamespace() + "atomic_fetch_and"},
      {"atomicOr", MapNames::getDpctNamespace() + "atomic_fetch_or"},
      {"atomicOr_system", MapNames::getDpctNamespace() + "atomic_fetch_or"},
      {"atomicXor", MapNames::getDpctNamespace() + "atomic_fetch_xor"},
      {"atomicXor_system", MapNames::getDpctNamespace() + "atomic_fetch_xor"},
      {"atomicMin", MapNames::getDpctNamespace() + "atomic_fetch_min"},
      {"atomicMin_system", MapNames::getDpctNamespace() + "atomic_fetch_min"},
      {"atomicMax", MapNames::getDpctNamespace() + "atomic_fetch_max"},
      {"atomicMax_system", MapNames::getDpctNamespace() + "atomic_fetch_max"},
      {"atomicExch", MapNames::getDpctNamespace() + "atomic_exchange"},
      {"atomicExch_system", MapNames::getDpctNamespace() + "atomic_exchange"},
      {"atomicCAS",
       MapNames::getDpctNamespace() + "atomic_compare_exchange_strong"},
      {"atomicCAS_system",
       MapNames::getDpctNamespace() + "atomic_compare_exchange_strong"},
      {"atomicInc", MapNames::getDpctNamespace() + "atomic_fetch_compare_inc"},
      {"atomicInc_system",
       MapNames::getDpctNamespace() + "atomic_fetch_compare_inc"},
      {"atomicDec", MapNames::getDpctNamespace() + "atomic_fetch_compare_dec"},
      {"atomicDec_system",
       MapNames::getDpctNamespace() + "atomic_fetch_compare_dec"},
  };
}
// Supported vector types
const MapNamesLang::SetTy MapNamesLang::SupportedVectorTypes{
    SUPPORTEDVECTORTYPENAMES};
const MapNamesLang::SetTy MapNamesLang::VectorTypes2MArray{
    VECTORTYPE2MARRAYNAMES};

const std::map<std::string, int> MapNamesLang::VectorTypeMigratedTypeSizeMap{
    {"char1", 1},       {"char2", 2},       {"char3", 4},
    {"char4", 4},       {"uchar1", 1},      {"uchar2", 2},
    {"uchar3", 4},      {"uchar4", 4},      {"short1", 2},
    {"short2", 4},      {"short3", 8},      {"short4", 8},
    {"ushort1", 2},     {"ushort2", 4},     {"ushort3", 8},
    {"ushort4", 8},     {"int1", 4},        {"int2", 8},
    {"int3", 16},       {"int4", 16},       {"uint1", 4},
    {"uint2", 8},       {"uint3", 16},      {"uint4", 16},
    {"long1", 8},       {"long2", 16},      {"long3", 32},
    {"long4", 32},      {"ulong1", 8},      {"ulong2", 16},
    {"ulong3", 32},     {"ulong4", 32},     {"longlong1", 8},
    {"longlong2", 16},  {"longlong3", 32},  {"longlong4", 32},
    {"ulonglong1", 8},  {"ulonglong2", 16}, {"ulonglong3", 32},
    {"ulonglong4", 32}, {"float1", 4},      {"float2", 8},
    {"float3", 16},     {"float4", 16},     {"double1", 8},
    {"double2", 16},    {"double3", 32},    {"double4", 32},
    {"__half", 2},      {"__half2", 4},     {"__half_raw", 2}};

const std::map<clang::dpct::KernelArgType, int>
    MapNamesLang::KernelArgTypeSizeMap{
        {clang::dpct::KernelArgType::KAT_Stream, 208},
        {clang::dpct::KernelArgType::KAT_Texture,
         48 /*32(image accessor) + 16(sampler)*/},
        {clang::dpct::KernelArgType::KAT_Accessor1D, 32},
        {clang::dpct::KernelArgType::KAT_Accessor2D, 56},
        {clang::dpct::KernelArgType::KAT_Accessor3D, 80},
        {clang::dpct::KernelArgType::KAT_Array1D, 8},
        {clang::dpct::KernelArgType::KAT_Array2D, 24},
        {clang::dpct::KernelArgType::KAT_Array3D, 32},
        {clang::dpct::KernelArgType::KAT_Default, 8},
        {clang::dpct::KernelArgType::KAT_MaxParameterSize, 1024}};

int MapNamesLang::getArrayTypeSize(const int Dim) {
  if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {
    if (Dim == 2) {
      return KernelArgTypeSizeMap.at(
          clang::dpct::KernelArgType::KAT_Accessor2D);
    } else if (Dim == 3) {
      return KernelArgTypeSizeMap.at(
          clang::dpct::KernelArgType::KAT_Accessor3D);
    } else {
      return KernelArgTypeSizeMap.at(
          clang::dpct::KernelArgType::KAT_Accessor1D);
    }
  } else {
    if (Dim == 2) {
      return KernelArgTypeSizeMap.at(clang::dpct::KernelArgType::KAT_Array2D);
    } else if (Dim == 3) {
      return KernelArgTypeSizeMap.at(clang::dpct::KernelArgType::KAT_Array3D);
    } else {
      return KernelArgTypeSizeMap.at(clang::dpct::KernelArgType::KAT_Array1D);
    }
  }
}

const MapNamesLang::MapTy MapNamesLang::Dim3MemberNamesMap{
    {"x", "[2]"}, {"y", "[1]"}, {"z", "[0]"},
    // ...
};

const std::map<unsigned, std::string> MapNamesLang::ArrayFlagMap{
    {0, "standard"},
    {1, "array"},
};

// Texture names mapping.
const MapNamesLang::MapTy TextureRule::TextureMemberNames{
    {"addressMode", "addressing_mode"},
    {"filterMode", "filtering_mode"},
    {"normalized", "coordinate_normalization_mode"},
    {"normalizedCoords", "coordinate_normalization_mode"},
    {"channelDesc", "channel"},
    {"Format", "channel_type"},
    {"NumChannels", "channel_num"},
    {"Width", "x"},
    {"Height", "y"},
    {"flags", "coordinate_normalization_mode"},
    {"maxAnisotropy", "max_anisotropy"},
    {"mipmapFilterMode", "mipmap_filtering"},
    {"minMipmapLevelClamp", "min_mipmap_level_clamp"},
    {"maxMipmapLevelClamp", "max_mipmap_level_clamp"},
};

// DeviceProp names mapping.
const MapNamesLang::MapTy DeviceInfoVarRule::PropNamesMap{
    {"clockRate", "max_clock_frequency"},
    {"major", "major_version"},
    {"minor", "minor_version"},
    {"integrated", "integrated"},
    {"warpSize", "max_sub_group_size"},
    {"multiProcessorCount", "max_compute_units"},
    {"maxThreadsPerBlock", "max_work_group_size"},
    {"maxThreadsPerMultiProcessor", "max_work_items_per_compute_unit"},
    {"name", "name"},
    {"totalGlobalMem", "global_mem_size"},
    {"sharedMemPerBlock", "local_mem_size"},
    {"sharedMemPerBlockOptin", "local_mem_size"},
    {"sharedMemPerMultiprocessor", "local_mem_size"},
    {"maxGridSize", "max_nd_range_size"},
    {"maxThreadsDim", "max_work_item_sizes"},
    {"memoryClockRate", "memory_clock_rate"},
    {"memoryBusWidth", "memory_bus_width"},
    {"pciDeviceID", "device_id"},
    {"uuid", "uuid"},
    {"l2CacheSize", "global_mem_cache_size"},
    {"maxTexture1D", "image1d_max"},
    {"maxTexture2D", "image2d_max"},
    {"maxTexture3D", "image3d_max"},
    {"regsPerBlock", "max_register_size_per_work_group"},
    // ...
};

const MapNamesLang::MapTy MapNamesLang::FunctionAttrMap{
    {"CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK", "max_work_group_size"},
    {"CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES",
     "shared_size_bytes /* statically allocated shared memory per work-group "
     "in bytes */"},
    {"CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES",
     "local_size_bytes /* local memory per work-item in bytes */"},
    {"CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES",
     "const_size_bytes /* user-defined constant kernel memory in bytes */"},
    {"CU_FUNC_ATTRIBUTE_NUM_REGS",
     "num_regs /* number of registers for each thread */"},
    // ...
};

// DeviceProp names mapping.
const MapNamesLang::MapTy MapNamesLang::MemberNamesMap{
    {"x", "x()"}, {"y", "y()"}, {"z", "z()"}, {"w", "w()"},
    // ...
};
const MapNamesLang::MapTy MapNamesLang::MArrayMemberNamesMap{
    {"x", "[0]"},
    {"y", "[1]"},
};

const MapNamesLang::SetTy MapNamesLang::HostAllocSet{
    "cudaHostAllocDefault",         "cudaHostAllocMapped",
    "cudaHostAllocPortable",        "cudaHostAllocWriteCombined",
    "CU_MEMHOSTALLOC_PORTABLE",     "CU_MEMHOSTALLOC_DEVICEMAP",
    "CU_MEMHOSTALLOC_WRITECOMBINED"};

// Function Attributes names migration
const MapNamesLang::MapTy KernelFunctionInfoRule::AttributesNamesMap{
    {"maxThreadsPerBlock", "max_work_group_size"},
};

MapNamesLang::MapTy TextureRule::ResourceTypeNames{
    {"devPtr", "data_ptr"},
    {"desc", "channel"},
    {"array", "data_ptr"},
    {"mipmap", "data_ptr"},
    {"width", "x"},
    {"height", "y"},
    {"pitchInBytes", "pitch"},
    {"sizeInBytes", "x"},
    {"hArray", "data_ptr"},
    {"format", "channel_type"},
    {"numChannels", "channel_num"}};

const MapNamesLang::MapTy MemoryDataTypeRule::PitchMemberNames{
    {"pitch", "pitch"}, {"ptr", "data_ptr"}, {"xsize", "x"}, {"ysize", "y"}};
const MapNamesLang::MapTy MemoryDataTypeRule::ExtentMemberNames{
    {"width", "[0]"}, {"height", "[1]"}, {"depth", "[2]"}};

const MapNamesLang::MapTy MemoryDataTypeRule::ArrayDescMemberNames{
    {"Width", "width"},
    {"Height", "height"},
    {"Depth", "depth"},
    {"Format", "channel_type"},
    {"NumChannels", "num_channels"}};

const MapNamesLang::MapTy MemoryDataTypeRule::DirectReplMemberNames{
    // cudaMemcpy3DParms fields.
    {"srcArray", "from.image"},
    {"srcPtr", "from.pitched"},
    {"srcPos", "from.pos"},
    {"dstArray", "to.image"},
    {"dstPtr", "to.pitched"},
    {"dstPos", "to.pos"},
    {"extent", "size"},
    {"kind", "direction"},
    // cudaMemcpy3DPeerParms fields.
    {"srcDevice", "from.dev_id"},
    {"dstDevice", "to.dev_id"},
    // CUDA_MEMCPY2D fields.
    {"Height", "size[1]"},
    {"WidthInBytes", "size_x_in_bytes"},
    {"dstXInBytes", "to.pos_x_in_bytes"},
    {"srcXInBytes", "from.pos_x_in_bytes"},
    {"dstY", "to.pos[1]"},
    {"srcY", "from.pos[1]"},
    // CUDA_MEMCPY3D fields.
    {"Depth", "size[2]"},
    {"dstZ", "to.pos[2]"},
    {"srcZ", "from.pos[2]"},
    // CUDA_MEMCPY3D_PEER fields.
    {"srcContext", "from.dev_id"},
    {"dstContext", "to.dev_id"},
};

const MapNamesLang::MapTy MemoryDataTypeRule::GetSetReplMemberNames{
    // CUDA_MEMCPY2D fields.
    {"dstPitch", "pitch"},
    {"srcPitch", "pitch"},
    {"dstDevice", "data_ptr"},
    {"dstHost", "data_ptr"},
    {"srcDevice", "data_ptr"},
    {"srcHost", "data_ptr"},
    // CUDA_MEMCPY3D fields.
    {"dstHeight", "y"},
    {"srcHeight", "y"},
};

const std::vector<std::string> MemoryDataTypeRule::RemoveMember{
    "dstLOD", "srcLOD", "dstMemoryType", "srcMemoryType", "Flags"};

const std::unordered_set<std::string> MapNamesLang::CooperativeGroupsAPISet{
    "this_thread_block",
    "this_grid",
    "sync",
    "tiled_partition",
    "thread_rank",
    "size",
    "shfl_down",
    "reduce",
    "num_threads",
    "shfl_up",
    "shfl",
    "shfl_xor",
    "meta_group_rank",
    "meta_group_size",
    "block_tile_memory",
    "thread_index",
    "group_index",
    "inclusive_scan",
    "exclusive_scan",
    "coalesced_threads",
    "num_blocks",
    "block_rank"};

const std::unordered_map<std::string, HelperFeatureEnum>
    MapNamesLang::SamplingInfoToSetFeatureMap = {
        {"coordinate_normalization_mode", HelperFeatureEnum::device_ext}};
const std::unordered_map<std::string, HelperFeatureEnum>
    MapNamesLang::SamplingInfoToGetFeatureMap = {
        {"addressing_mode", HelperFeatureEnum::device_ext},
        {"filtering_mode", HelperFeatureEnum::device_ext}};
const std::unordered_map<std::string, HelperFeatureEnum>
    MapNamesLang::ImageWrapperBaseToSetFeatureMap = {
        {"sampling_info", HelperFeatureEnum::device_ext},
        {"data", HelperFeatureEnum::device_ext},
        {"channel", HelperFeatureEnum::device_ext},
        {"channel_data_type", HelperFeatureEnum::device_ext},
        {"channel_size", HelperFeatureEnum::device_ext},
        {"coordinate_normalization_mode", HelperFeatureEnum::device_ext},
        {"channel_num", HelperFeatureEnum::device_ext},
        {"channel_type", HelperFeatureEnum::device_ext}};
const std::unordered_map<std::string, HelperFeatureEnum>
    MapNamesLang::ImageWrapperBaseToGetFeatureMap = {
        {"sampling_info", HelperFeatureEnum::device_ext},
        {"data", HelperFeatureEnum::device_ext},
        {"channel", HelperFeatureEnum::device_ext},
        {"channel_data_type", HelperFeatureEnum::device_ext},
        {"channel_size", HelperFeatureEnum::device_ext},
        {"addressing_mode", HelperFeatureEnum::device_ext},
        {"filtering_mode", HelperFeatureEnum::device_ext},
        {"coordinate_normalization_mode", HelperFeatureEnum::device_ext},
        {"channel_num", HelperFeatureEnum::device_ext},
        {"channel_type", HelperFeatureEnum::device_ext},
        {"sampler", HelperFeatureEnum::device_ext},
};

} // namespace dpct
} // namespace clang