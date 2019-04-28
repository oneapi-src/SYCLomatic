/******************************************************************************
* INTEL CONFIDENTIAL
*
* Copyright 2018-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted materials,
* and your use of them is governed by the express license under which they
* were provided to you ("License"). Unless the License provides otherwise,
* you may not use, modify, copy, publish, distribute, disclose or transmit
* this software or the related documents without Intel's prior written
* permission.

* This software and the related documents are provided as is, with no express
* or implied warranties, other than those that are expressly stated in the
* License.
*****************************************************************************/

//===--- syclct_util.hpp -------------------------------*- C++ -*-----===//

#ifndef SYCLCT_UTIL_H
#define SYCLCT_UTIL_H

#include <CL/sycl.hpp>

namespace syclct {

using namespace cl::sycl;
using short1 = vec<short, 1>;
using ushort1 = vec<ushort, 1>;
using int1 = vec<int, 1>;
using uint1 = vec<uint, 1>;
using long1 = vec<long ,1>;
using ulong1 = vec<ulong, 1>;
using longlong1 = vec<long long, 1>;
using ulonglong1 = vec<unsigned long long, 1>;
using half1 = vec<half, 1>;
using float1 = vec<float, 1>;
using double1 = vec<double, 1>;

half2 float22half2_rn(float2 f2) {
  return f2.convert<half, rounding_mode::rte>();
}

half float2half(float f) {
  float1 f1{f};
  half1 h1 = f1.convert<half, rounding_mode::automatic>();
  return h1.get_value(0);
}

half float2half_rd(float f) {
  float1 f1{f};
  half1 h1 = f1.convert<half, rounding_mode::rtn>();
  return h1.get_value(0);
}

half float2half_rn(float f) {
  float1 f1{f};
  half1 h1 = f1.convert<half, rounding_mode::rte>();
  return h1.get_value(0);
}

half float2half_ru(float f) {
  float1 f1{f};
  half1 h1 = f1.convert<half, rounding_mode::rtp>();
  return h1.get_value(0);
}

half float2half_rz(float f) {
  float1 f1{f};
  half1 h1 = f1.convert<half, rounding_mode::rtz>();
  return h1.get_value(0);
}

half2 float2half2_rn(float f) {
  float2 f2{f, f};
  return f2.convert<half, rounding_mode::rte>();
}

half2 floats2half2_rn(float f0, float f1) {
  float2 f2{f0, f1};
  return f2.convert<half, rounding_mode::rte>();
}

float2 half22float2(half2 h2) {
  return h2.convert<float, rounding_mode::automatic>();
}

float half2float(half h) {
  half1 h1{h};
  float1 f1 = h1.convert<float, rounding_mode::automatic>();
  return f1.get_value(0);
}

int half2int(half h) {
  half1 h1{h};
  int1 i1 = h1.convert<int, rounding_mode::automatic>();
  return i1.get_value(0);
}

int half2int_rd(half h) {
  half1 h1{h};
  int1 i1 = h1.convert<int, rounding_mode::rtn>();
  return i1.get_value(0);
}

int half2int_rn(half h) {
  half1 h1{h};
  int1 i1 = h1.convert<int, rounding_mode::rte>();
  return i1.get_value(0);
}

int half2int_ru(half h) {
  half1 h1{h};
  int1 i1 = h1.convert<int, rounding_mode::rtp>();
  return i1.get_value(0);
}

int half2int_rz(half h) {
  half1 h1{h};
  int1 i1 = h1.convert<int, rounding_mode::rtz>();
  return i1.get_value(0);
}

long long half2ll(half h) {
  half1 h1{h};
  longlong1 ll1 = h1.convert<long long, rounding_mode::automatic>();
  return ll1.get_value(0);
}

long long half2ll_rd(half h) {
  half1 h1{h};
  longlong1 ll1 = h1.convert<long long, rounding_mode::rtn>();
  return ll1.get_value(0);
}

long long half2ll_rn(half h) {
  half1 h1{h};
  longlong1 ll1 = h1.convert<long long, rounding_mode::rte>();
  return ll1.get_value(0);
}

long long half2ll_ru(half h) {
  half1 h1{h};
  longlong1 ll1 = h1.convert<long long, rounding_mode::rtp>();
  return ll1.get_value(0);
}

long long half2ll_rz(half h) {
  half1 h1{h};
  longlong1 ll1 = h1.convert<long long, rounding_mode::rtz>();
  return ll1.get_value(0);
}

short half2short(half h) {
  half1 h1 {h};
  short1 s1 = h1.convert<short, rounding_mode::automatic>();
  return s1.get_value(0);
}

short half2short_rd(half h) {
  half1 h1 {h};
  short1 s1 = h1.convert<short, rounding_mode::rtn>();
  return s1.get_value(0);
}

short half2short_rn(half h) {
  half1 h1 {h};
  short1 s1 = h1.convert<short, rounding_mode::rte>();
  return s1.get_value(0);
}

short half2short_ru(half h) {
  half1 h1 {h};
  short1 s1 = h1.convert<short, rounding_mode::rtp>();
  return s1.get_value(0);
}

short half2short_rz(half h) {
  half1 h1 {h};
  short1 s1 = h1.convert<short, rounding_mode::rtz>();
  return s1.get_value(0);
}

unsigned half2uint(half h) {
  half1 h1{h};
  uint1 u1 = h1.convert<unsigned, rounding_mode::automatic>();
  return u1.get_value(0);
}

unsigned half2uint_rd(half h) {
  half1 h1{h};
  uint1 u1 = h1.convert<unsigned, rounding_mode::rtn>();
  return u1.get_value(0);
}

unsigned half2uint_rn(half h) {
  half1 h1{h};
  uint1 u1 = h1.convert<unsigned, rounding_mode::rte>();
  return u1.get_value(0);
}

unsigned half2uint_ru(half h) {
  half1 h1{h};
  uint1 u1 = h1.convert<unsigned, rounding_mode::rtp>();
  return u1.get_value(0);
}

unsigned half2uint_rz(half h) {
  half1 h1{h};
  uint1 u1 = h1.convert<unsigned, rounding_mode::rtz>();
  return u1.get_value(0);
}

unsigned long long half2ull(half h) {
  half1 h1{h};
  ulonglong1 ull1 = h1.convert<unsigned long long, rounding_mode::automatic>();
  return ull1.get_value(0);
}

unsigned long long half2ull_rd(half h) {
  half1 h1{h};
  ulonglong1 ull1 = h1.convert<unsigned long long, rounding_mode::rtn>();
  return ull1.get_value(0);
}

unsigned long long half2ull_rn(half h) {
  half1 h1{h};
  ulonglong1 ull1 = h1.convert<unsigned long long, rounding_mode::rte>();
  return ull1.get_value(0);
}

unsigned long long half2ull_ru(half h) {
  half1 h1{h};
  ulonglong1 ull1 = h1.convert<unsigned long long, rounding_mode::rtp>();
  return ull1.get_value(0);
}

unsigned long long half2ull_rz(half h) {
  half1 h1{h};
  ulonglong1 ull1 = h1.convert<unsigned long long, rounding_mode::rtz>();
  return ull1.get_value(0);
}

ushort half2ushort(half h) {
  half1 h1{h};
  ushort1 us1 = h1.convert<ushort, rounding_mode::automatic>();
  return us1.get_value(0);
}

ushort half2ushort_rd(half h) {
  half1 h1{h};
  ushort1 us1 = h1.convert<ushort, rounding_mode::rtn>();
  return us1.get_value(0);
}

ushort half2ushort_rn(half h) {
  half1 h1{h};
  ushort1 us1 = h1.convert<ushort, rounding_mode::rte>();
  return us1.get_value(0);
}

ushort half2ushort_ru(half h) {
  half1 h1{h};
  ushort1 us1 = h1.convert<ushort, rounding_mode::rtp>();
  return us1.get_value(0);
}

ushort half2ushort_rz(half h) {
  half1 h1{h};
  ushort1 us1 = h1.convert<ushort, rounding_mode::rtz>();
  return us1.get_value(0);
}

half2 half2half2(half h) {
  return {h, h};
}

half2 halves2half2(half h0, half h1) {
  return {h0, h1};
}

float high2float(half2 h2) {
  float f = h2.get_value(0);
  return f;
}

half2 high2half2(half2 h2) {
  return {h2.get_value(0), h2.get_value(0)};
}

half2 highs2half2(half2 h2, half2 h2_1) {
  return {h2.get_value(0), h2_1.get_value(0)};
}

half int2half(int i) {
  int1 i1{i};
  half1 h1 = i1.convert<half, rounding_mode::automatic>();
  return h1.get_value(0);
}

half int2half_rd(int i) {
  int1 i1{i};
  half1 h1 = i1.convert<half, rounding_mode::rtn>();
  return h1.get_value(0);
}

half int2half_rn(int i) {
  int1 i1{i};
  half1 h1 = i1.convert<half, rounding_mode::rte>();
  return h1.get_value(0);
}

half int2half_ru(int i) {
  int1 i1{i};
  half1 h1 = i1.convert<half, rounding_mode::rtp>();
  return h1.get_value(0);
}

half int2half_rz(int i) {
  int1 i1{i};
  half1 h1 = i1.convert<half, rounding_mode::rtz>();
  return h1.get_value(0);
}

half ll2half(long long ll) {
  longlong1 ll1{ll};
  half1 h1 = ll1.convert<half, rounding_mode::automatic>();
  return h1.get_value(0);
}

half ll2half_rd(long long ll) {
  longlong1 ll1{ll};
  half1 h1 = ll1.convert<half, rounding_mode::rtn>();
  return h1.get_value(0);
}

half ll2half_rn(long long ll) {
  longlong1 ll1{ll};
  half1 h1 = ll1.convert<half, rounding_mode::rte>();
  return h1.get_value(0);
}

half ll2half_ru(long long ll) {
  longlong1 ll1{ll};
  half1 h1 = ll1.convert<half, rounding_mode::rtp>();
  return h1.get_value(0);
}

half ll2half_rz(long long ll) {
  longlong1 ll1{ll};
  half1 h1 = ll1.convert<half, rounding_mode::rtz>();
  return h1.get_value(0);
}

half high2half(half2 h2) {
  return h2.get_value(0);
}

float low2float(half2 h2) {
  float f = h2.get_value(1);
  return f;
}

half low2half(half2 h2) {
  return h2.get_value(1);
}

half2 low2half2(half2 h2) {
  return {h2.get_value(1), h2.get_value(1)};
}

half2 lowhigh2highlow(half2 h2) {
  return {h2.get_value(1), h2.get_value(0)};
}

half2 lows2half2(half2 h2, half2 h2_1) {
  return {h2.get_value(1), h2_1.get_value(1)};
}

half short2half(short s) {
  short1 s1{s};
  half1 h1 = s1.convert<half, rounding_mode::automatic>();
  return h1.get_value(0);
}

half short2half_rd(short s) {
  short1 s1{s};
  half1 h1 = s1.convert<half, rounding_mode::rtn>();
  return h1.get_value(0);
}

half short2half_rn(short s) {
  short1 s1{s};
  half1 h1 = s1.convert<half, rounding_mode::rte>();
  return h1.get_value(0);
}

half short2half_ru(short s) {
  short1 s1{s};
  half1 h1 = s1.convert<half, rounding_mode::rtp>();
  return h1.get_value(0);
}

half short2half_rz(short s) {
  short1 s1{s};
  half1 h1 = s1.convert<half, rounding_mode::rtz>();
  return h1.get_value(0);
}

half uint2half(uint u) {
  uint1 u1{u};
  half1 h1 = u1.convert<half, rounding_mode::automatic>();
  return h1.get_value(0);
}

half uint2half_rd(uint u) {
  uint1 u1{u};
  half1 h1 = u1.convert<half, rounding_mode::rtn>();
  return h1.get_value(0);
}

half uint2half_rn(uint u) {
  uint1 u1{u};
  half1 h1 = u1.convert<half, rounding_mode::rte>();
  return h1.get_value(0);
}

half uint2half_ru(uint u) {
  uint1 u1{u};
  half1 h1 = u1.convert<half, rounding_mode::rtp>();
  return h1.get_value(0);
}

half uint2half_rz(uint u) {
  uint1 u1{u};
  half1 h1 = u1.convert<half, rounding_mode::rtz>();
  return h1.get_value(0);
}

half ull2half(unsigned long long ull) {
  ulonglong1 ull1{ull};
  half1 h1 = ull1.convert<half, rounding_mode::automatic>();
  return h1.get_value(0);
}

half ull2half_rd(unsigned long long ull) {
  ulonglong1 ull1{ull};
  half1 h1 = ull1.convert<half, rounding_mode::rtn>();
  return h1.get_value(0);
}

half ull2half_rn(unsigned long long ull) {
  ulonglong1 ull1{ull};
  half1 h1 = ull1.convert<half, rounding_mode::rte>();
  return h1.get_value(0);
}

half ull2half_ru(unsigned long long ull) {
  ulonglong1 ull1{ull};
  half1 h1 = ull1.convert<half, rounding_mode::rtp>();
  return h1.get_value(0);
}

half ull2half_rz(unsigned long long ull) {
  ulonglong1 ull1{ull};
  half1 h1 = ull1.convert<half, rounding_mode::rtz>();
  return h1.get_value(0);
}

half ushort2half(ushort us) {
  ushort1 us1{us};
  half1 h1 = us1.convert<half, rounding_mode::automatic>();
  return h1.get_value(0);
}

half ushort2half_rd(ushort us) {
  ushort1 us1{us};
  half1 h1 = us1.convert<half, rounding_mode::rtn>();
  return h1.get_value(0);
}

half ushort2half_rn(ushort us) {
  ushort1 us1{us};
  half1 h1 = us1.convert<half, rounding_mode::rte>();
  return h1.get_value(0);
}

half ushort2half_ru(ushort us) {
  ushort1 us1{us};
  half1 h1 = us1.convert<half, rounding_mode::rtp>();
  return h1.get_value(0);
}

half ushort2half_rz(ushort us) {
  ushort1 us1{us};
  half1 h1 = us1.convert<half, rounding_mode::rtz>();
  return h1.get_value(0);
}

// Type Casting Intrinsics
float double2float(double d) {
  double1 d1{d};
  float1 f1 = d1.convert<float, rounding_mode::automatic>();
  return f1.get_value(0);
}

float double2float_rd(double d) {
  double1 d1{d};
  float1 f1 = d1.convert<float, rounding_mode::rtn>();
  return f1.get_value(0);
}

float double2float_rn(double d) {
  double1 d1{d};
  float1 f1 = d1.convert<float, rounding_mode::rte>();
  return f1.get_value(0);
}

float double2float_ru(double d) {
  double1 d1{d};
  float1 f1 = d1.convert<float, rounding_mode::rtp>();
  return f1.get_value(0);
}

float double2float_rz(double d) {
  double1 d1{d};
  float1 f1 = d1.convert<float, rounding_mode::rtz>();
  return f1.get_value(0);
}

int double2int(double d) {
  double1 d1{d};
  int1 i1 = d1.convert<int, rounding_mode::automatic>();
  return i1.get_value(0);
}

int double2int_rd(double d) {
  double1 d1{d};
  int1 i1 = d1.convert<int, rounding_mode::rtn>();
  return i1.get_value(0);
}

int double2int_rn(double d) {
  double1 d1{d};
  int1 i1 = d1.convert<int, rounding_mode::rte>();
  return i1.get_value(0);
}

int double2int_ru(double d) {
  double1 d1{d};
  int1 i1 = d1.convert<int, rounding_mode::rtp>();
  return i1.get_value(0);
}

int double2int_rz(double d) {
  double1 d1{d};
  int1 i1 = d1.convert<int, rounding_mode::rtz>();
  return i1.get_value(0);
}

long long double2ll(double d) {
  double1 d1{d};
  longlong1 ll1 = d1.convert<long long, rounding_mode::automatic>();
  return ll1.get_value(0);
}

long long double2ll_rd(double d) {
  double1 d1{d};
  longlong1 ll1 = d1.convert<long long, rounding_mode::rtn>();
  return ll1.get_value(0);
}

long long double2ll_rn(double d) {
  double1 d1{d};
  longlong1 ll1 = d1.convert<long long, rounding_mode::rte>();
  return ll1.get_value(0);
}

long long double2ll_ru(double d) {
  double1 d1{d};
  longlong1 ll1 = d1.convert<long long, rounding_mode::rtp>();
  return ll1.get_value(0);
}

long long double2ll_rz(double d) {
  double1 d1{d};
  longlong1 ll1 = d1.convert<long long, rounding_mode::rtz>();
  return ll1.get_value(0);
}

uint double2uint(double d) {
  double1 d1{d};
  uint1 ui1 = d1.convert<uint, rounding_mode::automatic>();
  return ui1.get_value(0);
}

uint double2uint_rd(double d) {
  double1 d1{d};
  uint1 ui1 = d1.convert<uint, rounding_mode::rtn>();
  return ui1.get_value(0);
}

uint double2uint_rn(double d) {
  double1 d1{d};
  uint1 ui1 = d1.convert<uint, rounding_mode::rte>();
  return ui1.get_value(0);
}

uint double2uint_ru(double d) {
  double1 d1{d};
  uint1 ui1 = d1.convert<uint, rounding_mode::rtp>();
  return ui1.get_value(0);
}

uint double2uint_rz(double d) {
  double1 d1{d};
  uint1 ui1 = d1.convert<uint, rounding_mode::rtz>();
  return ui1.get_value(0);
}

unsigned long long double2ull(double d) {
  double1 d1{d};
  ulonglong1 ull1 = d1.convert<unsigned long long, rounding_mode::automatic>();
  return ull1.get_value(0);
}

unsigned long long double2ull_rd(double d) {
  double1 d1{d};
  ulonglong1 ull1 = d1.convert<unsigned long long, rounding_mode::rtn>();
  return ull1.get_value(0);
}

unsigned long long double2ull_rn(double d) {
  double1 d1{d};
  ulonglong1 ull1 = d1.convert<unsigned long long, rounding_mode::rte>();
  return ull1.get_value(0);
}

unsigned long long double2ull_ru(double d) {
  double1 d1{d};
  ulonglong1 ull1 = d1.convert<unsigned long long, rounding_mode::rtp>();
  return ull1.get_value(0);
}

unsigned long long double2ull_rz(double d) {
  double1 d1{d};
  ulonglong1 ull1 = d1.convert<unsigned long long, rounding_mode::rtz>();
  return ull1.get_value(0);
}

int float2int(float f) {
  float1 f1{f};
  int1 i1 = f1.convert<int, rounding_mode::automatic>();
  return i1.get_value(0);
}

int float2int_rd(float f) {
  float1 f1{f};
  int1 i1 = f1.convert<int, rounding_mode::rtn>();
  return i1.get_value(0);
}

int float2int_rn(float f) {
  float1 f1{f};
  int1 i1 = f1.convert<int, rounding_mode::rte>();
  return i1.get_value(0);
}

int float2int_ru(float f) {
  float1 f1{f};
  int1 i1 = f1.convert<int, rounding_mode::rtp>();
  return i1.get_value(0);
}

int float2int_rz(float f) {
  float1 f1{f};
  int1 i1 = f1.convert<int, rounding_mode::rtz>();
  return i1.get_value(0);
}

long long float2ll(float f) {
  float1 f1{f};
  longlong1 ll1 = f1.convert<long long, rounding_mode::automatic>();
  return ll1.get_value(0);
}

long long float2ll_rd(float f) {
  float1 f1{f};
  longlong1 ll1 = f1.convert<long long, rounding_mode::rtn>();
  return ll1.get_value(0);
}

long long float2ll_rn(float f) {
  float1 f1{f};
  longlong1 ll1 = f1.convert<long long, rounding_mode::rte>();
  return ll1.get_value(0);
}

long long float2ll_ru(float f) {
  float1 f1{f};
  longlong1 ll1 = f1.convert<long long, rounding_mode::rtp>();
  return ll1.get_value(0);
}

long long float2ll_rz(float f) {
  float1 f1{f};
  longlong1 ll1 = f1.convert<long long, rounding_mode::rtz>();
  return ll1.get_value(0);
}

uint float2uint(float f) {
  float1 f1{f};
  uint1 u1 = f1.convert<uint, rounding_mode::automatic>();
  return u1.get_value(0);
}

uint float2uint_rd(float f) {
  float1 f1{f};
  uint1 u1 = f1.convert<uint, rounding_mode::rtn>();
  return u1.get_value(0);
}

uint float2uint_rn(float f) {
  float1 f1{f};
  uint1 u1 = f1.convert<uint, rounding_mode::rte>();
  return u1.get_value(0);
}

uint float2uint_ru(float f) {
  float1 f1{f};
  uint1 u1 = f1.convert<uint, rounding_mode::rtp>();
  return u1.get_value(0);
}

uint float2uint_rz(float f) {
  float1 f1{f};
  uint1 u1 = f1.convert<uint, rounding_mode::rtz>();
  return u1.get_value(0);
}

unsigned long long float2ull(float f) {
  float1 f1{f};
  ulonglong1 ull1 = f1.convert<unsigned long long, rounding_mode::automatic>();
  return ull1.get_value(0);
}

unsigned long long float2ull_rd(float f) {
  float1 f1{f};
  ulonglong1 ull1 = f1.convert<unsigned long long, rounding_mode::rtn>();
  return ull1.get_value(0);
}

unsigned long long float2ull_rn(float f) {
  float1 f1{f};
  ulonglong1 ull1 = f1.convert<unsigned long long, rounding_mode::rte>();
  return ull1.get_value(0);
}

unsigned long long float2ull_ru(float f) {
  float1 f1{f};
  ulonglong1 ull1 = f1.convert<unsigned long long, rounding_mode::rtp>();
  return ull1.get_value(0);
}

unsigned long long float2ull_rz(float f) {
  float1 f1{f};
  ulonglong1 ull1 = f1.convert<unsigned long long, rounding_mode::rtz>();
  return ull1.get_value(0);
}

double int2double_rn(int i) {
  int1 i1{i};
  double1 d1 = i1.convert<double, rounding_mode::rte>();
  return d1.get_value(0);
}

float int2float(int i) {
  int1 i1{i};
  float1 f1 = i1.convert<float, rounding_mode::automatic>();
  return f1.get_value(0);
}

float int2float_rd(int i) {
  int1 i1{i};
  float1 f1 = i1.convert<float, rounding_mode::rtn>();
  return f1.get_value(0);
}

float int2float_rn(int i) {
  int1 i1{i};
  float1 f1 = i1.convert<float, rounding_mode::rte>();
  return f1.get_value(0);
}

float int2float_ru(int i) {
  int1 i1{i};
  float1 f1 = i1.convert<float, rounding_mode::rtp>();
  return f1.get_value(0);
}

float int2float_rz(int i) {
  int1 i1{i};
  float1 f1 = i1.convert<float, rounding_mode::rtz>();
  return f1.get_value(0);
}

double ll2double(long long ll) {
  longlong1 ll1{ll};
  double1 d1 = ll1.convert<double, rounding_mode::automatic>();
  return d1.get_value(0);
}

double ll2double_rd(long long ll) {
  longlong1 ll1{ll};
  double1 d1 = ll1.convert<double, rounding_mode::rtn>();
  return d1.get_value(0);
}

double ll2double_rn(long long ll) {
  longlong1 ll1{ll};
  double1 d1 = ll1.convert<double, rounding_mode::rte>();
  return d1.get_value(0);
}

double ll2double_ru(long long ll) {
  longlong1 ll1{ll};
  double1 d1 = ll1.convert<double, rounding_mode::rtp>();
  return d1.get_value(0);
}

double ll2double_rz(long long ll) {
  longlong1 ll1{ll};
  double1 d1 = ll1.convert<double, rounding_mode::rtz>();
  return d1.get_value(0);
}

float ll2float(long long ll) {
  longlong1 ll1{ll};
  float1 f1 = ll1.convert<float, rounding_mode::automatic>();
  return f1.get_value(0);
}

float ll2float_rd(long long ll) {
  longlong1 ll1{ll};
  float1 f1 = ll1.convert<float, rounding_mode::rtn>();
  return f1.get_value(0);
}

float ll2float_rn(long long ll) {
  longlong1 ll1{ll};
  float1 f1 = ll1.convert<float, rounding_mode::rte>();
  return f1.get_value(0);
}

float ll2float_ru(long long ll) {
  longlong1 ll1{ll};
  float1 f1 = ll1.convert<float, rounding_mode::rtp>();
  return f1.get_value(0);
}

float ll2float_rz(long long ll) {
  longlong1 ll1{ll};
  float1 f1 = ll1.convert<float, rounding_mode::rtz>();
  return f1.get_value(0);
}

double uint2double_rn(uint u) {
  uint1 u1{u};
  double1 d1 = u1.convert<double, rounding_mode::automatic>();
  return d1.get_value(0);
}

float uint2float(uint u) {
  uint1 u1{u};
  float1 f1 = u1.convert<float, rounding_mode::automatic>();
  return f1.get_value(0);
}

float uint2float_rd(uint u) {
  uint1 u1{u};
  float1 f1 = u1.convert<float, rounding_mode::rtn>();
  return f1.get_value(0);
}

float uint2float_rn(uint u) {
  uint1 u1{u};
  float1 f1 = u1.convert<float, rounding_mode::rte>();
  return f1.get_value(0);
}

float uint2float_ru(uint u) {
  uint1 u1{u};
  float1 f1 = u1.convert<float, rounding_mode::rtp>();
  return f1.get_value(0);
}

float uint2float_rz(uint u) {
  uint1 u1{u};
  float1 f1 = u1.convert<float, rounding_mode::rtz>();
  return f1.get_value(0);
}

double ull2double(unsigned long long ull) {
  ulonglong1 ull1{ull};
  double1 d1 = ull1.convert<double, rounding_mode::automatic>();
  return d1.get_value(0);
}

double ull2double_rd(unsigned long long ull) {
  ulonglong1 ull1{ull};
  double1 d1 = ull1.convert<double, rounding_mode::rtn>();
  return d1.get_value(0);
}

double ull2double_rn(unsigned long long ull) {
  ulonglong1 ull1{ull};
  double1 d1 = ull1.convert<double, rounding_mode::rte>();
  return d1.get_value(0);
}

double ull2double_ru(unsigned long long ull) {
  ulonglong1 ull1{ull};
  double1 d1 = ull1.convert<double, rounding_mode::rtp>();
  return d1.get_value(0);
}

double ull2double_rz(unsigned long long ull) {
  ulonglong1 ull1{ull};
  double1 d1 = ull1.convert<double, rounding_mode::rtz>();
  return d1.get_value(0);
}

float ull2float(unsigned long long ull) {
  ulonglong1 ull1{ull};
  float1 f1 = ull1.convert<float, rounding_mode::automatic>();
  return f1.get_value(0);
}

float ull2float_rd(unsigned long long ull) {
  ulonglong1 ull1{ull};
  float1 f1 = ull1.convert<float, rounding_mode::rtn>();
  return f1.get_value(0);
}

float ull2float_rn(unsigned long long ull) {
  ulonglong1 ull1{ull};
  float1 f1 = ull1.convert<float, rounding_mode::rte>();
  return f1.get_value(0);
}

float ull2float_ru(unsigned long long ull) {
  ulonglong1 ull1{ull};
  float1 f1 = ull1.convert<float, rounding_mode::rtp>();
  return f1.get_value(0);
}

float ull2float_rz(unsigned long long ull) {
  ulonglong1 ull1{ull};
  float1 f1 = ull1.convert<float, rounding_mode::rtz>();
  return f1.get_value(0);
}

short half_as_short(half h) {
  return *reinterpret_cast<short*>(&h);
}

unsigned short half_as_ushort(half h) {
  return *reinterpret_cast<unsigned short*>(&h);
}

half short_as_half(short s) {
  return *reinterpret_cast<half*>(&s);
}

half ushort_as_half(ushort us) {
  return *reinterpret_cast<half*>(&us);
}

long long double_as_longlong(double d) {
  return *reinterpret_cast<long long*>(&d);
}

int float_as_int(float f) {
  return *reinterpret_cast<int*>(&f);
}

unsigned float_as_uint(float f) {
  return *reinterpret_cast<unsigned*>(&f);
}

float int_as_float(int i) {
  return *reinterpret_cast<float*>(&i);
}

double longlong_as_double(long long ll) {
  return *reinterpret_cast<long long*>(&ll);
}

float uint_as_float(unsigned ui) {
  return *reinterpret_cast<unsigned*>(&ui);
}

} // namespace syclct

#endif // SYCLCT_UTIL_H
