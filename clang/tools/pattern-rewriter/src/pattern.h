//===- pattern.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PATTERN_H
#define PATTERN_H

#include <map>
#include <string>

namespace pattern {

struct Rule {
  std::string Match;
  std::string Rewrite;
  std::map<std::string, Rule> Subrules;
};

std::string applyRule(const Rule &Rule, const std::string &Input);

} // namespace pattern

#endif
