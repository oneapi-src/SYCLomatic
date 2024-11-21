//===--------------- MigrationStatistics.h--------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
#include <map>
#include <set>
#include <string>
#include <vector>

#ifndef DPCT_RULEINFRA_MIGRATION_STATISTICS_H
#define DPCT_RULEINFRA_MIGRATION_STATISTICS_H

class MigrationStatistics {
private:
  static std::map<std::string /*API Name*/, bool /*Is Migrated*/>
      MigrationTable;
  static std::map<std::string /*Type Name*/, bool /*Is Migrated*/>
      TypeMigrationTable;

public:
  static bool IsMigrated(const std::string &APIName);
  static std::vector<std::string> GetAllAPINames(void);
  static std::map<std::string, bool> &GetTypeTable(void);
};

#endif //! DPCT_RULEINFRA_MIGRATION_STATISTICS_H