//===--- filelock.h-------------------------*- C++ -*---===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//
#pragma once

#include "GaNamespace.h"
#include "filesystem_util.h"

#include <string>
GAHELPER_NS_BEGIN

class filelock_t
{
public:
	filelock_t(const ustring& filename);
	~filelock_t();

	bool lock();
	void unlock();

	bool is_locked()const;

private:
	bool m_locked;
	ustring m_lock_filename;
};

GAHELPER_NS_END
