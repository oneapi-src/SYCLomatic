//===--- filelock.cpp-------------------------*- C++ -*---===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//
#include "filelock.h"

#include "filesystem_util.h"

#ifdef _MSC_VER
#	include <io.h>
#else
#	include <unistd.h>
#endif
#include <fcntl.h>
#include <sys/stat.h>
GAHELPER_NS_BEGIN

filelock_t::filelock_t(const ustring& filename)
{
	m_lock_filename = filename;
    //fixme
    //m_lock_filename.change_ext(_U("lock"));
	m_locked = false;
}

filelock_t::~filelock_t()
{
	if(is_locked())
		unlock();
}

bool filelock_t::lock()
{
	if(m_locked)
		return true;

	int fd = -1;
#ifdef _MSC_VER
	_wsopen_s(&fd, m_lock_filename.c_str(), _O_CREAT| _O_EXCL, _SH_DENYRW, _S_IREAD|_S_IWRITE);
#else
	fd = open(m_lock_filename.c_str(), O_CREAT|O_EXCL, (S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH));
#endif

	if(fd < 0)
		return false;

	m_locked = true;

#ifdef _MSC_VER
	_close(fd);
#else
	close(fd);
#endif

	return m_locked;
}

void filelock_t::unlock()
{
	if(m_locked)
	{
		//gh2::remove_file(m_lock_filename);
		m_locked = false;
	}
}

bool filelock_t::is_locked()const
{
	return m_locked;
}

GAHELPER_NS_END
