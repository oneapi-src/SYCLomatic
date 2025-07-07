# ===----- test.py --------------------------------------------------------=== #
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------=== #

from pathlib import Path
import os
import shutil
import subprocess

def run_command(cmd):
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Command fail: {e.stderr}")
        return None

def compare_with_diff(ref_file):
    result = subprocess.run(
        ["diff", "-u", ref_file, "temp.txt"],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        return True
    else:
        print("Found difference:\n" + result.stdout)
        return False

# Clean up previous test directory if it exists
test_dir = Path("tests")
if test_dir.exists() and test_dir.is_dir():
    shutil.rmtree(test_dir)

# Check if gitdiff2yaml exists
gitdiff2yaml_path = Path("gitdiff2yaml")
if not gitdiff2yaml_path.exists():
    print("gitdiff2yaml does not exist. Please build it first.")
    exit(1)

run_command(["unzip", "tests.zip"])
os.chdir("tests")

# Test 1
os.chdir("1")
old_commit_id = run_command(["git", "rev-parse", "HEAD~1"]).strip()
result = run_command(["../../gitdiff2yaml", old_commit_id, "-o", "temp.txt"])
test1_res = compare_with_diff("../../1.ref.txt")
if test1_res:
    print("Test 1 passed")
else:
    print("Test 1 failed")
os.chdir("..")

# Test 2
os.chdir("2")
old_commit_id = run_command(["git", "rev-parse", "HEAD~1"]).strip()
result = run_command(["../../gitdiff2yaml", old_commit_id, "-o", "temp.txt"])
test1_res = compare_with_diff("../../2.ref.txt")
if test1_res:
    print("Test 2 passed")
else:
    print("Test 2 failed")
os.chdir("..")

# Test 3
os.chdir("3")
old_commit_id = run_command(["git", "rev-parse", "HEAD~1"]).strip()
result = run_command(["../../gitdiff2yaml", old_commit_id, "-o", "temp.txt"])
test1_res = compare_with_diff("../../3.ref.txt")
if test1_res:
    print("Test 3 passed")
else:
    print("Test 3 failed")
os.chdir("..")

# Test 4
os.chdir("4")
old_commit_id = run_command(["git", "rev-parse", "HEAD~1"]).strip()
result = run_command(["../../gitdiff2yaml", old_commit_id, "-o", "temp.txt"])
test1_res = compare_with_diff("../../4.ref.txt")
if test1_res:
    print("Test 4 passed")
else:
    print("Test 4 failed")
os.chdir("..")

# Clean up
os.chdir("..")
shutil.rmtree(test_dir)
