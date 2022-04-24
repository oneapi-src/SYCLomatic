# Contributing

## License

SYCLomatic project is licensed under the terms of the Apache
License v2.0 with LLVM Exceptions license ([LICENSE.txt](llvm/LICENSE.TXT)).

By contributing to this project, you agree to the Apache License v2.0 with LLVM
Exceptions and copyright terms therein and release your contribution under
these terms.

## Contribution process

### Development

**NB**: For any changes not related to SYCLomatic, 
but rather to LLVM in general, it is strongly encouraged that you submit
the patch to https://llvm.org/ directly.
See [LLVM contribution guidelines](https://llvm.org/docs/Contributing.html)
for more information.

**NB**: A change in tool should be accompanied with
corresponding test changes.
See [Test SYCLomatic](GetStartedGuide.md#test-SYCLomatic)
section of Get Started guide for more information.

- Create a personal fork of the project on GitHub
  - For the SYCLomatic project, use **main** branch as baseline for your
    changes. See [Get Started Guide](GetStartedGuide.md).
- Prepare your patch
  - follow [LLVM coding standards](https://llvm.org/docs/CodingStandards.html)
  - [clang-format](https://clang.llvm.org/docs/ClangFormat.html) and
    [clang-tidy](https://clang.llvm.org/extra/clang-tidy/) tools can be
    integrated into your workflow to ensure formatting and stylistic
    compliance of your changes.
  - use

    ```bash
    ./clang/tools/clang-format/git-clang-format `git merge-base origin/main HEAD`
    ```

    to check the format of your current changes against the `origin/main`
    branch.
    - `-f` to also correct unstaged changes
    - `--diff` to only print the diff without applying
- Build the project following
[Get Started Guide instructions](GetStartedGuide.md#build-SYCLomatic-toolchain).
- Run regression tests -
[instructions](GetStartedGuide.md#test-SYCLomatic-toolchain).

### Tests development

Every product change should be accompanied with corresponding test modification
(adding new test(s), extending, removing or modifying existing test(s)).

There are 2 types of tests which are used for SYCLomatic validation:
* SYCLomatic in-tree LIT tests including [check-clang-c2s](../../clang/test/C2S)
 targets stored in this repository. These tests
should not have hardware (e.g. GPU, FPGA, etc.) or external software
dependencies (e.g. OpenCL, Level Zero, CUDA runtimes). These tests only have dependencies
on CUDA header files.  All tests not following
this approach should be moved to SYCLomatic end-to-end test repo.

    **Guidelines for adding SYCLomatic in-tree LIT tests**:
    - The LIT tests are used to check whether the migration result is expected. The
      test framework doesn't check whether LIT test itself can be built and run correctly. So
      make sure the each LIT test itself can be compiled with corresponding compiler
      tool chain before migration. Also it's better to make sure the migrated code
      can be built with SYCL compiler.

    - Add a helpful comment describing what the test does at the beginning and other comments throughout the test as necessary.

    - Try to follow descriptive naming convention for variables, functions as much as possible.
    Please refer [LLVM naming convention](https://llvm.org/docs/CodingStandards.html#name-types-functions-variables-and-enumerators-properly)

* [SYCLomatic end-to-end (E2E) tests](https://github.com/intel/SYCLomatic-test.git).
A test which requires full stack including backend runtimes (e.g. OpenCL,
Level Zero or CUDA) should be put to SYCLomatic E2E test suite following
[CONTRIBUTING](https://github.com/intel/SYCLomatic-test/blob/main/CONTRIBUTING.md).

### Commit message

- When writing your commit message, please make sure to follow
  [LLVM developer policies](
  https://llvm.org/docs/DeveloperPolicy.html#commit-messages) on the subject.
- For any SYCLomatic related commit, the `[SYCLomatic]` tag should be present in the
  commit message title. To a reasonable extent, additional tags can be used
  to signify the component changed, e.g.: `[ISSUE-NUM]`, `[DOC]`, `[NFC]`.
- For product changes which require modification in tests outside of the current repository
  (see [Test SYCLomatic toolchain](GetStartedGuide.md#test-SYCLomatic)).
  the commit message should contain the link to corresponding test PR.

### Review and acceptance testing

- Create a pull request for your changes following [Creating a pull request
instructions](https://help.github.com/articles/creating-a-pull-request/).
- CI will run a signed-off check as soon as your PR is created - see the
**check_pr** CI action results.
- CI will run several build and functional testing checks as soon as the PR is
approved by an Intel representative.
  - A new approval is needed if the PR was updated (e.g. during code review).
- Once the PR is approved and all checks have passed, the pull request is
ready for merge.

### Merge

Project maintainers merge pull requests using one of the following options:

- [Rebase and merge] The preferable choice for PRs containing a single commit
- [Squash and merge] Used when there are multiple commits in the PR
  - Squashing is done to make sure that the project is builable on any commit
- [Create a merge commit] Used for LLVM pull-down PRs to preserve hashes of the
commits pulled from the LLVM community repository

## Sign your work

Please use the sign-off line at the end of the patch. Your signature certifies that you wrote the patch or otherwise have the right to pass it on as an open-source patch. The rules are pretty simple: if you can certify
the below (from [developercertificate.org](http://developercertificate.org/)):

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
660 York Street, Suite 102,
San Francisco, CA 94110 USA

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

Then you just add a line to every git commit message:

    Signed-off-by: Joe Smith <joe.smith@email.com>

Use your real name (sorry, no pseudonyms or anonymous contributions.)

If you set your `user.name` and `user.email` git configs, you can sign your
commit automatically with `git commit -s`.


## [Legal information](legal_information.md)
