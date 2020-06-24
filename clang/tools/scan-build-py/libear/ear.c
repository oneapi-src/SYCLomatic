/* -*- coding: utf-8 -*-
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*/

/**
 * This file implements a shared library. This library can be pre-loaded by
 * the dynamic linker of the Operating System (OS). It implements a few function
 * related to process creation. By pre-load this library the executed process
 * uses these functions instead of those from the standard library.
 *
 * The idea here is to inject a logic before call the real methods. The logic is
 * to dump the call into a file. To call the real method this library is doing
 * the job of the dynamic linker.
 *
 * The only input for the log writing is about the destination directory.
 * This is passed as environment variable.
 */

#include "config.h"

#include <stddef.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <dlfcn.h>
#include <pthread.h>
#include <ctype.h>


#if defined HAVE_POSIX_SPAWN || defined HAVE_POSIX_SPAWNP
#include <spawn.h>
#endif

#if defined HAVE_NSGETENVIRON
# include <crt_externs.h>
#else
extern char **environ;
#endif
#define INTEL_CUSTOMIZATION

#define ENV_OUTPUT "INTERCEPT_BUILD_TARGET_DIR"
#ifdef APPLE
# define ENV_FLAT    "DYLD_FORCE_FLAT_NAMESPACE"
# define ENV_PRELOAD "DYLD_INSERT_LIBRARIES"
# define ENV_SIZE 3
#else
# define ENV_PRELOAD "LD_PRELOAD"
# define ENV_SIZE 2
#endif

#define DLSYM(TYPE_, VAR_, SYMBOL_)                                            \
    union {                                                                    \
        void *from;                                                            \
        TYPE_ to;                                                              \
    } cast;                                                                    \
    if (0 == (cast.from = dlsym(RTLD_NEXT, SYMBOL_))) {                        \
        perror("bear: dlsym");                                                 \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
    TYPE_ const VAR_ = cast.to;


typedef char const * bear_env_t[ENV_SIZE];

static int bear_capture_env_t(bear_env_t *env);
static int bear_reset_env_t(bear_env_t *env);
static void bear_release_env_t(bear_env_t *env);
static char const **bear_update_environment(char *const envp[], bear_env_t *env);
static char const **bear_update_environ(char const **in, char const *key, char const *value);
static char **bear_get_environment();
#ifdef INTEL_CUSTOMIZATION
static void bear_report_call(char const *fun, char const *argv[]);
#else
static void bear_report_call(char const *fun, char const *const argv[]);
#endif
static char const **bear_strings_build(char const *arg, va_list *ap);
static char const **bear_strings_copy(char const **const in);
static char const **bear_strings_append(char const **in, char const *e);
static size_t bear_strings_length(char const *const *in);
static void bear_strings_release(char const **);

int is_option_end(char* working);


static bear_env_t env_names =
    { ENV_OUTPUT
    , ENV_PRELOAD
#ifdef ENV_FLAT
    , ENV_FLAT
#endif
    };

static bear_env_t initial_env =
    { 0
    , 0
#ifdef ENV_FLAT
    , 0
#endif
    };

static int initialized = 0;
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

static void on_load(void) __attribute__((constructor));
static void on_unload(void) __attribute__((destructor));


#ifdef HAVE_EXECVE
static int call_execve(const char *path, char *const argv[],
                       char *const envp[]);
#endif
#ifdef HAVE_EXECVP
static int call_execvp(const char *file, char *const argv[]);
#endif
#ifdef HAVE_EXECVPE
static int call_execvpe(const char *file, char *const argv[],
                        char *const envp[]);
#endif
#ifdef HAVE_EXECVP2
static int call_execvP(const char *file, const char *search_path,
                       char *const argv[]);
#endif
#ifdef HAVE_EXECT
static int call_exect(const char *path, char *const argv[],
                      char *const envp[]);
#endif
#ifdef HAVE_POSIX_SPAWN
static int call_posix_spawn(pid_t *restrict pid, const char *restrict path,
                            const posix_spawn_file_actions_t *file_actions,
                            const posix_spawnattr_t *restrict attrp,
                            char *const argv[restrict],
                            char *const envp[restrict]);
#endif
#ifdef HAVE_POSIX_SPAWNP
static int call_posix_spawnp(pid_t *restrict pid, const char *restrict file,
                             const posix_spawn_file_actions_t *file_actions,
                             const posix_spawnattr_t *restrict attrp,
                             char *const argv[restrict],
                             char *const envp[restrict]);
#endif


/* Initialization method to Captures the relevant environment variables.
 */

static void on_load(void) {
    pthread_mutex_lock(&mutex);
    if (!initialized)
        initialized = bear_capture_env_t(&initial_env);
    pthread_mutex_unlock(&mutex);
}

static void on_unload(void) {
    pthread_mutex_lock(&mutex);
    bear_release_env_t(&initial_env);
    initialized = 0;
    pthread_mutex_unlock(&mutex);
}


/* These are the methods we try to hijack.
 */

#ifdef HAVE_EXECVE
int execve(const char *path, char *const argv[], char *const envp[]) {
#ifdef INTEL_CUSTOMIZATION
    bear_report_call(__func__, (char const **)argv);
#else
    bear_report_call(__func__, (char const *const *)argv);
#endif
    return call_execve(path, argv, envp);
}
#endif

#ifdef HAVE_EXECV
#ifndef HAVE_EXECVE
#error can not implement execv without execve
#endif
int execv(const char *path, char *const argv[]) {
#ifdef INTEL_CUSTOMIZATION
    bear_report_call(__func__, (char const **)argv);
#else
    bear_report_call(__func__, (char const *const *)argv);
#endif
    char * const * envp = bear_get_environment();
    return call_execve(path, argv, envp);
}
#endif

#ifdef HAVE_EXECVPE
int execvpe(const char *file, char *const argv[], char *const envp[]) {
#ifdef INTEL_CUSTOMIZATION
    bear_report_call(__func__, (char const **)argv);
    // To sync file name with argv[0], in case argv[0] is changed
    // by bear_report_call.
    file = argv[0];
#else
    bear_report_call(__func__, (char const *const *)argv);
#endif
    return call_execvpe(file, argv, envp);
}
#endif

#ifdef HAVE_EXECVP
int execvp(const char *file, char *const argv[]) {
#ifdef INTEL_CUSTOMIZATION
    bear_report_call(__func__, (char const **)argv);
    // To sync file name with argv[0], in case argv[0] is changed
    // by bear_report_call.
    file = argv[0];
#else
    bear_report_call(__func__, (char const *const *)argv);
#endif
    return call_execvp(file, argv);
}
#endif

#ifdef HAVE_EXECVP2
int execvP(const char *file, const char *search_path, char *const argv[]) {
#ifdef INTEL_CUSTOMIZATION
    bear_report_call(__func__, (char const **)argv);
    // To sync file name with argv[0], in case argv[0] is changed
    // by bear_report_call.
    file = argv[0];
#else
    bear_report_call(__func__, (char const *const *)argv);
#endif
    return call_execvP(file, search_path, argv);
}
#endif

#ifdef HAVE_EXECT
int exect(const char *path, char *const argv[], char *const envp[]) {
#ifdef INTEL_CUSTOMIZATION
    bear_report_call(__func__, (char const **)argv);
#else
    bear_report_call(__func__, (char const *const *)argv);
#endif
    return call_exect(path, argv, envp);
}
#endif

#ifdef HAVE_EXECL
# ifndef HAVE_EXECVE
#  error can not implement execl without execve
# endif
int execl(const char *path, const char *arg, ...) {
    va_list args;
    va_start(args, arg);
    char const **argv = bear_strings_build(arg, &args);
    va_end(args);

#ifdef INTEL_CUSTOMIZATION
    bear_report_call(__func__, (char const **)argv);
#else
    bear_report_call(__func__, (char const *const *)argv);
#endif
    char * const * envp = bear_get_environment();
    int const result = call_execve(path, (char *const *)argv, envp);

    bear_strings_release(argv);
    return result;
}
#endif

#ifdef HAVE_EXECLP
# ifndef HAVE_EXECVP
#  error can not implement execlp without execvp
# endif
int execlp(const char *file, const char *arg, ...) {
    va_list args;
    va_start(args, arg);
    char const **argv = bear_strings_build(arg, &args);
    va_end(args);

#ifdef INTEL_CUSTOMIZATION
    bear_report_call(__func__, (char const **)argv);
#else
    bear_report_call(__func__, (char const *const *)argv);
#endif
    int const result = call_execvp(file, (char *const *)argv);

    bear_strings_release(argv);
    return result;
}
#endif

#ifdef HAVE_EXECLE
# ifndef HAVE_EXECVE
#  error can not implement execle without execve
# endif
// int execle(const char *path, const char *arg, ..., char * const envp[]);
int execle(const char *path, const char *arg, ...) {
    va_list args;
    va_start(args, arg);
    char const **argv = bear_strings_build(arg, &args);
    char const **envp = va_arg(args, char const **);
    va_end(args);

#ifdef INTEL_CUSTOMIZATION
    bear_report_call(__func__, (char const **)argv);
#else
    bear_report_call(__func__, (char const *const *)argv);
#endif
    int const result =
        call_execve(path, (char *const *)argv, (char *const *)envp);

    bear_strings_release(argv);
    return result;
}
#endif

#ifdef HAVE_POSIX_SPAWN
int posix_spawn(pid_t *restrict pid, const char *restrict path,
                const posix_spawn_file_actions_t *file_actions,
                const posix_spawnattr_t *restrict attrp,
                char *const argv[restrict], char *const envp[restrict]) {
#ifdef INTEL_CUSTOMIZATION
    bear_report_call(__func__, (char const **)argv);
#else
    bear_report_call(__func__, (char const *const *)argv);
#endif
    return call_posix_spawn(pid, path, file_actions, attrp, argv, envp);
}
#endif

#ifdef HAVE_POSIX_SPAWNP
int posix_spawnp(pid_t *restrict pid, const char *restrict file,
                 const posix_spawn_file_actions_t *file_actions,
                 const posix_spawnattr_t *restrict attrp,
                 char *const argv[restrict], char *const envp[restrict]) {
#ifdef INTEL_CUSTOMIZATION
    bear_report_call(__func__, (char const **)argv);
#else
    bear_report_call(__func__, (char const *const *)argv);
#endif
    return call_posix_spawnp(pid, file, file_actions, attrp, argv, envp);
}
#endif

/* These are the methods which forward the call to the standard implementation.
 */

#ifdef HAVE_EXECVE
static int call_execve(const char *path, char *const argv[],
                       char *const envp[]) {
    typedef int (*func)(const char *, char *const *, char *const *);

    DLSYM(func, fp, "execve");

    char const **const menvp = bear_update_environment(envp, &initial_env);
    int const result = (*fp)(path, argv, (char *const *)menvp);
    bear_strings_release(menvp);
    return result;
}
#endif

#ifdef HAVE_EXECVPE
static int call_execvpe(const char *file, char *const argv[],
                        char *const envp[]) {
    typedef int (*func)(const char *, char *const *, char *const *);

    DLSYM(func, fp, "execvpe");

    char const **const menvp = bear_update_environment(envp, &initial_env);
    int const result = (*fp)(file, argv, (char *const *)menvp);
    bear_strings_release(menvp);
    return result;
}
#endif

#ifdef HAVE_EXECVP
static int call_execvp(const char *file, char *const argv[]) {
    typedef int (*func)(const char *file, char *const argv[]);

    DLSYM(func, fp, "execvp");

    bear_env_t current_env;
    bear_capture_env_t(&current_env);
    bear_reset_env_t(&initial_env);
    int const result = (*fp)(file, argv);
    bear_reset_env_t(&current_env);
    bear_release_env_t(&current_env);

    return result;
}
#endif

#ifdef HAVE_EXECVP2
static int call_execvP(const char *file, const char *search_path,
                       char *const argv[]) {
    typedef int (*func)(const char *, const char *, char *const *);

    DLSYM(func, fp, "execvP");

    bear_env_t current_env;
    bear_capture_env_t(&current_env);
    bear_reset_env_t(&initial_env);
    int const result = (*fp)(file, search_path, argv);
    bear_reset_env_t(&current_env);
    bear_release_env_t(&current_env);

    return result;
}
#endif

#ifdef HAVE_EXECT
static int call_exect(const char *path, char *const argv[],
                      char *const envp[]) {
    typedef int (*func)(const char *, char *const *, char *const *);

    DLSYM(func, fp, "exect");

    char const **const menvp = bear_update_environment(envp, &initial_env);
    int const result = (*fp)(path, argv, (char *const *)menvp);
    bear_strings_release(menvp);
    return result;
}
#endif

#ifdef HAVE_POSIX_SPAWN
static int call_posix_spawn(pid_t *restrict pid, const char *restrict path,
                            const posix_spawn_file_actions_t *file_actions,
                            const posix_spawnattr_t *restrict attrp,
                            char *const argv[restrict],
                            char *const envp[restrict]) {
    typedef int (*func)(pid_t *restrict, const char *restrict,
                        const posix_spawn_file_actions_t *,
                        const posix_spawnattr_t *restrict,
                        char *const *restrict, char *const *restrict);

    DLSYM(func, fp, "posix_spawn");

    char const **const menvp = bear_update_environment(envp, &initial_env);
    int const result =
        (*fp)(pid, path, file_actions, attrp, argv, (char *const *restrict)menvp);
    bear_strings_release(menvp);
    return result;
}
#endif

#ifdef HAVE_POSIX_SPAWNP
static int call_posix_spawnp(pid_t *restrict pid, const char *restrict file,
                             const posix_spawn_file_actions_t *file_actions,
                             const posix_spawnattr_t *restrict attrp,
                             char *const argv[restrict],
                             char *const envp[restrict]) {
    typedef int (*func)(pid_t *restrict, const char *restrict,
                        const posix_spawn_file_actions_t *,
                        const posix_spawnattr_t *restrict,
                        char *const *restrict, char *const *restrict);

    DLSYM(func, fp, "posix_spawnp");

    char const **const menvp = bear_update_environment(envp, &initial_env);
    int const result =
        (*fp)(pid, file, file_actions, attrp, argv, (char *const *restrict)menvp);
    bear_strings_release(menvp);
    return result;
}
#endif

#ifdef INTEL_CUSTOMIZATION
static int generate_file(char * filename){
    char buf[512];
    char cmd[512];
    int ret = 0;
    memset(cmd, '\0', 512);
    memset(buf, '\0', 512);
    int len=strlen(filename);
    if(len > 500) {
        perror("bear: generate file fail.");
        return -1;
    }
    strncpy(buf, filename, len);
    buf[len]='\0';
    while(len>0){
        if(buf[len]=='/'){
          buf[len]='\0';
          sprintf(cmd, "mkdir -p %s ", buf);
          ret = system(cmd);
          break;
        }
        len--;
    }
    FILE * fd = fopen(filename, "a+");
    if (0 == fd) {
        perror("bear: generate_file fopen fail.");
        return -1;
    }
    fprintf(fd, "emtpy-file");
    if (fclose(fd)) {
        perror("bear: fclose");
        return -1;
    }
    return ret;
}
// find xxx in "-o xxx"
// return value:
//  0 : found the project and create it.
//  1 : have not found the object, indicate next arg is object
//  -1: have not found the object.
int find_create_object(const char *str) {
    char *p = strstr(str, "-o");
    if(p && is_option_end(p + 2)) {
        p+=2;
        // skip emtpy
        while ((*p != '\0') && isblank(*p)) {
          p++;
        }
        if(*p == '\0'){
          return 1;
        }
        // find end of xxx.
        char *q=p;
        while( *q != '\0' && *q != ' ' && *q !='\t' ) {
          q++;
        }

        char ofilename[512];
        memset(ofilename, '\0', 512);
        memcpy(ofilename, p, q-p);
        ofilename[q-p]='\0';
        int ret = generate_file(ofilename);
        return ret;
    }
    return -1;
}

// check if 1st field of str contins =
int is_contain_eq(char *working) {
    while(*working != '\0') {
       if(isblank(*working))
            break;
       if(*working == '=')
            return 1;
       working++;
    }
    return 0;

}
// skip emtpy space.
char *skip_empty(char *working){
   if(working==NULL)
        return NULL;
   while(isblank(*working)) {
        working++;
   }
   return working;
}

// skip the option in "<option  value>"
char *skip_option(char *working) {
    if(working==NULL) {
        return NULL;
    }
    int len=strlen(working);
    while(len>0){
        len--;
        if(isblank(*working)) {
            working++;
            break;
        } else {
            working++;
        }
    }
    if(len==0) {
        return NULL;
    }
    return working;
}
// skip the value in "<option  value>"
char *skip_value(char *working){
    if(working==NULL) {
        return NULL;
    }
    int len=strlen(working);
    int require_quotation=0;
    while(len>0){
        len--;
        //process "
        if(require_quotation && *working != '"' ){
            working++;
            continue;
        }else if(require_quotation && *working == '"'){
            require_quotation=0;
            working++;
            continue;
        }
        if(*working== '"'){
            working++;
            require_quotation=1;
            continue;
        }
        //
        if(isblank(*working)) {
            working++;
            break;
        } else {
            working++;
        }

    }
    if(len==0) {
        return NULL;
    }
    return working;

}
// check if string is end
int is_finished(char *working){

    if(working == NULL || *working == '\0'|| *working == '|')
        return 1;
    else
        return 0;
}
int is_option_end(char* working){
    return  (isblank(working[0]) || working[0] == '\0');
}
int is_kv_option_special(char *working) {
    if((strncmp(working, "-D", 2) == 0 &&  !is_option_end(working + 2)) ||
        (strncmp(working, "-I", 2) == 0 &&  !is_option_end(working + 2)) ||
        (strncmp(working, "-O", 2) == 0 &&  !is_option_end(working + 2)) ||
        (strncmp(working, "-l", 2) == 0 &&  !is_option_end(working + 2))
        ) {
        return 1;
    }
    return 0;
}
int is_single_option(char *working){
    if(    strncmp(working, "--version", 9)==0
        || (strncmp(working, "-V" , 2)==0 && is_option_end(working+2))
        || strncmp(working, "--help", 6)==0
        || (strncmp(working, "-h" , 2)==0 && is_option_end(working+2))
        || strncmp(working, "--no-compress", 13)==0
        || strncmp(working, "-no-compress" , 12)==0
        || strncmp(working, "--extensible-whole-program", 26)==0
        || strncmp(working, "-ewp", 4)==0
        || strncmp(working, "--resource-usage", 16)==0
        || strncmp(working, "-res-usage", 10)==0
        || strncmp(working, "--Werror", 8)==0
        || strncmp(working, "-Werror", 7)==0
        || strncmp(working, "--Wno-deprecated-gpu-targets", 28)==0
        || strncmp(working, "-Wno-deprecated-gpu-targets", 27)==0
        || strncmp(working, "--Wno-deprecated-declarations", 29)==0
        || strncmp(working, "-Wno-deprecated-declarations", 28)==0
        || strncmp(working, "--Wreorder", 10)==0
        || strncmp(working, "-Wreorder", 9)==0
        || strncmp(working, "--restrict", 10)==0
        || strncmp(working, "-restrict", 9)==0
        || strncmp(working, "--source-in-ptx", 15)==0
        || strncmp(working, "-src-in-ptx", 11)==0
        || strncmp(working, "--keep-device-functions", 23)==0
        || strncmp(working, "-keep-device-functions", 22)==0
        || strncmp(working, "--disable-warnings", 18)==0
        || (strncmp(working, "-w", 2)==0 && is_option_end(working+2))
        || strncmp(working, "--use_fast_math", 15)==0
        || strncmp(working, "-use_fast_math", 14)==0
        || strncmp(working, "--no-device-link", 16)==0
        || strncmp(working, "-nodlink", 8)==0
        || strncmp(working, "--no-align-double", 17)==0
        || strncmp(working, "--no-align-double", 17)==0
        || strncmp(working, "--clean-targets", 15)==0
        || strncmp(working, "-clean", 6)==0
        || strncmp(working, "--save-temps", 12)==0
        || strncmp(working, "-save-temps", 11)==0
        || strncmp(working, "--keep", 6)==0
        || strncmp(working, "-keep", 5)==0
        || strncmp(working, "--verbose", 9)==0
        || (strncmp(working, "-v", 2)==0 && is_option_end(working+2))
        || strncmp(working, "--dryrun", 8)==0
        || strncmp(working, "-dryrun", 7)==0
        || strncmp(working, "--dont-use-profile", 18)==0
        || strncmp(working, "-noprof", 7)==0
        || strncmp(working, "--expt-extended-lambda", 22)==0
        || strncmp(working, "-expt-extended-lambda", 20)==0
        || strncmp(working, "--expt-relaxed-constexpr", 24)==0
        || strncmp(working, "-expt-relaxed-constexpr", 23)==0
        || strncmp(working, "--no-host-device-move-forward", 29)==0
        || strncmp(working, "-nohdmoveforward", 16)==0
        || strncmp(working, "--no-host-device-initializer-list", 33)==0
        || strncmp(working, "-nohdinitlist", 13)==0
        || strncmp(working, "--shared", 8)==0
        || strncmp(working, "-shared", 7)==0
        || strncmp(working, "--generate-line-info", 20)==0
        || strncmp(working, "-lineinfo", 9)==0
        || strncmp(working, "--device-debug", 14)==0
        || (strncmp(working, "-G", 2)==0 && is_option_end(working+2))
        || strncmp(working, "--debug", 7)==0
        || (strncmp(working, "-g", 2)==0 && is_option_end(working+2))
        || strncmp(working, "--profile", 9)==0
        || strncmp(working, "-pg", 3)==0
        || strncmp(working, "--use-local-env", 15)==0
        || strncmp(working, "--use-local-env", 15)==0
        || strncmp(working, "--run", 5)==0
        || strncmp(working, "-run", 4)==0
        || strncmp(working, "--lib", 5)==0
        || strncmp(working, "-lib", 4)==0
        || strncmp(working, "--link", 6)==0
        || strncmp(working, "-link", 5)==0
        || strncmp(working, "--device-link", 13)==0
        || strncmp(working, "-dlink", 6)==0
        || strncmp(working, "--device-w", 10)==0
        || strncmp(working, "-dw", 3)==0
        || strncmp(working, "--device-c", 10)==0
        || strncmp(working, "-dc", 3)==0
        || strncmp(working, "--compile", 9)==0
        || (strncmp(working, "-c", 2)==0 && is_option_end(working+2) )
        || strncmp(working, "--dependency-output", 19)==0
        || strncmp(working, "-MF", 3)==0
        || strncmp(working, "--generate-nonsystem-dependencies", 33)==0
        || strncmp(working, "-MM", 3)==0
        || strncmp(working, "--generate-dependencies", 23)==0
        || (strncmp(working, "-M", 2)==0 && is_option_end(working+2))
        || strncmp(working, "--preprocess", 12)==0
        || (strncmp(working, "-E", 2)==0 && is_option_end(working+2))
        || strncmp(working, "--ptx", 5)==0
        || strncmp(working, "-ptx", 4)==0
        || strncmp(working, "--fatbin", 8)==0
        || strncmp(working, "-fatbin", 7)==0
        || strncmp(working, "--cubin", 7)==0
        || strncmp(working, "-cubin", 6)==0
        || strncmp(working, "--cuda", 6)==0
        || strncmp(working, "-cuda", 5)==0
      ){
        return 1;
    }
    return 0;
  }
int is_c_option(char *working) {
    if(strncmp(working, "--compile", 9)==0
        || (strncmp(working, "-c" , 2)==0 && is_option_end(working+2) )){
        return 1;
    }else {
        return 0;
    }
}

// Target to identify the inputfile in the command.
//   command: the command that be parsed.
//   c_found: return whether -c option is used.
//   inputfile: return the inputfile in the command.
int parse_input_file(char *command, int *c_found, char *inputfile) {
    // this function try to find the inputfile from command.
    // 1. must have -c|--compile option available.
    // 2. option format:
    //   <[-|--]option>
    //   <[-|--]option> <value>:  values may contains ", eg. "-O2 "
    //   <[-|--]option=value>
    // return: 1 if parse out input file.
    char *working=command;
    int ret=0;
    //skip the exec-name
    working=skip_empty(working);
    working=skip_option(working);

    while(1) {
        working=skip_empty(working);
        if(is_finished(working)){
            return ret;
        }
        // is option
        if(*working=='-') {
            //process options
            if(is_single_option(working) || is_kv_option_special(working)) { // single option
                if(is_c_option(working)){
                    *c_found=1;
                }
                working=skip_option(working);
                working=skip_empty(working);
                if(is_finished(working)){
                    return ret;
                }
            } else { //option=value
                if(is_contain_eq(working)) {
                    working=skip_value(working);
                    working=skip_empty(working);
                } else {
                    //option,value
                    working=skip_option(working);
                    working=skip_empty(working);
                    working=skip_value(working);
                    working=skip_empty(working);
                }
                if(is_finished(working)){
                    return ret;
                }
            }
            continue;
        }
        // is input file.
        int len=0;
        char *begin=working;
        while(*working!=' ' && *working !='\t' && *working !='\0') {
            working++;
            len++;
        }
        memcpy(inputfile, begin, len);
        ret=1;

        working=skip_empty(working);
        if(is_finished(working)){
            return ret;
        }
    }
    return ret;
}

// dump the command and options to the trace file which
// will be parsed by scprits in scan-build-py.
// '<option>' | 'file' are expected to write out to the fd.
// while for 'file ' and 'file )' need to remove the space and ).
void dump_US_field(const char *str, FILE *fd, int US, int has_parenthesis){
   char *working=(char *)str;
   char *begin;
   working=skip_empty(working);

   if(working==NULL) {
        return;
   }

   char  tmpbuf[512];
   memset(tmpbuf, '\0', 512);
   begin=working;
   while(*working!='\0'){
       if(isblank(*working)) {
           memcpy(tmpbuf, begin, working-begin);
           tmpbuf[working-begin]='\0';
           // remove the right ).
           if(has_parenthesis) {
               char *p=tmpbuf;
               while(*p!='\0') {
                   if(*p==')') {
                       *p='\0';
                    }
                    p++;
                }
           }
           fprintf(fd, "%s%c", tmpbuf, US);
           //skip the emtpy
           while(isblank(*working)) {
               working++;
           }
           if(*working=='\0') {
               return;
           }
           begin=working;
       } else {
           working++;
       }
   }
   memcpy(tmpbuf, begin, working-begin);
   tmpbuf[working-begin]='\0';
   // remove the right ).
   if(has_parenthesis) {
       char *p=tmpbuf;
       while(*p!='\0') {
           if(*p==')') {
               *p='\0';
           }
           p++;
       }
   }
   fprintf(fd, "%s%c", tmpbuf, US);
   return;
}

/// Find command "nvcc" in \p str, and return the position of the character behind "nvcc".
/// e.g: str could be: "/usr/local/bin/nvcc  -Xcompiler ...",
///                    "/usr/local/bin/nvcc/gcc  -Xcompiler ...".
/// \returns the position of the character behind "nvcc" in str,
///          or NULL if no command "nvcc" found in str.
const char *find_nvcc(const char *str) {
  const char *pos = NULL;
  const char *ret = NULL;

  for (const char *ptr = str; *ptr != '\0'; ptr++) {
    if (isspace(*ptr)) {
      pos = ptr;
      break;
    }
  }

  if (pos == NULL) {
    int len = strlen(str);
    if (len >= 4 && str[len - 1] == 'c' && str[len - 2] == 'c' &&
        str[len - 3] == 'v' && str[len - 4] == 'n') {
      ret = str + len;
    }
  } else {
    int len = pos - str;
    if (len >= 4 && *(pos - 1) == 'c' && *(pos - 2) == 'c' &&
        *(pos - 3) == 'v' && *(pos - 4) == 'n') {
      ret = pos;
    }
  }

  return ret;
}

/// Repalce the command "nvcc" with path to command "intercept-stub" with path.
/// \param [in] src It could be:"/usr/local/bin/nvcc",
///                             "/usr/local/bin/nvcc -Xcompiler ...",
///                             "CPATH=...;/path/to/nvcc",
///                             "cd /path/to/dir && /path/to/nvcc".
/// \param [in] pos Points to the position of the character behind "nvcc".
/// \returns no return value.
char * replace_binary_name(const char *src, const char *pos){
    FILE *fp;
    char replacement[2048];
    fp = popen("which dpct", "r");
    if (fp == NULL) {
        perror("bear: failed to run command 'which dpct'\n" );
        exit(EXIT_FAILURE);
    }

    if(fgets(replacement, sizeof(replacement), fp) == NULL) {
        perror("bear: fgets\n" );
        exit(EXIT_FAILURE);
    }
    pclose(fp);
    strcpy(replacement + strlen(replacement) - strlen("bin/dpct") - 1, "libear/intercept-stub");

    char *buffer = (char *)malloc(strlen(src) + strlen(replacement) - strlen("nvcc"));
    char *insert_point = buffer;

    // To handle the situation that \psrc is
    // "CPATH=...;/path/to/nvcc" and "cd /path/to/dir && /path/to/nvcc"
    char *pos_prefix = pos - strlen("nvcc");
    for (; pos_prefix != src; pos_prefix--) {
        if (*pos_prefix == ';' || *pos_prefix == '&') {
            pos_prefix++;
            break;
        }
    }

    int len = pos_prefix - src;
    memcpy(insert_point, src, len);
    insert_point += len;
    memcpy(insert_point, replacement, strlen(replacement));
    insert_point += strlen(replacement);
    src = pos;
    strcpy(insert_point, src);
    return buffer;
}

int is_ubuntu_platform(void) {
  FILE *fp = NULL;
  char buffer[1024] = "";

  fp = popen("lsb_release -ds", "r");

  if (fp == NULL) {
    fp = popen("cat /proc/version", "r");
    if (fp == NULL) {
      perror("bear: failed to get Linux distribution info\n");
      exit(EXIT_FAILURE);
    }
  }

  if (fgets(buffer, 1024, fp) == NULL) {
    perror("bear: fgets\n");
    exit(EXIT_FAILURE);
  }
  pclose(fp);

  if (strstr(buffer, "Ubuntu")) {
    return 1;
  }
  return 0;
}

#endif

/* this method is to write log about the process creation. */
#ifdef INTEL_CUSTOMIZATION
// To indicate whether current OS is a Ubuntu system or other system.
// Value: 1 ubuntu system, Value: 0 other system.
static int ubuntu_platform = 0;

// To make sure global ubuntu_platform is only initialized once.
static int platform_initialized = 0;

static void bear_report_call(char const *fun, char const *argv[]) {
#else
static void bear_report_call(char const *fun, char const *const argv[]) {
#endif
    static int const GS = 0x1d;
    static int const RS = 0x1e;
    static int const US = 0x1f;

    if (!initialized)
#ifdef INTEL_CUSTOMIZATION
        initialized = bear_capture_env_t(&initial_env);
#else
        return;
#endif

    pthread_mutex_lock(&mutex);
    const char *cwd = getcwd(NULL, 0);
    if (0 == cwd) {
        perror("bear: getcwd");
        pthread_mutex_unlock(&mutex);
        exit(EXIT_FAILURE);
    }
    char const * const out_dir = initial_env[0];
    size_t const path_max_length = strlen(out_dir) + 32;
    char filename[path_max_length];
    if (-1 == snprintf(filename, path_max_length, "%s/%d.cmd", out_dir, getpid())) {
        perror("bear: snprintf");
        pthread_mutex_unlock(&mutex);
        exit(EXIT_FAILURE);
    }
    FILE * fd = fopen(filename, "a+");
    if (0 == fd) {
        perror("bear: fopen");
        pthread_mutex_unlock(&mutex);
        exit(EXIT_FAILURE);
    }
    fprintf(fd, "%d%c", getpid(), RS);
    fprintf(fd, "%d%c", getppid(), RS);
    fprintf(fd, "%s%c", fun, RS);
    fprintf(fd, "%s%c", cwd, RS);
    size_t const argc = bear_strings_length(argv);

#ifdef INTEL_CUSTOMIZATION
    // To indicate whether the captured argv[i] is a nvcc or ld command,
    // value: 1 yes, value 0 no.
    int is_nvcc_or_ld=0;

    // To indicate whether the object file has been fake generated,
    // value: 1 obj file generated, value: 0 not generated.
    int flag_object=0;

    // flag_optval is use for case: for options "-o xxx.o", "-o" and "xxx.o" is in
    // argv[i] and argv[i+1],  if "-o" is found in argv[i], then flag_optval
    // is set to show argv[i+1] contains the xxx.o
    int flag_optval=0;

    // value 1: means current command line is a nvcc comand, and the fake obj file
    // has been created, else ret is set to 0.
    int ret = 0;

    char *command_cp=NULL;
    size_t it_cp=0;
    // (CPATH=;command  args), need remove () around the command
    int has_parenthesis=0;

    if(strstr(argv[0], "nvcc") && !platform_initialized) {
        ubuntu_platform = is_ubuntu_platform();
        platform_initialized = 1;
    }
    // try to parse out nvcc and generate obj_file.
    for (size_t it = 0; it < argc; ++it) {
        const char *tail=argv[it];
        int len= strlen(tail);
        char *command=NULL;
        if(it<=3 /*eg. /bin/bash -c [CPATH=xxx;]command*/ && is_nvcc_or_ld==0 &&
                ((command=strstr(tail, "nvcc"))!=NULL)) {
          command_cp=command;
          it_cp=it;
          is_nvcc_or_ld=1;
          const char *tmpp=tail;
          while(tmpp!=command) {
            if(*tmpp=='(') {
                has_parenthesis=1;
                break;
            }
            tmpp++;
          }
          fprintf(fd, "%s%c", "nvcc", US);
        } else if((len ==2 && tail[0]=='l' && tail[1] =='d') ||
                  (len > 2 && tail[len-3]=='/' && tail[len-2] =='l' && tail[len-1] =='d')) {
            is_nvcc_or_ld=1;
            for(size_t i=it; i< argc; i++){
                if(strcmp(argv[i], "-o") == 0){
                    char ofilename[512];
                    int olen=strlen(argv[i+1]);
                    memset(ofilename,'\0',512);
                    if(olen >= 512) {
                        perror("bear: filename length too long.");
                        pthread_mutex_unlock(&mutex);
                        exit(EXIT_FAILURE);
                    }
                    strncpy(ofilename,argv[i+1], olen);
                    if(generate_file(ofilename) != 0) {
                        pthread_mutex_unlock(&mutex);
                        exit(EXIT_FAILURE);
                    }
                    flag_object=1;
                }
            }
        }
        if(flag_optval==1){
          char ofilename[512];
          int olen=strlen(argv[it]);
          memset(ofilename,'\0',512);
          if(olen >= 512) {
            perror("bear: filename length too long.");
            pthread_mutex_unlock(&mutex);
            exit(EXIT_FAILURE);
          }
          strncpy(ofilename,argv[it], olen);
          if(generate_file(ofilename) != 0) {
            pthread_mutex_unlock(&mutex);
            exit(EXIT_FAILURE);
          }
          flag_optval=0;
          flag_object=1;
        }
        if(flag_object==0) {
          // here we need parse out the object file if -o option is used.
          // find xxx in the -o xxx of the command, generate it.
          int r=find_create_object(tail);
          if(r==0){
            flag_object=1;
          }
          if(r==1){
            flag_optval=1;
          }
        }

    }
    for (size_t it = it_cp; it < argc; ++it) {
        if(it==it_cp && command_cp!=NULL) {
           dump_US_field(command_cp + strlen("nvcc"), fd, US, has_parenthesis);
        } else {
           dump_US_field(argv[it], fd, US, has_parenthesis);
        }
    }
#else
    for (size_t it = 0; it < argc; ++it) {
        fprintf(fd, "%s%c", argv[it], US);
    }
#endif

    fprintf(fd, "%c", GS);
    if (fclose(fd)) {
        perror("bear: fclose");
        pthread_mutex_unlock(&mutex);
        exit(EXIT_FAILURE);
    }
    free((void *)cwd);
    pthread_mutex_unlock(&mutex);
#ifdef INTEL_CUSTOMIZATION
    if(is_nvcc_or_ld == 1 && flag_object == 1){
      ret=1;
    } else  if(is_nvcc_or_ld == 1) {
        // object is not give by -o. Need figure out the default output for cmd "gcc -c xx.c"
        char *tmp=malloc(4096);
        if(tmp==NULL) {
            perror("bear: malloc memory fail.");
            exit(EXIT_FAILURE);
        }
        memset(tmp, '\0', 4096);
        int idx = 0;
        int c_option;
        char ofilename[512];
        memset(ofilename, '\0', 512);
        int parse_ret = 0;
        for (size_t it = it_cp; it < argc; ++it) {
           memcpy(tmp+idx, argv[it], strlen(argv[it]));
           idx+=strlen(argv[it]);
           tmp[idx]=' ';
           idx++;
        }
        parse_ret=parse_input_file(tmp, &c_option, ofilename);
        if(parse_ret==1 && c_option==1) {
            //change the suffix of the ofilename from .c .cpp => .o)
            int olen=strlen(ofilename);
            while(olen>=0 && ofilename[olen]!= '.') {
                olen--;
            }
            if(olen==-1) {
                olen=strlen(ofilename);
                if(olen>499) {
                    perror("bear: filename length too long.");
                    exit(EXIT_FAILURE);
                }
                ofilename[olen] = '.';
                ofilename[olen+1] = 'o';
                ofilename[olen+2] = '\0';
            } else {
                if(olen>500) {
                    perror("bear: filename length too long.");
                    exit(EXIT_FAILURE);
                }
                ofilename[olen+1] = 'o';
                ofilename[olen+2] = '\0';
            }

          pthread_mutex_lock(&mutex);
          if(generate_file(ofilename) != 0) {
            pthread_mutex_unlock(&mutex);
            exit(EXIT_FAILURE);
          }
          pthread_mutex_unlock(&mutex);
        }
        free(tmp);
        ret = 1;
    }

    // try to replace nvcc with intercept-stub,
    // e.g: "/path/to/nvcc -ccbin ... ",
    //      "/bin/sh -c "/path/to"/bin/nvcc -ccbin ..."
    const char *pos = find_nvcc(argv[it_cp]);

    int is_stub_need = 0;
    if (ubuntu_platform) {
        is_stub_need = (pos != NULL && *pos != '\0');
    } else {
        is_stub_need = (pos != NULL);
    }

    if(is_stub_need)
    {
        ret = 0; // intercept-stub should continue to run.

        // intercept-stub is used to handle the nvcc command like
        // "/bin/sh -c nvcc -c `echo ./`hello.c", it will change the nvcc command to
        // "/bin/sh -c /path/to/libear/intercept-stub -c `echo ./`hello.c", then the coming
        // command "/path/to/libear/intercept-stub -c ./hello.c" will be run, and the source file
        // name "hello.c" will be captured by intercept.py.
        argv[it_cp] = replace_binary_name(argv[it_cp], pos);
    }

    if(ret == 1 && it_cp == 0){
        exit(0);
    }
#endif
}

/* update environment assure that chilren processes will copy the desired
 * behaviour */

static int bear_capture_env_t(bear_env_t *env) {
    int status = 1;
    for (size_t it = 0; it < ENV_SIZE; ++it) {
        char const * const env_value = getenv(env_names[it]);
        char const * const env_copy = (env_value) ? strdup(env_value) : env_value;
        (*env)[it] = env_copy;
        status &= (env_copy) ? 1 : 0;
    }
    return status;
}

static int bear_reset_env_t(bear_env_t *env) {
    int status = 1;
    for (size_t it = 0; it < ENV_SIZE; ++it) {
        if ((*env)[it]) {
            setenv(env_names[it], (*env)[it], 1);
        } else {
            unsetenv(env_names[it]);
        }
    }
    return status;
}

static void bear_release_env_t(bear_env_t *env) {
    for (size_t it = 0; it < ENV_SIZE; ++it) {
        free((void *)(*env)[it]);
        (*env)[it] = 0;
    }
}

static char const **bear_update_environment(char *const envp[], bear_env_t *env) {
    char const **result = bear_strings_copy((char const **)envp);
    for (size_t it = 0; it < ENV_SIZE && (*env)[it]; ++it)
        result = bear_update_environ(result, env_names[it], (*env)[it]);
    return result;
}

static char const **bear_update_environ(char const *envs[], char const *key, char const * const value) {
    // find the key if it's there
    size_t const key_length = strlen(key);
    char const **it = envs;
    for (; (it) && (*it); ++it) {
        if ((0 == strncmp(*it, key, key_length)) &&
            (strlen(*it) > key_length) && ('=' == (*it)[key_length]))
            break;
    }
    // allocate a environment entry
    size_t const value_length = strlen(value);
    size_t const env_length = key_length + value_length + 2;
    char *env = malloc(env_length);
    if (0 == env) {
        perror("bear: malloc [in env_update]");
        exit(EXIT_FAILURE);
    }
    if (-1 == snprintf(env, env_length, "%s=%s", key, value)) {
        perror("bear: snprintf");
        exit(EXIT_FAILURE);
    }
    // replace or append the environment entry
    if (it && *it) {
        free((void *)*it);
        *it = env;
	return envs;
    }
    return bear_strings_append(envs, env);
}

static char **bear_get_environment() {
#if defined HAVE_NSGETENVIRON
    return *_NSGetEnviron();
#else
    return environ;
#endif
}

/* util methods to deal with string arrays. environment and process arguments
 * are both represented as string arrays. */

static char const **bear_strings_build(char const *const arg, va_list *args) {
    char const **result = 0;
    size_t size = 0;
    for (char const *it = arg; it; it = va_arg(*args, char const *)) {
        result = realloc(result, (size + 1) * sizeof(char const *));
        if (0 == result) {
            perror("bear: realloc");
            exit(EXIT_FAILURE);
        }
        char const *copy = strdup(it);
        if (0 == copy) {
            perror("bear: strdup");
            exit(EXIT_FAILURE);
        }
        result[size++] = copy;
    }
    result = realloc(result, (size + 1) * sizeof(char const *));
    if (0 == result) {
        perror("bear: realloc");
        exit(EXIT_FAILURE);
    }
    result[size++] = 0;

    return result;
}

static char const **bear_strings_copy(char const **const in) {
    size_t const size = bear_strings_length(in);

    char const **const result = malloc((size + 1) * sizeof(char const *));
    if (0 == result) {
        perror("bear: malloc");
        exit(EXIT_FAILURE);
    }

    char const **out_it = result;
    for (char const *const *in_it = in; (in_it) && (*in_it);
         ++in_it, ++out_it) {
        *out_it = strdup(*in_it);
        if (0 == *out_it) {
            perror("bear: strdup");
            exit(EXIT_FAILURE);
        }
    }
    *out_it = 0;
    return result;
}

static char const **bear_strings_append(char const **const in,
                                        char const *const e) {
    size_t size = bear_strings_length(in);
    char const **result = realloc(in, (size + 2) * sizeof(char const *));
    if (0 == result) {
        perror("bear: realloc");
        exit(EXIT_FAILURE);
    }
    result[size++] = e;
    result[size++] = 0;
    return result;
}

static size_t bear_strings_length(char const *const *const in) {
    size_t result = 0;
    for (char const *const *it = in; (it) && (*it); ++it)
        ++result;
    return result;
}

static void bear_strings_release(char const **in) {
    for (char const *const *it = in; (it) && (*it); ++it) {
        free((void *)*it);
    }
    free((void *)in);
}
