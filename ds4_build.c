#include "ds4.h"

#include <string.h>

/* The Makefile passes DS4_BUILD_GIT_SHA as a bare token (no embedded quotes):
 * quoted -D values lose their quotes in recursive targets, where CFLAGS goes
 * through one extra shell evaluation. Stringify it here instead. */
#ifndef DS4_BUILD_GIT_SHA
#define DS4_BUILD_GIT_SHA unknown
#endif
#define DS4_BUILD_STR2(x) #x
#define DS4_BUILD_STR(x) DS4_BUILD_STR2(x)

const char *ds4_build_backend(void) {
#ifdef DS4_NO_GPU
    return "cpu";
#elif defined(__APPLE__)
    return "metal";
#elif defined(DS4_ROCM_BUILD)
    return "rocm";
#else
    return "cuda";
#endif
}

const char *ds4_build_arch(void) {
#if defined(__aarch64__) || defined(__arm64__)
    return "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#else
    return "unknown";
#endif
}

const char *ds4_build_git_sha(void) {
    return DS4_BUILD_STR(DS4_BUILD_GIT_SHA);
}

void ds4_build_info_print(FILE *fp) {
    if (!fp) fp = stdout;
    fprintf(fp,
            "ds4 build\n"
            "git:     %s\n"
            "backend: %s\n"
            "arch:    %s\n",
            ds4_build_git_sha(),
            ds4_build_backend(),
            ds4_build_arch());
}

bool ds4_build_info_requested(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        if (argv[i] && strcmp(argv[i], "--build-info") == 0) return true;
    }
    return false;
}
