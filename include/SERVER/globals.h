#ifndef GLOBALS_H
#define GLOBALS_H
#define UNIX_PATH_MAX 108

#include <stdint.h>

#include <SERVER/policy.h>

extern char SOCK_NAME[UNIX_PATH_MAX];
extern char LOG_FILE[UNIX_PATH_MAX];
extern int64_t CORE_POOL_SIZE;
extern int64_t MAX_POOL_SIZE;
extern int64_t MAX_FILES;
extern int64_t MAX_MEMORY;
extern int ENABLE_COMPRESSION;
extern int VERBOSE_PRINT;
extern int POLICY;

void load_config(char *path);
#endif