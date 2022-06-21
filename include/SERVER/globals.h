#ifndef GLOBALS_H
#define GLOBALS_H
#define UNIX_PATH_MAX 108


#include <stdint.h>


extern char SOCK_NAME[UNIX_PATH_MAX];
extern char LOG_FILE[UNIX_PATH_MAX];
extern uint64_t CORE_POOL_SIZE;
extern uint64_t MAX_POOL_SIZE;
extern uint64_t MAX_FILES;
extern uint64_t MAX_MEMORY;



void load_config(char *path);
#endif