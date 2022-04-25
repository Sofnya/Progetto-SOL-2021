#ifndef GLOBALS_H
#define GLOBALS_H
#define UNIX_PATH_MAX 108

extern char SOCK_NAME[UNIX_PATH_MAX];
extern long POOL_SIZE;
extern long long MAX_FILES;
extern long long MAX_MEMORY;


void load_config(char *path);
#endif