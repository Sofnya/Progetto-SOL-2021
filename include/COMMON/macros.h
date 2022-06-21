#ifndef MACROS_H
#define MACROS_H
#include <stdlib.h>
#include <errno.h>
#include <stdio.h>

#define ERROR_CHECK(arg) if((arg) == -1) { perror("Error"); exit(EXIT_FAILURE); }
#define SAFE_ERROR_CHECK(arg) if((arg) == -1) { perror("Error"); return -1; }
#define UNSAFE_NULL_CHECK(arg) if((arg) == NULL) { perror("Error"); exit(EXIT_FAILURE); }
#define SAFE_NULL_CHECK(arg) if((arg) == NULL) { perror("Error"); return -1; }
#define PTHREAD_CHECK(arg) if((errno = (arg)) != 0) { perror("Error on a pthread call"); exit(EXIT_FAILURE); }
#define READ_CHECK(arg) if((arg) <= 0) {if(errno != 0) perror("Error on a read"); return -1;}

#define CLEANUP_ERROR_CHECK(arg, cleanup) if((arg) == -1) {perror("Error"); cleanup; return -1;}

#define CLEANUP_CHECK(arg, err, cleanup) if((arg) == (err)) {perror("Error"); cleanup; return -1;}
#endif