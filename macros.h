#ifndef MACROS_H
#define MACROS_H
#include <stdlib.h>
#include <errno.h>
#include <stdio.h>

#define ERROR_CHECK(arg) if((arg) == -1) { perror("Error"); exit(EXIT_FAILURE); }
#define UNSAFE_NULL_CHECK(arg) if((arg) == NULL) { perror("Error"); exit(EXIT_FAILURE); }
#define SAFE_NULL_CHECK(arg) if((arg) == NULL) { return -1; }
#define PTHREAD_CHECK(arg) if((errno = (arg)) != 0) { perror("Error on a pthread call"); exit(EXIT_FAILURE); }

#endif