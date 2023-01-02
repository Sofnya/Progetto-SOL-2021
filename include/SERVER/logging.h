#ifndef LOGGING_H
#define LOGGING_H
#include <pthread.h>

extern volatile pthread_mutex_t LOGLOCK;
int logger(char *msg, char *type);

#endif