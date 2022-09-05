#ifndef HELPERS_H
#define HELPERS_H

#include <time.h>
#include <pthread.h>

struct _Hexec
{
    int (*fnc)(void *);
    void *arg;
    volatile int *result;
    pthread_cond_t *done;
    volatile int *isDone;
};

int timeoutCall(int (*fnc)(void *), void *arg, struct timespec maxWait);

void *_innerCall(void *exec);

#endif