#include <time.h>
#include <pthread.h>
#include <errno.h>
#include <stdio.h>


#include "COMMON/helpers.h"
#include "COMMON/macros.h"


int timeoutCall(int (*fnc)(void *), void *arg, struct timespec maxWait)
{
    struct timespec absTime;
    pthread_t pid;
    pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t done = PTHREAD_COND_INITIALIZER;
    struct _Hexec exec;
    volatile int isDone = 0;
    volatile int result = 0;
    int err = 0;

    exec.fnc = fnc;
    exec.arg = arg;
    exec.done = &done;
    exec.isDone = &isDone;
    exec.result = &result;


    PTHREAD_CHECK(pthread_mutex_lock(&mtx));

    clock_gettime(CLOCK_REALTIME, &absTime);
    absTime.tv_sec += maxWait.tv_sec;
    absTime.tv_nsec += maxWait.tv_nsec;

    PTHREAD_CHECK(pthread_create(&pid, NULL, _innerCall, (void *)&exec));


    while(!isDone)
    {
        err = pthread_cond_timedwait(&done, &mtx, &absTime);
    }
    PTHREAD_CHECK(pthread_mutex_unlock(&mtx));

    if(err) 
    {
        puts("Error, cancelling!");
        pthread_cancel(pid);
        puts("Joining...");
        PTHREAD_CHECK(pthread_join(pid, NULL));
        puts("Done?");
        errno = err;
        return 0;
    }

    PTHREAD_CHECK(pthread_join(pid, NULL));
    errno = 0;
    return *(exec.result);
}


void *_innerCall(void *arg)
{   
    struct _Hexec exec = * (struct _Hexec *)arg;

    PTHREAD_CHECK(pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL));

    *(exec.result) = exec.fnc(exec.arg);

    *(exec.isDone) = 1;
    PTHREAD_CHECK(pthread_cond_signal(exec.done));
    return NULL;
}
