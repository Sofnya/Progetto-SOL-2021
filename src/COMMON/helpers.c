#include <time.h>
#include <pthread.h>
#include <errno.h>
#include <stdio.h>
#include <signal.h>
#include <time.h>

#include <unistd.h>
#include <sys/syscall.h>

#include "COMMON/helpers.h"
#include "COMMON/macros.h"

/**
 * @brief Evaluates fnc(arg), terminating either on evaluation termination or after a timeout.
 *
 * @param fnc the function to be evaluated.
 * @param arg the arguments on which to evaluate fnc.
 * @param maxWait how many seconds/nanoseconds to wait since starting to evaluate fnc.
 * @return int the result of fnc on success, with errno=0, on failure returns 0 and sets errno.
 */
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

    // All threads except the main root thread should ignore termination signals, and let the root thread handle termination.
    sigset_t sigmask;
    sigemptyset(&sigmask);
    sigaddset(&sigmask, SIGINT);
    sigaddset(&sigmask, SIGHUP);
    sigaddset(&sigmask, SIGQUIT);
    pthread_sigmask(SIG_BLOCK, &sigmask, NULL);

    exec.fnc = fnc;
    exec.arg = arg;
    exec.done = &done;
    exec.isDone = &isDone;
    exec.result = &result;

    // We need to hold a mutex while waiting on a condition.
    PTHREAD_CHECK(pthread_mutex_lock(&mtx));

    // Start the clock, by creating a timespec which tells us when to stop waiting.
    clock_gettime(CLOCK_REALTIME, &absTime);
    absTime.tv_sec += maxWait.tv_sec;
    absTime.tv_nsec += maxWait.tv_nsec;

    // Start the evaluation, through special _innerCall wrapper.
    PTHREAD_CHECK(pthread_create(&pid, NULL, _innerCall, (void *)&exec));

    // Wait on the done condition.
    while (!isDone)
    {
        err = pthread_cond_timedwait(&done, &mtx, &absTime);
        if (err != 0)
        {
            break;
        }
    }
    PTHREAD_CHECK(pthread_mutex_unlock(&mtx));

    if (err)
    {
        pthread_cancel(pid);
        PTHREAD_CHECK(pthread_join(pid, NULL));
        errno = err;
        return 0;
    }

    PTHREAD_CHECK(pthread_join(pid, NULL));
    errno = 0;
    return *(exec.result);
}

/**
 * @brief Evaluates fnc(arg), and notifies the caller when it's done.
 *
 * @param arg a valid _Hexec structure.
 * @return void* nothing.
 */
void *_innerCall(void *arg)
{
    struct _Hexec exec = *(struct _Hexec *)arg;

    // We need to be cancellable always. A timeout call shouln't be used on a function that's not safe to cancel.
    NULL_PTHREAD_CHECK(pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL));

    // The actual execution. We store the result in exec.result, to which the caller holds a pointer.
    *(exec.result) = exec.fnc(exec.arg);

    // And we notify the caller that we are done.
    *(exec.isDone) = 1;
    NULL_PTHREAD_CHECK(pthread_cond_signal(exec.done));
    return NULL;
}

/**
 * @brief Generates an UUID.
 *
 * @param uuid where the UUID will be generated.
 */
void genUUID(char *uuid)
{
    sprintf(uuid, "%x%x-%x-%x-%x-%x%x%x",
            rand(), rand(),               // Generates a 64-bit Hex number
            rand(),                       // Generates a 32-bit Hex number
            ((rand() & 0x0fff) | 0x4000), // Generates a 32-bit Hex number of the form 4xxx (4 indicates the UUID version)
            rand() % 0x3fff + 0x8000,     // Generates a 32-bit Hex number in the range [0x8000, 0xbfff]
            rand(), rand(), rand());      // Generates a 96-bit Hex number
};

/**
 * @brief Gets the current thread id.
 *
 * @return long the current thread id.
 */
long getTID()
{
    return syscall(SYS_gettid);
}