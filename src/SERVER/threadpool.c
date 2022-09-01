#include <pthread.h>
#include <stdbool.h>
#include <assert.h>
#include <unistd.h>
#include <signal.h>

#include "SERVER/threadpool.h"
#include "COMMON/macros.h"
#include "SERVER/logging.h"

void threadpoolInit(size_t coreSize, size_t maxSize, ThreadPool *pool)
{
    pool->_coreSize = coreSize;
    pool->_maxSize = maxSize;

    UNSAFE_NULL_CHECK(pool->alive = malloc(sizeof(AtomicInt)));
    atomicInit(pool->alive);

    UNSAFE_NULL_CHECK(pool->_queue = malloc(sizeof(SyncQueue)));
    syncqueueInit(pool->_queue);

    pool->_closed = false;
    pool->_terminate = false;

    pthread_create(&pool->_manager, NULL, _manage, (void *)pool);
}

void threadpoolDestroy(ThreadPool *pool)
{
    struct _exec *cur;

    logger("Destroying threadpool!", "STATUS");
    while (syncqueueLen(*pool->_queue) > 0)
    {
        cur = syncqueuePop(pool->_queue);
        free(cur);
    }
    syncqueueDestroy(pool->_queue);
    atomicDestroy(pool->alive);
    free(pool->_queue);
    free(pool->alive);
}

void threadpoolClose(ThreadPool *pool)
{
    pool->_closed = true;
}

void threadpoolTerminate(ThreadPool *pool)
{
    pool->_terminate = true;
}

void threadpoolCleanExit(ThreadPool *pool)
{
    threadpoolClose(pool);
    while (syncqueueLen(*pool->_queue) > 0)
    {
    }
    threadpoolTerminate(pool);

    logger("Joining manager...", "STATUS");
    PTHREAD_CHECK(pthread_join(pool->_manager, NULL));
    logger("Manager joined!", "STATUS");
}

int threadpoolSubmit(void (*fnc)(void *), void *arg, ThreadPool *pool)
{
    struct _exec *newExec;

    if (pool->_closed)
    {
        errno = EINVAL;
        return -1;
    }

    SAFE_NULL_CHECK(newExec = malloc(sizeof(struct _exec)));
    newExec->arg = arg;
    newExec->fnc = fnc;

    syncqueuePush(newExec, pool->_queue);

    return 0;
}

int _spawnThread(ThreadPool *pool)
{
    struct _execLoopArgs *curArgs;
    pthread_t pid;

    SAFE_NULL_CHECK(curArgs = malloc(sizeof(struct _execLoopArgs)));
    curArgs->queue = pool->_queue;
    curArgs->terminate = &(pool->_terminate);
    curArgs->alive = pool->alive;

    PTHREAD_CHECK(pthread_create(&pid, NULL, _execLoop, (void *)curArgs));
    PTHREAD_CHECK(pthread_detach(pid));

    return 1;
}

void *_execLoop(void *args)
{
    struct _exec *curTask;
    void (*curFunc)(void *);
    void *curArgs;

    bool volatile *terminate = ((struct _execLoopArgs *)args)->terminate;
    SyncQueue *queue = ((struct _execLoopArgs *)args)->queue;
    AtomicInt *alive = ((struct _execLoopArgs *)args)->alive;

    free(args);

    atomicInc(1, alive);
    pthread_cleanup_push(&_threadCleanup, (void *)alive);

    while (!(*terminate))
    {
        errno = 0;
        if ((curTask = (struct _exec *)syncqueuePop(queue)) == NULL)
        {
            if (errno == EINVAL)
                return 0;

            continue;
        }
        curFunc = curTask->fnc;
        curArgs = curTask->arg;

        free(curTask);
        curFunc(curArgs);
    }

    pthread_cleanup_pop(true);
    pthread_exit(NULL);
}

void *_manage(void *args)
{
    ThreadPool *pool = (ThreadPool *)args;
    size_t curSize;
    struct _exec *task;
    size_t i;

    while (!pool->_terminate)
    {
        curSize = atomicGet(pool->alive);

#ifdef TP_DEBUG
        puts("Running a round of threadpool management:");
        printf("There are %ld alive threads.", curSize);
#endif
        if (curSize < pool->_coreSize)
        {
            size_t threadsToSpawn = pool->_coreSize - curSize;
            for (i = threadsToSpawn; i > 0; i--)
            {
                _spawnThread(pool);
            }
#ifdef TP_DEBUG
            printf("Spawned %ld threads to get to coresize...\n", threadsToSpawn);
#endif
        }

        if (pool->_queue->_len == 0 && curSize > pool->_coreSize)
        {
            task = malloc(sizeof(struct _exec));
            task->fnc = &_die;
            task->arg = NULL;
            syncqueuePush(task, pool->_queue);
#ifdef TP_DEBUG
            puts("Killed a thread.");
#endif
        }
        else if (syncqueueLen(*pool->_queue) > 0 && curSize < pool->_maxSize)
        {
            size_t threadsToSpawn = _min(syncqueueLen(*pool->_queue), (pool->_maxSize - curSize));
            for (i = threadsToSpawn; i > 0; i--)
            {
                _spawnThread(pool);
            }
#ifdef TP_DEBUG
            printf("Spawned %ld threads.\n", threadsToSpawn);
#endif
        }
#ifdef TP_DEBUG
        puts("Done with round of management.");
#endif
        usleep(1000);
    }

    _threadpoolKILL(pool);

    while (atomicGet(pool->alive) > 0)
    {
#ifdef TP_DEBUG
        printf("%ld threads are still alive, waiting for them to die!\n", atomicGet(pool->alive));
#endif
        usleep(100000);
    }
    return 0;
}

void _threadCleanup(void *args)
{
#ifdef TP_DEBUG
    puts("Thread cleaning up!");
#endif
    AtomicInt *alive = (AtomicInt *)args;
    atomicDec(1, alive);
}

void _die(void *args)
{
#ifdef TP_DEBUG
    puts("Dying!");
#endif
    pthread_exit(NULL);
}

size_t _min(size_t a, size_t b)
{
    if (a < b)
        return a;
    return b;
}

void _threadpoolKILL(ThreadPool *pool)
{
    struct _exec *task;
    int i;

    while (syncqueueLen(*pool->_queue) > 0)
    {
        task = (struct _exec *)syncqueuePop(pool->_queue);
        free(task);
    }

#ifdef TP_DEBUG
    printf("Killing %ld threads\n", atomicGet(pool->alive));
#endif
    for (i = atomicGet(pool->alive); i > 0; i--)
    {
        UNSAFE_NULL_CHECK(task = malloc(sizeof(struct _exec)));
        task->fnc = &_die;
        task->arg = NULL;
        syncqueuePush(task, pool->_queue);
    }
}
