#define _GNU_SOURCE

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

    UNSAFE_NULL_CHECK(pool->_pids = malloc(sizeof(List)));
    listInit(pool->_pids);

    UNSAFE_NULL_CHECK(pool->_pidsmtx = malloc(sizeof(pthread_mutex_t)));
    PTHREAD_CHECK(pthread_mutex_init(pool->_pidsmtx, NULL));

    pool->_closed = false;
    pool->_terminate = false;

    pthread_create(&pool->_manager, NULL, _manage, (void *)pool);
}

void threadpoolDestroy(ThreadPool *pool)
{
    struct _exec *cur;
    void *el;

    logger("Destroying threadpool!", "STATUS");
    while (syncqueueLen(*pool->_queue) > 0)
    {
        cur = syncqueuePop(pool->_queue);
        free(cur);
    }
    syncqueueDestroy(pool->_queue);
    atomicDestroy(pool->alive);

    while (listSize(*pool->_pids) > 0)
    {
        listPop(&el, pool->_pids);
        free(el);
    }
    listDestroy(pool->_pids);

    PTHREAD_CHECK(pthread_mutex_destroy(pool->_pidsmtx));

    free(pool->_queue);
    free(pool->alive);
    free(pool->_pids);
    free(pool->_pidsmtx);
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
    struct timespec abstime;

    threadpoolClose(pool);
    while (syncqueueLen(*pool->_queue) > 0)
    {
    }
    threadpoolTerminate(pool);

    timespec_get(&abstime, TIME_UTC);
    abstime.tv_sec += 10;
    logger("Joining manager...", "STATUS");
    printf("Calling pthread_join from PID:%d and PTHREAD:%ld on PTHREAD:%ld\n", getpid(), pthread_self(), pool->_manager);
    PTHREAD_CHECK(pthread_timedjoin_np(pool->_manager, NULL, &abstime));
    logger("Manager joined!", "STATUS");
}

void threadpoolFastExit(ThreadPool *pool)
{
    int i;
    threadpoolClose(pool);
    threadpoolTerminate(pool);
    syncqueueClose(pool->_queue);

    // Waits for up to 5 seconds for all threads to terminate.
    for (i = 0; i < 500; i++)
    {
        if (atomicGet(pool->alive) == 0)
        {
            break;
        }

        // 10ms sleep done 500 times is 5s
        usleep(10000);
    }
    // Otherwise cancel them.
    threadpoolCancel(pool);

    logger("FastExit done", "STATUS");
}

void threadpoolCancel(ThreadPool *pool)
{
    void *el;
    pthread_t *pid;
    char log[500];

    printf("Canceling, alive:%ld\n", atomicGet(pool->alive));
    printf("ListSize:%d\n", listSize(*pool->_pids));
    printList(pool->_pids);

    PTHREAD_CHECK(pthread_mutex_lock(pool->_pidsmtx));
    while (listSize(*pool->_pids) > 0)
    {
        listPop(&el, pool->_pids);
        pid = (pthread_t *)el;
        sprintf(log, ">Canceling:%ld", *pid);
        logger(log, "THREADPOOL");
        pthread_cancel(*pid);
        free(el);
    }
    PTHREAD_CHECK(pthread_mutex_unlock(pool->_pidsmtx));
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
    pthread_t *pid;

    SAFE_NULL_CHECK(pid = malloc(sizeof(pthread_t)));
    SAFE_NULL_CHECK(curArgs = malloc(sizeof(struct _execLoopArgs)));
    curArgs->queue = pool->_queue;
    curArgs->terminate = &(pool->_terminate);
    curArgs->alive = pool->alive;
    curArgs->pids = pool->_pids;
    curArgs->pidsmtx = pool->_pidsmtx;

    PTHREAD_CHECK(pthread_mutex_lock(pool->_pidsmtx));

    PTHREAD_CHECK(pthread_create(pid, NULL, _execLoop, (void *)curArgs));

    listAppend((void *)pid, pool->_pids);
    PTHREAD_CHECK(pthread_mutex_unlock(pool->_pidsmtx));

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
    struct _cleanup *cln;
    SAFE_NULL_CHECK(cln = malloc(sizeof(struct _cleanup)));

    cln->alive = alive;
    cln->pids = ((struct _execLoopArgs *)args)->pids;
    cln->pidsmtx = ((struct _execLoopArgs *)args)->pidsmtx;
    free(args);

    atomicInc(1, alive);
    pthread_cleanup_push(&_threadCleanup, (void *)cln);

    pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);

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
    size_t lastSize = 0;
    struct _exec *task;
    size_t i;
    char log[500];

    while (!pool->_terminate)
    {
        curSize = atomicGet(pool->alive);

#ifdef TP_DEBUG
        puts("Running a round of threadpool management:");
        printf("There are %ld alive threads.", curSize);
#endif
        if (curSize != lastSize)
        {
            sprintf(log, ">Alive:%ld", curSize);
            logger(log, "THREADPOOL");
            lastSize = curSize;
        }
        if (curSize < pool->_coreSize)
        {
            size_t threadsToSpawn = pool->_coreSize - curSize;
            for (i = threadsToSpawn; i > 0; i--)
            {
                _spawnThread(pool);
            }

            sprintf(log, ">Spawned:%ld", threadsToSpawn);
            logger(log, "THREADPOOL");
        }

        if (pool->_queue->_len == 0 && curSize > pool->_coreSize)
        {
            task = malloc(sizeof(struct _exec));
            task->fnc = &_die;
            task->arg = NULL;
            syncqueuePush(task, pool->_queue);
            logger(">Killed:1", "THREADPOOL");
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

            sprintf(log, ">Spawned:%ld", threadsToSpawn);
            logger(log, "THREADPOOL");
#ifdef TP_DEBUG
            printf("Spawned %ld threads.\n", threadsToSpawn);
#endif
        }
#ifdef TP_DEBUG
        puts("Done with round of management.");
#endif
        usleep(10000);
    }

    _threadpoolKILL(pool);

    return 0;
}

void _threadCleanup(void *args)
{
    struct _cleanup *cln = (struct _cleanup *)args;

    AtomicInt *alive = cln->alive;
    List *pids = cln->pids;
    pthread_mutex_t *poolmtx = cln->pidsmtx;
    pthread_t mine, *el;
    void *saveptr = NULL;
    int i = 0, found = 0;

    PTHREAD_CHECK(pthread_mutex_lock(poolmtx));
    mine = pthread_self();

    while (listScan((void **)&el, &saveptr, pids) == 0)
    {
        if (pthread_equal(*el, mine))
        {
            found = 1;
            listRemove(i, (void **)&el, pids);
            free(el);
            break;
        }
        i++;
    }
    if (errno == EOF)
    {
        if (pthread_equal(*el, mine))
        {
            found = 1;
            listRemove(i, (void **)&el, pids);
            free(el);
        }
    }
    PTHREAD_CHECK(pthread_mutex_unlock(poolmtx));

    atomicDec(1, alive);
    if (!found)
    {
        puts("\n\n------------\nWARNING: NOT FOUND!!\n-----------\n");
    }
    free(cln);
    PTHREAD_CHECK(pthread_detach(mine));
}

void _die(void *args)
{
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
