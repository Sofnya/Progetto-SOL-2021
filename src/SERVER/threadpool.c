#include <pthread.h>
#include <stdbool.h>
#include <assert.h>
#include <unistd.h>
#include <signal.h>

#include "SERVER/threadpool.h"
#include "COMMON/macros.h"
#include "SERVER/logging.h"
#include "COMMON/helpers.h"

/**
 * @brief Initializes given ThreadPool with given core and max size.
 *
 * @param coreSize the coreSize of the new ThreadPool.
 * @param maxSize the maxSize of the new ThreadPool.
 * @param pool the ThreadPool to be initialized.
 */
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
    VOID_PTHREAD_CHECK(pthread_mutex_init(pool->_pidsmtx, NULL));

    pool->_closed = false;
    pool->_terminate = false;

    // At init, we also spawn our manager and automatically start the pool.
    pthread_create(&pool->_manager, NULL, _manage, (void *)pool);
}

/**
 * @brief Destroys given ThreadPool, freeing it's resources. Should only be called after a ThreadPoolExit has been called.
 *
 * @param pool the ThreadPool to be destroyed.
 */
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

    VOID_PTHREAD_CHECK(pthread_mutex_destroy(pool->_pidsmtx));

    free(pool->_queue);
    free(pool->alive);
    free(pool->_pids);
    free(pool->_pidsmtx);
}

/**
 * @brief Closes the ThreadPool, stopping it from accepting any further jobs.
 *
 * @param pool
 */
void threadpoolClose(ThreadPool *pool)
{
    pool->_closed = true;
}

/**
 * @brief Terminates the ThreadPool, telling all threads, once free, to terminate.
 *
 * @param pool
 */
void threadpoolTerminate(ThreadPool *pool)
{
    pool->_terminate = true;
}

/**
 * @brief Makes given ThreadPool exit cleanly, closing it but waiting for all jobs to finish before terminating it.
 *
 * @param pool the ThreadPool to terminate.
 */
void threadpoolCleanExit(ThreadPool *pool)
{
    // First close the pool to further submissions.
    threadpoolClose(pool);

    // We busy wait for the queue to be empty.
    while (syncqueueLen(*pool->_queue) > 0)
    {
    }

    // Once we are done we terminate it.
    threadpoolTerminate(pool);

    logger("Joining manager...", "STATUS");
    VOID_PTHREAD_CHECK(pthread_join(pool->_manager, NULL));
    logger("Manager joined!", "STATUS");
}

/**
 * @brief Makes given ThreadPool exit fast, closing it, terminating it instantly, and cancelling all Threads violently if they are not done within 5s.
 * This can obviously cause resource leaks.
 *
 * @param pool the ThreadPool to terminate.
 */
void threadpoolFastExit(ThreadPool *pool)
{
    int i;

    // We first close the pool to further submissions.
    threadpoolClose(pool);
    // We then tell all threads to terminate as soon as possible, without consuming the pool.
    threadpoolTerminate(pool);

    // Waits for up to 5 seconds for all threads to terminate.
    // While this is busy waiting, and could be donem with a timed wait on a condition variable, the added complexity to the codebase isn't worth it.
    for (i = 0; i < 500; i++)
    {
        if (atomicGet(pool->alive) == 0)
        {
            break;
        }

        // 10ms sleep done 500 times is 5s
        usleep(10000);
    }
    // Otherwise cancel them. Note that if no threads are alive threadpoolCancel has no effect.
    // Cancelling the threadpool is only done as a last resort, in case any threads are stuck, and will lead to loss of resources.
    threadpoolCancel(pool);

    logger("Joining manager...", "STATUS");
    VOID_PTHREAD_CHECK(pthread_join(pool->_manager, NULL));
    logger("Manager joined!", "STATUS");

    logger("FastExit done", "STATUS");
}

/**
 * @brief Cancels all alive threads in given ThreadPool. This is very violent and will result in resource leaks. Should only be used as a last resort.
 *
 * @param pool
 */
void threadpoolCancel(ThreadPool *pool)
{
    void *el;
    pthread_t *pid;
    char log[500];

    if (atomicGet(pool->alive) == 0)
    {
        return;
    }

    logger("Cancelling ThreadPool", "STATUS");

    VOID_PTHREAD_CHECK(pthread_mutex_lock(pool->_pidsmtx));

    sprintf(log, ">Alive:%ld >ListSize:%d", atomicGet(pool->alive), listSize(*pool->_pids));
    logger(log, "THREADPOOL");
    while (listSize(*pool->_pids) > 0)
    {
        listPop(&el, pool->_pids);
        pid = (pthread_t *)el;
        sprintf(log, ">Cancelling:%ld", *pid);
        logger(log, "THREADPOOL");
        // All pids in pool->_pids are guaranteed alive, as threads automatically remove themselves from the list on death.
        // As such we can safely cancel them.
        VOID_PTHREAD_CHECK(pthread_cancel(*pid));
        sprintf(log, ">Cancelled:%ld", *pid);
        logger(log, "THREADPOOL");
        free(el);
    }
    VOID_PTHREAD_CHECK(pthread_mutex_unlock(pool->_pidsmtx));
}

/**
 * @brief Submits given task fnc(arg) to the ThreadPool.
 *
 * @param fnc the function to be evaluated.
 * @param arg the arguments on which to evaluate fnc.
 * @param pool the ThreadPool to which to submit.
 * @return int 0 on success, -1 and sets errno on failure.
 */
int threadpoolSubmit(void (*fnc)(void *), void *arg, ThreadPool *pool)
{
    struct _exec *newExec;

    // If the ThreadPool is closed we return an error without accepting the job.
    if (pool->_closed)
    {
        errno = EINVAL;
        return -1;
    }

    // Otherwise we initialize the job and push it on the queue.
    SAFE_NULL_CHECK(newExec = malloc(sizeof(struct _exec)));
    newExec->arg = arg;
    newExec->fnc = fnc;

    syncqueuePush(newExec, pool->_queue);

    return 0;
}

/**
 * @brief Called by the pool manager to spawn a new thread in the ThreadPool.
 *
 * @param pool the ThreadPool in which to spawn a new thread.
 * @return int 0 on success, -1 and sets errno on failure.
 */
int _spawnThread(ThreadPool *pool)
{
    struct _execLoopArgs *curArgs;
    pthread_t *pid;

    // We first initialize the new thread's arguments.
    SAFE_NULL_CHECK(pid = malloc(sizeof(pthread_t)));
    SAFE_NULL_CHECK(curArgs = malloc(sizeof(struct _execLoopArgs)));
    curArgs->queue = pool->_queue;
    curArgs->terminate = &(pool->_terminate);
    curArgs->alive = pool->alive;
    curArgs->pids = pool->_pids;
    curArgs->pidsmtx = pool->_pidsmtx;

    PTHREAD_CHECK(pthread_mutex_lock(pool->_pidsmtx));

    // We start the thread inside of the critical zone pidsmtx to make sure that the thread doesn't die before we put it's pid in the _pids list.
    PTHREAD_CHECK(pthread_create(pid, NULL, _execLoop, (void *)curArgs));

    listAppend((void *)pid, pool->_pids);
    PTHREAD_CHECK(pthread_mutex_unlock(pool->_pidsmtx));

    return 0;
}

/**
 * @brief The loop that a thread of the ThreadPool executes, taking jobs from the queue and running them until terminated.
 *
 * @param args should point to a valid struct _execLoopArgs, which the thread will need.
 * @return void* nothing.
 */
void *_execLoop(void *args)
{
    struct _exec *curTask;
    void (*curFunc)(void *);
    void *curArgs;

    // All threads except the main root thread should ignore termination signals, and let the root thread handle termination.
    sigset_t sigmask;
    sigemptyset(&sigmask);
    sigaddset(&sigmask, SIGINT);
    sigaddset(&sigmask, SIGHUP);
    sigaddset(&sigmask, SIGQUIT);
    pthread_sigmask(SIG_BLOCK, &sigmask, NULL);

    // First we parse the given args.
    bool volatile *terminate = ((struct _execLoopArgs *)args)->terminate;
    SyncQueue *queue = ((struct _execLoopArgs *)args)->queue;
    AtomicInt *alive = ((struct _execLoopArgs *)args)->alive;
    struct _cleanup *cln;
    UNSAFE_NULL_CHECK(cln = malloc(sizeof(struct _cleanup)));

    cln->alive = alive;
    cln->pids = ((struct _execLoopArgs *)args)->pids;
    cln->pidsmtx = ((struct _execLoopArgs *)args)->pidsmtx;
    free(args);

    // We then notify others we are alive.
    atomicInc(1, alive);
    // And push our threadCleanup to always be executed at exit.
    pthread_cleanup_push(&_threadCleanup, (void *)cln);

    // The thread should always be cancellable, as usually it will need to be cancelled in blocking situations, such as waiting for a lock.
    pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);

    // We check terminate between every execution of a job.
    while (!(*terminate))
    {
        errno = 0;
        // syncqueuePop is blocking, so if no jobs are present we will wait here.
        if ((curTask = (struct _exec *)syncqueuePop(queue)) == NULL)
        {
            if (errno == EINVAL)
                pthread_exit(NULL);

            continue;
        }
        // Check terminate again after the blocking call.
        if (*terminate)
        {
            free(curTask);
            break;
        }

        curFunc = curTask->fnc;
        curArgs = curTask->arg;

        free(curTask);

        // And evaluate the function.
        curFunc(curArgs);
    }

    // If we terminate, execute the cleanup and exit.
    pthread_cleanup_pop(true);
    pthread_exit(NULL);
}

/**
 * @brief The cleanup of a thread, will always be executed on a threads death.
 *
 * @param args should point to a valid struct _cleanup which contains necessary information for the dying thread.
 */
void _threadCleanup(void *args)
{
    // First parse the args.
    struct _cleanup *cln = (struct _cleanup *)args;

    AtomicInt *alive = cln->alive;
    List *pids = cln->pids;
    pthread_mutex_t *poolmtx = cln->pidsmtx;
    pthread_t mine, *el;
    void *saveptr = NULL;
    int i = 0, found = 0;

    pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL);
    // Enter the critical poolmtx portion, as we will be removing ourselves from the pool->_pids list.
    VOID_PTHREAD_CHECK(pthread_mutex_lock(poolmtx));
    mine = pthread_self();

    // Look for ourselves in the list.
    errno = 0;
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
    // List scan always misses the last element.
    if (errno == EOF)
    {
        if (pthread_equal(*el, mine))
        {
            found = 1;
            listRemove(i, (void **)&el, pids);
            free(el);
        }
    }
    VOID_PTHREAD_CHECK(pthread_mutex_unlock(poolmtx));

    // Now notify others we are no longer alive.
    atomicDec(1, alive);

    if (!found)
    {
        // This really shouldn't happen, so a very large warning is printed.
        puts("\n\n------------\nWARNING: NOT FOUND!!\n-----------\n");
    }

    free(cln);
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);

    // Detach ourselves right before exit, as noone joins threads. It's not done before as threads should be cancellable while they are still in the pool->_pids list.
    VOID_PTHREAD_CHECK(pthread_detach(mine));
}

/**
 * @brief The main of the manager thread. Handles the resizing of the pool, spawning and killing threads as needed.
 *
 * @param args should point to the ThreadPool to be managed.
 * @return void* nothing.
 */
void *_manage(void *args)
{
    ThreadPool *pool = (ThreadPool *)args;
    size_t curSize;
    size_t lastSize = 0;
    struct _exec *task;
    size_t i;
    char log[500];

    // All threads except the main root thread should ignore signals, and let the root thread handle termination.
    sigset_t sigmask;
    sigemptyset(&sigmask);
    sigaddset(&sigmask, SIGINT);
    sigaddset(&sigmask, SIGHUP);
    sigaddset(&sigmask, SIGQUIT);
    pthread_sigmask(SIG_BLOCK, &sigmask, NULL);

    // stability controls how many rounds should go a certain way before the manager reacts.
    const int stability = 5;
    // balance keeps track of how many rounds we are in the negative(not enough threads) or positive(too many threads), and is reset each time we spawn or kill a thread, and every stability*2 rounds.
    int balance = 0;
    int roundCount = 0;

    // The manager also has to check for ThreadPool termination.
    while (!pool->_terminate)
    {
        // sprintf(log, ">QueueSize:%ld", syncqueueLen(*pool->_queue));
        // logger(log, "THREADPOOL");

        if (roundCount >= stability * 2)
        {
            balance = 0;
            roundCount = 0;
        }
        roundCount++;

        // curSize is the curreng number of alive threads.
        curSize = atomicGet(pool->alive);

        // To avoid filling the log with alives, we only log a changed size.
        if (curSize != lastSize)
        {
            sprintf(log, ">Alive:%ld", curSize);
            logger(log, "THREADPOOL");
            lastSize = curSize;
        }

        // We need to always have at least coreSize threads.
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

        // If the job queue is empty, we probably have too many threads, and we start killing one per round of management.
        else if (syncqueueLen(*pool->_queue) == 0 && curSize > pool->_coreSize)
        {
            balance++;
            if (balance > stability)
            {
                balance = 0;
                roundCount = 0;

                task = malloc(sizeof(struct _exec));
                task->fnc = &_die;
                task->arg = NULL;

                // To kill a thread we just push one _die task on the queue, once a thread pops it it will spontanously die.
                syncqueuePush(task, pool->_queue);
                logger(">Killed:1", "THREADPOOL");
            }
        }

        // If we have jobs in the queue and haven't reached maxSize, we spawn one thread.
        else if (syncqueueLen(*pool->_queue) > 0 && curSize < pool->_maxSize)
        {
            balance--;
            if (balance < -stability)
            {
                balance = 0;
                roundCount = 0;

                _spawnThread(pool);

                logger(">Spawned:1", "THREADPOOL");
            }
        }

        // We run a round of management about 100 times a second.
        usleep(10000);
    }

    // Before finishing we try to kill any remaining threads.
    _threadpoolKILL(pool);

    // Busy wait for all threads to die.
    while (atomicGet(pool->alive) > 0)
    {
        // 10ms
        usleep(10000);
    }

    return 0;
}

/**
 * @brief A dummy task, which kills the calling thread.
 *
 * @param args nothing.
 */
void _die(void *args)
{
    pthread_exit(NULL);
}

/**
 * @brief A simple min function, returns the smallest between a and b.
 *
 * @param a the first argument.
 * @param b the second argument.
 * @return size_t
 */
size_t _min(size_t a, size_t b)
{
    if (a < b)
        return a;
    return b;
}

/**
 * @brief Tries to kill threads cleanly, pushing many _die tasks on the queue which will be executed as soon as the threads are available.
 *
 * @param pool
 */
void _threadpoolKILL(ThreadPool *pool)
{
    struct _exec *task;
    int i;

    // We first empty the queue.
    while (syncqueueLen(*pool->_queue) > 0)
    {
        task = (struct _exec *)syncqueuePop(pool->_queue);
        free(task);
    }

    // Then we push as many _die tasks on the queue as there are alive threads.
    for (i = atomicGet(pool->alive); i > 0; i--)
    {
        UNSAFE_NULL_CHECK(task = malloc(sizeof(struct _exec)));

        task->fnc = &_die;
        task->arg = NULL;
        if (syncqueuePush((void *)task, pool->_queue) == -1)
        {
            free(task);
        }
    }
}
