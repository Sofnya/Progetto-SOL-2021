#include <pthread.h>
#include <stdbool.h>
#include <assert.h>
#include <unistd.h>
#include <signal.h>


#include "COMMON/threadpool.h"
#include "COMMON/macros.h"


void threadpoolInit(uint64_t coreSize, uint64_t maxSize, ThreadPool *pool)
{
    pool->_coreSize = coreSize;
    pool->_maxSize = maxSize;


    UNSAFE_NULL_CHECK(pool->pidList = malloc(sizeof(List)));
    ERROR_CHECK(listInit(pool->pidList));

    UNSAFE_NULL_CHECK(pool->_queue = malloc(sizeof(SyncQueue)));
    syncqueueInit(pool->_queue);

    pool->_closed = false;
    pool->_terminate = false;
    
    pthread_create(&pool->_manager, NULL, _manage, (void *)pool);
}


void threadpoolDestroy(ThreadPool *pool)
{
    struct _exec *cur;
    pthread_t *pid;

    puts("Destroying threadpool!");
    while(syncqueueLen(*pool->_queue) > 0)
    {
        cur = syncqueuePop(pool->_queue);
        free(cur);
    }
    while(listPop((void **)&pid, pool->pidList) != -1) free(pid);
    syncqueueDestroy(pool->_queue);
    listDestroy(pool->pidList);
    free(pool->_queue);
    free(pool->pidList);
}


void threadpoolClose(ThreadPool *pool)
{
    syncqueueClose(pool->_queue);
    pool->_closed = true;
}


void threadpoolTerminate(ThreadPool *pool)
{
    pool->_terminate = true;
}


void threadpoolCleanExit(ThreadPool *pool)
{
    threadpoolClose(pool);
    while(syncqueueLen(*pool->_queue) > 0){}
    threadpoolTerminate(pool);
    
    puts("Joining manager...");
    PTHREAD_CHECK(pthread_join(pool->_manager, NULL));
    puts("Manager joined!");
}


int threadpoolSubmit(void (*fnc)(void*), void* arg, ThreadPool *pool)
{
    struct _exec *newExec;
    
    if(pool->_closed){ errno = EINVAL; return -1;}
    
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

    SAFE_NULL_CHECK(curArgs = malloc(sizeof(struct _execLoopArgs)));
    curArgs->queue = pool->_queue;
    curArgs->terminate = &pool->_terminate;

    SAFE_NULL_CHECK(pid = malloc(sizeof(pthread_t)));
    PTHREAD_CHECK(pthread_create(pid, NULL, _execLoop, (void *)curArgs));
    PTHREAD_CHECK(pthread_detach(*pid));

    return listPush(pid, pool->pidList);
}




void *_execLoop(void *args)
{
    struct _exec *curTask;
    void (*curFunc)(void*);
    void* curArgs;

    bool volatile *terminate = ((struct _execLoopArgs *)args)->terminate;
    SyncQueue *queue = ((struct _execLoopArgs *)args)->queue; 

    free(args);
    
    while(!(*terminate))
    {
        errno = 0;
        if((curTask = (struct _exec*) syncqueuePop(queue)) == NULL)
        {
            if(errno == EINVAL) return 0;
            
            continue;
        }
        curFunc = curTask->fnc;
        curArgs = curTask->arg;
        
        free(curTask);
        curFunc(curArgs);
    }

    return 0;
}


void *_manage(void *args)
{
    ThreadPool *pool = (ThreadPool *) args;
    struct _exec *task;
    uint64_t i;

    while(!pool->_terminate)
    {
        //puts("Running a round of threadpool management:");
        // First we update the list of pids by removing dead threads.
        _updatePids(pool);

        while(listSize(*pool->pidList) < pool->_coreSize)
        {
            _spawnThread(pool);
            //count ++;
        }
        /**if(count != 0)
        {printf("Spawned %ld threads to get to coresize...\n", count);}*/
        
        if(pool->_queue->_len == 0 && listSize(*pool->pidList) > pool->_coreSize)
        {
            task = malloc(sizeof(struct _exec));
            task->fnc = &_die;
            task->arg = NULL;
            syncqueuePush(task, pool->_queue);
            //puts("Killed a thread.");
        }
        else if(syncqueueLen(*pool->_queue) > 0 && listSize(*pool->pidList) < pool->_maxSize)
        {
            uint64_t threadsToSpawn = _min(syncqueueLen(*pool->_queue), (pool->_maxSize - listSize(*pool->pidList)));
            for(i = threadsToSpawn; i > 0; i--)
            {
                _spawnThread(pool);
            }
            //printf("Spawned %ld threads.\n", threadsToSpawn);
        }
        //puts("Done with round of management.");
        usleep(10000);
    }

    _threadpoolKILL(pool);

    while(listSize(*pool->pidList) > 0)
    {
        puts("Some threads are still alive, waiting for them to die!");
        _updatePids(pool);
    }
    return 0;
}



void _die(void *args)
{
    pthread_exit(NULL);
}




uint64_t _min(uint64_t a, uint64_t b)
{
    if(a < b) return a;
    return b;
}


void _threadpoolKILL(ThreadPool *pool)
{
    struct _exec *task;
    int i;
    
    puts("Killing threadpool!");
    while(syncqueueLen(*pool->_queue) > 0)
    {
        task = (struct _exec *)syncqueuePop(pool->_queue);
        free(task);
    }
    for(i = listSize(*pool->pidList) + 1; i > 0; i--)
    {
        UNSAFE_NULL_CHECK(task = malloc(sizeof(struct _exec)));
        task->fnc = &_die;
        task->arg = NULL;
        syncqueuePush(task, pool->_queue);
    }
}   

void _updatePids(ThreadPool *pool)
{
    uint64_t i;
    pthread_t *pid;
    // First we update the list of pids by removing dead threads.
    i = 0;
    //count = 0;
    while(i < listSize(*pool->pidList))
    {
        ERROR_CHECK(listGet(i, (void **)&pid, pool->pidList));

        // If the thread is dead we remove it.
        if(pthread_kill(*pid, 0) != 0)
        {
            listRemove(i, NULL, pool->pidList);
            free(pid);
            //count ++;
        }
        else i++;
    }
}