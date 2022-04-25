#include <pthread.h>
#include <stdbool.h>
#include <assert.h>
#include <unistd.h>


#include "threadpool.h"
#include "macros.h"


void threadpoolInit(int size, ThreadPool *pool)
{
    int i;
    struct _execLoopArgs *curArgs;
    

    pool->_size = size;
    if(size <= 0) pool->_size = 4;

    NULL_CHECK(pool->_pids = malloc(sizeof(pthread_t) * pool->_size));

    NULL_CHECK(pool->_queue = malloc(sizeof(SyncQueue)));
    syncqueueInit(pool->_queue);

    pool->_closed = false;
    pool->_terminate = false;
    

    for(i = 0; i < pool->_size; i++)
    {

        NULL_CHECK(curArgs = malloc(sizeof(struct _execLoopArgs)));
        curArgs->queue = pool->_queue;
        curArgs->terminate = &pool->_terminate;

        PTHREAD_CHECK(pthread_create(pool->_pids + i, NULL, _execLoop, (void *)curArgs));
    }
}


void threadpoolDestroy(ThreadPool *pool)
{
    struct _exec *cur;

    while(syncqueueLen(*pool->_queue) > 0)
    {
        cur = syncqueuePop(pool->_queue);
        free(cur);
    }
    syncqueueDestroy(pool->_queue);
    
    free(pool->_pids);
    free(pool->_queue);
}


void threadpoolClose(ThreadPool *pool)
{
    pool->_closed = true;
    syncqueueClose(pool->_queue);

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
    threadpoolClose(pool);
    threadpoolJoin(pool);
}


void threadpoolSubmit(void (*fnc)(void*), void* arg, ThreadPool *pool)
{
    struct _exec *newExec;
    
    if(pool->_closed){ errno = EINVAL; return;}
    
    NULL_CHECK(newExec = malloc(sizeof(struct _exec)));
    newExec->arg = arg;
    newExec->fnc = fnc;

    syncqueuePush(newExec, pool->_queue);
}


void threadpoolJoin(ThreadPool *pool)
{
    int i;
    
    for(i = 0; i < pool->_size; i++)
    {
        PTHREAD_CHECK(pthread_join(pool->_pids[i], NULL));
    }
}


void threadpoolCancel(ThreadPool *pool)
{
    int i;


    threadpoolTerminate(pool);
    for(i = 0; i < pool->_size; i++)
    {
        PTHREAD_CHECK(pthread_cancel(pool->_pids[i]));
    }
}


void *_execLoop(void *args)
{
    struct _exec *curTask;
    void (*curFunc)(void*);
    void* curArgs;

    bool *terminate = ((struct _execLoopArgs *)args)->terminate;
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

