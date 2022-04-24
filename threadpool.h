#ifndef THREADPOOL_H
#define THREADPOOL_H

#include "syncqueue.h"


struct _exec {
    void (*fnc)(void*);
    void* arg;
};


typedef struct _threadPool {
    syncQueue *_queue;
    pthread_t *_pids;
    bool _closed;
    bool _terminate;
    int _size;
} threadPool;


struct _execLoopArgs {
    syncQueue *queue;
    bool *terminate;
};


void threadpoolInit(int size, threadPool *pool);
void threadpoolClear(threadPool *pool);
void threadpoolClose(threadPool *pool);
void threadpoolTerminate(threadPool *pool);
void threadpoolCleanExit(threadPool *pool);
void threadpoolSubmit(void (*fnc)(void*), void* arg, threadPool *pool);

void threadpoolJoin(threadPool *pool);
void threadpoolCancel(threadPool *pool);

void *_execLoop(void *args);


#endif