#ifndef THREADPOOL_H
#define THREADPOOL_H

#include "syncqueue.h"


struct _exec {
    void (*fnc)(void*);
    void* arg;
};


typedef struct _ThreadPool {
    SyncQueue *_queue;
    pthread_t *_pids;
    bool _closed;
    bool _terminate;
    int _size;
} ThreadPool;


struct _execLoopArgs {
    SyncQueue *queue;
    bool *terminate;
};


void threadpoolInit(int size, ThreadPool *pool);
void threadpoolDestroy(ThreadPool *pool);
void threadpoolClose(ThreadPool *pool);
void threadpoolTerminate(ThreadPool *pool);
void threadpoolCleanExit(ThreadPool *pool);
int threadpoolSubmit(void (*fnc)(void*), void* arg, ThreadPool *pool);

void threadpoolJoin(ThreadPool *pool);
void threadpoolCancel(ThreadPool *pool);

void *_execLoop(void *args);


#endif