#ifndef THREADPOOL_H
#define THREADPOOL_H


#include <stdint.h>


#include "syncqueue.h"
#include "list.h"


struct _exec {
    void (*fnc)(void*);
    void* arg;
};


typedef struct _ThreadPool {
    SyncQueue *_queue;
    pthread_t _manager;
    List *pidList;
    bool volatile _closed;
    bool volatile _terminate;
    uint64_t _coreSize;
    uint64_t _maxSize;
} ThreadPool;


struct _execLoopArgs {
    SyncQueue *queue;
    bool volatile *terminate;
};


void threadpoolInit(uint64_t coreSize, uint64_t maxSize, ThreadPool *pool);
void threadpoolDestroy(ThreadPool *pool);

void threadpoolClose(ThreadPool *pool);

void threadpoolTerminate(ThreadPool *pool);
void threadpoolCleanExit(ThreadPool *pool);

int threadpoolSubmit(void (*fnc)(void*), void* arg, ThreadPool *pool);

int _spawnThread(ThreadPool *pool);

void *_execLoop(void *args);
void *_manage(void *args);
void _die(void *args);
void _threadpoolKILL(ThreadPool *pool);
void _updatePids(ThreadPool *pool);

uint64_t _min(uint64_t a, uint64_t b); 
#endif