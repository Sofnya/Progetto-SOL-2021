#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <stdint.h>

#include "COMMON/syncqueue.h"
#include "COMMON/list.h"
#include "COMMON/atomicint.h"

struct _exec
{
    void (*fnc)(void *);
    void *arg;
};

typedef struct _ThreadPool
{
    SyncQueue *_queue;
    pthread_t _manager;
    AtomicInt *alive;
    bool volatile _closed;
    bool volatile _terminate;
    size_t _coreSize;
    size_t _maxSize;
} ThreadPool;

struct _execLoopArgs
{
    SyncQueue *queue;
    bool volatile *terminate;
    AtomicInt *alive;
};

void threadpoolInit(size_t coreSize, size_t maxSize, ThreadPool *pool);
void threadpoolDestroy(ThreadPool *pool);

void threadpoolClose(ThreadPool *pool);

void threadpoolTerminate(ThreadPool *pool);
void threadpoolCleanExit(ThreadPool *pool);

int threadpoolSubmit(void (*fnc)(void *), void *arg, ThreadPool *pool);

int _spawnThread(ThreadPool *pool);

void *_execLoop(void *args);
void *_manage(void *args);
void _threadCleanup(void *args);
void _die(void *args);
void _threadpoolKILL(ThreadPool *pool);

size_t _min(size_t a, size_t b);

#endif