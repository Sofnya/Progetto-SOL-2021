#ifndef LOCK_HANDLER_H
#define LOCK_HANDLER_H

#include "SERVER/filesystem.h"
#include "SERVER/threadpool.h"
#include "COMMON/syncqueue.h"

#define R_LOCK 0
#define R_UNLOCK 1
#define R_REMOVE 2
#define R_OPENLOCK 3

typedef struct _handlerRequest
{
    int type;
    char *name;
    char *uuid;
    void *args;
} HandlerRequest;

struct _handlerArgs
{
    FileSystem *fs;
    SyncQueue *msgQueue;
    ThreadPool *tp;
    volatile int *terminate;
};

void lockHandler(void *args);
int lockHandlerLock(char *name, char *uuid, SyncQueue *msgQueue);
int lockHandlerUnlock(char *name, char *uuid, SyncQueue *msgQueue);
int lockHandlerRemove(char *name, SyncQueue *msgQueue);
int lockHandlerOpenLock(char *name, char *uuid, SyncQueue *msgQueue);

#endif