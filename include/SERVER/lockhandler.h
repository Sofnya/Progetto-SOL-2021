#ifndef LOCK_HANDLER_H
#define LOCK_HANDLER_H

#include "SERVER/filesystem.h"
#include "COMMON/syncqueue.h"

#define R_LOCK 0
#define R_UNLOCK 1
#define R_REMOVE 2
#define R_OPENLOCK 3

typedef struct _handlerRequest
{
    int type;
    char *name;
    void *args;
} HandlerRequest;

struct _handlerArgs
{
    FileSystem *fs;
    SyncQueue *msgQueue;
    volatile int *terminate;
};

void lockHandler(void *args);

#endif