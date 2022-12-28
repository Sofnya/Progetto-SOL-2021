#ifndef LOCK_HANDLER_H
#define LOCK_HANDLER_H

#include "SERVER/filesystem.h"
#include "SERVER/threadpool.h"
#include "COMMON/syncqueue.h"

#define R_LOCK 0
#define R_UNLOCK 1
#define R_REMOVE 2
#define R_OPENLOCK 3
#define R_UNLOCK_NOTIFY 4
#define R_LOCK_CREATE_NOTIFY 5
#define R_DISCONNECT_NOTIFY 6

typedef struct _handlerRequest
{
    int type;
    char *name;
    char *uuid;
    struct _handleArgs *args;
} HandlerRequest;

struct _handlerArgs
{
    FileSystem *fs;
    SyncQueue *msgQueue;
    ThreadPool *tp;
    volatile int *terminate;
};

void *lockHandler(void *args);

int handlerRequestInit(int type, char *name, char *uuid, struct _handleArgs *args, HandlerRequest *request);
void handlerRequestDestroy(HandlerRequest *request);
#endif