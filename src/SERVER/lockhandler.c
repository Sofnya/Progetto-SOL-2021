#include "SERVER/lockhandler.h"

#include "SERVER/filesystem.h"
#include "COMMON/hashtable.h"
#include "COMMON/list.h"
#include "SERVER/globals.h"
#include "SERVER/server.h"
#include "SERVER/threadpool.h"

void lockHandler(void *args)
{
    struct _handlerArgs *hargs = (struct _handlerArgs *)args;

    FileSystem *fs = hargs->fs;
    SyncQueue *queue = hargs->msgQueue;
    ThreadPool *tp = hargs->tp;
    volatile int *terminate = hargs->terminate;
    free(hargs);

    HashTable waitingLocks;
    List *waitingList;

    HandlerRequest *request;

    while (!terminate)
    {
        request = (HandlerRequest *)syncqueuePop(queue);
        if (request == NULL)
        {
            continue;
        }

        switch (request->type)
        {
        case (R_LOCK):
        {
            if (isLockedFile(request->name, fs) == -1)
            {
                lockFile(request->name, request->uuid, fs);
                PRINT_ERROR_CHECK(threadpoolSubmit(&handleRequest, request->args, tp));
            }
            else
            {
            }
            break;
        }
        case (R_UNLOCK):
        case (R_REMOVE):
        {
            break;
        }
        case (R_OPENLOCK):
        {
            break;
        }
        }
    }
}

/**
 * @brief Sens a Lock request to the LockHandler, for File with given name, from connection of given UUID.
 *
 * @param name the name of the File to Lock.
 * @param uuid the UUID of the requesting connection.
 * @param msgQueue the SyncQueue shared with the LockHandler thread.
 */
int lockHandlerLock(char *name, char *uuid, SyncQueue *msgQueue)
{
    HandlerRequest request;
    request.type = R_LOCK;
    request.name = name;
    request.uuid = uuid;
}

int lockHandlerUnlock(char *name, char *uuid, SyncQueue *msgQueue);
int lockHandlerRemove(char *name, SyncQueue *msgQueue);
int lockHandlerOpenLock(char *name, char *uuid, SyncQueue *msgQueue);