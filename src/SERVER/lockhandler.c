#include "SERVER/lockhandler.h"

#include "SERVER/filesystem.h"
#include "COMMON/hashtable.h"
#include "COMMON/list.h"

void lockHandler(void *args)
{
    struct _handlerArgs *hargs = (struct _handlerArgs *)args;

    FileSystem *fs = hargs->fs;
    SyncQueue *queue = hargs->msgQueue;
    volatile int *terminate = hargs->terminate;
    free(hargs);

    HashTable lockedFiles;

    HashTable waitingLocks;
    List *waitingList;

    hashTableInit(1024, &lockedFiles);

    HandlerRequest *request;

    while (!terminate)
    {
        request = syncqueuePop(queue);
        if (request == NULL)
        {
            continue;
        }

        switch (request->type)
        {
        case (R_LOCK):
        {
            if (hashTableGet(request->name, NULL, lockedFiles) == -1)
            {
                if (hashTableGet)
            }
            else
            {
            }
            break;
        }
        case (R_UNLOCK):
        case (R_REMOVE):
        {
            hashTableRemove(request->name, NULL, lockedFiles);
            break;
        }
        case (R_OPENLOCK):
        {
            hashTablePut(request->name, NULL, lockedFiles);
            break;
        }
        }
    }
}