#include <string.h>
#include <stdlib.h>

#include "SERVER/lockhandler.h"

#include "SERVER/filesystem.h"
#include "COMMON/hashtable.h"
#include "COMMON/list.h"
#include "SERVER/globals.h"
#include "SERVER/server.h"
#include "SERVER/threadpool.h"
#include "COMMON/macros.h"
#include "SERVER/logging.h"

void _requestRespond(HandlerRequest *request, int status, ThreadPool *tp)
{
    request->args->request->status = status;
    PRINT_ERROR_CHECK(threadpoolSubmit(&handleRequest, request->args, tp));
}

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
    char log[200];

    PRINT_ERROR_CHECK(hashTableInit(MAX_FILES, &waitingLocks));

    HandlerRequest *request;
    HandlerRequest *request_tmp;

    logger("Starting", "LOCKHANDLER");
    while (!(*terminate))
    {
        request = (HandlerRequest *)syncqueuePop(queue);
        logger("Received request", "LOCKHANDLER");
        if (request == NULL)
        {
            continue;
        }

        switch (request->type)
        {
            // An open lock is handled the same as pre-locking the file. We can do this since open-creates are handled directly by the FileSystem.
        case (R_OPENLOCK):
        case (R_LOCK):
        {
            if (isLockedFile(request->name, fs) == -1)
            {

                logger("Request:LOCK >Satisfied", "LOCKHANDLER");
                lockFile(request->name, request->uuid, fs);
                _requestRespond(request, MS_OK, tp);
                handlerRequestDestroy(request);
                free(request);
            }
            else
            {
                logger("Request:LOCK >Put in queue", "LOCKHANDLER");
                if (hashTableGet(request->name, (void **)&waitingList, waitingLocks) == -1)
                {
                    UNSAFE_NULL_CHECK(waitingList = malloc(sizeof(List)));
                    PRINT_ERROR_CHECK(listInit(waitingList));
                    PRINT_ERROR_CHECK(listPush((void *)request, waitingList));
                    PRINT_ERROR_CHECK(hashTablePut(request->name, (void **)&waitingList, waitingLocks));
                }
                else
                {
                    PRINT_ERROR_CHECK(listAppend((void *)request, waitingList));
                }
            }
            break;
        }
        case (R_UNLOCK):
        {

            if (isLockedByFile(request->name, request->uuid, fs))
            {

                logger("Request:UNLOCK >Success", "LOCKHANDLER");
                PRINT_ERROR_CHECK(unlockFile(request->name, fs));
                if (hashTableGet(request->name, (void **)waitingList, waitingLocks) == 0)
                {
                    PRINT_ERROR_CHECK(listPop((void **)&request_tmp, waitingList));
                    logger("Waking 1 waiting request", "LOCKHANDLER");

                    lockFile(request->name, request->uuid, fs);
                    _requestRespond(request_tmp, MS_OK, tp);
                    handlerRequestDestroy(request_tmp);
                    free(request_tmp);
                    if (listSize(*waitingList) == 0)
                    {
                        hashTableRemove(request->name, NULL, waitingLocks);
                        listDestroy(waitingList);
                        free(waitingList);
                    }
                }

                _requestRespond(request, MS_OK, tp);
            }
            else
            {

                logger("Request:UNLOCK >Error", "LOCKHANDLER");
                _requestRespond(request, MS_ERR, tp);
            }
            handlerRequestDestroy(request);
            free(request);
            break;
        }
        case (R_REMOVE):
        {

            logger("Request:REMOVE", "LOCKHANDLER");
            if (hashTableRemove(request->name, (void **)&waitingList, waitingLocks) == 0)
            {
                sprintf(log, "Waking %d waiting requests", listSize(*waitingList));
                logger(log, "LOCKHANDLER");

                while (listPop((void **)&request_tmp, waitingList) == 0)
                {
                    _requestRespond(request_tmp, MS_ERR, tp);
                    handlerRequestDestroy(request_tmp);
                    free(request_tmp);
                }
                listDestroy(waitingList);
                free(waitingList);
            }
            handlerRequestDestroy(request);
            free(request);
            break;
        }
        case (R_UNLOCK_NOTIFY):
        {

            logger("Request:UNLOCK_NOTIFY", "LOCKHANDLER");
            if (hashTableGet(request->name, (void **)waitingList, waitingLocks) == 0)
            {
                PRINT_ERROR_CHECK(listPop((void **)&request_tmp, waitingList));

                logger("Waking 1 waiting request", "LOCKHANDLER");

                lockFile(request->name, request->uuid, fs);
                _requestRespond(request_tmp, MS_OK, tp);
                handlerRequestDestroy(request_tmp);
                free(request_tmp);
                if (listSize(*waitingList) == 0)
                {
                    hashTableRemove(request->name, NULL, waitingLocks);
                    listDestroy(waitingList);
                    free(waitingList);
                }
            }
            handlerRequestDestroy(request);
            free(request);
            break;
        }
        }
    }

    logger("Terminating", "LOCKHANDLER");
    while (hashTablePop(NULL, (void **)&waitingList, waitingLocks) != -1)
    {
        while (listPop((void **)&request_tmp, waitingList) != -1)
        {
            free(request_tmp->args);
            handlerRequestDestroy(request_tmp);
        }
        listDestroy(waitingList);
        free(waitingList);
    }
    hashTableDestroy(&waitingLocks);
}

/**
 * @brief Initializes given HandlerRequest with given values.
 *
 * @param type the type of the HandlerRequest.
 * @param name the name of the HandlerRequest.
 * @param uuid the UUID of the HandlerRequest.
 * @param args the handleArgs of the HandlerRequest.
 * @param request the HandlerRequest to initialize.
 * @return int 0 on success, -1 on failure.
 */
int handlerRequestInit(int type, char *name, char *uuid, struct _handleArgs *args, HandlerRequest *request)
{
    request->type = type;
    SAFE_NULL_CHECK(request->name = malloc(strlen(name) + 1));
    SAFE_NULL_CHECK(request->uuid = malloc(strlen(uuid) + 1));

    strcpy(request->name, name);
    strcpy(request->uuid, uuid);
    request->args = args;

    return 0;
}

/**
 * @brief Destroys given HandlerRequest, freeing it's resources.
 *
 * @param request the HandlerRequest to destroy.
 */
void handlerRequestDestroy(HandlerRequest *request)
{
    free(request->name);
    free(request->uuid);
}