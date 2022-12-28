#include <string.h>
#include <stdlib.h>
#include <signal.h>
#include <errno.h>

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

char *_printWaitingList(void *el)
{
    struct _entry a = *(struct _entry *)el;
    char *res;
    HandlerRequest *cur;
    List *waitingList = (List *)a.value;
    void *saveptr = NULL;
    int len;
    int count;

    len = strlen(a.key) + 100;
    len += listSize(*waitingList) * 90;

    UNSAFE_NULL_CHECK(res = malloc(len));

    count = sprintf(res, "Key: %s\n", a.key);
    while (listScan((void **)&cur, &saveptr, waitingList) != -1)
    {
        count += sprintf(res + count, "%s | ", cur->uuid);
    }
    return res;
}

char *_printLockedFiles(void *el)
{
    struct _entry a = *(struct _entry *)el;
    char *res;
    int len;

    len = strlen(a.key);
    len += strlen((const char *)a.value);
    len += 100;
    res = malloc(len);
    sprintf(res, "%s : %s", a.key, (char *)a.value);
    return res;
}
void _printAllLocks(HashTable lockedFiles, HashTable waitingLocks)
{
    if (ONE_LOCK_POLICY && (hashTableSize(lockedFiles) > 0))
    {
        puts("\n------------\nLocked Files:");
        customPrintHashTable(lockedFiles, &_printLockedFiles);
        puts("\n------------");
    }
    if (hashTableSize(waitingLocks) > 0)
    {
        puts("\n------------\nWaiting Locks:");
        customPrintHashTable(waitingLocks, &_printWaitingList);
        puts("\n------------");
    }
}

void *lockHandler(void *args)
{
    struct _handlerArgs *hargs = (struct _handlerArgs *)args;

    FileSystem *fs = hargs->fs;
    SyncQueue *queue = hargs->msgQueue;
    ThreadPool *tp = hargs->tp;
    volatile int *terminate = hargs->terminate;
    free(hargs);

    HashTable waitingLocks;
    List *waitingList;
    char log[500];

    PRINT_ERROR_CHECK(hashTableInit(MAX_FILES, &waitingLocks));

    HandlerRequest *request;
    HandlerRequest *request_tmp;

    // For the implementation of ONE_LOCK_POLICY
    // We keep an HashTable mapping an UUID to it's current lockedFile.
    HashTable lockedFiles;
    char *lockedFile;

    if (ONE_LOCK_POLICY)
    {
        PRINT_ERROR_CHECK(hashTableInit(1024, &lockedFiles));
    }

    // All threads except the main root thread should ignore termination signals, and let the root thread handle termination.
    sigset_t sigmask;
    sigemptyset(&sigmask);
    sigaddset(&sigmask, SIGINT);
    sigaddset(&sigmask, SIGHUP);
    sigaddset(&sigmask, SIGQUIT);
    pthread_sigmask(SIG_BLOCK, &sigmask, NULL);

    logger("Starting", "LOCKHANDLER");
    while (!(*terminate))
    {
        request = (HandlerRequest *)syncqueuePop(queue);
        // sprintf(log, ">Queue size:%ld >Waiting size:%lld", syncqueueLen(*queue), hashTableSize(waitingLocks));
        // logger(log, "LOCKHANDLER");

        _printAllLocks(lockedFiles, waitingLocks);
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
            // If ONE_LOCK_POLICY is activated in the config, we keep track of each connstates one locked file, and automatically unlock it before trying to lock a new file.
            if (ONE_LOCK_POLICY)
            {
                if (hashTableRemove(request->uuid, (void **)&lockedFile, lockedFiles) != -1)
                {

                    sprintf(log, "ONELOCKPOLICY UUID:%s  >Automatically unlocking file %20s in favour of %20s", request->uuid, lockedFile, request->name);
                    logger(log, "LOCKHANDLER");

                    PRINT_ERROR_CHECK(unlockFile(lockedFile, fs));

                    // We notify ourselves for code reuse.
                    UNSAFE_NULL_CHECK(request_tmp = malloc(sizeof(HandlerRequest)));
                    handlerRequestInit(R_UNLOCK_NOTIFY, lockedFile, request->uuid, NULL, request_tmp);
                    syncqueuePush((void *)request_tmp, queue);

                    free(lockedFile);
                }
            }
            errno = 0;
            if (isLockedFile(request->name, fs) == -1 && errno != EBADF)
            {

                sprintf(log, "Request:LOCK >Success >UUID:%s >File:%s", request->uuid, request->name);
                logger(log, "LOCKHANDLER");
                lockFile(request->name, request->uuid, fs);

                if (ONE_LOCK_POLICY)
                {
                    UNSAFE_NULL_CHECK(lockedFile = malloc(strlen(request->name) + 1));
                    strcpy(lockedFile, request->name);
                    hashTablePut(request->uuid, lockedFile, lockedFiles);
                }

                _requestRespond(request, MS_OK, tp);
                handlerRequestDestroy(request);
                free(request);
            }
            else if (errno == EBADF)
            {

                sprintf(log, "Request:LOCK >Error >UUID:%s >File:%s", request->uuid, request->name);
                logger(log, "LOCKHANDLER");

                _requestRespond(request, MS_ERR, tp);
                handlerRequestDestroy(request);
                free(request);
            }
            else
            {
                sprintf(log, "Request:LOCK >Queued >UUID:%s >File:%s", request->uuid, request->name);
                logger(log, "LOCKHANDLER");
                if (hashTableGet(request->name, (void **)&waitingList, waitingLocks) == -1)
                {
                    UNSAFE_NULL_CHECK(waitingList = malloc(sizeof(List)));
                    PRINT_ERROR_CHECK(listInit(waitingList));
                    PRINT_ERROR_CHECK(listPush((void *)request, waitingList));
                    PRINT_ERROR_CHECK(hashTablePut(request->name, (void *)waitingList, waitingLocks));
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
            if (ONE_LOCK_POLICY)
            {
                if (hashTableGet(request->uuid, (void **)&lockedFile, lockedFiles) != -1 && !strcmp(lockedFile, request->name))
                {
                    hashTableRemove(request->uuid, NULL, lockedFiles);
                    free(lockedFile);
                }
            }
            if (isLockedByFile(request->name, request->uuid, fs) == 0)
            {

                sprintf(log, "Request:UNLOCK >Success >UUID:%s >File:%s", request->uuid, request->name);
                logger(log, "LOCKHANDLER");
                PRINT_ERROR_CHECK(unlockFile(request->name, fs));
                if (hashTableGet(request->name, (void **)&waitingList, waitingLocks) == 0)
                {
                    PRINT_ERROR_CHECK(listPop((void **)&request_tmp, waitingList));
                    logger("Waking 1 waiting request", "LOCKHANDLER");

                    PRINT_ERROR_CHECK(lockFile(request_tmp->name, request_tmp->uuid, fs));

                    // In case of a woken up lock, we have to update it's lockedFile.
                    if (ONE_LOCK_POLICY)
                    {
                        UNSAFE_NULL_CHECK(lockedFile = malloc(strlen(request_tmp->name) + 1));
                        strcpy(lockedFile, request_tmp->name);
                        hashTablePut(request_tmp->uuid, lockedFile, lockedFiles);
                    }

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

                sprintf(log, "Request:UNLOCK >Error >UUID:%s >File:%s", request->uuid, request->name);
                logger(log, "LOCKHANDLER");
                _requestRespond(request, MS_ERR, tp);
            }
            handlerRequestDestroy(request);
            free(request);
            break;
        }
        case (R_REMOVE):
        {
            if (ONE_LOCK_POLICY)
            {
                if (hashTableGet(request->uuid, (void **)&lockedFile, lockedFiles) != -1 && !strcmp(lockedFile, request->name))
                {
                    hashTableRemove(request->uuid, NULL, lockedFiles);
                    free(lockedFile);
                }
            }

            sprintf(log, "Request:REMOVE >UUID:%s >File:%s", request->uuid, request->name);
            logger(log, "LOCKHANDLER");
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

            if (ONE_LOCK_POLICY)
            {
                if (hashTableGet(request->uuid, (void **)&lockedFile, lockedFiles) != -1 && !strcmp(lockedFile, request->name))
                {
                    hashTableRemove(request->uuid, NULL, lockedFiles);
                    free(lockedFile);
                }
            }
            sprintf(log, "Request:UNLOCK_NOTIFY >UUID:%s >File:%s", request->uuid, request->name);
            logger(log, "LOCKHANDLER");
            if (hashTableGet(request->name, (void **)&waitingList, waitingLocks) == 0)
            {
                PRINT_ERROR_CHECK(listPop((void **)&request_tmp, waitingList));

                sprintf(log, "Waking lock >UUID:%s >File:%s", request_tmp->uuid, request_tmp->name);
                logger(log, "LOCKHANDLER");

                PRINT_ERROR_CHECK(lockFile(request_tmp->name, request_tmp->uuid, fs));

                if (ONE_LOCK_POLICY)
                {
                    UNSAFE_NULL_CHECK(lockedFile = malloc(strlen(request_tmp->name) + 1));
                    strcpy(lockedFile, request_tmp->name);
                    hashTablePut(request_tmp->uuid, lockedFile, lockedFiles);
                }

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
        case (R_LOCK_CREATE_NOTIFY):
        {
            sprintf(log, "Request:LOCK_CREATE_NOTIFY >UUID:%s >File:%s", request->uuid, request->name);
            logger(log, "LOCKHANDLER");
            if (ONE_LOCK_POLICY)
            {
                if (hashTableRemove(request->uuid, (void **)&lockedFile, lockedFiles) != -1)
                {

                    sprintf(log, "ONELOCKPOLICY UUID:%s >Automatically unlocking file %20s in favour of %20s", request->uuid, lockedFile, request->name);
                    logger(log, "LOCKHANDLER");

                    PRINT_ERROR_CHECK(unlockFile(lockedFile, fs));

                    // We notify ourselves for code reuse.
                    UNSAFE_NULL_CHECK(request_tmp = malloc(sizeof(HandlerRequest)));
                    handlerRequestInit(R_UNLOCK_NOTIFY, lockedFile, request->uuid, NULL, request_tmp);
                    syncqueuePush((void *)request_tmp, queue);

                    free(lockedFile);
                }

                UNSAFE_NULL_CHECK(lockedFile = malloc(strlen(request->name) + 1));
                strcpy(lockedFile, request->name);
                hashTablePut(request->uuid, lockedFile, lockedFiles);
            }
            handlerRequestDestroy(request);
            free(request);
            break;
        }
        case (R_DISCONNECT_NOTIFY):
        {
            if (ONE_LOCK_POLICY)
            {
                if (hashTableRemove(request->uuid, (void **)&lockedFile, lockedFiles) != -1)
                {
                    free(lockedFile);
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

    if (ONE_LOCK_POLICY)
    {
        while (hashTablePop(NULL, (void **)&lockedFile, lockedFiles) != -1)
        {
            free(lockedFile);
        }
        hashTableDestroy(&lockedFiles);
    }

    return NULL;
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