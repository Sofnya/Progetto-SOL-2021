#include <sys/socket.h>
#include <sys/un.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>
#include <signal.h>

#include "COMMON/macros.h"
#include "SERVER/threadpool.h"
#include "COMMON/message.h"
#include "COMMON/helpers.h"
#include "SERVER/filesystem.h"
#include "SERVER/globals.h"
#include "SERVER/connstate.h"
#include "SERVER/logging.h"

#define UNIX_PATH_MAX 108
#define N 100

int sfd;
ThreadPool pool;
FileSystem fs;
List connections;
volatile sig_atomic_t signalNumber = 0;

struct _messageArgs
{
    int fd;
    Message *m;
};

struct _handleArgs
{
    int fd;
    size_t counter;
    Message *request;
    ConnState *state;
};

void acceptRequests();
void handleRequest(void *args);
Message *parseRequest(Message *request, ConnState *state);

void logConnection(ConnState state);
void logDisconnect(ConnState state);
void logRequest(Message *request, ConnState state);
void logResponse(Message *response, ConnState state);

void cleanup(void);

long _heuristic(void *el);
char *a(void *);

// Just tells the main to terminate.
void signalHandler(int signum)
{
    signalNumber = signum;
    return;
}

// This is called at every clean exit, as it's registered atexit(). We use it to free all resources, and print an account of the session.
void cleanup(void)
{
    int *cur;

    logger("Cleaning up!", "STATUS");
    close(sfd);
    unlink(SOCK_NAME);

    threadpoolDestroy(&pool);

    prettyPrintStats(fs.fsStats);
    prettyPrintFiles(&fs);
    fsDestroy(&fs);

    // Need to properly destroy the connections list.
    while (listPop((void **)&cur, &connections) != -1)
    {
        free(cur);
    }
    listDestroy(&connections);

    logger("Done cleaning up, goodbye!", "STATUS");
}

int main(int argc, char *argv[])
{
    struct sockaddr_un sa;
    struct sigaction action;

    action.sa_handler = &signalHandler;
    action.sa_flags = 0;
    sigfillset(&action.sa_mask);

    // Register our signalHandler.
    sigaction(SIGQUIT, &action, NULL);
    sigaction(SIGINT, &action, NULL);
    sigaction(SIGHUP, &action, NULL);

    // Ignore SIGPIPE to avoid having problems with reads/sends.
    sigaction(SIGPIPE, &(struct sigaction){{SIG_IGN}}, NULL);

    // Register our cleanup.
    atexit(&cleanup);

    // We start by parsing our config file.
    if (argc == 2)
    {
        load_config(argv[1]);
    }
    else
    {
        load_config("config.txt");
    }

    // Always use a clean log.
    remove(LOG_FILE);

    logger("Starting", "STATUS");

    listInit(&connections);
    // We now have the config options needed to initialize our FileSystem.
    fsInit(MAX_FILES, MAX_MEMORY, ENABLE_COMPRESSION, &fs);

    // We initialize our ThreadPool.
    threadpoolInit(CORE_POOL_SIZE, MAX_POOL_SIZE, &pool);

    // And open our socket.
    ERROR_CHECK(sfd = socket(AF_UNIX, SOCK_STREAM, 0));

    sa.sun_family = AF_UNIX;
    strncpy(sa.sun_path, SOCK_NAME, UNIX_PATH_MAX - 1);
    sa.sun_path[UNIX_PATH_MAX - 1] = '\00';

    if (bind(sfd, (struct sockaddr *)&sa, sizeof(sa)) == -1)
    {
        puts("Couldn't bind to requested address, terminating.");
        threadpoolFastExit(&pool);
        exit(EXIT_FAILURE);
    }

    if (listen(sfd, SOMAXCONN) == -1)
    {
        puts("Couldn't listen on requested address, terminating.");
        threadpoolFastExit(&pool);
        exit(EXIT_FAILURE);
    }

    // And now we start waiting for new requests, accepting them and passing them to the ThreadPool to handle.
    acceptRequests();

    // If we are here, we received a termination signal.

    logger("Exiting", "STATUS");

    int *cur;

    // First we disconnect all clients.
    while (listPop((void **)&cur, &connections) != -1)
    {
        close(*cur);
        free(cur);
    }

    // Then we terminate the threadpool.
    threadpoolCleanExit(&pool);

    exit(EXIT_SUCCESS);
}

void acceptRequests()
{
    int fd_sk = sfd, fd_c, fd_num = 0, fd, i = 0;
    int *curFD;
    fd_set set, rdset;
    Message *request;
    struct _handleArgs *args;
    HashTable connStates;
    ConnState *cur;
    void *saveptr = NULL;

    // An int is at most 10^19, so 19 chars long, we set length to 30 just to be safe :/.
    char curKey[30];

    sigset_t old_sigmask;
    sigset_t new_sigmask;

    hashTableInit(1024, &connStates);

    if (fd_sk > fd_num)
    {
        fd_num = fd_sk;
    }
    FD_ZERO(&set);
    FD_SET(fd_sk, &set);

    // We block termination signals when outside of pselect, to avoid race conditions.
    sigemptyset(&new_sigmask);
    sigaddset(&new_sigmask, SIGHUP);
    sigaddset(&new_sigmask, SIGINT);
    sigaddset(&new_sigmask, SIGQUIT);
    sigaddset(&new_sigmask, SIGPIPE);

    pthread_sigmask(SIG_SETMASK, &new_sigmask, &old_sigmask);

    while (signalNumber == 0 || ((signalNumber == SIGHUP) && (listSize(connections) > 0)))
    {
        rdset = set;

        if (pselect(fd_num + 1, &rdset, NULL, NULL, NULL, &old_sigmask) == -1)
        {
            continue;
        }
        else
        {
            for (fd = 0; fd <= fd_num; fd++)
            {
                if (FD_ISSET(fd, &rdset))
                {
                    // Accept new connections.
                    if (fd == fd_sk)
                    {
                        if (signalNumber == SIGHUP)
                        {
                            continue;
                        }

                        PRINT_ERROR_CHECK(fd_c = accept(fd_sk, NULL, 0));

                        if (fd_c > fd_num)
                        {
                            fd_num = fd_c;
                        }
                        FD_SET(fd_c, &set);
                        UNSAFE_NULL_CHECK(cur = malloc(sizeof(ConnState)));

                        // We use the fd as a key, as it's guaranteed to be unique as long as the connection is alive.
                        sprintf(curKey, "%d", fd_c);
                        PRINT_ERROR_CHECK(connStateInit(&fs, cur));

                        PRINT_ERROR_CHECK(hashTablePut(curKey, (void *)cur, connStates));

                        UNSAFE_NULL_CHECK(curFD = malloc(sizeof(int)));
                        *curFD = fd_c;
                        PRINT_ERROR_CHECK(listPush((void *)curFD, &connections));

                        logConnection(*cur);
                    }
                    else
                    {
                        request = malloc(sizeof(Message));
                        // A new request is ready.
                        if (receiveMessage(fd, request) != -1)
                        {
                            sprintf(curKey, "%d", fd);

                            PRINT_ERROR_CHECK(hashTableGet(curKey, (void **)&cur, connStates));

                            UNSAFE_NULL_CHECK(args = malloc(sizeof(struct _handleArgs)));
                            args->fd = fd;
                            args->request = request;
                            args->state = cur;
                            args->counter = atomicInc(1, &cur->requestN) - 1;

                            atomicInc(1, &cur->inUse);

                            logRequest(request, *cur);
                            PRINT_ERROR_CHECK(threadpoolSubmit(&handleRequest, args, &pool));
                        }
                        // Or a socket was closed, we handle the disconnection.
                        else
                        {
                            free(request);

                            // Remove the corresponding ConnState from the HashTable.
                            sprintf(curKey, "%d", fd);
                            PRINT_ERROR_CHECK(hashTableRemove(curKey, (void **)&cur, connStates));
                            logDisconnect(*cur);

                            // And mark it for destruction by the last thread using it.
                            cur->shouldDestroy = 1;
                            if (atomicComp(0, &cur->inUse) == 0)
                            {
                                connStateDestroy(cur);
                                free(cur);
                            }

                            FD_CLR(fd, &set);
                            PRINT_ERROR_CHECK(close(fd));

                            // Keep our list of connections updated.
                            saveptr = NULL;
                            i = 0;
                            while (listScan((void **)&curFD, &saveptr, &connections) != -1)
                            {
                                if (*curFD == fd)
                                {
                                    PRINT_ERROR_CHECK(listRemove(i, (void **)&curFD, &connections));
                                    free(curFD);
                                    errno = 0;
                                    break;
                                }
                                i++;
                            }
                            if (errno == EOF)
                            {
                                if (*curFD == fd)
                                {
                                    PRINT_ERROR_CHECK(listRemove(i, (void **)&curFD, &connections));
                                    free(curFD);
                                }
                            }

                            PRINT_ERROR_CHECK(listSort(&connections, &_heuristic));

                            // customPrintList(&connections, &a);
                            if (listSize(connections) == 0)
                            {
                                fd_num = fd_sk;
                            }
                            else
                            {
                                PRINT_ERROR_CHECK(listGet(0, (void **)&curFD, &connections));
                                if (*curFD > fd_sk)
                                {
                                    fd_num = *curFD;
                                }
                                else
                                {
                                    fd_num = fd_sk;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Gotta properly destroy our connstates at the end.
    while (hashTablePop(NULL, (void **)&cur, connStates) != -1)
    {
        cur->shouldDestroy = 1;
        if (atomicGet(&cur->inUse) == 0)
        {
            connStateDestroy(cur);
            free(cur);
        }
    }
    hashTableDestroy(&connStates);
}

void handleRequest(void *args)
{
    struct _handleArgs handleArgs = *(struct _handleArgs *)args;
    Message *request = handleArgs.request;
    ConnState *state = handleArgs.state;
    int fd = handleArgs.fd;
    size_t counter = handleArgs.counter;
    Message *response;

    // If the connState is being destroyed, returns immediately, destroying it if we are the last to use it.
    if (state->shouldDestroy == 1)
    {
        if (atomicDec(1, &state->inUse) == 0)
        {
            connStateDestroy(state);
            free(state);
        }
        return;
    }

    // If the request is out of order we push it back on the threadpool and execute something else.
    if (atomicComp(counter, &state->parsedN) != 0)
    {
        threadpoolSubmit(&handleRequest, args, &pool);
        puts("Request out of order");
        return;
    }

    free(args);
    // Parse it and generate an appropriate response. This is where we actually modify the FileSystem.
    response = parseRequest(request, state);
    logResponse(response, *state);

    // And send our response. No error handling as we need to terminate anyway.
    sendMessage(fd, response);

    // Update our counter.
    atomicInc(1, &state->parsedN);

    messageDestroy(request);
    messageDestroy(response);
    free(request);
    free(response);

    if (atomicDec(1, &state->inUse) == 0)
    {
        if (state->shouldDestroy == 1)
        {
            connStateDestroy(state);
            free(state);
        }
    }

    return;
}

// Here we parse a request, making all necessary calls to the FileSystem and generating an appropriate response with the results.
Message *parseRequest(Message *request, ConnState *state)
{
    Message *response;

    UNSAFE_NULL_CHECK(response = malloc(sizeof(Message)));
    switch (request->type)
    {
    case (MT_INFO):
    {
        messageInit(0, NULL, "INFO", MT_INFO, MS_OK, response);
        return response;
    }

    case (MT_FOPEN):
    {
        int flags;
        FileContainer *fcs = NULL;
        int fcsSize = 0, i;
        void *buf;
        size_t size;

        // A safety check.
        if (request->size == sizeof(int))
        {
            flags = *((int *)(request->content));
            if (conn_openFile(request->info, flags, &fcs, &fcsSize, state) == 0)
            {
                // I wanted more information in the message so that we can tell which flags where used in the log.
                if ((flags & O_CREATE) && (flags & O_LOCK))
                {
                    messageInit(0, NULL, "OPEN|CREATE|LOCK", MT_INFO, MS_OK, response);
                }
                else if (flags & O_CREATE)
                {
                    messageInit(0, NULL, "OPEN|CREATE", MT_INFO, MS_OK, response);
                }
                else if (flags & O_LOCK)
                {
                    messageInit(0, NULL, "OPEN|LOCK", MT_INFO, MS_OK, response);
                }
                else
                {
                    messageInit(0, NULL, "OPEN", MT_INFO, MS_OK, response);
                }
            }
            // We handle capacity misses by serializing our array of FileContainers in a buffer, and send it that way.
            else if (errno == EOVERFLOW)
            {
                if (serializeContainerArray(fcs, fcsSize, &size, &buf) == 0)
                {
                    messageInit(size, buf, "OPEN|CAPMISS", MT_INFO, MS_OKCAP, response);
                    free(buf);
                }
                else
                {
                    perror("Something went wrong while serializing a container array...");
                    messageInit(0, NULL, "ERROR|CAPMISS", MT_INFO, MS_ERR, response);
                }
            }
            else
            {
                messageInit(0, NULL, "ERROR", MT_INFO, MS_ERR, response);
            }
            if (fcs != NULL)
            {
                for (i = 0; i < fcsSize; i++)
                {
                    destroyContainer(fcs + i);
                }
                free(fcs);
            }
            return response;
        }
    }

    case (MT_FCLOSE):
    {
        if (conn_closeFile(request->info, state) == 0)
        {
            messageInit(0, NULL, "CLOSED", MT_INFO, MS_OK, response);
        }
        else
        {
            messageInit(0, NULL, "ERROR", MT_INFO, MS_ERR, response);
        }
        return response;
    }

    case (MT_FREAD):
    {
        void *buf;
        size_t size;

        // We need the File's uncompressed size to allocate an appropriate buffer.
        size = getSize(request->info, state->fs);
        if (size == 0)
        {
            messageInit(0, NULL, "ERROR|MISSING", MT_INFO, MS_ERR, response);
            return response;
        }

        buf = malloc(size);
        if (buf == NULL)
        {
            messageInit(0, NULL, "ERROR", MT_INFO, MS_INTERR, response);
            return response;
        }
        if (conn_readFile(request->info, &buf, size, state) == 0)
        {
            messageInit(size, buf, "READ", MT_INFO, MS_OK, response);
        }
        else
        {
            messageInit(0, NULL, "ERROR", MT_INFO, MS_ERR, response);
        }
        free(buf);
        return response;
    }

    case (MT_FWRITE):
    {
        FileContainer *fcs = NULL;
        int fcsSize = 0, i;
        void *buf;
        size_t size;

        if (conn_writeFile(request->info, request->content, request->size, &fcs, &fcsSize, state) == 0)
        {
            messageInit(0, NULL, "WRITE", MT_INFO, MS_OK, response);
        }
        // Need to handle capacity missess.
        else if (errno == EOVERFLOW)
        {
            if (serializeContainerArray(fcs, fcsSize, &size, &buf) == 0)
            {
                messageInit(size, buf, "WRITE|CAPMISS", MT_INFO, MS_OKCAP, response);
                free(buf);
            }
            else
            {
                perror("Something went wrong while serializing a container array...");
                messageInit(0, NULL, "ERROR|CAPMISS", MT_INFO, MS_ERR, response);
            }
        }
        else
        {
            messageInit(0, NULL, "ERROR", MT_INFO, MS_ERR, response);
        }
        if (fcs != NULL)
        {
            for (i = 0; i < fcsSize; i++)
            {
                destroyContainer(fcs + i);
            }
            free(fcs);
        }
        return response;
    }

    // Appends are actually never called by the client, but still supported as per the API.
    case (MT_FAPPEND):
    {
        FileContainer *fcs = NULL;
        int fcsSize = 0, i;
        void *buf;
        size_t size;

        if (conn_appendFile(request->info, request->content, request->size, &fcs, &fcsSize, state) == 0)
        {
            messageInit(0, NULL, "APPEND", MT_INFO, MS_OK, response);
        }
        else if (errno == EOVERFLOW)
        {
            if (serializeContainerArray(fcs, fcsSize, &size, &buf) == 0)
            {
                messageInit(size, buf, "APPEND|CAPMISS", MT_INFO, MS_OKCAP, response);
                free(buf);
            }
            else
            {
                perror("Something went wrong while serializing a container array...");
                messageInit(0, NULL, "ERROR|CAPMISS", MT_INFO, MS_ERR, response);
            }
        }
        else
        {
            messageInit(0, NULL, "ERROR", MT_INFO, MS_ERR, response);
        }

        if (fcs != NULL)
        {
            for (i = 0; i < fcsSize; i++)
            {
                destroyContainer(fcs + i);
            }
            free(fcs);
        }
        return response;
    }

    case (MT_FREM):
    {
        if (conn_removeFile(request->info, state) == 0)
        {
            messageInit(0, NULL, "REMOVED", MT_INFO, MS_OK, response);
        }
        else
        {
            messageInit(0, NULL, "ERROR", MT_INFO, MS_ERR, response);
        }
        return response;
    }

    case (MT_DISCONNECT):
    {
        messageInit(0, NULL, "DISCONNECTED", MT_INFO, MS_OK, response);
        return response;
    }

    case (MT_FLOCK):
    {
        if (conn_lockFile(request->info, state) == 0)
        {
            messageInit(0, NULL, "LOCKED", MT_INFO, MS_OK, response);
        }
        else
        {
            messageInit(0, NULL, "ERROR", MT_INFO, MS_ERR, response);
        }
        return response;
    }

    case (MT_FUNLOCK):
    {
        if (conn_unlockFile(request->info, state) == 0)
        {
            messageInit(0, NULL, "UNLOCKED", MT_INFO, MS_OK, response);
        }
        else
        {
            messageInit(0, NULL, "ERROR", MT_INFO, MS_ERR, response);
        }
        return response;
    }

    case (MT_FREADN):
    {
        FileContainer *fc;
        void *buf = NULL;
        int amount, i, n;
        size_t size;
        char info[200];

        n = *(int *)request->content;
        amount = conn_readNFiles(n, &fc, state);

        // As for a capacity miss, we use a serialized FileContainer array to send many files back.
        if (amount > 0)
        {
            if (serializeContainerArray(fc, amount, &size, &buf) == -1)
            {
                messageInit(0, NULL, "ERROR", MT_INFO, MS_ERR, response);
            }
            else
            {
                sprintf(info, "READN:%d", amount);
                messageInit(size, buf, info, MT_INFO, MS_OK, response);
                for (i = 0; i < amount; i++)
                {
                    destroyContainer(&fc[i]);
                }
            }
            free(fc);
            free(buf);
        }
        else
        {
            free(fc);
            messageInit(0, NULL, "ERROR", MT_INFO, MS_ERR, response);
        }
        return response;
    }

    default:
    {
        messageInit(0, NULL, "Invalid request.", MT_INFO, MS_INV, response);
        return response;
    }
    }
}

// Just some pretty logging.
void logConnection(ConnState state)
{
    char parsed[36 + 100];
    sprintf(parsed, ">Connected >UUID:%s", state.uuid);
    logger(parsed, "CONN_OPEN");
}

void logDisconnect(ConnState state)
{
    char parsed[36 + 100];
    sprintf(parsed, ">Disconnected >UUID:%s", state.uuid);
    logger(parsed, "CONN_CLOSE");
}

void logRequest(Message *request, ConnState state)
{
    char *parsed;
    if (request->info == NULL)
    {
        parsed = malloc(500);
        sprintf(parsed, ">MALFORMED >UUID:%s", state.uuid);
    }
    else
    {
        parsed = malloc(strlen(request->info) + 500);
        switch (request->type)
        {
        case (MT_INFO):
        {
            sprintf(parsed, ">INFO >UUID:%s >%s", state.uuid, request->info);
            break;
        }
        case (MT_FOPEN):
        {
            sprintf(parsed, ">OPEN >UUID:%s >%s", state.uuid, request->info);
            break;
        }
        case (MT_FCLOSE):
        {
            sprintf(parsed, ">CLOSE >UUID:%s >%s", state.uuid, request->info);
            break;
        }
        case (MT_FREAD):
        {
            sprintf(parsed, ">READ >UUID:%s >%s", state.uuid, request->info);
            break;
        }
        case (MT_FWRITE):
        {
            sprintf(parsed, ">WRITE >UUID:%s >Size:%ld >%s", state.uuid, request->size, request->info);
            break;
        }
        case (MT_FAPPEND):
        {
            sprintf(parsed, ">APPEND >UUID:%s >Size:%ld >%s", state.uuid, request->size, request->info);
            break;
        }
        case (MT_FREM):
        {
            sprintf(parsed, ">REMOVE >UUID:%s >%s", state.uuid, request->info);
            break;
        }
        case (MT_DISCONNECT):
        {
            sprintf(parsed, ">DISCONNECT >UUID:%s >%s", state.uuid, request->info);
            break;
        }
        case (MT_FLOCK):
        {
            sprintf(parsed, ">LOCK >UUID:%s >%s", state.uuid, request->info);
            break;
        }
        case (MT_FUNLOCK):
        {
            sprintf(parsed, ">UNLOCK >UUID:%s >%s", state.uuid, request->info);
            break;
        }
        case (MT_FREADN):
        {
            sprintf(parsed, ">READN >UUID:%s >%s", state.uuid, request->info);
            break;
        }
        }
    }
    logger(parsed, "REQUEST");
    free(parsed);
}

void logResponse(Message *response, ConnState state)
{
    char *parsed;
    if (response->info == NULL)
    {
        parsed = malloc(500);
        sprintf(parsed, ">MALFORMED >UUID:%s", state.uuid);
    }
    else
    {
        parsed = malloc(500 + strlen(response->info));
        sprintf(parsed, ">%s >Size:%ld >UUID:%s", response->info, response->size, state.uuid);
    }
    logger(parsed, "RESPONSE");
    free(parsed);
}

// A simple heuristic to sort our connections list in descending order.
long _heuristic(void *el)
{
    int *cur = (int *)el;
    return -*cur;
}

char *a(void *el)
{
    char *res = malloc(30);
    int cur = *(int *)el;
    sprintf(res, "%d", cur);
    return res;
}