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

struct _messageArgs
{
    int fd;
    Message *m;
};

void handleConnection(void *fdc);
Message *parseRequest(Message *request, ConnState state);

void logConnection(ConnState state);
void logDisconnect(ConnState state);
void logRequest(Message *request, ConnState state);
void logResponse(Message *response, ConnState state);

void cleanup(void);

int _receiveMessageWrapper(void *args);
int _sendMessageWrapper(void *args);

void signalHandler(int signum)
{
    logger("Exiting", "STATUS");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[])
{
    int fdc;
    int *curFd;
    struct sockaddr_un sa;

    signal(SIGTERM, &signalHandler);
    signal(SIGQUIT, &signalHandler);
    signal(SIGINT, &signalHandler);
    signal(SIGHUP, &signalHandler);
    sigaction(SIGPIPE, &(struct sigaction){{SIG_IGN}}, NULL);

    atexit(&cleanup);

    if (argc == 2)
    {
        load_config(argv[1]);
    }
    else
    {
        load_config("config.txt");
    }

    remove(LOG_FILE);

    logger("Starting", "STATUS");

    fsInit(MAX_FILES, MAX_MEMORY, ENABLE_COMPRESSION, &fs);

    ERROR_CHECK(sfd = socket(AF_UNIX, SOCK_STREAM, 0));

    sa.sun_family = AF_UNIX;
    strncpy(sa.sun_path, SOCK_NAME, UNIX_PATH_MAX - 1);
    sa.sun_path[UNIX_PATH_MAX - 1] = '\00';

    ERROR_CHECK(bind(sfd, (struct sockaddr *)&sa, sizeof(sa)))

    ERROR_CHECK(listen(sfd, SOMAXCONN))

    threadpoolInit(CORE_POOL_SIZE, MAX_POOL_SIZE, &pool);
    while (1)
    {
        ERROR_CHECK(fdc = accept(sfd, NULL, NULL));
        SAFE_NULL_CHECK(curFd = malloc(sizeof(int)));
        *curFd = fdc;
        threadpoolSubmit(&handleConnection, curFd, &pool);

        // puts("Got connection!");
    }

    exit(EXIT_SUCCESS);
}

void handleConnection(void *fdc)
{
    Message *request;
    Message *response;
    ConnState state;

    struct timespec maxWait;
    struct _messageArgs args;
    maxWait.tv_nsec = 0;
    maxWait.tv_sec = 5;

    int fd = *(int *)fdc;
    int err;
    args.fd = fd;

    bool done = false;
    free(fdc);

    connStateInit(&fs, &state);

    logConnection(state);
    while (!done)
    {
        if ((request = malloc(sizeof(Message))) == NULL)
        {
            perror("Error on malloc");
            done = true;
            continue;
        }

        request->size = 0;
        request->content = NULL;
        request->info = NULL;
        request->type = 0;
        request->status = 0;

        args.m = request;
        err = timeoutCall(_receiveMessageWrapper, (void *)&args, maxWait);
        if (err == -1 || err == ETIMEDOUT)
        {
            if (err != -1)
                perror("Error on receive");
            done = true;
            messageDestroy(request);
            free(request);
            continue;
        }

        done = (request->type == MT_DISCONNECT);

        logRequest(request, state);
        response = parseRequest(request, state);
        logResponse(response, state);

        args.m = response;
        err = timeoutCall(_sendMessageWrapper, (void *)&args, maxWait);
        if (err == -1 || err == ETIMEDOUT)
        {
            if (err != -1)
                perror("Error on send");
            done = true;
            messageDestroy(request);
            free(request);
            messageDestroy(response);
            free(response);
            continue;
        }

        messageDestroy(request);
        messageDestroy(response);
        free(request);
        free(response);
    }

    logDisconnect(state);
    close(fd);
    connStateDestroy(&state);
    return;
}

Message *parseRequest(Message *request, ConnState state)
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
        FileContainer *fcs;
        int fcsSize = 0, i;
        void *buf;
        uint64_t size;

        if (request->size == sizeof(int))
        {
            flags = *((int *)(request->content));
            if (conn_openFile(request->info, flags, &fcs, &fcsSize, state) == 0)
            {
                messageInit(0, NULL, "OPEN", MT_INFO, MS_OK, response);
            }
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

                for (i = 0; i < fcsSize; i++)
                {
                    destroyContainer(fcs + i);
                }
                free(fcs);
            }
            else
            {
                messageInit(0, NULL, "ERROR", MT_INFO, MS_ERR, response);
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
        uint64_t size;

        size = getSize(request->info, state.fs);
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
        FileContainer *fcs;
        int fcsSize = 0, i;
        void *buf;
        uint64_t size;

        if (conn_writeFile(request->info, request->content, request->size, &fcs, &fcsSize, state) == 0)
        {
            messageInit(0, NULL, "WRITE", MT_INFO, MS_OK, response);
        }
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
            for (i = 0; i < fcsSize; i++)
            {
                destroyContainer(fcs + i);
            }
            free(fcs);
        }
        else
        {
            messageInit(0, NULL, "ERROR", MT_INFO, MS_ERR, response);
        }
        return response;
    }

    case (MT_FAPPEND):
    {
        FileContainer *fcs;
        int fcsSize = 0, i;
        void *buf;
        uint64_t size;

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
            for (i = 0; i < fcsSize; i++)
            {
                destroyContainer(fcs + i);
            }
            free(fcs);
        }
        else
        {
            messageInit(0, NULL, "ERROR", MT_INFO, MS_ERR, response);
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
        uint64_t size;
        char info[200];

        n = *(int *)request->content;
        amount = conn_readNFiles(n, &fc, state);

        if (amount > 0)
        {
            serializeContainerArray(fc, amount, &size, &buf);
            sprintf(info, "READN:%d", amount);
            messageInit(size, buf, info, MT_INFO, MS_OK, response);
            for (i = 0; i < amount; i++)
            {
                destroyContainer(&fc[i]);
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

void cleanup(void)
{
    logger("Cleaning up!", "STATUS");
    close(sfd);
    unlink(SOCK_NAME);
    threadpoolCleanExit(&pool);

    threadpoolDestroy(&pool);

    fsDestroy(&fs);

    logger("Done cleaning up!", "STATUS");
}

int _receiveMessageWrapper(void *args)
{
    struct _messageArgs tmp = *((struct _messageArgs *)args);
    return receiveMessage(tmp.fd, tmp.m);
}
int _sendMessageWrapper(void *args)
{
    struct _messageArgs tmp = *((struct _messageArgs *)args);
    return sendMessage(tmp.fd, tmp.m);
}

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
        sprintf(parsed, ">%s >Size:%ld >UUID:%s ", response->info, response->size, state.uuid);
    }
    logger(parsed, "RESPONSE");
    free(parsed);
}
