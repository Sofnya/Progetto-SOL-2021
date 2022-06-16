#include <sys/socket.h>
#include <sys/un.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>
#include <signal.h>


#include "COMMON/macros.h"
#include "COMMON/threadpool.h"
#include "COMMON/message.h"
#include "COMMON/helpers.h"
#include "SERVER/filesystem.h"
#include "SERVER/globals.h"
#include "SERVER/connstate.h"


#define UNIX_PATH_MAX 108
#define N 100


int sfd;
ThreadPool pool;
FileSystem fs;

struct _messageArgs{
    int fd;
    Message *m;
};

void handleConnection(void *fdc);
Message *parseRequest(Message *request, ConnState state);
void cleanup(void);

int _receiveMessageWrapper(void *args);
int _sendMessageWrapper(void *args);


void signalHandler(const int signum) {
    puts("Exiting");
    exit(EXIT_FAILURE);
}

int main()
{
    int fdc;
    int *curFd;
    struct sockaddr_un sa;

    signal(SIGTERM, &signalHandler);
    signal(SIGQUIT, &signalHandler);
    signal(SIGINT, &signalHandler);
    sigaction(SIGPIPE, &(struct sigaction){SIG_IGN}, NULL);

    atexit(&cleanup);

    load_config("config");

    fsInit(MAX_FILES, MAX_MEMORY, &fs);


    ERROR_CHECK(sfd = socket(AF_UNIX, SOCK_STREAM, 0));

    sa.sun_family = AF_UNIX;
    strncpy(sa.sun_path, SOCK_NAME, UNIX_PATH_MAX - 1);
    sa.sun_path[UNIX_PATH_MAX - 1] = '\00';

    ERROR_CHECK(bind(sfd, (struct sockaddr *)&sa, sizeof(sa)))
    
    ERROR_CHECK(listen(sfd, SOMAXCONN))

    threadpoolInit(CORE_POOL_SIZE, MAX_POOL_SIZE, &pool);
    while(1){
        ERROR_CHECK(fdc = accept(sfd, NULL, NULL));
        SAFE_NULL_CHECK(curFd = malloc(sizeof(int)));
        *curFd = fdc;
        threadpoolSubmit(&handleConnection, curFd, &pool);
        
        //puts("Got connection!");
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
    while(!done){
        if((request = malloc(sizeof(Message))) == NULL)
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
        err = timeoutCall(_receiveMessageWrapper, (void *)&args ,maxWait);
        if(err == -1 || err == ETIMEDOUT)
        {
            if (err != -1) perror("Error on receive");
            done = true;
            messageDestroy(request);
            free(request);
            continue;
        }

        done = (request->type == MT_DISCONNECT);
        response = parseRequest(request, state);
        
        args.m = response;
        err = timeoutCall(_sendMessageWrapper, (void *)&args ,maxWait);
        if(err == -1 || err == ETIMEDOUT)
        {
            if (err != -1) perror("Error on send");
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

    close(fd);
    connStateDestroy(&state);
    puts("Done bye");
    return;
}


Message *parseRequest(Message *request, ConnState state)
{
    Message *response;
    UNSAFE_NULL_CHECK(response = malloc(sizeof(Message)));
    switch(request->type)
    {
        case (MT_INFO):
        {
            messageInit(0,NULL, "Hello to you too!", MT_INFO, MS_OK, response);
            return response;
        }

        case (MT_FOPEN):
        {
            int flags;
            if(request->size == sizeof(int))
            {
                flags = *((int *)(request->content));
                if(conn_openFile(request->info, flags, state) == 0)
                {
                    messageInit(0, NULL, "File opened!", MT_INFO, MS_OK, response);
                }
                else
                {
                    messageInit(0, NULL, "Error!", MT_INFO, MS_ERR, response);
                }
                return response;
            }
        }

        case (MT_FCLOSE):
        {
            if(conn_closeFile(request->info, state) == 0)
            {
                    messageInit(0, NULL, "File closed!", MT_INFO, MS_OK, response);
            }
            else
            {
                    messageInit(0, NULL, "Error!", MT_INFO, MS_ERR, response); 
            }
            return response;
        }

        case (MT_FREAD):
        {
            void *buf;
            uint64_t size;
            
            size = getSize(request->info, state.fs);
            if(size == 0)
            {
                messageInit(0, NULL, "No such file!", MT_INFO, MS_ERR, response);
                return response;
            }

            buf = malloc(size);
            if(buf == NULL)
            {
                messageInit(0, NULL, "Server error!", MT_INFO, MS_INTERR, response);
                return response;
            }
            if(conn_readFile(request->info, &buf, size, state) == 0)
            {
                messageInit(size, buf, "Read done!", MT_INFO, MS_OK, response);
            }
            else
            {
                messageInit(0, NULL, "Error!", MT_INFO, MS_ERR, response);
            }
            free(buf);
            return response;
        }

        case (MT_FWRITE):
        {
            if(conn_writeFile(request->info, request->content, request->size, state) == 0)
            {
                messageInit(0, NULL, "Write done!", MT_INFO, MS_OK, response);
            }
            else
            {
                messageInit(0, NULL, "Error!", MT_INFO, MS_ERR, response);
            }
            return response;
        }

        case (MT_FAPPEND):
        {
            if(conn_appendFile(request->info, request->content, request->size, state) == 0)
            {
                messageInit(0, NULL, "Append done!", MT_INFO, MS_OK, response);
            }
            else
            {
                messageInit(0, NULL, "Error!", MT_INFO, MS_ERR, response);
            }
            return response;
        }

        case (MT_FREM):
        {
            if(conn_removeFile(request->info, state) == 0)
            {
                messageInit(0, NULL, "File Removed!", MT_INFO, MS_OK, response);
            }
            else
            {
                messageInit(0, NULL, "Error!", MT_INFO, MS_ERR, response);
            }
            return response;
        }

        case (MT_DISCONNECT):
        {
            messageInit(0, NULL, "Ok, bye!", MT_INFO, MS_OK, response);
            return response;
        }
        
        case (MT_FLOCK):
        {
            if(conn_lockFile(request->info, state) == 0)
            {
                messageInit(0, NULL, "File Locked!", MT_INFO, MS_OK, response);
            }
            else
            {
                messageInit(0, NULL, "Error!", MT_INFO, MS_ERR, response);
            }
            return response;
        }

        case(MT_FUNLOCK):
        {
            if(conn_unlockFile(request->info, state) == 0)
            {
                messageInit(0, NULL, "File Unlocked!", MT_INFO, MS_OK, response);
            }
            else
            {
                messageInit(0, NULL, "Error!", MT_INFO, MS_ERR, response);
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
    puts("Cleaning up!");
    close(sfd);
    unlink(SOCK_NAME);
    threadpoolCleanExit(&pool);

    threadpoolDestroy(&pool);

    fsDestroy(&fs);

    puts("Done cleaning up!");
}




int _receiveMessageWrapper(void *args)
{
    struct _messageArgs tmp = *((struct _messageArgs *) args);
    return receiveMessage(tmp.fd, tmp.m);
}
int _sendMessageWrapper(void *args)
{
    struct _messageArgs tmp = *((struct _messageArgs *) args);
    return sendMessage(tmp.fd, tmp.m);
}