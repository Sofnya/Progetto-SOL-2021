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
#include "SERVER/filesystem.h"
#include "SERVER/globals.h"


#define UNIX_PATH_MAX 108
#define N 100


int sfd;
ThreadPool pool;
FileSystem fs;

void handleConnection(void *fdc);
void cleanup(void);
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
        curFd = malloc(sizeof(int));
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
    int fd = *(int *)fdc;
    free(fdc);

    UNSAFE_NULL_CHECK(request = malloc(sizeof(Message)));

    receiveMessage(fd, request);

    puts(request->info);
    

    response = parseRequest(request);
    sendMessage(fd, response);    
    
    messageDestroy(request);
    messageDestroy(response);
    
    free(request);
    free(response);
    close(fd);
    
    return;
}


Message *parseRequest(Message *request)
{
    Message *response;
    UNSAFE_NULL_CHECK(response = malloc(sizeof(Message)));
    switch(request->type)
    {
        case(0):
            messageInit(0,NULL, "Hey to u too", 0, 200, response);
            return response;
        default:
            messageInit(0, NULL, "Invalid request.", 0, 400, response);
            return response;
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
}