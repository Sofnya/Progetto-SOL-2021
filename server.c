#include <sys/socket.h>
#include <sys/un.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>
#include <signal.h>


#include "macros.h"
#include "threadpool.h"
#include "globals.h"


#define UNIX_PATH_MAX 108
#define N 100


int sfd;
threadPool pool;

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

    ERROR_CHECK(sfd = socket(AF_UNIX, SOCK_STREAM, 0));

    sa.sun_family = AF_UNIX;
    strncpy(sa.sun_path, SOCK_NAME, UNIX_PATH_MAX - 1);
    sa.sun_path[UNIX_PATH_MAX - 1] = '\00';

    ERROR_CHECK(bind(sfd, (struct sockaddr *)&sa, sizeof(sa)))
    
    ERROR_CHECK(listen(sfd, SOMAXCONN))

    threadpoolInit(8, &pool);
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
    char buf[N];
    int fd = *(int *)fdc;
    free(fdc);
    read(fd, buf, N);
    puts(buf);
    close(fd);
    
    return;
}

void cleanup(void)
{
    close(sfd);
    unlink(SOCK_NAME);
    threadpoolDestroy(&pool);
}