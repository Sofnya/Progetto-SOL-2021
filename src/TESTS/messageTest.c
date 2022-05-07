#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/un.h>
#include <sys/socket.h>
#include <pthread.h>


#include "COMMON/message.h"

#define SOCK_NAME "testSocket"

void cleanup(void);
void *senderClient(void *a);

int sfd;
int main(int argc, char const *argv[])
{   
    pthread_t pid;
    struct sockaddr_un sa;
    Message *m;
    int fdc, reuse=1;
    atexit(&cleanup);

    sfd = socket(AF_UNIX, SOCK_STREAM, 0);
    if(sfd == -1) {perror("Error when creating socket"); exit(0);}

    if(setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) == -1)
    {
        perror("Error when setting sock opt");
        exit(0);
    }

    puts("Socket succesfully created");

    sa.sun_family = AF_UNIX;
    strcpy(sa.sun_path, SOCK_NAME);
    bind(sfd, (struct sockaddr *)&sa, sizeof(sa));

    puts("Socket bound!");

    pthread_create(&pid, NULL, &senderClient, NULL);

    listen(sfd, SOMAXCONN);
    fdc = accept(sfd, NULL, NULL);

    if(fdc == -1) {puts("bad socket...");
    perror("Error");}

    puts("Listening server got connection..");

    m = malloc(sizeof(Message));
    assert(!receiveMessage(fdc, m));

    assert(!memcmp("abcdefghij", m->content, 10));
    assert(!strcmp("This is your info", m->info));
    assert(m->size == 10);
    assert(m->status == 2);
    assert(m->type == 1);
    puts(m->info);
    
    puts("All tests succesfull");
    exit(0);
}

void *senderClient(void *a)
{
    // Sending client.
    struct sockaddr_un sa;
    Message *m;
    int fdc;

    fdc = socket(AF_UNIX, SOCK_STREAM, 0);
    if(fdc == -1){
        perror("Client error on creating socket");
        return 0;
    }

    strcpy(sa.sun_path, SOCK_NAME);
    sa.sun_family = AF_UNIX;

    if(connect(fdc, (struct sockaddr *)&sa, sizeof(sa)) == -1)
    {
        perror("Client error on connecting");
        return 0;
    }

    m = malloc(sizeof(Message));

    messageInit(10, (void *)"abcdefghij", "This is your info", 1, 2, m);

    sendMessage(fdc, m);

    messageDestroy(m);
    free(m);
    return 0;
}

void cleanup(void)
{
    puts("Cleaning up!");
    close(sfd);
    unlink(SOCK_NAME);
    exit(-1);
}