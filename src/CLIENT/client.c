#include <sys/socket.h>
#include <sys/un.h>
#include <stdlib.h>
#include <errno.h>
#include <stdio.h>
#include <unistd.h>


#include "COMMON/message.h"
#include "COMMON/macros.h"

#define UNIX_PATH_MAX 108
#define SOCKNAME "fakeAddress"
#define N 100


int main()
{
    int sfd;
    struct sockaddr_un sa;
    Message *request;

    for(int i = 0; i < 1000; i++){
        if((sfd = socket(AF_UNIX, SOCK_STREAM, 0)) == -1)
        { perror("Something went wrong while creating socket"); }

        strncpy(sa.sun_path, SOCKNAME, UNIX_PATH_MAX);
        sa.sun_family = AF_UNIX;

        while (connect(sfd,(struct sockaddr*)&sa, sizeof(sa)) == -1 ) {
            if (errno == ENOENT)
            sleep(1); /* sock non esiste */
            else exit(EXIT_FAILURE); 
        }
        UNSAFE_NULL_CHECK(request = malloc(sizeof(Message)));
        messageInit(0, NULL, "Hallo!", 0, 0, request);
        sendMessage(sfd, request);
        messageDestroy(request);
        free(request);
        
        close(sfd);
    }
    exit(EXIT_SUCCESS);
}