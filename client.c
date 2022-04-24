#include <sys/socket.h>
#include <sys/un.h>
#include <stdlib.h>
#include <errno.h>
#include <stdio.h>
#include <unistd.h>


#define UNIX_PATH_MAX 108
#define SOCKNAME "./myVeryOwnAddress"
#define N 100


int main()
{
    int sfd;
    struct sockaddr_un sa;

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
        write(sfd, "Hallo!", 7);
        close(sfd);
    }
    exit(EXIT_SUCCESS);
}