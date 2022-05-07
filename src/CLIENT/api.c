#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <stdbool.h>
#include <unistd.h>
#include <stdio.h>


#include "CLIENT/api.h"
#include "COMMON/message.h"
#include "COMMON/macros.h"


#define UNIX_PATH_MAX 108


int sfd;


int openConnection(const char* sockname, int msec, const struct timespec abstime)
{
    struct sockaddr_un sa;
    struct timespec curTime;

    SAFE_ERROR_CHECK(sfd = socket(AF_UNIX, SOCK_STREAM, 0));

    strncpy(sa.sun_path, sockname, UNIX_PATH_MAX);
    sa.sun_path[UNIX_PATH_MAX] = '\00';
    sa.sun_family = AF_UNIX;

    timespec_get(&curTime, TIME_UTC);
    while (curTime.tv_sec < abstime.tv_sec)
    {
        if(connect(sfd, (struct sockaddr*)&sa, sizeof(struct sockaddr_un)) == 0)
        {
            puts("Connection opened!");
            return 0;
        }
        usleep(msec * 1000);

        timespec_get(&curTime, TIME_UTC);
    }

    errno = ECOMM;
    return -1;
}


int closeConnection(const char* sockname)
{
    Message m;
    bool success;
    
    SAFE_ERROR_CHECK(messageInit(0, NULL, NULL, MT_DISCONNECT, MS_REQ, &m));
    SAFE_ERROR_CHECK(sendMessage(sfd, &m));
    messageDestroy(&m);

    SAFE_ERROR_CHECK(receiveMessage(sfd, &m));
    success = (m.status == MS_OK);
    puts(m.info);
    messageDestroy(&m);


    close(sfd);
    if(success) return 0;
    return -1;
}


int openFile(const char* pathname, int flags)
{
    Message m;
    bool success;

    SAFE_ERROR_CHECK(messageInit(sizeof(int), &flags, pathname, MT_FOPEN, MS_REQ, &m));
    SAFE_ERROR_CHECK(sendMessage(sfd, &m));

    messageDestroy(&m);
    SAFE_ERROR_CHECK(receiveMessage(sfd, &m));
    success = (m.status == MS_OK);
    puts(m.info);
    messageDestroy(&m);

    if(success) return 0;
    return -1;
}


int readFile(const char* pathname, void** buf, size_t* size)
{
    Message m;
    bool success;

    SAFE_ERROR_CHECK(messageInit(0 , NULL, pathname, MT_FREAD, MS_REQ, &m));
    SAFE_ERROR_CHECK(sendMessage(sfd, &m));

    messageDestroy(&m);
    SAFE_ERROR_CHECK(receiveMessage(sfd, &m));

    success = (m.status == MS_OK);
    puts(m.info);
    if(success)
    {
        *size = m.size;
        SAFE_NULL_CHECK(*buf = malloc(m.size));
        memcpy(*buf, m.content, m.size);
    }
    messageDestroy(&m);

    if(success) return 0;
    return -1;
}

//TODO
int readNFiles(int N, const char* dirname){return -1;}


int writeFile(const char* pathname, const char* dirname)
{
    FILE *fd;
    long size;
    void *buffer;
    Message m;
    bool success;


    SAFE_NULL_CHECK(fd = fopen(pathname, "r"));
    fseek(fd , 0 , SEEK_END);
    size = ftell(fd);
    rewind(fd);

    SAFE_NULL_CHECK(buffer = malloc(size));
    
    if(fread(buffer, 1, size, fd) != size)
    {
        errno = EIO;
        return -1;
    }
    fclose(fd);
    SAFE_ERROR_CHECK(messageInit(size, buffer, pathname, MT_FWRITE, MS_REQ, &m));
    SAFE_ERROR_CHECK(sendMessage(sfd, &m));
    messageDestroy(&m);
    SAFE_ERROR_CHECK(receiveMessage(sfd, &m));

    success = (m.status == MS_OK || m.status == MS_OKCAP);
    puts(m.info);

    if((m.status == MS_OKCAP) && (dirname != NULL))
    {
        char *pathname;
        size_t dirlen, pathlen;
        dirlen = strlen(dirname);
        pathlen = strlen(m.info);

        SAFE_NULL_CHECK(pathname = malloc(dirlen + pathlen + 1));
        strncpy(pathname, dirname, dirlen);
        strncpy(pathname + dirlen, m.info, pathlen + 1);

        SAFE_NULL_CHECK(fd = fopen(pathname, "w+"));
        free(pathname);

        if(fwrite(m.content, 1, m.size, fd) != m.size)
        {
            errno = EIO;
            return -1;
        }

    }
    messageDestroy(&m);

    if(success) return 0;
    return -1;
}


int appendToFile(const char* pathname, void* buf, size_t size, const char* dirname)
{
    FILE *fd;
    Message m;
    bool success;


    
    SAFE_ERROR_CHECK(messageInit(size, buf, pathname, MT_FAPPEND, MS_REQ, &m));
    SAFE_ERROR_CHECK(sendMessage(sfd, &m));
    messageDestroy(&m);
    SAFE_ERROR_CHECK(receiveMessage(sfd, &m));

    success = (m.status == MS_OK || m.status == MS_OKCAP);
    puts(m.info);

    if((m.status == MS_OKCAP) && (dirname != NULL))
    {
        char *pathname;
        size_t dirlen, pathlen;
        dirlen = strlen(dirname);
        pathlen = strlen(m.info);

        SAFE_NULL_CHECK(pathname = malloc(dirlen + pathlen + 1));
        strncpy(pathname, dirname, dirlen);
        strncpy(pathname + dirlen, m.info, pathlen + 1);

        SAFE_NULL_CHECK(fd = fopen(pathname, "w+"));
        free(pathname);

        if(fwrite(m.content, 1, m.size, fd) != m.size)
        {
            errno = EIO;
            return -1;
        }

    }
    messageDestroy(&m);

    if(success) return 0;
    return -1;
}

//TODO
int lockFile(const char* pathname){return -1;}
int unlockFile(const char* pathname){return -1;}


int closeFile(const char* pathname)
{
    Message m;
    bool success;

    SAFE_ERROR_CHECK(messageInit(0, NULL, pathname, MT_FCLOSE, MS_REQ, &m));
    SAFE_ERROR_CHECK(sendMessage(sfd, &m));

    messageDestroy(&m);
    SAFE_ERROR_CHECK(receiveMessage(sfd, &m));
    success = (m.status == MS_OK);
    puts(m.info);
    messageDestroy(&m);

    if(success) return 0;
    return -1;
}


int removeFile(const char* pathname)
{
    Message m;
    bool success;

    SAFE_ERROR_CHECK(messageInit(0, NULL, pathname, MT_FREM, MS_REQ, &m));
    SAFE_ERROR_CHECK(sendMessage(sfd, &m));

    messageDestroy(&m);
    SAFE_ERROR_CHECK(receiveMessage(sfd, &m));
    success = (m.status == MS_OK);
    puts(m.info);
    messageDestroy(&m);

    if(success) return 0;
    return -1;
}

