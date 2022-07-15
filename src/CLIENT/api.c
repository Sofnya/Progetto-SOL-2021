#include <sys/socket.h>
#include <sys/un.h>

#include <time.h>
#include <stdbool.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>

#include "CLIENT/api.h"
#include "COMMON/message.h"
#include "COMMON/macros.h"
#include "COMMON/fileContainer.h"
#include "CLIENT/clientHelpers.h"

#define UNIX_PATH_MAX 108

int sfd;
int verbose = 0;

int openConnection(const char *sockname, int msec, const struct timespec abstime)
{
    struct sockaddr_un sa;
    struct timespec curTime;

    if (verbose)
    {
        printf("Opening connection to %s\n", sockname);
    }

    SAFE_ERROR_CHECK(sfd = socket(AF_UNIX, SOCK_STREAM, 0));

    strncpy(sa.sun_path, sockname, UNIX_PATH_MAX);
    sa.sun_path[UNIX_PATH_MAX] = '\00';
    sa.sun_family = AF_UNIX;

    timespec_get(&curTime, TIME_UTC);
    while (curTime.tv_sec < abstime.tv_sec)
    {
        if (connect(sfd, (struct sockaddr *)&sa, sizeof(struct sockaddr_un)) == 0)
        {
            if (verbose)
            {
                printf("Connection opened! %d\n", sfd);
            }
            return 0;
        }
        usleep(msec * 1000);

        timespec_get(&curTime, TIME_UTC);
    }

    errno = ECOMM;
    return -1;
}

int closeConnection(const char *sockname)
{
    Message m;
    bool success;

    if (verbose)
    {
        printf("Closing connection to %s\n", sockname);
    }

    SAFE_ERROR_CHECK(messageInit(0, NULL, "Goodbye", MT_DISCONNECT, MS_REQ, &m));
    SAFE_ERROR_CHECK(sendMessage(sfd, &m));
    messageDestroy(&m);

    SAFE_ERROR_CHECK(receiveMessage(sfd, &m));
    success = (m.status == MS_OK || m.status == MS_OKCAP);
    if (verbose)
    {
        puts(m.info);
    }
    if (m.status == MS_OKCAP)
    {
    }
    messageDestroy(&m);

    close(sfd);
    if (success)
        return 0;
    return -1;
}

int openFile(const char *pathname, int flags)
{
    Message m;
    bool success;

    if (verbose)
    {
        printf("Opening file %s\n", pathname);
    }
    SAFE_ERROR_CHECK(messageInit(sizeof(int), &flags, pathname, MT_FOPEN, MS_REQ, &m));
    SAFE_ERROR_CHECK(sendMessage(sfd, &m));

    messageDestroy(&m);
    SAFE_ERROR_CHECK(receiveMessage(sfd, &m));
    success = (m.status == MS_OK);
    if (verbose)
    {
        puts(m.info);
    }
    messageDestroy(&m);

    if (success)
        return 0;
    return -1;
}

int readFile(const char *pathname, void **buf, size_t *size)
{
    Message m;
    bool success;

    if (verbose)
    {
        printf("Reading file %s\n", pathname);
    }

    SAFE_ERROR_CHECK(messageInit(0, NULL, pathname, MT_FREAD, MS_REQ, &m));
    SAFE_ERROR_CHECK(sendMessage(sfd, &m));

    messageDestroy(&m);
    SAFE_ERROR_CHECK(receiveMessage(sfd, &m));

    success = (m.status == MS_OK);
    if (verbose)
    {
        puts(m.info);
    }
    if (success)
    {
        *size = m.size;
        SAFE_NULL_CHECK(*buf = malloc(m.size));
        memcpy(*buf, m.content, m.size);
    }
    messageDestroy(&m);

    if (success)
        return 0;
    return -1;
}

int readNFiles(int N, const char *dirname)
{
    Message m;
    bool success;
    FileContainer *fc;
    uint64_t amount;
    char info[100];

    sprintf(info, "Requesting %d files", N);

    if (verbose)
    {
        puts(info);
    }

    SAFE_ERROR_CHECK(messageInit(sizeof(int), &N, info, MT_FREADN, MS_REQ, &m));
    SAFE_ERROR_CHECK(sendMessage(sfd, &m));

    messageDestroy(&m);
    SAFE_ERROR_CHECK(receiveMessage(sfd, &m));

    success = (m.status == MS_OK);
    if (verbose)
    {
        puts(m.info);
    }

    if (!success)
    {
        messageDestroy(&m);
        return -1;
    }

    fc = deserializeContainerArray(m.content, m.size, &amount);

    __writeToDir(fc, amount, dirname);

    free(fc);
    messageDestroy(&m);

    return 0;
}

int writeFile(const char *pathname, const char *dirname)
{
    FILE *fd;
    long size;
    void *buffer;
    FileContainer *fc;
    uint64_t amount;
    Message m;
    bool success;

    if (verbose)
    {
        printf("Writing file %s\n", pathname);
    }

    SAFE_NULL_CHECK(fd = fopen(pathname, "r"));
    SAFE_ERROR_CHECK(fseek(fd, 0, SEEK_END));
    SAFE_ERROR_CHECK(size = ftell(fd));
    rewind(fd);

    SAFE_NULL_CHECK(buffer = malloc(size));

    if (fread(buffer, 1, size, fd) != size)
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
    if (verbose)
    {
        puts(m.info);
    }

    if ((m.status == MS_OKCAP) && (dirname != NULL))
    {
        fc = deserializeContainerArray(m.content, m.size, &amount);

        __writeToDir(fc, amount, dirname);
        free(fc);
    }
    messageDestroy(&m);

    if (success)
        return 0;
    return -1;
}

int appendToFile(const char *pathname, void *buf, size_t size, const char *dirname)
{
    Message m;
    bool success;
    FileContainer *fc;
    uint64_t amount;

    if (verbose)
    {
        printf("Appending to file %s\n", pathname);
    }

    SAFE_ERROR_CHECK(messageInit(size, buf, pathname, MT_FAPPEND, MS_REQ, &m));
    SAFE_ERROR_CHECK(sendMessage(sfd, &m));
    messageDestroy(&m);
    SAFE_ERROR_CHECK(receiveMessage(sfd, &m));

    success = (m.status == MS_OK || m.status == MS_OKCAP);
    if (verbose)
    {
        puts(m.info);
    }

    if ((m.status == MS_OKCAP) && (dirname != NULL))
    {
        fc = deserializeContainerArray(m.content, m.size, &amount);

        __writeToDir(fc, amount, dirname);
        free(fc);
    }
    messageDestroy(&m);

    if (success)
        return 0;
    return -1;
}

int lockFile(const char *pathname)
{
    Message m;
    bool success;

    if (verbose)
    {
        printf("Locking file %s\n", pathname);
    }

    SAFE_ERROR_CHECK(messageInit(0, NULL, pathname, MT_FLOCK, MS_REQ, &m));
    SAFE_ERROR_CHECK(sendMessage(sfd, &m));

    messageDestroy(&m);
    SAFE_ERROR_CHECK(receiveMessage(sfd, &m));
    success = (m.status == MS_OK);
    if (verbose)
    {
        puts(m.info);
    }
    messageDestroy(&m);

    if (success)
        return 0;
    return -1;
}

int unlockFile(const char *pathname)
{
    Message m;
    bool success;

    if (verbose)
    {
        printf("Unlocking file %s\n", pathname);
    }

    SAFE_ERROR_CHECK(messageInit(0, NULL, pathname, MT_FUNLOCK, MS_REQ, &m));
    SAFE_ERROR_CHECK(sendMessage(sfd, &m));

    messageDestroy(&m);
    SAFE_ERROR_CHECK(receiveMessage(sfd, &m));
    success = (m.status == MS_OK);
    if (verbose)
    {
        puts(m.info);
    }
    messageDestroy(&m);

    if (success)
        return 0;
    return -1;
}

int closeFile(const char *pathname)
{
    Message m;
    bool success;

    if (verbose)
    {
        printf("Closing file %s\n", pathname);
    }

    SAFE_ERROR_CHECK(messageInit(0, NULL, pathname, MT_FCLOSE, MS_REQ, &m));
    SAFE_ERROR_CHECK(sendMessage(sfd, &m));

    messageDestroy(&m);
    SAFE_ERROR_CHECK(receiveMessage(sfd, &m));
    success = (m.status == MS_OK);
    if (verbose)
    {
        puts(m.info);
    }
    messageDestroy(&m);

    if (success)
        return 0;
    return -1;
}

int removeFile(const char *pathname)
{
    Message m;
    bool success;

    if (verbose)
    {
        printf("Removing file %s\n", pathname);
    }

    SAFE_ERROR_CHECK(messageInit(0, NULL, pathname, MT_FREM, MS_REQ, &m));
    SAFE_ERROR_CHECK(sendMessage(sfd, &m));

    messageDestroy(&m);
    SAFE_ERROR_CHECK(receiveMessage(sfd, &m));
    success = (m.status == MS_OK);
    if (verbose)
    {
        puts(m.info);
    }
    messageDestroy(&m);

    if (success)
        return 0;
    return -1;
}

int create_file(const char *pathname, int flags, const char *dirname)
{
    Message m;
    bool success;
    FileContainer *fc;
    uint64_t amount;

    if (verbose)
    {
        printf("Creating file %s\n", pathname);
    }

    flags |= O_CREATE;

    SAFE_ERROR_CHECK(messageInit(sizeof(int), &flags, pathname, MT_FOPEN, MS_REQ, &m));
    SAFE_ERROR_CHECK(sendMessage(sfd, &m));

    messageDestroy(&m);
    SAFE_ERROR_CHECK(receiveMessage(sfd, &m));
    success = (m.status != MS_ERR);
    if (verbose)
    {
        puts(m.info);
    }

    if ((m.status == MS_OKCAP) && (dirname != NULL))
    {
        fc = deserializeContainerArray(m.content, m.size, &amount);

        __writeToDir(fc, amount, dirname);
        free(fc);
    }
    messageDestroy(&m);

    if (success)
        return 0;
    return -1;
}