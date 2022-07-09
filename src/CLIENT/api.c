#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <stdbool.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>

#include "CLIENT/api.h"
#include "COMMON/message.h"
#include "COMMON/macros.h"
#include "COMMON/fileContainer.h"

#define UNIX_PATH_MAX 108

int sfd;

void __mkdir(char *path)
{
    char *prev, *cur;
    prev = strtok(path, "/");
    while ((cur = strtok(NULL, "/")) != NULL)
    {
        printf("Making dir:%s\n", prev);
        if (mkdir(prev, 0777) && errno != EEXIST)
            perror("error while creating dir");
        prev[strlen(prev)] = '/';
    }
}

void __writeToDir(FileContainer *fc, uint64_t size, const char *dirname)
{
    int i;
    char *path;
    FILE *file;

    if (dirname == NULL)
        return;

    printf("Writing %ld files to dir:%s\n", size, dirname);
    for (i = 0; i < size; i++)
    {
        path = malloc(strlen(fc[i].name) + 10 + strlen(dirname));
        sprintf(path, "%s/%s", dirname, fc[i].name);

        __mkdir(path);

        if ((file = fopen(path, "w+")) != NULL)
        {
            fwrite(fc[i].content, 1, fc[i].size, file);
            fclose(file);
            printf("Wrote %s\n", path);
        }
        else
        {
            perror("Couldn't open local path");
        }

        free(path);
        destroyContainer(&fc[i]);
    }
}

int openConnection(const char *sockname, int msec, const struct timespec abstime)
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
        if (connect(sfd, (struct sockaddr *)&sa, sizeof(struct sockaddr_un)) == 0)
        {
            printf("Connection opened! %d\n", sfd);
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

    SAFE_ERROR_CHECK(messageInit(0, NULL, "Goodbye", MT_DISCONNECT, MS_REQ, &m));
    SAFE_ERROR_CHECK(sendMessage(sfd, &m));
    messageDestroy(&m);

    SAFE_ERROR_CHECK(receiveMessage(sfd, &m));
    success = (m.status == MS_OK || m.status == MS_OKCAP);
    puts(m.info);

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

    SAFE_ERROR_CHECK(messageInit(sizeof(int), &flags, pathname, MT_FOPEN, MS_REQ, &m));
    SAFE_ERROR_CHECK(sendMessage(sfd, &m));

    messageDestroy(&m);
    SAFE_ERROR_CHECK(receiveMessage(sfd, &m));
    success = (m.status == MS_OK);
    puts(m.info);
    messageDestroy(&m);

    if (success)
        return 0;
    return -1;
}

int readFile(const char *pathname, void **buf, size_t *size)
{
    Message m;
    bool success;

    SAFE_ERROR_CHECK(messageInit(0, NULL, pathname, MT_FREAD, MS_REQ, &m));
    SAFE_ERROR_CHECK(sendMessage(sfd, &m));

    messageDestroy(&m);
    SAFE_ERROR_CHECK(receiveMessage(sfd, &m));

    success = (m.status == MS_OK);
    puts(m.info);
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

    SAFE_ERROR_CHECK(messageInit(sizeof(int), &N, info, MT_FREADN, MS_REQ, &m));
    SAFE_ERROR_CHECK(sendMessage(sfd, &m));

    messageDestroy(&m);
    SAFE_ERROR_CHECK(receiveMessage(sfd, &m));

    success = (m.status == MS_OK);
    puts(m.info);

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

    SAFE_NULL_CHECK(fd = fopen(pathname, "r"));
    fseek(fd, 0, SEEK_END);
    size = ftell(fd);
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
    puts(m.info);

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

    SAFE_ERROR_CHECK(messageInit(size, buf, pathname, MT_FAPPEND, MS_REQ, &m));
    SAFE_ERROR_CHECK(sendMessage(sfd, &m));
    messageDestroy(&m);
    SAFE_ERROR_CHECK(receiveMessage(sfd, &m));

    success = (m.status == MS_OK || m.status == MS_OKCAP);
    puts(m.info);

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

    SAFE_ERROR_CHECK(messageInit(0, NULL, pathname, MT_FLOCK, MS_REQ, &m));
    SAFE_ERROR_CHECK(sendMessage(sfd, &m));

    messageDestroy(&m);
    SAFE_ERROR_CHECK(receiveMessage(sfd, &m));
    success = (m.status == MS_OK);
    puts(m.info);
    messageDestroy(&m);

    if (success)
        return 0;
    return -1;
}

int unlockFile(const char *pathname)
{
    Message m;
    bool success;

    SAFE_ERROR_CHECK(messageInit(0, NULL, pathname, MT_FUNLOCK, MS_REQ, &m));
    SAFE_ERROR_CHECK(sendMessage(sfd, &m));

    messageDestroy(&m);
    SAFE_ERROR_CHECK(receiveMessage(sfd, &m));
    success = (m.status == MS_OK);
    puts(m.info);
    messageDestroy(&m);

    if (success)
        return 0;
    return -1;
}

int closeFile(const char *pathname)
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

    if (success)
        return 0;
    return -1;
}

int removeFile(const char *pathname)
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

    flags |= O_CREATE;

    SAFE_ERROR_CHECK(messageInit(sizeof(int), &flags, pathname, MT_FOPEN, MS_REQ, &m));
    SAFE_ERROR_CHECK(sendMessage(sfd, &m));

    messageDestroy(&m);
    SAFE_ERROR_CHECK(receiveMessage(sfd, &m));
    success = (m.status == MS_OK || m.status == MS_OKCAP);
    puts(m.info);

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