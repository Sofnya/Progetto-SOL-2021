#include <stdlib.h>
#include <errno.h>
#include <uuid/uuid.h>

#include "SERVER/connstate.h"
#include "SERVER/filesystem.h"
#include "COMMON/hashtable.h"
#include "COMMON/macros.h"

int connStateInit(FileSystem *fs, ConnState *state)
{
    uuid_t tmp;
    SAFE_NULL_CHECK(state->fds = malloc(sizeof(HashTable)));
    state->fs = fs;
    uuid_generate(tmp);
    uuid_unparse(tmp, state->uuid);

    return hashTableInit(16, state->fds);
}

void connStateDestroy(ConnState *state)
{
    char *key;
    FileDescriptor *fd;

    puts(">>>>>DESTROYING CONNSTATE!!!");
    while (hashTablePop(&key, (void **)&fd, *(state->fds)) != -1)
    {
        if (fd->flags & FI_LOCK)
        {
            unlockFile(fd, state->fs);
        }
        free(key);
        free(fd);
    }

    hashTableDestroy(state->fds);
    free(state->fds);
}

int conn_openFile(char *path, int flags, FileContainer **fcs, int *fcsSize, ConnState state)
{
    FileDescriptor *fd, *tmp;

    if (openFile(path, flags, &fd, state.fs) == -1)
    {
        if (errno != EOVERFLOW)
        {
            return -1;
        }

        *fcsSize = freeSpace(0, fcs, state.fs);
        if (*fcsSize == -1)
        {
            perror("Error on freeSpace");
            return -1;
        }
        if (openFile(path, flags, &fd, state.fs) == -1)
        {
            perror("Double error on open after capacity miss.. BAD");
        }

        hashTablePut((char *)fd->name, fd, *state.fds);

        errno = EOVERFLOW;
        return -1;
    }
    if (hashTableGet(path, (void **)&tmp, *state.fds) != -1)
    {
        free(fd);
        errno = EINVAL;
        return -1;
    }

    return hashTablePut((char *)fd->name, fd, *state.fds);
}

int conn_closeFile(const char *path, ConnState state)
{
    FileDescriptor *fd;

    if (hashTableRemove(path, (void **)&fd, *state.fds) == -1)
    {
        errno = EINVAL;
        return -1;
    }

    return closeFile(fd, state.fs);
}

int conn_readFile(const char *path, void **buf, uint64_t size, ConnState state)
{
    FileDescriptor *fd;

    if (hashTableGet(path, (void **)&fd, *state.fds) == -1)
    {
        errno = EINVAL;
        return -1;
    }

    return readFile(fd, buf, size, state.fs);
}

int conn_writeFile(const char *path, void *buf, uint64_t size, FileContainer **fcs, int *fcsSize, ConnState state)
{
    FileDescriptor *fd;

    if (size > state.fs->maxSize)
    {
        errno = EINVAL;
        return -1;
    }

    if (hashTableGet(path, (void **)&fd, *state.fds) == -1)
    {
        errno = EINVAL;
        return -1;
    }

    if (writeFile(fd, buf, size, state.fs) == -1)
    {
        if (errno != EOVERFLOW)
        {
            return -1;
        }
        *fcsSize = freeSpace(size, fcs, state.fs);
        if (*fcsSize == -1)
        {
            perror("Couldn't free space for capacity miss...");
            return -1;
        }

        if (writeFile(fd, buf, size, state.fs) == -1)
        {
            perror("Double error on write after capacity miss.. BAD");
        }
        errno = EOVERFLOW;
        return -1;
    }
    return 0;
}

int conn_appendFile(const char *path, void *buf, uint64_t size, FileContainer **fcs, int *fcsSize, ConnState state)
{
    FileDescriptor *fd;

    if (size + getSize(path, state.fs) > state.fs->maxSize)
    {
        errno = EINVAL;
        return -1;
    }

    if (hashTableGet(path, (void **)&fd, *state.fds) == -1)
    {
        errno = EINVAL;
        return -1;
    }

    if (appendToFile(fd, buf, size, state.fs) == -1)
    {
        if (errno != EOVERFLOW)
        {
            return -1;
        }

        *fcsSize = freeSpace(size, fcs, state.fs);
        if (*fcsSize == -1)
        {
            perror("Couldn't free space for capacity miss...");
            return -1;
        }

        if (appendToFile(fd, buf, size, state.fs) == -1)
        {
            perror("Double error on append after capacity miss.. BAD");
        }
        errno = EOVERFLOW;
        return -1;
    }
    return 0;
}

int conn_readNFiles(int N, FileContainer **fcs, ConnState state)
{
    int tmp;
    tmp = readNFiles(N, fcs, state.fs);

    return tmp;
}

int conn_removeFile(const char *path, ConnState state)
{
    FileDescriptor *fd;
    int tmp;

    if (hashTableRemove(path, (void **)&fd, *state.fds) == -1)
    {
        errno = EINVAL;
        return -1;
    }

    tmp = removeFile(fd, state.fs);
    if (tmp == -1)
    {
        unlockFile(fd, state.fs);
    }
    free(fd);
    return tmp;
}

int conn_lockFile(const char *path, ConnState state)
{
    FileDescriptor *fd;

    if (hashTableGet(path, (void **)&fd, *state.fds) == -1)
    {
        errno = EINVAL;
        return -1;
    }

    return lockFile(fd, state.fs);
}

int conn_unlockFile(const char *path, ConnState state)
{
    FileDescriptor *fd;

    if (hashTableGet(path, (void **)&fd, *state.fds) == -1)
    {
        errno = EINVAL;
        return -1;
    }

    return unlockFile(fd, state.fs);
}
