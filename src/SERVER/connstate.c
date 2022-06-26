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

    while(hashTablePop(&key, (void **)&fd, *(state->fds)) != -1)
    {
        if(fd->flags & FI_LOCK)
        {
            unlockFile(fd, state->fs);
        }
        free(key);
        free(fd);
    }

    hashTableDestroy(state->fds);
    free(state->fds);
}


int conn_openFile(const char *path, int flags, ConnState state)
{
    FileDescriptor *fd;

    if(openFile(path, flags, &fd, state.fs) == -1) return -1;
    
    if(hashTableGet(path, (void **)&fd, *state.fds) != -1)
    {
        free(fd);
        errno = EINVAL;
        return -1;
    }

    return hashTablePut(fd->name, fd, *state.fds);
}


int conn_closeFile(const char *path, ConnState state)
{
    FileDescriptor *fd;

    if(hashTableRemove(path, (void **)&fd, *state.fds) == -1)
    {
        errno = EINVAL; 
        return -1;
    }

    return closeFile(fd, state.fs);
}


int conn_readFile(const char *path, void** buf, size_t size, ConnState state)
{
    FileDescriptor *fd;
    
    if(hashTableGet(path, (void **)&fd, *state.fds) == -1)
    {
        errno = EINVAL;
        return -1;
    }

    return readFile(fd, buf, size, state.fs);
}


int conn_writeFile(const char *path, void *buf, size_t size, ConnState state)
{
    FileDescriptor *fd;

    if(hashTableGet(path, (void **)&fd, *state.fds) == -1)
    {
        errno = EINVAL;
        return -1;
    }

    return writeFile(fd, buf, size, state.fs);
}


int conn_appendFile(const char *path, void *buf, size_t size, ConnState state)
{
    FileDescriptor *fd;

    if(hashTableGet(path, (void **)&fd, *state.fds) == -1)
    {
        errno = EINVAL;
        return -1;
    }

    return appendToFile(fd, buf, size, state.fs);
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

    if(hashTableRemove(path, (void **)&fd, *state.fds) == -1)
    {
        errno = EINVAL;
        return -1;
    }

    tmp = removeFile(fd, state.fs);
    if(tmp == -1)
    {
        unlockFile(fd, state.fs);
    }
    free(fd);
    return tmp;
}


int conn_lockFile(const char *path, ConnState state)
{
    FileDescriptor *fd;

    if(hashTableGet(path, (void **)&fd, *state.fds) == -1)
    {
        errno = EINVAL;
        return -1;
    }

    return lockFile(fd, state.fs);
}


int conn_unlockFile(const char *path, ConnState state)
{
    FileDescriptor *fd;

    if(hashTableGet(path, (void **)&fd, *state.fds) == -1)
    {
        errno = EINVAL;
        return -1;
    }

    return unlockFile(fd, state.fs);
}
