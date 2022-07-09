#ifndef CONNSTATE_H
#define CONNSTATE_H

#include <stdint.h>
#include <uuid/uuid.h>

#include "COMMON/fileContainer.h"
#include "COMMON/hashtable.h"
#include "SERVER/filesystem.h"

typedef struct _connState
{
    HashTable *fds;
    FileSystem *fs;
    char uuid[36];
} ConnState;

int connStateInit(FileSystem *fs, ConnState *state);
void connStateDestroy(ConnState *state);

int conn_openFile(const char *path, int flags, FileContainer **fcs, int *fcsSize, ConnState state);
int conn_closeFile(const char *path, ConnState state);

int conn_readFile(const char *path, void **buf, size_t size, ConnState state);
int conn_writeFile(const char *path, void *buf, size_t size, FileContainer **fcs, int *fcsSize, ConnState state);
int conn_appendFile(const char *path, void *buf, size_t size, FileContainer **fcs, int *fcsSize, ConnState state);

int conn_readNFiles(int N, FileContainer **fcs, ConnState state);

int conn_removeFile(const char *path, ConnState state);

int conn_lockFile(const char *path, ConnState state);
int conn_unlockFile(const char *path, ConnState state);

#endif