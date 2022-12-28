#ifndef CONNSTATE_H
#define CONNSTATE_H

#include <stdint.h>
#include <pthread.h>

#include "COMMON/fileContainer.h"
#include "COMMON/hashtable.h"
#include "COMMON/atomicint.h"
#include "SERVER/filesystem.h"

typedef struct _connState
{
    HashTable *fds;
    FileSystem *fs;
    char uuid[100];

    // We keep track of how many threads are using the ConnState, so that only the last thread will destroy it.
    AtomicInt inUse;
    volatile int shouldDestroy;

    // Counts parsed requests, to ensure that requests are processed in order for any connection.
    // While this is not necessary for our client, that awaits a response before sending any new requests, a form of synchronization is needed to avoid concurrent accesses that could otherwise be caused by an efficient/malicious client.
    AtomicInt requestN;
    AtomicInt parsedN;

} ConnState;

int connStateInit(FileSystem *fs, ConnState *state);
void connStateDestroy(ConnState *state);

int conn_openFile(char *path, int flags, FileContainer **fcs, int *fcsSize, ConnState *state);
int conn_lockFile(char *path, ConnState *state);
int conn_unlockFile(char *path, ConnState *state);
int conn_closeFile(const char *path, ConnState *state);

int conn_readFile(const char *path, void **buf, size_t size, ConnState *state);
int conn_writeFile(const char *path, void *buf, size_t size, FileContainer **fcs, int *fcsSize, ConnState *state);
int conn_appendFile(const char *path, void *buf, size_t size, FileContainer **fcs, int *fcsSize, ConnState *state);

int conn_readNFiles(int N, FileContainer **fcs, ConnState *state);

int conn_removeFile(const char *path, ConnState *state);

#endif