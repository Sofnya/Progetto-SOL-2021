#ifndef FILESYSTEM_H
#define FILESYSTEM_H

#include <stdint.h>
#include <pthread.h>
#include <sys/types.h>

#include "files.h"
#include "COMMON/hashtable.h"
#include "COMMON/list.h"
#include "COMMON/atomicint.h"
#include "COMMON/fileContainer.h"
#include "COMMON/syncqueue.h"

#define O_LOCK 00000001
#define O_CREATE 00000002

#define FI_READ 00000001
#define FI_WRITE 00000002
#define FI_APPEND 00000004
#define FI_LOCK 00000010
#define FI_CREATED 00000020

typedef struct _stats
{
    AtomicInt *maxSize;
    AtomicInt *maxN;
    pthread_mutex_t *sizeMtx;

    AtomicInt *filesCreated;
    AtomicInt *capMisses;

    AtomicInt *open, *close;
    AtomicInt *read, *write, *append;
    AtomicInt *readN;
    AtomicInt *lock, *unlock;
    AtomicInt *remove;

} FSStats;
typedef struct _filesystem
{
    List *filesList;
    pthread_mutex_t *filesListMtx;
    HashTable *filesTable;
    AtomicInt *curSize, *curN;
    size_t maxSize, maxN;
    int isCompressed;

    pthread_rwlock_t *rwLock;

    FSStats *fsStats;

    SyncQueue *lockHandlerQueue;
} FileSystem;

typedef struct _fileDescriptor
{
    char *name;
    pid_t pid;
    int flags;
    char *uuid;
} FileDescriptor;

int fsInit(size_t maxN, size_t maxSize, int isCompressed, FileSystem *fs);
void fsDestroy(FileSystem *fs);

int fdInit(const char *name, pid_t pid, int flags, char *uuid, FileDescriptor *fd);
void fdDestroy(FileDescriptor *fd);

int statsInit(FSStats *stats);
void statsDestroy(FSStats *stats);

int statsUpdateSize(FSStats *stats, size_t curSize, size_t curN);

int openFile(char *pathname, int flags, char *uuid, FileDescriptor **fd, FileSystem *fs);
int closeFile(FileDescriptor *fd, FileSystem *fs);

int readFile(FileDescriptor *fd, void **buf, size_t size, FileSystem *fs);
int writeFile(FileDescriptor *fd, void *buf, size_t size, FileSystem *fs);
int appendToFile(FileDescriptor *fd, void *buf, size_t size, FileSystem *fs);

int readNFiles(int N, FileContainer **buf, char *uuid, FileSystem *fs);

int lockFile(char *name, char *uuid, FileSystem *fs);
int unlockFile(char *name, FileSystem *fs);

int isLockedFile(char *name, FileSystem *fs);
int isLockedByFile(char *name, char *uuid, FileSystem *fs);

int removeFile(FileDescriptor *fd, FileSystem *fs);

size_t getSize(const char *pathname, FileSystem *fs);
size_t getTrueSize(const char *pathname, FileSystem *fs);

size_t getCurSize(FileSystem *fs);
size_t getCurN(FileSystem *fs);

int freeSpace(size_t size, FileContainer **buf, FileSystem *fs);

void prettyPrintFiles(FileSystem *fs);
void prettyPrintStats(FSStats *stats);

#endif