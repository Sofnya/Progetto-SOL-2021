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

#define O_LOCK 00000001
#define O_CREATE 00000002

#define FI_READ 00000001
#define FI_WRITE 00000002
#define FI_APPEND 00000004
#define FI_LOCK 00000010

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
} FileSystem;

typedef struct _fileDescriptor
{
    char *name;
    pid_t pid;
    int flags;
} FileDescriptor;

int fsInit(size_t maxN, size_t maxSize, int isCompressed, FileSystem *fs);
void fsDestroy(FileSystem *fs);

int fdInit(const char *name, pid_t pid, int flags, FileDescriptor *fd);
void fdDestroy(FileDescriptor *fd);

int statsInit(FSStats *stats);
void statsDestroy(FSStats *stats);

int statsUpdateSize(FSStats *stats, size_t curSize, size_t curN);

int openFile(char *pathname, int flags, FileDescriptor **fd, FileSystem *fs);
int closeFile(FileDescriptor *fd, FileSystem *fs);

int readFile(FileDescriptor *fd, void **buf, size_t size, FileSystem *fs);
int writeFile(FileDescriptor *fd, void *buf, size_t size, FileSystem *fs);
int appendToFile(FileDescriptor *fd, void *buf, size_t size, FileSystem *fs);

int readNFiles(int N, FileContainer **buf, FileSystem *fs);

int lockFile(FileDescriptor *fd, FileSystem *fs);
int unlockFile(FileDescriptor *fd, FileSystem *fs);
int tryLockFile(FileDescriptor *fd, FileSystem *fs);

int removeFile(FileDescriptor *fd, FileSystem *fs);

size_t getSize(const char *pathname, FileSystem *fs);
size_t getTrueSize(const char *pathname, FileSystem *fs);

size_t getCurSize(FileSystem *fs);
size_t getCurN(FileSystem *fs);

int freeSpace(size_t size, FileContainer **buf, FileSystem *fs);

void prettyPrintFiles(FileSystem *fs);
void prettyPrintStats(FSStats *stats);

#endif