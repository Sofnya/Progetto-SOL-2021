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

typedef struct _filesystem
{
    List *filesList;
    pthread_mutex_t *filesListMtx;
    HashTable *filesTable;
    AtomicInt *curSize, *curN;
    uint64_t maxSize, maxN;
    int isCompressed;

    pthread_rwlock_t *rwLock;
} FileSystem;

typedef struct _fileDescriptor
{
    const char *name;
    pid_t pid;
    int flags;
} FileDescriptor;

int fsInit(uint64_t maxN, uint64_t maxSize, int isCompressed, FileSystem *fs);
void fsDestroy(FileSystem *fs);

int openFile(char *pathname, int flags, FileDescriptor **fd, FileSystem *fs);
int closeFile(FileDescriptor *fd, FileSystem *fs);

int readFile(FileDescriptor *fd, void **buf, uint64_t size, FileSystem *fs);
int writeFile(FileDescriptor *fd, void *buf, uint64_t size, FileSystem *fs);
int appendToFile(FileDescriptor *fd, void *buf, uint64_t size, FileSystem *fs);

int readNFiles(int N, FileContainer **buf, FileSystem *fs);

int lockFile(FileDescriptor *fd, FileSystem *fs);
int unlockFile(FileDescriptor *fd, FileSystem *fs);
int tryLockFile(FileDescriptor *fd, FileSystem *fs);

int removeFile(FileDescriptor *fd, FileSystem *fs);

uint64_t getSize(const char *pathname, FileSystem *fs);
uint64_t getTrueSize(const char *pathname, FileSystem *fs);

uint64_t getCurSize(FileSystem *fs);
uint64_t getCurN(FileSystem *fs);

int freeSpace(uint64_t size, FileContainer **buf, FileSystem *fs);

#endif