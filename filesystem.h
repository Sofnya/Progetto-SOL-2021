#ifndef FILESYSTEM_H
#define FILESYSTEM_H

#include <stdint.h>
#include <pthread.h>
#include <sys/types.h>


#include "files.h"
#include "COMMON/hashtable.h"
#include "COMMON/list.h"
#include "COMMON/atomicint.h"


#define O_LOCK      00000001
#define O_CREATE    00000002
#define O_APPEND    00000010

#define FI_READ      00000001
#define FI_WRITE     00000002
#define FI_LOCK      00000010


typedef struct _filesystem{
    List *filesList;
    pthread_mutex_t *filesListMtx;
    HashTable *filesTable;
    AtomicInt *curSize, *curN;
    uint64_t maxSize, maxN;
} FileSystem;


typedef struct _fileDescriptor {
    char *name;
    pid_t pid;
    int flags;
} FileDescriptor;


int fsInit(uint64_t maxN, uint64_t maxSize, FileSystem *fs);
void fsDestroy(FileSystem *fs);

int openFile(char* pathname, int flags, FileDescriptor **fd, FileSystem *fs);
int closeFile(FileDescriptor *fd, FileSystem *fs);

int readFile(FileDescriptor *fd, void** buf, size_t size, FileSystem *fs);
int writeFile(FileDescriptor *fd, void *buf, size_t size, FileSystem *fs);
int appendToFile(FileDescriptor *fd, void* buf, size_t size, FileSystem *fs);

int lockFile(const char* pathname, FileSystem *fs);
int unlockFile(const char* pathname, FileSystem *fs);
int removeFile(FileDescriptor *fd, FileSystem *fs);


#endif