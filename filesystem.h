#ifndef FILESYSTEM_H
#define FILESYSTEM_H

#include <stdint.h>
#include <pthread.h>
#include <sys/types.h>


#include "files.h"
#include "hashtable.h"
#include "list.h"
#include "atomicint.h"


#define O_LOCK      00000001
#define O_CREATE    00000002
#define O_APPEND    00000010



typedef struct _filesystem{
    List *filesList;
    List *openFiles;
    pthread_mutex_t *filesListMtx, *openFilesMtx;
    HashTable *filesTable;
    AtomicInt curSize, curN;
    uint64_t maxSize, maxN;
} FileSystem;


typedef struct _fileDescriptor {
    char *name;
    pid_t pid;
    int flags;
} FileDescriptor;


int fsInit(uint64_t maxN, uint64_t maxSize, FileSystem *fs);
void fsDestroy(FileSystem *fs);

int openFile(const char* pathname, int flags, FileSystem *fs);
int closeFile(const char* pathname, FileSystem *fs);

int readFile(const char* pathname, void** buf, size_t* size, FileSystem *fs);
int writeFile(const char* pathname, FileSystem *fs);
int appendToFile(const char* pathname, void* buf, size_t size, FileSystem *fs);

int lockFile(const char* pathname, FileSystem *fs);
int unlockFile(const char* pathname, FileSystem *fs);
int removeFile(const char* pathname, FileSystem *fs);


#endif