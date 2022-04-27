#ifndef FILESYSTEM_H
#define FILESYSTEM_H

#include <stdint.h>


#include "files.h"
#include "hashtable.h"
#include "list.h"


typedef struct _filesystem{
    List *filesList;
    HashTable *filesTable;
    uint64_t curN, maxN;
    uint64_t curSize, maxSize;
} FileSystem;


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