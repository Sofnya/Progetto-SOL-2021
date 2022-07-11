#ifndef FILES_H
#define FILES_H

#include <pthread.h>
#include <stdint.h>

typedef struct _file
{
    char *name;
    void *content;
    uint64_t size;
    uint64_t compressedSize;
    int isCompressed;
    pthread_mutex_t *mtx;
} File;

int fileInit(const char *name, File *file);
void fileDestroy(File *file);

int fileWrite(const void *content, uint64_t size, File *file);
int fileAppend(const void *content, uint64_t size, File *file);

int fileRead(void *buf, uint64_t bufsize, File *file);

int fileTryLock(File *file);
int fileLock(File *file);
int fileUnlock(File *file);

int fileCompress(File *file);
int fileDecompress(File *file);

uint64_t getFileSize(File *file);
uint64_t getFileTrueSize(File *file);
const char *getFileName(File *file);

#endif