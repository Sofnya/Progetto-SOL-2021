#ifndef FILES_H
#define FILES_H

#include <pthread.h>
#include <stdint.h>

typedef struct _file
{
    char *name;
    void *content;
    size_t size;
    size_t compressedSize;
    int isCompressed;
    pthread_mutex_t *mtx;
} File;

int fileInit(const char *name, int isCompressed, File *file);
void fileDestroy(File *file);

int fileWrite(const void *content, size_t size, File *file);
int fileAppend(const void *content, size_t size, File *file);

int fileRead(void *buf, size_t bufsize, File *file);

int fileTryLock(File *file);
int fileLock(File *file);
int fileUnlock(File *file);

int fileCompress(File *file);
int fileDecompress(File *file);

size_t getFileSize(File *file);
size_t getFileTrueSize(File *file);
const char *getFileName(File *file);

#endif