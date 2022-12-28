#ifndef FILES_H
#define FILES_H

#include <pthread.h>
#include <stdint.h>
#include <time.h>

typedef struct _metadata
{
    char *name;
    pthread_mutex_t *metadataLock;
    time_t creationTime;
    time_t lastAccess;
    long numberAccesses;
    size_t size;
} Metadata;

typedef struct _file
{
    char *name;
    void *content;
    size_t size;
    size_t compressedSize;
    volatile int isCompressed;
    char *lockedBy;
    pthread_spinlock_t *lock;

    Metadata *metadata;
} File;

int fileInit(const char *name, int isCompressed, File *file);
int fileDestroy(File *file);

int metadataInit(const char *name, Metadata *metadata);
void metadataDestroy(Metadata *metadata);

int metadataAccess(Metadata *metadata);
int metadataUpdateSize(Metadata *metadata, size_t size);

int fileWrite(const void *content, size_t size, File *file);
int fileAppend(const void *content, size_t size, File *file);

int fileRead(void *buf, size_t bufsize, File *file);

int fileIsLockedBy(File *file, char *uuid);
int fileIsLocked(File *file);
int fileLock(File *file, char *uuid);
int fileUnlock(File *file);

int fileCompress(File *file);
int fileDecompress(File *file);

size_t getFileSize(File *file);
size_t getFileTrueSize(File *file);
const char *getFileName(File *file);

#endif