#ifndef FILES_H
#define FILES_H


#define O_READ      00000001
#define O_WRITE     00000002
#define O_LOCK      00000020
#define O_APPEND    00000010
#define O_CREATE    00000100
#define O_OPEN      00000200


#include <pthread.h>
#include <stdint.h>

#include "hashtable.h"


typedef struct _file{
    char *name;
    void *content;
    uint64_t size;
    int flags;
    pthread_mutex_t *mtx;
} File;


int fileInit(char *name, File *file);
void fileDestroy(File *file);

int fileWrite(void *content, uint64_t size, File *file);
int fileRead(void *buf, uint64_t bufsize, File *file);

int fileLock(File *file);
int fileUnlock(File *file);

int fileOpen(int flags, File *file);
int fileClose(File *file);

uint64_t getFileSize(File *file);
char *getFileName(File *file);

#endif