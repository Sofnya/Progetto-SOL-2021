#ifndef FILEWRAPPER_H
#define FILEWRAPPER_H

#include <stdint.h>


typedef struct _fileContainer {
    uint64_t size;
    void *content;
    char *name;
} FileContainer;


int containerInit(uint64_t size, void *content, char *name, FileContainer *fc);
void destroyContainer(FileContainer *fc);
int serializeContainer(FileContainer fc, void **buf, uint64_t *size);
FileContainer deserializeContainer(void *buf, uint64_t size);

#endif