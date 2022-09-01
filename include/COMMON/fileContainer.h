#ifndef FILEWRAPPER_H
#define FILEWRAPPER_H

#include <stdint.h>

typedef struct _fileContainer
{
    size_t size;
    void *content;
    char *name;
} FileContainer;

int containerInit(size_t size, void *content, const char *name, FileContainer *fc);
void destroyContainer(FileContainer *fc);
int serializeContainer(FileContainer fc, void **buf, size_t *size);
FileContainer deserializeContainer(void *buf, size_t size);

int serializeContainerArray(FileContainer *fc, size_t n, size_t *size, void **buf);
FileContainer *deserializeContainerArray(void *buf, size_t size, size_t *n);

size_t calcSize(FileContainer fc);

#endif