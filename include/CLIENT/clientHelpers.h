#ifndef CLIENTHELPERS_H
#define CLIENTHELPERS_H

#include <stdint.h>

#include "COMMON/fileContainer.h"

void __mkdir(char *path);
void __writeBufToDir(void *buf, size_t size, const char *fileName, const char *dirname);
void __writeToDir(FileContainer *fc, size_t size, const char *dirname);

#endif