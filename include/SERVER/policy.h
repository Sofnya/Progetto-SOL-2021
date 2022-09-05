#ifndef POLICY_H
#define POLICY_H

#include "SERVER/filesystem.h"

#define P_RAND 0001
#define P_FIFO 0002
#define P_LIFO 0004
#define P_LRU 0010
#define P_MRU 0020
#define P_LU 0040
#define P_MU 0100
#define P_SMOL 0200
#define P_BIGG 0400
#define P_LFU 01000

int missPolicy(FileDescriptor **fd, FileSystem *fs);

#endif