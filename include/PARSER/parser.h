#ifndef PARSER_H
#define PARSER_H

#include <stdint.h>

typedef struct _stats
{
    long long readN, writeN, lockN, unlockN, openN, closeN, removeN;
    long long readSize, writeSize;
    long long nErrors;
    long long maxSize, maxN;
    long long missN, missRemoved;
    long long connN;
    long long maxConn;
    long long spaceSaved;
    long long nRequests, nResponse;
    long long maxThreads, threadsSpawned, threadsKilled;
    long long handlerRequests;
} Stats;

Stats parse(const char *path);
void printStats(Stats stats);

#endif