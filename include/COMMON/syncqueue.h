#ifndef SYNCQUEUE_H
#define SYNCQUEUE_H

#include <pthread.h>
#include <stdbool.h>

struct _slist
{
    struct _slist *next;
    struct _slist *prev;
    void *data;
};

typedef struct _SyncQueue
{
    struct _slist *_head;
    struct _slist *_tail;
    pthread_mutex_t _mtx;
    pthread_cond_t _newEl;
    bool _isOpen;

    long _len;
} SyncQueue;

void syncqueueInit(SyncQueue *queue);
void syncqueueDestroy(SyncQueue *queue);

int syncqueuePush(void *el, SyncQueue *queue);
void *syncqueuePop(SyncQueue *queue);
long syncqueueLen(SyncQueue queue);
void syncqueueClose(SyncQueue *queue);
void syncqueueOpen(SyncQueue *queue);

#endif