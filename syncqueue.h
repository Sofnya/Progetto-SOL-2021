#ifndef SYNCQUEUE_H
#define SYNCQUEUE_H

#include <pthread.h>
#include <stdbool.h>



struct _list {
    struct _list *next;
    struct _list *prev;
    void *data;
};

typedef struct _SyncQueue {
    struct _list *_head;
    struct _list *_tail;
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


#endif