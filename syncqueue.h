#ifndef SYNCQUEUE_H
#define SYNCQUEUE_H

#include <pthread.h>
#include <stdbool.h>



struct _list {
    struct _list *next;
    struct _list *prev;
    void *data;
};

typedef struct _syncQueue {
    struct _list *_head;
    struct _list *_tail;
    pthread_mutex_t _mtx;
    pthread_cond_t _newEl;
    bool _isOpen;
    
    int _len;
} syncQueue;


void syncqueueInit(syncQueue *queue);
void syncqueueDestroy(syncQueue *queue);

void syncqueuePush(void *el, syncQueue *queue);
void *syncqueuePop(syncQueue *queue);
int syncqueueLen(syncQueue queue);
void syncqueueClose(syncQueue *queue);


#endif