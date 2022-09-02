#ifndef LIST_H
#define LIST_H

struct _listEl
{
    struct _listEl *next;
    struct _listEl *prev;
    void *data;
};

typedef struct _list
{
    struct _listEl *_head;
    struct _listEl *_tail;
    long long size;
} List;

int listInit(List *list);
int listDestroy(List *list);
int listPush(void *el, List *list);
int listPop(void **el, List *list);
int listAppend(void *el, List *list);
int listPut(long long pos, void *el, List *list);
int listGet(long long pos, void **el, List *list);
int listRemove(long long pos, void **el, List *list);
int listSize(List list);

int listScan(void **el, void **saveptr, List *list);

void printList(List *list);
int listSort(List *list, int (*heuristic)(void *));

#endif