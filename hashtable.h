#ifndef HASHTABLE_H
#define HASHTABLE_H

#include <pthread.h>
#include <stdint.h>


#include "list.h"
#include "murmur3.h"


struct _row {
    List *row;
    pthread_mutex_t *mtx;
};

struct _entry {
    char *key;
    void *value;
};


typedef struct _HashTable{
    uint64_t size;
    struct _row *_table;    
} HashTable;




int _rowGet(const char *key, struct _entry *el, struct _row row);
int _rowPut(struct _entry el, struct _row row);
int _rowRemove(const char *key, struct _entry *el, struct _row row);
int _rowInit(struct _row *row);
void _rowDestroy(struct _row row);


int hashTableInit(uint64_t size, HashTable *table);
void hashTableDestroy(HashTable *table);
int hashTableGet(const char *key, void **value, HashTable table);
int hashTableRemove(const char *key, void **value, HashTable table);
int hashTablePut(const char *key, void *value, HashTable table);

uint64_t _getLoc(const char *key, uint64_t size);

#endif