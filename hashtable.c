#include "hashtable.h"
#include <string.h>
#include <errno.h>
#include <stdbool.h>

#ifdef DEBUG
#include <assert.h>
#endif

#include "macros.h"


#define SEED 1337

/**
 * @brief Searches for the element of key key in the given _row, and returns it in el if found.
 * 
 * @param key the key of the element to be found.
 * @param el if not NULL, where the entry will be stored, if found.
 * @param row the row on which to search.
 * @return int 0 on success, -1 if not found.
 */
int _rowGet(char *key, struct _entry *el, struct _row row)
{
    struct _entry *cur;
    List *list;    
    void *curel, *saveptr = NULL;
    PTHREAD_CHECK(pthread_mutex_lock(row.mtx));
    list = row.row;

    if(list->size == 0) 
    {
        PTHREAD_CHECK(pthread_mutex_unlock(row.mtx));
        return -1;
    }

    while(listScan(&curel, &saveptr, list) != -1)
    {
        cur = curel;
        if(!strcmp(cur->key, key))
        {
            if(el != NULL) *el = *cur;
            PTHREAD_CHECK(pthread_mutex_unlock(row.mtx));
            return 0;
        }
    }
    if(errno == EOF){
        cur = curel;
        if(!strcmp(cur->key, key))
        {
            if(el != NULL) *el = *cur;
            PTHREAD_CHECK(pthread_mutex_unlock(row.mtx));
            return 0;
        }
    }

    PTHREAD_CHECK(pthread_mutex_unlock(row.mtx));
    return -1;
    
}

/**
 * @brief Inserts the given entry in the given row, substituting previously existing entry of same key.
 * 
 * @param el the entry to be inserted.
 * @param row the _row to modify
 * @return int 0 on a success, an errorcode and sets errno otherwise.
 */
int _rowPut(struct _entry el, struct _row row)
{
    List *list;
    struct _entry *new;
    
    
    PTHREAD_CHECK(pthread_mutex_lock(row.mtx));
    if(_rowGet(el.key, NULL, row) == 0)
    {
        _rowRemove(el.key, NULL, row);
    }

    SAFE_NULL_CHECK(new = malloc(sizeof(struct _entry)));
    *new = el;

    list = row.row;
    listPush(new, list);
    PTHREAD_CHECK(pthread_mutex_unlock(row.mtx));
    
    return 0;
}

/**
 * @brief Removes an entry of given key from row. If el not NULL returns the removed element inside of it.
 * 
 * @param key the key of the entry to be removed.
 * @param el if not NULL, where the entry will be stored.
 * @param row the _row to modify
 * @return int 0 on success, -1 if not found
 */
int _rowRemove(char *key, struct _entry *el, struct _row row)
{
    struct _entry *cur;
    bool found = false;
    int pos = 0;
    List *list;    
    void *curel, *saveptr = NULL;


    PTHREAD_CHECK(pthread_mutex_lock(row.mtx));
    list = row.row;

    if(list->size == 0) 
    {
        PTHREAD_CHECK(pthread_mutex_unlock(row.mtx));
        return -1;
    }

    while(listScan(&curel, &saveptr, list) != -1)
    {
        cur = curel;
        if(!strcmp(cur->key, key))
        {
            found = true;
            break;
        }
        pos++;
    }
    if(errno == EOF){
        cur = curel;
        if(!strcmp(cur->key, key))
        {
            found = true;
        }
    }

    if(found)
    {
        ERROR_CHECK(listRemove(pos, &curel, list));
        cur = curel;
        if(el != NULL) *el = *cur;
        free(cur);
    }

    PTHREAD_CHECK(pthread_mutex_unlock(row.mtx));
    if(found) return 0;
    return -1;
}

/**
 * @brief Initializes the given _row.
 * 
 * @param row the row to initialize.
 * @return int 0 on success, -1 and sets errno on error.
 */
int _rowInit(struct _row *row)
{
    pthread_mutexattr_t mtxattr;
    pthread_mutexattr_init(&mtxattr);
    ERROR_CHECK(pthread_mutexattr_settype(&mtxattr, PTHREAD_MUTEX_RECURSIVE));

    SAFE_NULL_CHECK(row->row = malloc(sizeof(List)));
    SAFE_NULL_CHECK(row->mtx = malloc(sizeof(pthread_mutex_t)));

    listInit(row->row);
    ERROR_CHECK(pthread_mutex_init(row->mtx, &mtxattr));

    pthread_mutexattr_destroy(&mtxattr);
 
    return 0;
}

/**
 * @brief Destroys the given row, freeing all memory.
 * 
 * @param row the row to destroy.
 */
void _rowDestroy(struct _row row)
{
    void *cur, *saveptr = NULL;

    if(row.row->size != 0)
    {
        errno = 0;
        while(listScan(&cur, &saveptr, row.row) != -1)
        {
            free(cur);
        }
        if(errno == EOF) free(cur);
    }

    listDestroy(row.row);
    free(row.row);
    pthread_mutex_destroy(row.mtx);
    free(row.mtx);
}


/**
 * @brief Initializes the given HashTable, with given size.
 * 
 * @param size the size of the table. Should be about 2 times the predicted number of elements for best performance.
 * @param table the table to be initialized.
 * @return int 0 on success, -1 and sets errno on failure.
 */
int hashTableInit(uint64_t size, HashTable *table)
{
    int i;

    if(size <= 0) return -1;

    table->size = size;
    SAFE_NULL_CHECK(table->_table = malloc(sizeof(struct _row) * size));

    for(i = 0; i < size; i++)
    {
        if(_rowInit(table->_table + i) == -1) return -1;
    }

    return 0;
}

/**
 * @brief Destroys the given HashTable, freeing all memory.
 * 
 * @param table the table to be destroyed.
 */
void hashTableDestroy(HashTable *table)
{
    int i;

    for(i = 0; i < table->size; i++)
    {
        _rowDestroy(table->_table[i]);
    }

    free(table->_table);
}

/**
 * @brief Gets the value corresponding to given key from the HashTable.
 * 
 * @param key the key.
 * @param value where the found value will be stored.
 * @param table the table in which to search.
 * @return int 0 on a success, -1 if not found.
 */
int hashTableGet(char *key, void **value, HashTable table)
{
    uint64_t loc;
    struct _entry entry;


    loc = _getLoc(key, table.size);

    if(_rowGet(key, &entry, table._table[loc]) == -1) return -1;

    #ifdef DEBUG
    assert(!strcmp(entry.key, key));
    #endif

    *value = entry.value;

    return 0;
}

/**
 * @brief Removes the value corresponding to the given key form the table. Returns it in value if value is not NULL.
 * 
 * @param key the key.
 * @param value if not NULL, where the value will be stored.
 * @param table the table to modify.
 * @return int 0 on a success, -1 if not found.
 */
int hashTableRemove(char *key, void **value, HashTable table)
{
   
    uint64_t loc;
    struct _entry entry;

    
    loc = _getLoc(key, table.size);

    if(_rowRemove(key, &entry, table._table[loc]) == -1) return -1;

    #ifdef DEBUG
    assert(!strcmp(entry.key, key));
    #endif

    if(value != NULL) *value = entry.value;

    return 0;
}

/**
 * @brief Inserts the given key:value pair inside of the table. If selected key already exists, substitutes it's value.
 * 
 * @param key the key.
 * @param value the value.
 * @param table the table to be modified.
 * @return int 0 on success, -1 and sets errno otherwise.
 */
int hashTablePut(char *key, void *value, HashTable table)
{
    uint64_t loc;
    struct _entry entry;
    
    loc = _getLoc(key, table.size);

    entry.key = key;
    entry.value = value;

    return _rowPut(entry, table._table[loc]);
}

/**
 * @brief Helper function for internal use only. Returns the location in the _table of given key.
 * 
 * @param key the key to be hashed.
 * @param size the size of the table.
 * @return uint64_t the index inside of which the key should be.
 */
uint64_t _getLoc(char *key, uint64_t size)
{
     uint64_t hash[2];
     MurmurHash3_x64_128(key, strlen(key), SEED, hash);

     return hash[0] % size;
}
