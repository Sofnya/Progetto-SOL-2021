#include <string.h>
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <pthread.h>

#ifdef DEBUG
#include <assert.h>
#endif

#include "COMMON/hashtable.h"
#include "COMMON/macros.h"

#define SEED 1337

/**
 * @brief Searches for the element of given key in the given _row, and returns it in el if found.
 *
 * @param key the key of the element to get.
 * @param el if not NULL, where the entry will be stored, if found.
 * @param row the row in which to search.
 * @return int 0 on success, -1 if not found.
 */
int _rowGet(const char *key, struct _entry *el, struct _row row)
{
    struct _entry *cur;
    List *list;
    void *curel, *saveptr = NULL;
    PTHREAD_CHECK(pthread_mutex_lock(row.mtx));
    list = row.row;

    if (list->size == 0)
    {
        PTHREAD_CHECK(pthread_mutex_unlock(row.mtx));
        return -1;
    }

    errno = 0;
    // We just scan the list and strcmp every element's key with our given key.
    while (listScan(&curel, &saveptr, list) != -1)
    {
        cur = curel;
        if (!strcmp(cur->key, key))
        {
            if (el != NULL)
                *el = *cur;
            PTHREAD_CHECK(pthread_mutex_unlock(row.mtx));
            return 0;
        }
    }
    // Since listScan always misses the last element we have to treat it separately.
    if (errno == EOF)
    {
        cur = curel;
        if (!strcmp(cur->key, key))
        {
            if (el != NULL)
                *el = *cur;
            PTHREAD_CHECK(pthread_mutex_unlock(row.mtx));
            return 0;
        }
    }

    // If we got here we found nothing.
    PTHREAD_CHECK(pthread_mutex_unlock(row.mtx));
    return -1;
}

/**
 * @brief Inserts given entry in the given row, substituting previously existing entry of same key.
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

    // First we check if an entry with the same key is already present, if so we remove it.
    // We can do this without deadlocking since we initialized the lock as reentrant.
    if (_rowGet(el.key, NULL, row) == 0)
    {
        _rowRemove(el.key, NULL, row);
    }

    // Then just initialize our entry and push it on the row list.
    SAFE_NULL_CHECK(new = malloc(sizeof(struct _entry)));
    new->value = el.value;
    new->key = el.key;

    list = row.row;
    listPush(new, list);
    PTHREAD_CHECK(pthread_mutex_unlock(row.mtx));

    return 0;
}

/**
 * @brief Removes an entry of given key from row. If el is not NULL returns the removed element inside of it.
 *
 * @param key the key of the entry to remove.
 * @param el if not NULL, where the entry will be stored.
 * @param row the _row to modify
 * @return int 0 on success, -1 if not found
 */
int _rowRemove(const char *key, struct _entry *el, struct _row row)
{
    struct _entry *cur;
    bool found = false;
    int pos = 0;
    List *list;
    void *curel, *saveptr = NULL;

    PTHREAD_CHECK(pthread_mutex_lock(row.mtx));
    list = row.row;

    if (list->size == 0)
    {
        PTHREAD_CHECK(pthread_mutex_unlock(row.mtx));
        return -1;
    }

    // We scan the list until we find the element's position.
    errno = 0;
    while (listScan(&curel, &saveptr, list) != -1)
    {
        cur = curel;
        if (!strcmp(cur->key, key))
        {
            found = true;
            break;
        }
        pos++;
    }
    if (errno == EOF)
    {
        cur = curel;
        if (!strcmp(cur->key, key))
        {
            found = true;
        }
    }

    // If we found it we can just call listRemove on the found position to remove it.
    if (found)
    {
        ERROR_CHECK(listRemove(pos, &curel, list));
        cur = curel;
        if (el != NULL)
            *el = *cur;
        free(cur);
    }

    PTHREAD_CHECK(pthread_mutex_unlock(row.mtx));
    if (found)
        return 0;
    return -1;
}

/**
 * @brief Initializes given _row.
 *
 * @param row the row to initialize.
 * @return int 0 on success, -1 and sets errno on error.
 */
int _rowInit(struct _row *row)
{
    pthread_mutexattr_t mtxattr;
    pthread_mutexattr_init(&mtxattr);

    // We use a reentrant mutex so that we can call our own functions while still holding the lock.
    ERROR_CHECK(pthread_mutexattr_settype(&mtxattr, PTHREAD_MUTEX_RECURSIVE));

    SAFE_NULL_CHECK(row->row = malloc(sizeof(List)));
    SAFE_NULL_CHECK(row->mtx = malloc(sizeof(pthread_mutex_t)));

    listInit(row->row);
    ERROR_CHECK(pthread_mutex_init(row->mtx, &mtxattr));

    pthread_mutexattr_destroy(&mtxattr);

    return 0;
}

/**
 * @brief Destroys given row, freeing it's resources.
 *
 * @param row the row to destroy.
 */
void _rowDestroy(struct _row row)
{
    void *cur, *saveptr = NULL;
    struct _entry *curEntry;

    if (row.row->size != 0)
    {
        errno = 0;
        // We free all entry's in the row list before destroying it.
        while (listScan(&cur, &saveptr, row.row) != -1)
        {
            curEntry = cur;
            free(curEntry->key);
            free(curEntry);
        }
        if (errno == EOF)
        {
            curEntry = cur;
            free(curEntry->key);
            free(curEntry);
        }
    }

    listDestroy(row.row);
    free(row.row);
    pthread_mutex_destroy(row.mtx);
    free(row.mtx);
}

/**
 * @brief Pops an element from given row.
 *
 * @param el where the entry will be stored.
 * @param row the row to be modified.
 * @return int 0 on success, -1 on failure.
 */
int _rowPop(struct _entry *el, struct _row row)
{
    struct _entry *curel;
    int tmp;

    PTHREAD_CHECK(pthread_mutex_lock(row.mtx));

    tmp = listPop((void **)&curel, row.row);
    if (tmp == 0)
    {
        *el = *curel;
    }
    PTHREAD_CHECK(pthread_mutex_unlock(row.mtx));

    return tmp;
}

/**
 * @brief Initializes given HashTable, with given size.
 *
 * @param size the size of the HashTable. Should be about 2 times the predicted number of elements for best performance.
 * @param table the HashTable to be initialized.
 * @return int 0 on success, -1 and sets errno on failure.
 */
int hashTableInit(size_t size, HashTable *table)
{
    int i;

    if (size <= 0)
        return -1;

    table->size = size;
    SAFE_NULL_CHECK(table->_table = malloc(sizeof(struct _row) * size));

    for (i = 0; i < size; i++)
    {
        if (_rowInit(table->_table + i) == -1)
            return -1;
    }

    return 0;
}

/**
 * @brief Destroys given HashTable, freeing it's resources.
 *
 * @param table the HashTable to be destroyed.
 */
void hashTableDestroy(HashTable *table)
{
    int i;

    for (i = 0; i < table->size; i++)
    {
        _rowDestroy(table->_table[i]);
    }

    free(table->_table);
}

/**
 * @brief Gets the value corresponding to given key from given HashTable.
 *
 * @param key the key.
 * @param value where the found value will be stored.
 * @param table the HashTable in which to search.
 * @return int 0 on a success, -1 if not found.
 */
int hashTableGet(const char *key, void **value, HashTable table)
{
    size_t loc;
    struct _entry entry;

    // We just check in which row we stored the entry, then call _rowGet on that row.
    loc = _getLoc(key, table.size);

    if (_rowGet(key, &entry, table._table[loc]) == -1)
        return -1;

#ifdef DEBUG
    assert(!strcmp(entry.key, key));
#endif

    *value = entry.value;

    return 0;
}

/**
 * @brief Removes the value corresponding to given key from given HashTable. Returns it in value if value is not NULL.
 *
 * @param key the key.
 * @param value if not NULL, where the value will be stored.
 * @param table the HashTable to modify.
 * @return int 0 on a success, -1 if not found.
 */
int hashTableRemove(const char *key, void **value, HashTable table)
{

    size_t loc;
    struct _entry entry;

    // As in get, we just find the right row before calling _rowRemove on it.
    loc = _getLoc(key, table.size);

    if (_rowRemove(key, &entry, table._table[loc]) == -1)
        return -1;

#ifdef DEBUG
    assert(!strcmp(entry.key, key));
#endif

    if (value != NULL)
        *value = entry.value;

    free(entry.key);

    return 0;
}

/**
 * @brief Inserts given key:value pair inside of given HashTable. If selected key already exists, substitutes it's value.
 *
 * @param key the key.
 * @param value the value.
 * @param table the HashTable to modify.
 * @return int 0 on success, -1 and sets errno otherwise.
 */
int hashTablePut(char *key, void *value, HashTable table)
{
    size_t loc;
    struct _entry entry;

    loc = _getLoc(key, table.size);

    entry.key = malloc(strlen(key) + 1);
    strcpy(entry.key, key);
    entry.value = value;

    return _rowPut(entry, table._table[loc]);
}

/**
 * @brief Pops an element from given HashTable. A copy of it's key will be stored in key and it's value will be stored in value.
 *
 * @param key where the element's key will be stored, if not NULL.
 * @param value where the element's value will be stored.
 * @param table the HashTable to modify.
 * @return int 0 on success, -1 and sets errno otherwise (if table is empty).
 */
int hashTablePop(char **key, void **value, HashTable table)
{
    struct _entry result;
    int i = 0, tmp = -1;

    while (tmp == -1 && i < table.size)
    {
        tmp = _rowPop(&result, table._table[i]);
        i++;
    }

    if (tmp == -1)
    {
        errno = ENOENT;
        return -1;
    }

    if (key != NULL)
    {
        *key = result.key;
    }
    else
    {
        free(result.key);
    }
    *value = result.value;

    return 0;
}

/**
 *
 * @brief Gets the number of elements currently stored in HashTable, roughly.
 *
 * @param table the HashTable to query.
 * @return size_t the number of elements currently stored in table.
 */
long long hashTableSize(HashTable table)
{
    int i;
    long long count;

    // This isn't guarateed to be accurate as it's not thread-safe, but in a non thread-safe situation this will still get a roughly accurate result.
    for (i = 0; i < table.size; i++)
    {
        count += listSize(*table._table[i].row);
    }

    return count;
}

/**
 * @brief Helper function for internal use only. Returns the location in the _table of given key.
 *
 * @param key the key to be hashed.
 * @param size the size of the table.
 * @return size_t the index inside of which the key should be.
 */
size_t _getLoc(const char *key, size_t size)
{
    size_t hash[2];
    MurmurHash3_x64_128(key, strlen(key), SEED, hash);

    return hash[0] % size;
}

/**
 * @brief For debugging, prints given row's contents.
 *
 * @param row the row to print.
 */
void _printRow(struct _row row)
{
    void *saveptr = NULL;
    struct _entry *el;

    errno = 0;
    while (listScan((void **)&el, &saveptr, row.row) != -1)
    {
        printf("|%s : %p|", el->key, el->value);
        printf("->");
    }
    if (errno == EOF)
    {
        printf("|%s : %p|\n", el->key, el->value);
    }
}

/**
 * @brief For debugging, prints given HashTable's contents.
 *
 * @param table the HashTable to print.
 */
void printHashTable(HashTable table)
{
    int i;
    for (i = 0; i < table.size; i++)
    {
        _printRow(table._table[i]);
    }
}
