#include "COMMON/hashtable.h"
#include <assert.h>
#include <stdio.h>


int main(int argc, char const *argv[])
{
    struct _row testRow;

    struct _entry entry1, entry2, entry3, entry4;
    entry1.key = "entry1";
    entry1.value = (void*)1;
    entry2.key = "entry2";
    entry2.value = (void*)2;
    entry3.key = "entry3";
    entry3.value = (void*)3;


    _rowInit(&testRow);

    assert(_rowGet("entry1", &entry4, testRow) == -1);

    _rowPut(entry1, testRow);
    _rowPut(entry2, testRow);
    _rowPut(entry3, testRow);
    
    assert(_rowGet("entry1", &entry4, testRow) == 0);
    assert(entry4.value == (void*)1);

    _rowRemove("entry1", NULL, testRow);
    assert(_rowGet("entry1", &entry4, testRow) == -1);
    
    _rowDestroy(testRow);

    puts("Tests on _row successfull!");

    HashTable table;
    int j;
    hashTableInit(2, &table);

    hashTablePut("entry1", (void *)1, table);
    hashTablePut("entry2", (void *)2, table);
    hashTablePut("entry3", (void *)3, table);
    
    hashTableGet("entry1", (void **)&j, table);
    assert(j == 1);
    hashTableGet("entry2", (void **)&j, table);
    assert(j == 2);
    hashTableGet("entry3",(void **) &j, table);
    assert(j == 3);

    hashTablePut("entry1", (void *)1231, table);
    hashTableGet("entry1",(void **) &j, table);
    assert(j == 1231);
    hashTableRemove("entry1", NULL, table);
    assert(hashTableGet("entry1", (void **)&j, table) == -1);


    hashTableGet("entry2", (void **)&j, table);
    assert(j == 2);
    hashTableGet("entry3", (void **)&j, table);
    assert(j == 3);

    hashTableDestroy(&table);

    
    puts("All tests succesfull!");
    return 0;
}
