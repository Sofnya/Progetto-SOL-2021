#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <sys/un.h>
#include <sys/socket.h>
#include <pthread.h>
#include <string.h>

#include "COMMON/message.h"
#include "SERVER/threadpool.h"
#include "COMMON/syncqueue.h"
#include "COMMON/macros.h"
#include "COMMON/list.h"
#include "COMMON/macros.h"
#include "COMMON/hashtable.h"
#include "COMMON/fileContainer.h"
#include "SERVER/filesystem.h"
#include "SERVER/files.h"

#define SOCK_NAME "testSocket"

void test(void *par)
{
    int p = *(int *)par;
    free(par);
    assert(p == 10);
    usleep(9831);
    return;
}

long testHeuristic(void *arg)
{
    return (int)arg;
}

int threadpoolTest()
{
    ThreadPool pool;
    int i;
    threadpoolInit(1, 300, &pool);

    for (i = 0; i < 1000; i++)
    {
        int *j = malloc(sizeof(int));
        *j = 10;
        threadpoolSubmit(&test, j, &pool);
        usleep(1000);
    }

    threadpoolCleanExit(&pool);
    threadpoolDestroy(&pool);

    return 0;
}

int syncqueueTest()
{
    SyncQueue *queue;
    int i, j = 99999;

    UNSAFE_NULL_CHECK(queue = malloc(sizeof(SyncQueue)));
    syncqueueInit(queue);
    for (i = 0; i < 100000; i++)
    {
        syncqueuePush(&i, queue);
        assert(*(int *)syncqueuePop(queue) == i);
    }
    for (i = 0; i < 999; i++)
    {
        syncqueuePush(&j, queue);
    }
    for (i = 0; i < 900; i++)
    {
        assert(j == *(int *)syncqueuePop(queue));
    }

    syncqueueDestroy(queue);
    free(queue);
    return 0;
}

void cleanup(void);
void *senderClient(void *a);

int sfd;
int messageTest()
{
    pthread_t pid;
    struct sockaddr_un sa;
    Message *m;
    int fdc, reuse = 1;
    atexit(&cleanup);

    sfd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sfd == -1)
    {
        perror("Error when creating socket");
        exit(0);
    }

    if (setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) == -1)
    {
        perror("Error when setting sock opt");
        exit(0);
    }

    sa.sun_family = AF_UNIX;
    strcpy(sa.sun_path, SOCK_NAME);
    bind(sfd, (struct sockaddr *)&sa, sizeof(sa));

    pthread_create(&pid, NULL, &senderClient, NULL);

    listen(sfd, SOMAXCONN);
    fdc = accept(sfd, NULL, NULL);

    if (fdc == -1)
    {
        perror("Error");
        return -1;
    }

    m = malloc(sizeof(Message));
    assert(!receiveMessage(fdc, m));

    // puts(m->info);
    assert(!memcmp("abcdefghij", m->content, 10));
    assert(!strcmp("This is your info", m->info));
    assert(m->size == 10);
    assert(m->status == 2);
    assert(m->type == 1);

    pthread_join(pid, NULL);
    messageDestroy(m);
    free(m);
    cleanup();
    return 0;
}

void *senderClient(void *a)
{
    // Sending client.
    struct sockaddr_un sa;
    Message *m;
    int fdc;

    fdc = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fdc == -1)
    {
        perror("Client error on creating socket");
        return 0;
    }

    strcpy(sa.sun_path, SOCK_NAME);
    sa.sun_family = AF_UNIX;

    if (connect(fdc, (struct sockaddr *)&sa, sizeof(sa)) == -1)
    {
        perror("Client error on connecting");
        return 0;
    }

    m = malloc(sizeof(Message));

    messageInit(10, (void *)"abcdefghij", "This is your info", 1, 2, m);

    sendMessage(fdc, m);

    messageDestroy(m);
    free(m);
    return 0;
}

void cleanup(void)
{
    close(sfd);
    unlink(SOCK_NAME);
}

int listTest()
{
    List list, list2;
    void *saveptr;
    int i;
    int *j;

    listInit(&list);
    listInit(&list2);

    for (i = 0; i <= 9999; i++)
    {
        UNSAFE_NULL_CHECK(j = malloc(sizeof(int)));
        *j = i;
        // printf("Inserting %d...\n", *j);
        ERROR_CHECK(listAppend((void *)j, &list));
    }

    UNSAFE_NULL_CHECK(j = malloc(sizeof(int)));
    *j = 1337;
    listPut(0, (void *)j, &list);

    for (i = 0; i <= 9999; i++)
    {
        ERROR_CHECK(listGet(i + 1, (void **)&j, &list));
        assert(i == *j);
    }

    saveptr = NULL;
    while (listScan((void **)&j, &saveptr, &list) != -1)
    {
        free(j);
    }
    free(j);

    listDestroy(&list);

    for (i = 0; i <= 9999; i++)
    {
        UNSAFE_NULL_CHECK(j = malloc(sizeof(int)));
        *j = i;
        ERROR_CHECK(listPush(j, &list2));
    }

    for (i = 9999; i >= 0; i--)
    {
        ERROR_CHECK(listPop((void **)&j, &list2));
        assert(*j == i);
        free(j);
    }

    listDestroy(&list2);

    puts("Sorting test!");
    int size = rand() % 100;
    listInit(&list);
    for (i = 0; i < size; i++)
    {
        listPush((void *)(rand() % 100), &list);
    }
    printList(&list);
    puts("\nNow sorting it!\n");
    listSort(&list, &testHeuristic);
    printList(&list);
    puts("\nAll done!");

    return 0;
}

int hashtableTest()
{
    struct _row testRow;

    struct _entry entry1, entry2, entry3, entry4;
    entry1.key = "entry1";
    entry1.value = (void *)1;
    entry2.key = "entry2";
    entry2.value = (void *)2;
    entry3.key = "entry3";
    entry3.value = (void *)3;

    _rowInit(&testRow);

    assert(_rowGet("entry1", &entry4, testRow) == -1);

    _rowPut(entry1, testRow);
    _rowPut(entry2, testRow);
    _rowPut(entry3, testRow);

    assert(_rowGet("entry1", &entry4, testRow) == 0);
    assert(entry4.value == (void *)1);

    _rowRemove("entry1", NULL, testRow);
    assert(_rowGet("entry1", &entry4, testRow) == -1);

    _rowDestroy(testRow);

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
    hashTableGet("entry3", (void **)&j, table);
    assert(j == 3);

    hashTablePut("entry1", (void *)1231, table);
    hashTableGet("entry1", (void **)&j, table);
    assert(j == 1231);
    hashTableRemove("entry1", NULL, table);
    assert(hashTableGet("entry1", (void **)&j, table) == -1);

    hashTableGet("entry2", (void **)&j, table);
    assert(j == 2);
    hashTableGet("entry3", (void **)&j, table);
    assert(j == 3);

    hashTableDestroy(&table);

    return 0;
}

int filesystemTest()
{
    FileSystem *fs;
    FileDescriptor *fd;
    char *contents = "abcdefghijklmnopqrstuvwxyz", *tmp;

    UNSAFE_NULL_CHECK(fs = malloc(sizeof(FileSystem)));

    ERROR_CHECK(fsInit(100, 2048, 1, fs));

    ERROR_CHECK(openFile("file1", O_CREATE, &fd, fs));
    ERROR_CHECK(lockFile(fd, fs));
    ERROR_CHECK(writeFile(fd, (void *)contents, 5, fs));
    ERROR_CHECK(appendToFile(fd, (void *)contents, 5, fs));

    ERROR_CHECK(unlockFile(fd, fs));
    UNSAFE_NULL_CHECK(tmp = malloc(11));
    tmp[10] = '\00';

    ERROR_CHECK(readFile(fd, (void **)&tmp, 10, fs));
    assert(!strcmp(tmp, "abcdeabcde"));

    ERROR_CHECK(closeFile(fd, fs));
    free(tmp);
    fsDestroy(fs);
    free(fs);

    return 0;
}

int filesTest()
{
    File *file1;
    char *alphabet = "abcdefghijklmnopqrstuvwxyz", *tmp;
    void *content;

    UNSAFE_NULL_CHECK(file1 = malloc(sizeof(File)));
    ERROR_CHECK(fileInit("file1", 0, file1));

    content = alphabet;

    ERROR_CHECK(fileLock(file1));
    ERROR_CHECK(fileWrite(content, 5, file1));
    ERROR_CHECK(fileUnlock(file1));
    ERROR_CHECK(fileAppend(content, 5, file1));

    ERROR_CHECK(fileCompress(file1));

    assert(!strcmp(getFileName(file1), "file1"));

    ERROR_CHECK(fileDecompress(file1));
    assert(getFileSize(file1) == 10);

    tmp = malloc(11);
    tmp[10] = '\00';

    ERROR_CHECK(fileRead(tmp, 11, file1));
    assert(!strcmp("abcdeabcde", tmp));

    free(tmp);
    fileLock(file1);
    fileDestroy(file1);
    free(file1);

    return 0;
}

int fileContainerTest()
{
    FileContainer fc, fc2, array[2], *result;
    void *buf;
    void *content = "abcdefghijk";
    char *name = "fileName.txt", *name2 = "anotherFile.txt";
    size_t size, n;

    assert(containerInit(5, content, name, &fc) == 0);

    assert(strcmp(fc.name, name) == 0);
    assert(strncmp(fc.content, content, 5) == 0);
    assert(fc.size == 5);

    assert(serializeContainer(fc, &buf, &size) == 0);

    destroyContainer(&fc);

    fc2 = deserializeContainer(buf, size);
    assert(strcmp(fc2.name, name) == 0);
    assert(strncmp(fc2.content, content, 5) == 0);
    assert(fc2.size == 5);
    destroyContainer(&fc2);
    free(buf);
    buf = NULL;

    puts("Now testing arrays...");
    assert(containerInit(5, content, name, &fc) == 0);
    assert(containerInit(10, content, name2, &fc2) == 0);

    puts("Serializing");
    array[0] = fc;
    array[1] = fc2;
    assert(serializeContainerArray(array, 2, &size, &buf) == 0);

    puts("Deserializing");
    result = deserializeContainerArray(buf, size, &n);
    assert(n == 2);

    puts("Checking coherence");
    assert(strcmp(result[0].name, name) == 0);
    assert(strcmp(result[1].name, name2) == 0);

    assert(strncmp(result[0].content, content, fc.size) == 0);
    assert(strncmp(result[1].content, content, fc2.size) == 0);
    assert(result[0].size == 5);
    assert(result[1].size == 10);

    destroyContainer(&result[0]);
    destroyContainer(&result[1]);
    destroyContainer(&fc);
    destroyContainer(&fc2);

    free(buf);

    return 0;
}

int main(int argc, char const *argv[])
{
    puts("Starting filesTest");
    assert(filesTest() == 0);
    puts("filesTest succesfull!\n");

    puts("Starting filesystemTest");
    assert(filesystemTest() == 0);
    puts("filesystemTest succesfull!\n");

    puts("Starting listTest");
    assert(listTest() == 0);
    puts("listTest succesfull!\n");

    puts("Starting messageTest");
    assert(messageTest() == 0);
    puts("messageTest succesfull!\n");

    puts("Starting syncqueueTest");
    assert(syncqueueTest() == 0);
    puts("syncqueueTest succesfull!\n");

    puts("Starting threadpoolTest");
    assert(threadpoolTest() == 0);
    puts("threadpoolTest succesfull!\n");

    puts("Starting fileContainerTest");
    assert(fileContainerTest() == 0);
    puts("fileContainerTest succesfull!\n");

    puts("OK!\nALL TESTS SUCCESFULL!");
    return 0;
}
