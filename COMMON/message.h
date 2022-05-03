#ifndef MESSAGE_H
#define MESSAGE_H


#include <stdint.h>

typedef struct _message {
    uint64_t size;
    void *content;
    char *info;
    int type, status;
} Message;


int messageInit(uint64_t size, void *content, char *info, int type, int status, Message *m);
void messageDestroy(Message *m);


int sendMessage(int fd, Message *m);
int receiveMessage(int fd, Message *m);


#endif