#ifndef MESSAGE_H
#define MESSAGE_H


#include <stdint.h>


#define MT_FOPEN 1
#define MT_FCLOSE 2
#define MT_FREAD 3
#define MT_FWRITE 4
#define MT_FAPPEND 5
#define MT_FREM 6


#define MS_REQ 0
#define MS_OK 200
#define MS_OKCAP 201


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