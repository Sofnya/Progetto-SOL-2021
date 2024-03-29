#ifndef MESSAGE_H
#define MESSAGE_H

#include <stdint.h>

#define MT_INFO 0
#define MT_FOPEN 1
#define MT_FCLOSE 2
#define MT_FREAD 3
#define MT_FWRITE 4
#define MT_FAPPEND 5
#define MT_FREM 6
#define MT_DISCONNECT 7
#define MT_FLOCK 8
#define MT_FUNLOCK 9
#define MT_FREADN 10

#define MS_REQ 0
#define MS_OK 200
#define MS_OKCAP 201
#define MS_INV 400
#define MS_ERR 401
#define MS_INTERR 402

typedef struct _message
{
    size_t size;
    void *content;
    char *info;
    int type, status;
} Message;

int messageInit(size_t size, void *content, const char *info, int type, int status, Message *m);
void messageDestroy(Message *m);

int sendMessage(int fd, Message *m);
int receiveMessage(int fd, Message *m);

int readWrapper(int fd, void *buf, size_t count);
int writeWrapper(int fd, void *buf, size_t count);

#endif