#ifndef SERVER_H
#define SERVER_H

#include "COMMON/message.h"
#include "SERVER/connstate.h"

struct _handleArgs
{
    int fd;
    size_t counter;
    Message *request;
    ConnState *state;
};
void handleRequest(void *args);

#endif