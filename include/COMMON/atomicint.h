#ifndef ATOMICINT_H
#define ATOMICINT_H

#include <stdint.h>
#include <pthread.h>

typedef struct _atomicInt
{
    size_t _value;
    pthread_mutex_t *_mtx;
} AtomicInt;

void atomicInit(AtomicInt *el);
void atomicDestroy(AtomicInt *el);

size_t atomicGet(AtomicInt *el);
size_t atomicPut(size_t value, AtomicInt *el);
size_t atomicInc(size_t value, AtomicInt *el);
size_t atomicDec(size_t value, AtomicInt *el);

#endif