#ifndef ATOMICINT_H
#define ATOMICINT_H

#include <stdint.h>
#include <pthread.h>


typedef struct _atomicInt{
    uint64_t _value;
    pthread_mutex_t *_mtx;
} AtomicInt;




void atomicInit(AtomicInt *el);
void atomicDestroy(AtomicInt *el);

uint64_t atomicGet(AtomicInt *el);
uint64_t atomicPut(uint64_t value, AtomicInt *el);
uint64_t atomicInc(uint64_t value, AtomicInt *el);
uint64_t atomicDec(uint64_t value, AtomicInt *el);

#endif