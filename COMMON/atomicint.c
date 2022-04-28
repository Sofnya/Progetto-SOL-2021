#include "atomicint.h"
#include "macros.h"


/**
 * @brief Initializes the given AtomicInt.
 * 
 * @param el 
 */
void atomicInit(AtomicInt *el)
{
    /**pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    */
    pthread_mutex_init(el->_mtx, NULL);
    el->_value = 0;
}

/**
 * @brief Destroys the given AtomicInt.
 * 
 * @param el 
 */
void atomicDestroy(AtomicInt *el)
{
    pthread_mutex_destroy(el->_mtx);
}

/**
 * @brief Returns the value of the given AtomicInt.
 * 
 * @param el the AtomicInt to query.
 * @return uint64_t the value of el.
 */
uint64_t atomicGet(AtomicInt *el)
{
    uint64_t res;
    PTHREAD_CHECK(pthread_mutex_lock(el->_mtx));
    res = el->_value;
    PTHREAD_CHECK(pthread_mutex_unlock(el->_mtx));
    
    return res;
}

/**
 * @brief Assigns a new value atomically to the given AtomicInt, and returns it's old value.
 * 
 * @param value the value to be set.
 * @param el the AtomicInt to update.
 * @return uint64_t the old value of el.
 */
uint64_t atomicPut(uint64_t value, AtomicInt *el)
{
    uint64_t res;
    PTHREAD_CHECK(pthread_mutex_lock(el->_mtx));
    res = el->_value;
    el->_value = value;
    PTHREAD_CHECK(pthread_mutex_unlock(el->_mtx));
    
    return res;
}

/**
 * @brief Increments the given AtomicInt atomically by given value, returning it's new value.
 * 
 * @param value by how much to increment.
 * @param el the AtomicInt to increment.
 * @return uint64_t the new value of el.
 */
uint64_t atomicInc(uint64_t value, AtomicInt *el)
{
    uint64_t res;
    PTHREAD_CHECK(pthread_mutex_lock(el->_mtx));
    el->_value += value;
    res = el->_value;
    PTHREAD_CHECK(pthread_mutex_unlock(el->_mtx));
    
    return res;
}

/**
 * @brief Decrements the given AtomicInt atomically by given value, returning it's new value.
 * 
 * @param value by how much to decrement it.
 * @param el the AtomicInt to decrement
 * @return uint64_t the new value of el
 */
uint64_t atomicDec(uint64_t value, AtomicInt *el)
{
    uint64_t res;
    PTHREAD_CHECK(pthread_mutex_lock(el->_mtx));
    el->_value -= value;
    res = el->_value;
    PTHREAD_CHECK(pthread_mutex_unlock(el->_mtx));
    
    return res;
}
