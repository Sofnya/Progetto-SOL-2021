#include "COMMON/atomicint.h"
#include "COMMON/macros.h"

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

    UNSAFE_NULL_CHECK(el->_mtx = malloc(sizeof(pthread_mutex_t)));
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
    free(el->_mtx);
}

/**
 * @brief Returns the value of the given AtomicInt.
 *
 * @param el the AtomicInt to query.
 * @return size_t the value of el.
 */
size_t atomicGet(AtomicInt *el)
{
    size_t res;
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
 * @return size_t the old value of el.
 */
size_t atomicPut(size_t value, AtomicInt *el)
{
    size_t res;
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
 * @return size_t the new value of el.
 */
size_t atomicInc(size_t value, AtomicInt *el)
{
    size_t res;
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
 * @return size_t the new value of el
 */
size_t atomicDec(size_t value, AtomicInt *el)
{
    size_t res;
    PTHREAD_CHECK(pthread_mutex_lock(el->_mtx));
    el->_value -= value;
    res = el->_value;
    PTHREAD_CHECK(pthread_mutex_unlock(el->_mtx));

    return res;
}
