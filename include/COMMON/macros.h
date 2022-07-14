#ifndef MACROS_H
#define MACROS_H
#include <stdlib.h>
#include <errno.h>
#include <stdio.h>

#define ERROR_CHECK(arg)                                             \
    if ((arg) == -1)                                                 \
    {                                                                \
        printf("Error on line:%d of file:%s\n", __LINE__, __FILE__); \
        perror("Error");                                             \
        exit(EXIT_FAILURE);                                          \
    }
#define SAFE_ERROR_CHECK(arg)                                        \
    if ((arg) == -1)                                                 \
    {                                                                \
        printf("Error on line:%d of file:%s\n", __LINE__, __FILE__); \
        perror("Error");                                             \
        return -1;                                                   \
    }
#define UNSAFE_NULL_CHECK(arg)                                       \
    if ((arg) == NULL)                                               \
    {                                                                \
        printf("Error on line:%d of file:%s\n", __LINE__, __FILE__); \
        perror("Error");                                             \
        exit(EXIT_FAILURE);                                          \
    }
#define SAFE_NULL_CHECK(arg)                                         \
    if ((arg) == NULL)                                               \
    {                                                                \
        printf("Error on line:%d of file:%s\n", __LINE__, __FILE__); \
        perror("Error");                                             \
        return -1;                                                   \
    }
#define PTHREAD_CHECK(arg)                                           \
    if ((errno = (arg)) != 0)                                        \
    {                                                                \
        printf("Error on line:%d of file:%s\n", __LINE__, __FILE__); \
        perror("Error on a pthread call");                           \
        exit(EXIT_FAILURE);                                          \
    }
#define READ_CHECK(arg)                \
    if ((arg) <= 0)                    \
    {                                  \
        if (errno != 0)                \
            perror("Error on a read"); \
        return -1;                     \
    }

#define CLEANUP_ERROR_CHECK(arg, cleanup)                            \
    if ((arg) == -1)                                                 \
    {                                                                \
        printf("Error on line:%d of file:%s\n", __LINE__, __FILE__); \
        perror("Error");                                             \
        cleanup;                                                     \
        return -1;                                                   \
    }

#define CLEANUP_CHECK(arg, err, cleanup)                             \
    if ((arg) == (err))                                              \
    {                                                                \
        printf("Error on line:%d of file:%s\n", __LINE__, __FILE__); \
        perror("Error");                                             \
        cleanup;                                                     \
        return -1;                                                   \
    }
#define CLEANUP_PTHREAD_CHECK(arg, cleanup)                          \
    if ((arg) != 0)                                                  \
    {                                                                \
        printf("Error on line:%d of file:%s\n", __LINE__, __FILE__); \
        perror("Error on pthread");                                  \
        cleanup;                                                     \
        return -1;                                                   \
    }

#endif