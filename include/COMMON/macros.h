#ifndef MACROS_H
#define MACROS_H
#include <stdlib.h>
#include <errno.h>
#include <stdio.h>

#define ERROR_CHECK(arg)                                                      \
    if ((arg) == -1)                                                          \
    {                                                                         \
        fprintf(stderr, "Error on line:%d of file:%s\n", __LINE__, __FILE__); \
        perror("Error");                                                      \
        return -1;                                                            \
    }
#define SAFE_ERROR_CHECK(arg)                                                 \
    if ((arg) == -1)                                                          \
    {                                                                         \
        fprintf(stderr, "Error on line:%d of file:%s\n", __LINE__, __FILE__); \
        perror("Error");                                                      \
        return -1;                                                            \
    }
#define PRINT_ERROR_CHECK(arg)                                                \
    if ((arg) == -1)                                                          \
    {                                                                         \
        fprintf(stderr, "Error on line:%d of file:%s\n", __LINE__, __FILE__); \
        perror("Error");                                                      \
    }
// A quieter error check, since we expect many disconnections and such.
#define SAFE_PIPE_CHECK(arg) \
    if ((arg) == -1)         \
    {                        \
        return -1;           \
    }
#define UNSAFE_NULL_CHECK(arg)                                                \
    if ((arg) == NULL)                                                        \
    {                                                                         \
        fprintf(stderr, "Error on line:%d of file:%s\n", __LINE__, __FILE__); \
        perror("Error");                                                      \
        exit(EXIT_FAILURE);                                                   \
    }
#define SAFE_NULL_CHECK(arg)                                                  \
    if ((arg) == NULL)                                                        \
    {                                                                         \
        fprintf(stderr, "Error on line:%d of file:%s\n", __LINE__, __FILE__); \
        perror("Error");                                                      \
        return -1;                                                            \
    }
#define PTHREAD_CHECK(arg)                                                    \
    if ((errno = (arg)) != 0)                                                 \
    {                                                                         \
        fprintf(stderr, "Error on line:%d of file:%s\n", __LINE__, __FILE__); \
        perror("Error on a pthread call");                                    \
        return -1;                                                            \
    }
#define VOID_PTHREAD_CHECK(arg)                                               \
    if ((errno = (arg)) != 0)                                                 \
    {                                                                         \
        fprintf(stderr, "Error on line:%d of file:%s\n", __LINE__, __FILE__); \
        perror("Error on a pthread call");                                    \
        return;                                                               \
    }
#define NULL_PTHREAD_CHECK(arg)                                               \
    if ((errno = (arg)) != 0)                                                 \
    {                                                                         \
        fprintf(stderr, "Error on line:%d of file:%s\n", __LINE__, __FILE__); \
        perror("Error on a pthread call");                                    \
        return NULL;                                                          \
    }
#define READ_CHECK(arg, expected)                                                 \
    {                                                                             \
        ssize_t tmp;                                                              \
        if ((tmp = (arg)) != (expected))                                          \
        {                                                                         \
            if (errno != 0)                                                       \
            {                                                                     \
                perror("Error on a read");                                        \
            }                                                                     \
            fprintf(stderr, "Error on line:%d of file:%s\n", __LINE__, __FILE__); \
            printf("Expected to read:%ld, instead read:%ld\n", (expected), tmp);  \
        }                                                                         \
    }

#define CLEANUP_ERROR_CHECK(arg, cleanup)                                     \
    if ((arg) == -1)                                                          \
    {                                                                         \
        fprintf(stderr, "Error on line:%d of file:%s\n", __LINE__, __FILE__); \
        perror("Error");                                                      \
        cleanup;                                                              \
        return -1;                                                            \
    }

#define CLEANUP_CHECK(arg, err, cleanup)                                      \
    if ((arg) == (err))                                                       \
    {                                                                         \
        fprintf(stderr, "Error on line:%d of file:%s\n", __LINE__, __FILE__); \
        perror("Error");                                                      \
        cleanup;                                                              \
        return -1;                                                            \
    }
#define CLEANUP_PTHREAD_CHECK(arg, cleanup)                                   \
    if ((arg) != 0)                                                           \
    {                                                                         \
        fprintf(stderr, "Error on line:%d of file:%s\n", __LINE__, __FILE__); \
        perror("Error on pthread");                                           \
        cleanup;                                                              \
        return -1;                                                            \
    }

#endif