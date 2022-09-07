#include <stdlib.h>
#include <errno.h>

#include "COMMON/list.h"
#include "COMMON/macros.h"

/**
 * @brief Initializes given List.
 *
 * @param list the List to initialize.
 * @return int 0 on success, -1 and sets errno otherwise.
 */
int listInit(List *list)
{
    list->_head = NULL;
    list->_tail = NULL;
    list->size = 0;

    return 0;
}

/**
 * @brief Destroys given List, freeing it's resources.
 *
 * @param list the List to destroy.
 * @return int 0 on success, -1 and sets errno otherwise.
 */
int listDestroy(List *list)
{
    struct _listEl *cur;

    while (list->_head != NULL)
    {
        cur = list->_head;
        list->_head = list->_head->next;
        free(cur);
    }

    list->size = 0;
    list->_tail = NULL;

    return 0;
}

/**
 * @brief Inserts given element in position 0 of given List.
 *
 * @param el the element to insert.
 * @param list the List to update.
 * @return int 0 if successfull, -1 and sets errno on an error.
 */
int listPush(void *el, List *list)
{
    struct _listEl *new;

    // If List is empty we handle it separately.
    if (list->_head == NULL)
    {
        SAFE_NULL_CHECK(list->_head = malloc(sizeof(struct _listEl)));
        list->_tail = list->_head;
        list->_head->next = NULL;
        list->_head->prev = NULL;
        list->_head->data = el;
    }
    // Otherwise we just create a new head.
    else
    {
        SAFE_NULL_CHECK(new = malloc(sizeof(struct _listEl)));
        new->data = el;
        new->next = list->_head;
        new->prev = NULL;
        list->_head->prev = new;
        list->_head = new;
    }

    list->size++;

    return 0;
}

/**
 * @brief Gets and removes the element at position 0 of given List.
 *
 * @param el if not NULL, where the removed element will be saved.
 * @param list the target List.
 * @return int 0 if successfull, -1 and sets errno on an error.
 */
int listPop(void **el, List *list)
{
    struct _listEl *tmp;

    if (list->size == 0)
    {
        errno = EINVAL;
        return -1;
    }

    tmp = list->_head;
    list->_head = list->_head->next;

    if (el != NULL)
        *el = tmp->data;

    free(tmp);
    list->size--;

    // Need to handle the case in which the List was only one element long, and update it's tail.
    if (list->_head == NULL)
    {
        list->_tail = NULL;
    }
    // And update the prev to completely remove el from the List.
    else
    {
        list->_head->prev = NULL;
    }

    return 0;
}

/**
 * @brief Puts given element el at the end of given List.
 *
 * @param el the element to append.
 * @param list the List to update.
 * @return int 0 if successfull, -1 and sets errno on error.
 */
int listAppend(void *el, List *list)
{
    struct _listEl *new;

    SAFE_NULL_CHECK(new = malloc(sizeof(struct _listEl)));
    new->data = el;
    new->prev = list->_tail;
    new->next = NULL;
    if (list->size == 0)
    {
        list->_head = new;
        list->_tail = new;
        list->size++;

        return 0;
    }

    list->_tail->next = new;
    list->_tail = new;
    list->size++;
    return 0;
}

/**
 * @brief Puts given element el in position pos in given List.
 * @param pos the position the element should take in the updated List.
 * @param el the element to add.
 * @param list the List to update.
 * @return int 0 if successfull, -1 and sets errno on an error.
 */
int listPut(long long pos, void *el, List *list)
{
    long long curPos;
    struct _listEl *cur, *new;
    ;

    // Valid positions go from 0 (head) to size (tail + 1, an append.)
    if ((pos > list->size) | (pos < 0))
    {
        errno = EINVAL;
        return -1;
    }

    // Pushes and appends are treated separately.
    if (pos == 0)
    {
        return listPush(el, list);
    }
    if (pos == list->size)
    {
        return listAppend(el, list);
    }

    // Is in second half, proceed from tail.
    if ((list->size / 2 - pos) < 0)
    {
        // We find the item immediately preceding our new element's position.
        curPos = list->size - 1;
        cur = list->_tail;
        while (curPos > pos - 1)
        {
            cur = cur->prev;
            curPos--;
        }
    }
    // Is in first half, proceed from head.
    else
    {
        // Same as above, but forwards.
        curPos = 0;
        cur = list->_head;
        while (curPos < pos - 1)
        {
            cur = cur->next;
            curPos++;
        }
    }

    // And add a new element right after the found element.
    SAFE_NULL_CHECK(new = malloc(sizeof(struct _listEl)));
    new->data = el;
    new->prev = cur;
    new->next = cur->next;
    cur->next = new;
    new->next->prev = new;

    if (new->next == NULL)
    {
        list->_tail = new;
    }

    list->size++;

    return 0;
}

/**
 * @brief Returns the element in position pos inside of el.
 *
 * @param pos the position from which to get the element.
 * @param el a pointer in which the element will be stored.
 * @param list the List from which to get the element.
 * @return int 0 if successfull, -1 and sets errno on an error (if pos is an invalid position or el is NULL).
 */
int listGet(long long pos, void **el, List *list)
{
    long long curPos;
    struct _listEl *cur;

    if ((pos >= list->size) | (pos < 0) | (el == NULL))
    {
        errno = EINVAL;
        return -1;
    }

    // In second half, proceed from tail.
    if (list->size / 2 - pos < 0)
    {
        cur = list->_tail;
        curPos = list->size - 1;

        while (curPos != pos)
        {
            cur = cur->prev;
            curPos--;
        }
    }
    // In first half, proceed from head.
    else
    {
        cur = list->_head;
        curPos = 0;
        while (curPos != pos)
        {
            cur = cur->next;
            curPos++;
        }
    }
    *el = cur->data;
    return 0;
}

/**
 * @brief Removes the element in position pos from given List. If el != NULL returns the removed element in el.
 *
 * @param pos the position from which to remove the element.
 * @param el if not NULL, will be set to the removed element.
 * @param list the List to update.
 * @return int 0 if successfull, -1 and sets errno on an error (if pos is an invalid position).
 */
int listRemove(long long pos, void **el, List *list)
{
    long long curPos;
    struct _listEl *cur;

    if ((pos >= list->size) | (pos < 0))
    {
        errno = EINVAL;
        return -1;
    }

    // First element is just a pop.
    if (pos == 0)
    {
        return listPop(el, list);
    }
    // Last element needs special treatment.
    if (pos == list->size - 1)
    {
        cur = list->_tail;
        list->_tail = list->_tail->prev;

        if (el != NULL)
            *el = cur->data;
        list->_tail->next = NULL;

        list->size--;

        free(cur);

        return 0;
    }

    // Otherwise we gotta find it.
    // In second half, proceed from tail.
    if (list->size / 2 - pos < 0)
    {
        cur = list->_tail;
        curPos = list->size - 1;

        while (curPos != pos)
        {
            cur = cur->prev;
            curPos--;
        }
    }
    // In first half, proceed from head.
    else
    {
        cur = list->_head;
        curPos = 0;
        while (curPos != pos)
        {
            cur = cur->next;
            curPos++;
        }
    }
    // Now cur points to the correct position.
    if (el != NULL)
        *el = cur->data;

    // It's not the first or last element so we can safely do this.
    cur->prev->next = cur->next;
    cur->next->prev = cur->prev;

    list->size--;
    free(cur);

    return 0;
}

/**
 * @brief Returns the size of given List. Element positions go from 0 to size - 1.
 *
 * @param list the List to query.
 * @return int the size of given List.
 */
int listSize(List list)
{
    return list.size;
}

/**
 * @brief Scans given List sequentially, giving a new element at every call. Should be first called with a NULL saveptr, and successively with the same unmodified saveptr.
 *
 * @param el where the element will be stored.
 * @param saveptr should point to NULL on the first call, unmodified afterwards.
 * @param list the List to scan.
 * @return int 0 on a successfull read, -1 and sets errno to EOF on List end, -1 and EINVAL on an invalid scan.
 */
int listScan(void **el, void **saveptr, List *list)
{
    struct _listEl *cur;

    if (list->size == 0)
    {
        errno = EINVAL;
        return -1;
    }

    // Our saveptr is actually just an internal listElement.
    if (*saveptr == NULL)
    {
        cur = list->_head;
        *saveptr = cur;
    }
    else
    {
        cur = *saveptr;
    }

    *el = cur->data;

    if (cur->next != NULL)
    {
        *saveptr = cur->next;
        return 0;
    }
    else
    {
        errno = EOF;
        return -1;
    }
}

/**
 * @brief For debugging, prints a list's contents.
 *
 * @param list the List to be print.
 */
void printList(List *list)
{
    void *saveptr = NULL;
    void *el;
    int i = 0;

    if (listSize(*list) == 0)
        return;

    while (listScan(&el, &saveptr, list) != -1)
    {
        printf("|%p|", el);
        printf("->");
        i++;
    }
    printf("|%p|\n", el);
}

/**
 * @brief Prints a List's contents, using the custom fnc provided for every element.
 *
 * @param list the List to be print.
 * @param fnc a function that turns every element in given List in a malloced printable string.
 */
void customPrintList(List *list, char *(*fnc)(void *))
{
    void *saveptr = NULL;
    void *el;
    char *cur;
    int i = 0;

    if (listSize(*list) == 0)
        return;

    while (listScan(&el, &saveptr, list) != -1)
    {
        cur = fnc(el);
        puts(cur);
        free(cur);
        i++;
    }
    cur = fnc(el);
    puts(cur);

    free(cur);
}

/**
 * @brief Sorts given List in ascending order, according to given custom heuristic.
 *
 * @param list the List to sort.
 * @param heuristic a function turning every List element in a long.
 * @return int 0 on success, -1 on failure.
 */
int listSort(List *list, long (*heuristic)(void *))
{
    size_t sortedPos = 1;
    void *saveptr = NULL;
    void *curEl;
    void *prevEl;
    long curVal;
    int i;

    // If List is one or zero elements it's already sorted.
    if (listSize(*list) <= 1)
        return 0;

    // We implement a simple insertion sort.
    // We keep track of the sorted portion of the list, the elements from position 0 to sortedPos - 1 are already sorted.
    while (sortedPos < listSize(*list))
    {
        // We work on the first element after the sorted portion.
        SAFE_ERROR_CHECK(listGet(sortedPos - 1, &prevEl, list));
        SAFE_ERROR_CHECK(listGet(sortedPos, &curEl, list));
        curVal = heuristic(curEl);

        // If curVal is bigger than the maximum value of the sorted portion, it should stay where it is.
        if (curVal >= heuristic(prevEl))
        {
            sortedPos++;
            continue;
        }

        // Otherwise we gotta find it's right place inside of the sorted portion, by scanning all elements.
        saveptr = NULL;
        i = 0;
        while (listScan(&prevEl, &saveptr, list) == 0)
        {
            // We are searching for the first element bigger than our element, to put ourselves right before it.
            if (curVal <= heuristic(prevEl))
            {
                // We are guaranteed to find this element before the end of the sorted portion of the List.
                // Once found we remove the current element from it's previous position, and put it right before the found element.
                SAFE_ERROR_CHECK(listRemove(sortedPos, &curEl, list));
                SAFE_ERROR_CHECK(listPut(i, curEl, list));
                sortedPos++;
                break;
            }
            i++;
        }
    }

    return 0;
}