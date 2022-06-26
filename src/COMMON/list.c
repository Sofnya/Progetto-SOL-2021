#include <stdlib.h>
#include <errno.h>


#include "COMMON/list.h"
#include "COMMON/macros.h"

/**
 * @brief Initializes the list. Should always be called before using it.
 * 
 * @param list the list to be initialized.
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
 * @brief Frees the list, it should not be used afterwards. Take care to free all data contained within it yourself.
 * 
 * @param list the list to be destroyed.
 * @return int 0 on success, -1 and sets errno otherwise.
 */
int listDestroy(List *list)
{
    struct _listEl *cur;

    while(list->_head != NULL)
    {
        cur = list->_head;
        list->_head = list->_head->next;
        free(cur);
    }

    list->size = 0;
    list->_tail= NULL;

    return 0;
}

/**
 * @brief Inserts an element in position 0.
 * 
 * @param el the element to be inserted.
 * @param list the list to be updated.
 * @return int 0 if successfull, -1 and sets errno on an error.
 */
int listPush(void *el, List *list)
{
    struct _listEl *new;

    if(list->_head == NULL)
    {
        SAFE_NULL_CHECK(list->_head = malloc(sizeof(struct _listEl)));
        list->_tail = list->_head;
        list->_head->next = NULL;
        list->_head->prev = NULL;
        list->_head->data = el;
    }
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
 * @brief Gets and removes the element at position 0.
 * 
 * @param el if not NULL, where the removed element will be saved.
 * @param list the target list.
 * @return int 0 if successfull, -1 and sets errno on an error.
 */
int listPop(void **el, List *list)
{
    struct _listEl *tmp;

    if(list->size == 0)
    {
        errno = EINVAL;
        return -1;
    }

    tmp = list->_head;
    list->_head = list->_head->next;

    if(el != NULL) *el = tmp->data;

    free(tmp);
    list->size--;

    if(list->_head == NULL)
    {
        list->_tail = NULL;
    }
    else
    {
        list->_head->prev = NULL;
    }

    return 0;
}

/**
 * @brief Puts the element el at the end of the list (position size).
 * 
 * @param el the element to be appended.
 * @param list the list to be updated.
 * @return int 0 if successfull, -1 and sets errno on error.
 */
int listAppend(void *el, List *list)
{
    struct _listEl *new;

    SAFE_NULL_CHECK(new = malloc(sizeof(struct _listEl)));
    new->data = el;
    new->prev = list->_tail;
    new->next = NULL;
    if(list->size == 0)
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
 * @brief Puts the element el in position pos in the list. 
 * @param pos the position the element should take in the updated list.
 * @param el the element to be added.
 * @param list the list to be updated.
 * @return int 0 if successfull, -1 and sets errno on an error (if pos is an invalid position).
 */
int listPut(long long pos, void *el, List *list)
{
    long long curPos;
    struct _listEl *cur, *new;;

    //Valid positions go from 0 (head) to size (tail + 1, an append.)
    if((pos > list->size) | (pos < 0))
    {
        errno = EINVAL;
        return -1;
    }

    //Pushes and appends are treated separately.
    if(pos == 0)
    {
        return listPush(el, list);
    }
    if(pos == list->size)
    {
        return listAppend(el, list);
    }

    //Is in second half, proceed from tail.
    if((list->size/2 - pos) < 0)
    {
        //We find the item immediately preceding our new element's position.
        curPos = list->size - 1;
        cur = list->_tail;
        while(curPos > pos - 1)
        {
            cur = cur->prev;
            curPos--;
        }

    }
    //Is in first half, proceed from head.
    else
    {
        //Same as above, but forwards.
        curPos = 0;
        cur = list->_head;
        while(curPos < pos - 1)
        {
            cur = cur->next;
            curPos++;
        }
    }
        //And add a new element right after..
        SAFE_NULL_CHECK(new = malloc(sizeof(struct _listEl)));
        new->data = el;
        new->prev = cur;
        new->next = cur->next;
        cur->next = new;

        list->size++;

        return 0;

}

/**
 * @brief Returns the element in position pos inside of el.
 * 
 * @param pos the position from which to get the element.
 * @param el a pointer in which the element will be stored.
 * @param list the list from which to get the element.
 * @return int 0 if successfull, -1 and sets errno on an error (if pos is an invalid position or el is NULL).
 */
int listGet(long long pos,void **el, List *list)
{
    long long curPos;
    struct _listEl *cur;

    if((pos >= list->size) | (pos < 0) | (el == NULL))
    {
        errno = EINVAL;
        return -1;
    }

    //In second half, proceed from tail.
    if(list->size/2 - pos < 0)
    {
        cur = list->_tail;
        curPos = list->size -1;

        while(curPos != pos)
        {
            cur = cur->prev;
            curPos--;
        }
    }
    //In first half, proceed from head.
    else
    {
        cur = list->_head;
        curPos = 0;
        while(curPos != pos)
        {
            cur = cur->next;
            curPos++;
        }
    }
    *el = cur->data;
    return 0;
}

/**
 * @brief Removes the element in position pos from the list. If el != NULL returns the removed element in there.
 * 
 * @param pos the position from which to remove the element.
 * @param el if not NULL, will be set to the removed element.
 * @param list the list to be updated.
 * @return int 0 if successfull, -1 and sets errno on an error (if pos is an invalid position).
 */
int listRemove(long long pos, void **el, List *list)
{
    long long curPos;
    struct _listEl *cur;

    if((pos >= list->size) | (pos < 0))
    {
        errno = EINVAL;
        return -1;
    }

    // First element is just a pop.
    if(pos == 0)
    {
        return listPop(el, list);
    }
    // Last element needs special treatment.
    if(pos == list->size - 1)
    {
        cur = list->_tail;
        list->_tail = list->_tail->prev;

        if(el != NULL) *el = cur->data;
        list->_tail->next = NULL;

        list->size--;

        free(cur);

        return 0;
    }

    //Otherwise we gotta find it.
    //In second half, proceed from tail.
    if(list->size/2 - pos < 0)
    {
        cur = list->_tail;
        curPos = list->size -1;

        while(curPos != pos)
        {
            cur = cur->prev;
            curPos--;
        }
    }
    //In first half, proceed from head.
    else
    {
        cur = list->_head;
        curPos = 0;
        while(curPos != pos)
        {
            cur = cur->next;
            curPos++;
        }
    }
    //Now cur points to the correct position.
    if(el != NULL) *el = cur->data;

    //It's not the first or last element so we can safely do this.
    cur->prev->next = cur->next;
    cur->next->prev = cur->prev;

    list->size--;
    free(cur);
    
    return 0;

}

/**
 * @brief Returns the size of list. Element positions go from 0 to size - 1.
 * 
 * @param list the list of which one needs the size.
 * @return int the size of list.
 */
int listSize(List list)
{
    return list.size;
}


/**
 * @brief Scans the list sequentially, giving a new element at every call. Should be first called with a NULL saveptr, and successively with the same unmodified saveptr.
 * 
 * @param el where the element will be stored.
 * @param saveptr should point to NULL on the first call, unmodified afterwards.
 * @param list the list to scan.
 * @return int 0 on a successfull read, -1 and sets errno to EOF on list end, -1 and EINVAL on an invalid scan.
 */
int listScan(void **el, void **saveptr, List *list)
{
    struct _listEl *cur;

    if(list->size == 0)
    {
        errno = EINVAL;
        return -1;
    }

    if(*saveptr == NULL)
    {
        cur = list->_head;
        *saveptr = cur;
    }
    else
    {
        cur = *saveptr;
    }

    *el = cur->data;

    if(cur->next != NULL)
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
 * @param list the list to be printed.
 */
void printList(List *list)
{
    void *saveptr = NULL;
    void *el;

    while(listScan(&el, &saveptr, list) != -1)
    {
        printf("|%p|", el);
        printf("->");
    }
    printf("|%p|", el);
}
