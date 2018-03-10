""" In this file, I'll code up solutions to some common & fun data structures
and algorithms problems!"""



""" Yelp Interview """
def max_subset(arr):
    """Return the value of the maximum subset of an array in terms of
    product. Consider efficiency and shoot for a runtime of 0(N)."""

    max_lst = [arr[0]]
    min_lst = [arr[0]]

    for i in range(1, len(arr)):
        max_lst.append(max(arr[i], arr[i] * max_lst[i - 1], arr[i] * min_lst[i - 1]))
        min_lst.append(min(arr[i], arr[i] * min_lst[i - 1], arr[i] * max_lst[i - 1]))

    return max(max_lst)


""" Workday Interview """


def num_times(x, arr):
    """Given a sorted array, arr, find the amount of times that integer
    x appears in a. Optimize for efficiency."""
    def found_num(x, arr, index):
        count = 1
        left_index, right_index = index - 1, index + 1
        while left_index > -1:
            if arr[left_index] == x:
                count += 1
            else:
                break
            left_index -= 1
        while right_index < len(arr):
            if arr[right_index] == x:
                count += 1
            else:
                break
            right_index += 1
        return count

    def find_nums(x, arr, index=0):
        if arr[index] != x and len(arr) == 1:
            return 0
        elif arr[index] == x:
            return found_num(x, arr, index)
        elif arr[index] < x:
            arr = arr[(index + 1):]
        else:
            arr = arr[:index]
        return find_nums(x, arr, len(arr) // 2)

    return find_nums(x, arr, len(arr) // 2)


"""GM AutonomV Final Interview"""

class Cache:
    """Create an API for a LRU or Least-Recently-Used Cache.
    One should be able to assign a certain number of key value pairs to the cache,
    and should be able to access a value given a key. If a new key-value pair is
    added to the cache and the memory is full, the least-recently accessed or
    inserted pair should be removed from memory.

    The cache should have constant time insertion and deletion.
    """

    def __init__(self, max_size):
        self.d = {}
        self.head = () #Pointer to Head of Doubly Linked List
        self.tail = () ##Pointer to tail of Doubly Linked List
        self.size = 0
        self.max_size = max_size #Max Cache Size

    def get(self, key):
        if key in self.d:
            output = self.d[key]
            if output is self.tail and self.size > 1:
                # checks if tail pointer needs to be reassigned
                self.tail = self.tail.prev
            if output.next:
                 # reassigns pointers in the middle of the DLL to remove recently used item
                output.next.prev = output.prev
                output.prev.next = output.next
            else:
                output.prev.next = ()
            self.head.prev = output
             # moves recently accessed item to the front of the DLL
            output.next = self.head
            self.head = output
            output.prev = ()
            return output.val
        else:
            return "Please enter valid key."

    def set(self, key, value):
        if self.size == self.max_size:
            old_key = self.tail.key
            del self.d[old_key]
            self.tail = self.tail.prev
        else:
            self.size += 1
        new_node = Node(key, value)
        if self.head:
            #reassigns pointers to add the new item to the front of the Doubly linked list
            self.head.prev = new_node
        else:
            self.head = new_node
            self.tail = new_node
        new_node.next = self.head
        self.head = new_node
        self.d[key] = new_node

class Node:
    def __init__(self, key, value):
        self.val = value
        self.key = key
        self.next = ()
        self.prev = ()
