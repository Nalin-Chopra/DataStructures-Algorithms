""" In this file, I'll code up solutions to some common & fun data structures
and algorithms problems!"""


def max_subset(arr):
    """Return the value of the maximum subset of an array in terms of
    product. Consider efficiency and shoot for a runtime of 0(N)."""

    max_lst = [arr[0]]
    min_lst = [arr[0]]

    for i in range(1, len(arr)):
        max_lst.append(max(arr[i], arr[i] * max_lst[i - 1], arr[i] * min_lst[i - 1]))
        min_lst.append(min(arr[i], arr[i] * min_lst[i - 1], arr[i] * max_lst[i - 1]))

    return max(max_lst)


class Cache:
    """Create an API for a LRU or Least-Recently-Used Cache.
    One should be able to assign a certain number of key value pairs to the cache,
    and should be able to access a value given a key. If a new key-value pair is
    added to the cache and the memory is full, the least-recently accessed or
    inserted pair should be removed from memory.

    The cache should have constant time insertion and deletion.
    """

    def __init__(self, max_size):
        self.d = {} #Dictionary with values as nodes of a DLL
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

def linked_lst(btree):
	"""Create a linked_lst function for Binary Trees. The method should return a
	Linked List that contains all the elemnts of the tree in sorted order."""

	if btree is BTree.empty:
		return Link.empty
	rest = Link(t.entry, linked_lst(btree.right))
	if BTree.left:
		front = linked_lst(btree.left)
		result = front
		while front.rest is not Link.empty:
			front = front.rest
		front.rest = rest
		return result
	else:
		return rest

def reverse(lnk):
    """
    >>> a = Link(1, Link(2, Link(3)))
    >>> r = reverse(a)
    >>> r.first
    3
    >>> r.rest.first
    2
    """
# iterative
    """cur = lnk
    prev = Link.empty
    while cur is not Link.empty:
        nex = cur.rest
        cur.rest = prev
        prev = cur
        cur = nex
    return prev"""

# reverse reverse recursion

    if lnk is lnk.empty or lnk.rest is lnk.empty:
        return lnk
    rest_rev = reverse(lnk.rest)
    lnk.rest.rest = lnk
    lnk.rest = Link.empty
    return rest_rev


def widest_level(t):
    """Return a list of node values of the widest level of a tree.

    >>> sum([[1], [2]], [])
    [1, 2]
    >>> t = Tree(3, [Tree(1, [Tree(1), Tree(5)]), Tree(4, [Tree(9, [Tree(2)])])])
    >>> widest_level(t)
    [1, 5, 9]
    """
    levels = []
    x = [t]
    while x:
        levels.append([t.label for t in x])
        x = sum([t.branches for t in x], [])
    return max(levels, key = len)

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


def gen_path(t):
	""" Return a generator for all the paths from a root to leaf.
	>>> tree = Tree(10, [Tree(1, [Tree(5), Tree(6, [Tree(10)])])])
	>>> g = gen_path(t)
	>>> next(g)
	[10, 1, 5]
	>>> for path in g:
		print(path)
	[10, 1, 6, 10]
	[10, 10]
	"""
	if not t.is_leaf():
		for paths in [gen_path(b) for b in t.branches]:
			for path in paths:
				yield [t.label] + path
	else:
		yield [t.label]


"""Below are the implementations for the Tree & Linked list data
structures that I have used throughout the file."""

# Tree Implementation
class Tree:
    def __init__(self, label, branches=[]):
        for c in branches:
            assert isinstance(c, Tree)
        self.label = label
        self.branches = list(branches)

    def __repr__(self):
        if self.branches:
            branches_str = ', ' + repr(self.branches)
        else:
            branches_str = ''
        return 'Tree({0}{1})'.format(self.label, branches_str)

    def is_leaf(self):
        return not self.branches

    def __eq__(self, other):
        return type(other) is type(self) and self.label == other.label \
               and self.branches == other.branches

    def __str__(self):
        def print_tree(t, indent=0):
            tree_str = '  ' * indent + str(t.label) + "\n"
            for b in t.branches:
                tree_str += print_tree(b, indent + 1)
            return tree_str
        return print_tree(self).rstrip()

    def copy_tree(self):
        return Tree(self.label, [b.copy_tree() for b in self.branches])

class BTree:
    empty = ()
    def __init__(self, label, left=(), right=()):
        assert left is BTree.empty or isinstance(left, BTree)
        assert right is BTree.empty or isinstance(right, BTree)
        self.label = label
        self.left = left
        self.right = right

    def __repr__(self):
        if self.left:
            left_branches = ', ' + repr(self.left)
        else:
            left_branches = ''
        if self.right:
            right_branches = ', ' + repr(self.right)
        else:
            right_branches = ''

        return 'BTree({0}{1}{2})'.format(self.label, left_branches, right_branches)

    def is_leaf(self):
        return not self.left and not self.right


# Link
class Link:
    """A linked list.

    >>> s = Link(1, Link(2, Link(3)))
    >>> s.first
    1
    >>> s.rest
    Link(2, Link(3))
    """
    empty = ()

    def __init__(self, first, rest=empty):
        assert rest is Link.empty or isinstance(rest, Link)
        self.first = first
        self.rest = rest

    def __repr__(self):
        if self.rest is Link.empty:
            return 'Link({})'.format(self.first)
        else:
            return 'Link({}, {})'.format(self.first, repr(self.rest))

    def __str__(self):
        """Returns a human-readable string representation of the Link

        >>> s = Link(1, Link(2, Link(3, Link(4))))
        >>> str(s)
        '<1 2 3 4>'
        >>> str(Link(1))
        '<1>'
        >>> str(Link.empty)  # empty tuple
        '()'
        """
        string = '<'
        while self.rest is not Link.empty:
            string += str(self.first) + ' '
            self = self.rest
        return string + str(self.first) + '>'
