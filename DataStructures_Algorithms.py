""" In this file, I'll code up solutions to some common & fun data structures
and algorithms problems!"""

from itertools import *

def closest_random_points(x_list, y_list):

    def compute_dist(pt1, pt2):
        return ((pt1[0] - pt2[0]) ** 2 + (pt1[0] + pt2[0]) ** 2) ** 0.5

    min_dist = float('-inf')

    for i in range(len(x_list)):
        for j in range(len(y_list)):

            if i != j:

                pt1 = (x_list[i], y_list[i])
                pt2 = (x_list[j], y_list[j])

                cur_dist = compute_dist(pt1, pt2)
                min_dist = min(min_dist, cur_dist)
    
    return min_dist





def compute_path(x, y, index):
    """Given a grid world coordinate goal, x,y find the lexographically index-th smallest path, where there are only two actions, 'H' or 'V' and you can
    only move down or to the right. Assume you start from 0,0."""
    potential_paths = []
    chose_x = x > y

    for combination in combinations(range(x + y), min(x, y)):

        if chose_x:
            net = ['H'] * (x + y)

            for i in combination:
                net[i] = 'V'
            
            potential_paths.append(''.join(net))
        else:
            net = ['V'] * (x + y)

            for i in combination:
                net[i] = 'H'
            
            potential_paths.append(''.join(net))

    potential_paths.sort()

    return potential_paths[index]




def partition_binary_array(arr):
    """ Partition an array of 0's and 1's into 3 parts such that the binary representation
    of all the digits in each partition are equal.
    Ex: arr = [0, 1, 0, 0, 1, 0, 0, 1, 0, 0] => [3, 7]
    
    If it cannot be split, return [-1, -1]"""

    if len(arr) < 3:
        return [-1, -1]

    num_ones = arr.count(1)

    if num_ones % 3 or not num_ones:
        print("Non divisible number of 1's or no 1's.")
        return [-1, -1]
    
    num_ones_each_partition = num_ones // 3

    j = len(arr)
    cur_num_ones = 0
    required_padded_zeros = 0

    while j >= 0 and cur_num_ones < num_ones_each_partition:
        j -= 1
        if arr[j]:
            cur_num_ones += 1
        elif not arr[j] and not cur_num_ones:
            required_padded_zeros += 1
    
    adjusted_j = j - 1

    while not arr[adjusted_j]:
        adjusted_j -= 1

    count = 0
    adjusted_j += 1
    while not arr[adjusted_j] and count < required_padded_zeros:
        adjusted_j += 1
        count += 1
    
    j = adjusted_j

    # print("j", j)
    # print("required_padded_zeros", required_padded_zeros)
    # print("num ones each partition", num_ones_each_partition)

    i = -1
    cur_num_ones = 0

    while i < len(arr) and cur_num_ones < num_ones_each_partition:
        i += 1
        if arr[i]:
            cur_num_ones += 1
    
    num_padded_zeros = 0
    while i < len(arr) and num_padded_zeros < required_padded_zeros:
        i += 1
        num_padded_zeros += 1
        if arr[i]:
            print("Not proper amount of leading zeros on left partition.")
            return [-1, -1]

    left_partition_val = compute_binary_rep(arr, 0, i)
    middle_partition_val = compute_binary_rep(arr, i + 1, j - 1)
    right_partition_val = compute_binary_rep(arr, j, len(arr) - 1)

    if not (left_partition_val == middle_partition_val == right_partition_val):
        print("Indices: ", i, j)
        print("Non equal partitions: ", left_partition_val, middle_partition_val, right_partition_val)
        return [-1, -1]
    
    return [i, j]
    

def compute_binary_rep(arr, start, end):
    total = 0
    power = 0

    for i in range(end, start - 1, -1):

        cur_val = arr[i]
        total += cur_val * 2 ** power
        power += 1

    return total

def median_of_two_sorted_arrays(lst1, lst2):
    """ Find the median of two individually sorted arrays as if they were merged.
    Runtime should be proportional to log(m + n) where m and n are the sizes of the
    two arrays."""
    def find_kth_val(k, lst1, lst2):

        if not lst1:
            return lst2[k]
        if not lst2:
            return lst1[k]

        mid1_index = len(lst1) // 2
        mid2_index = len(lst2) // 2

        if k > mid1_index + mid2_index:
            # Value of k changes in recursive calls as we are removing from the front of lists
            if lst1[mid1_index] > lst2[mid2_index]:
                return find_kth_val(k - mid2_index - 1, lst1, lst2[mid2_index + 1:])
            else:
                return find_kth_val(k - mid1_index - 1, lst1[mid1_index + 1:], lst2)
        else:
            # Vale of k doesn't change in recursive calls as we are removing from end of lists
            if lst1[mid1_index] > lst2[mid2_index]:
                return find_kth_val(k, lst1[:mid1_index], lst2)
            else:
                return find_kth_val(k, lst1, lst2[:mid2_index])

    net_length = len(lst1) + len(lst2)
    if net_length % 2 == 1:
        return find_kth_val(net_length // 2, lst1, lst2)
    else:
        # if the number of total values is even, calculate the median by
        # taking the average of the two middle values
        left_median = find_kth_val(net_length // 2 - 1, lst1, lst2)
        right_median = find_kth_val(net_length // 2, lst1, lst2)
        return (left_median + right_median) / 2


def max_subset(arr):
    """Return the value of the maximum product of a single subset of an array.
     Consider efficiency and shoot for a linear asymptotic runtime."""

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
        self.max_size = max(max_size, 1) #Max Cache Size is min 1

    def get(self, key):
        if key in self.d:
            output = self.d[key]
            if output is self.tail and self.size > 1:
                # moves global tail attribute to second item to the end
                self.tail = self.tail.prev
                #reassigns new tails 'next' attribute to empty tuple
                self.tail.next = ()

            elif output.next and output.prev:
                 # if output is in middle of DLL reassigns pointers in the middle
                 # to remove recently used item
                output.next.prev = output.prev
                output.prev.next = output.next
            if output is not self.head:
                # moves most recently accessed item to front of DLL
                self.head.prev = output
                output.next = self.head
                self.head = output
                self.head.prev = ()
            return output.val
        else:
            return "Please enter valid key."

    def set(self, key, value):
        if self.size == self.max_size:
            # delete least recently used item from cache if size is full
            old_key = self.tail.key
            del self.d[old_key]
            self.tail = self.tail.prev
            self.size -= 1
        if key not in self.d:
            new_node = Node(key, value)
            if self.head:
                #reassigns pointers to add the new item to the front of the Doubly linked list
                self.head.prev = new_node
                new_node.next = self.head
            else:
                #initialize head and tail
                self.head = new_node
                self.tail = new_node
            self.head = new_node
            self.d[key] = new_node
            self.size += 1
        else:
            #replace whatever old key was with new key, move to front of DLL (reuse get method from above)
            self.d[key].val = value
            self.get(key)

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

def three_sum(nums, target_sum):
    """ Given a list of numbers, nums, find all of the combinations of three numbers that add up
    to target_sum. Shoot for quadratic asymptotic runtime."""

    if len(nums) < 3:
            return []

    nums.sort()
    results = set()

    for i in range(len(nums) - 2):
        j = i + 1
        k = len(nums) - 1
        while j < k:
            cur = nums[i] + nums[j] + nums[k]
            if cur > target_sum:
                k -= 1
            elif cur < target_sum:
                j += 1
            else:
                results.add((nums[i], nums[j], nums[k]))
                k-= 1
                j += 1
    return list(results)

def count_change(n):
    """Given a world in which there are coins of every power of two, going upwards in size infinitely,
    find the number of ways one can split up a change value of size n with this currency set."""
    ways = [0 for x in range(n+1)]
    ways[0] = 1
    i = 1
    while i <= n:
        j = i
        while j <= n:
            ways[j] += ways[j - i]
            j += 1
        i *= 2
    return ways[n]

def three_sum_closest(nums, target_sum):
    """ Given a list of numbers, nums, find the value of the sum of any three numbers which is closest to the target_sum.
     Shoot for quadratic asymptotic runtime. """
    nums.sort()
    closest = None

    for i in range(len(nums) - 2):
        j = i + 1
        k = len(nums) - 1
        while j < k:
            cur = nums[i] + nums[j] + nums[k]
            if closest is None:
                closest = cur
                smallest_diff = abs(cur - target_sum)
            elif abs(cur - target_sum) < smallest_diff:
                closest = cur
                smallest_diff = abs(cur - target_sum)

            if cur > target_sum:
                k -= 1
            elif cur < target_sum:
                j += 1
            else:
                k-= 1
                j += 1
    return closest


"""Below are the implementations for the Tree & Linked list data
structures that I have used throughout the file. Credit to UC Berkeley's CS61A
course for developing the implementations of these data structures."""

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
