'''-----------------------------------List-------------------------------------'''
'''
Python的列表就相当于其它语言的Array
'''
'''-----------------------------------Initialization初始化-------------------------------------'''
# 1d array
l = [0 for _ in range(len([1,2,3,4,5]))]
print(l)
# 2d array
l = [[0] for i in range(3) for j in range(4)]
print(l)
'''-----------------------------------Start from behind-------------------------------------'''
lastElement = l[-1]
lastTwo = l[-2:]
for i in range(0, -10, -1):
    print(i)
'''-----------------------------------Copy复制-------------------------------------'''
# shallow copy 浅拷贝
l1 = [1,2,3,4]
l2 = l1[:]
print(l2)
# or
l2 = l1.copy()
print(l2)
"""浅复制的问题在于，如果l1内部还有list，那么这种嵌套的索引不能被复制"""
a = [1,2, [3,4]]
b = a[:]
print(b)
a[2].append(5)
print(b)
# deep copy 深拷贝
import copy
b = copy.deepcopy(a)
print(b)
'''-----------------------------------Enumerate 枚举-------------------------------------'''
l = ["a", "b", "c"]
# value and index together
for i, v in enumerate(l):
    print(i, v)
'''-----------------------------------Zip-------------------------------------'''
# zip的本意就是拉链，可以想象成将两个数组像拉链一样挨个聚合
x = [1, 2, 3]
y = [4, 5, 6]
zipped = zip(x, y)
print(list(zipped))
'''-----------------------------------Reduce-------------------------------------'''
# reduce 可以分别对相邻元素使用同一种计算规则，同时每一步结果作为下一步的参数，很典型的函数式编程用法
import functools
l = [1,3,5,6,2,]
print("The sum of the list element is : ", end="")
print(functools.reduce(lambda a,b : a+b, l))
'''-----------------------------------Map-------------------------------------'''
# 可以将参数一一映射来计算
date = "2019-06-25"
Y, M, D = map(int, date.split('-'))
print("Y:", Y, " M:", M, " D:", D)
'''-----------------------------------Deque-------------------------------------'''
"""
list删除末尾的操作是O(1)的，但是删除头操作就是O(n),这时候需要一个双端队列deque。首尾的常规操作为：
    - append:添加到末尾
    - appendleft: 添加到开头
    - pop: 剔除末尾
    - popleft: 移除开头
"""
'''-----------------------------------Sorted-------------------------------------'''
"""
list自身有自带的sort函数，但是它不返回新的list，sorted能返回一个新的list，并且支持传入参数的reverse
"""
# 比如我们有一个tuple数组，按照tuple的第一个元素进行排序
l1 = [(1,2), (0,1), (3,10)]
l2 = sorted(l1, key=lambda x: x[0])
print(l2)
# 这里的key允许传入一个自定义参数，也可以用自带函数进行比较，比如在一个string数组里只想比较小写,可以传入key=str.lower
l1 = ["banana","Apple","Watermelon"]
l2 = sorted(l1, key=str.lower)
print(l2)
# ['Apple', 'banana', 'Watermelon']
'''-----------------------------------Lambda-------------------------------------'''
# lambda 定义匿名函数，可以理解成一个callback function，参数名一一对应就行
'''-----------------------------------cmp_to_key-------------------------------------'''
"""
python3 中， sorted函数取消了自带的cmp函数，需要借助functools库中的cmp_to_key来作比较
比如如果要按照数组元素的绝对值来排序
"""
from functools import cmp_to_key
def absSort(arr):
    newarr = sorted(arr, key=cmp_to_key(mycmp))
    return newarr
def mycmp(a, b):
    if abs(a) < abs(b):
        return -1
    elif abs(a) > abs(b):
        return 1
    else:
        return 0
print(absSort([-3, -2, -1]))
# [-1, -2, -3]
print(absSort([3, 2, 1]))
# [1, 2, 3]
'''-----------------------------------set-------------------------------------'''
"""
set的查找操作的复杂度为O(1)，有时候可以替代dict来存储中间过程
    - add: set的添加是add不是append
    - remove vs discard： 都是删除操作，区别在于remove不存在的元素会报错，discard会报错
    - union, intersection 快速获得并集和交集，方便一些去重操作
"""
'''-----------------------------------dict-------------------------------------'''
"""字典，相当于其他语言中的map, hashtable, hashmap之类的，读取操作也是O(1)复杂度"""
"""
    -   keys()
    -   values()
    -   items()
    分别可以获得key, value, {key: value}的数组 
"""
'''-----------------------------------setdefaule-------------------------------------'''
"""
这个函数经常在初始化字典的时候使用，如果某个key在字典中存在，返回它的value，否则返回你给的default值
"""
# 比如在建立一个trie tree的时候
# node = self.root
# for char in word:
#     node = node.setdefault(char, {})
'''-----------------------------------OrderedDict-------------------------------------'''
"""
OrderedDict 能记录你key和value的插入顺序，底层其实是一个双向链表加哈希表的实现，我们甚至可以使用move_to_end这样的函数
import collections
imports the collections module into the current namespace, so you could work with this import like this:

import collections
orderedDict = collections.OrderedDict()

equals to expression: 

from collections import OrderedDict
"""
from collections import OrderedDict
d = OrderedDict.fromkeys('abcde')
d.move_to_end('b')
print(''.join(d.keys()))
d.move_to_end('b', last=False)
print(''.join(d.keys()))
'''-----------------------------------defaultdict-------------------------------------'''
"""
defaultdict可以很好地解决一些初始化的问题，比如value是一个list，每次需要判断key是否存在的情况
这时我们可以直接定义
"""
from collections import defaultdict
from collections import *
d = defaultdict(list)
s = [('yellow', 1), ('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)]
for key, value in s:
    print(key, value)
    d[key].append(value)
print(sorted(d.items()))
# [('blue', [2, 4]), ('red', [1]), ('yellow', [1, 3])]
'''-----------------------------------heapq-------------------------------------'''
"""
heapq就是python的priority queue, heapq[0]即为栈顶元素
heapq的实现是小顶堆，如果需要一个大顶堆，常规的做法就是把值取负存入，取出时再反转
以下是借助heapq来实现heapsort的例子
"""
def heapsort(iterable):
    h = []
    for value in iterable:
        heappush(h, value)
    return [heappop(h) for _ in rnage(len(h))]
# [('blue', [2, 4]), ('red', [1]), ('yellow', [1, 3])]
'''-----------------------------------bisect-------------------------------------'''
"""
python自带二分查找的库，在一些不要求实现binary search, 但是借助它能加速的场景下可以直接使用
Python 有一个 bisect 模块，用于维护有序列表。bisect 模块实现了一个算法用于插入元素到有序列表。
在一些情况下，这比反复排序列表或构造一个大的列表再排序的效率更高。Bisect 是二分法的意思，
这里使用二分法来排序，它会将一个元素插入到一个有序列表的合适位置，
这使得不需要每次调用 sort 的方式维护有序列表。
"""
import bisect
a = [1, 2, 3, 4, 5]
print(bisect.bisect(a, 2, lo=0, hi=len(a)))
position = bisect.bisect(a, 6)
bisect.insort(a, 3)
print(a)
# 这里的参数分别为数组，要查找的数，范围起始点，范围结束点
# bisect.bisect_left, bisect_right 分别返回可以插入x的最左和最右index
'''-----------------------------------Counter-------------------------------------'''
"""
Counter接受的参数可以是一个string， 或者一个list， mapping
"""
c = Counter()
c = Counter('gallahad')
c = Counter({'red': 4, 'blue': 2})
c = Counter(cats=4, dogs=8)
print(Counter('abracadabra').most_common(3))
# [('a', 5), ('b', 2), ('r', 2)]
'''-----------------------------------Strings-------------------------------------'''
# ord返回的单个字符的unicode：
print(ord('a'))
# char则是反向操作
print(chr(100))
'''-----------------------------------strip-------------------------------------'''
# 移除string前后的字符串，默认来移除空格，但是也可以给一个字符串，然后会移除含有这个字符串的部分
print('    spacious   '.strip())
# spacious
print('www.example.com'.strip('cmowz.'))
# example
'''-----------------------------------spilt-------------------------------------'''
# 按照某个字符串来切分，返回一个list,可以传入一个参数maxspilt来限定隔离
print('1,2,3'.split(','))
# ['1', '2', '3']
print('1,2,3'.split(',', maxsplit=1))
# ['1', '2,3']
print('1,2,,3,'.split(','))
# ['1', '2', '', '3', '']
'''-----------------------------------int/float-------------------------------------'''
"""
最大，最小number,有时候初始化我们需要设定Math.max()和Math.min()在python中分别以float('inf')和float('-inf')表示
"""
import sys
Max = float('inf')
Min = float('-inf')
print(Max, Min)
# 除法
"""
/ 会保留浮点，相当于float相除
// int 相除
"""
print(3 / 2)
print(3 // 2)
# 次方 **
print(2 ** 10)
'''-----------------------------------conditions-------------------------------------'''
"""
在python的三项表达式中ternary operation,于其他语言不太一样
"""
# res = a if condition else b
# 如果condition满足，那么res = a, 不然res = b,
# 在类C的语言里即为： res = condition ? a : b;
'''-----------------------------------any, all-------------------------------------'''
"""
any(), all()很好理解，就是字面意思，即参数中任何一个为true或者全部为true则返回true
经常可以秀一些骚操作：
比如 36. Valid Sudoku 这题：
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        row = [[x for x in y if x != '.'] for y in board]
        col = [[x for x in y if x != '.'] for y in zip(*board)]
        pal = [[board[i+m][j+n] for m in range(3) for n in range(3) if board[i+m][j+n] != '.'] for i in (0, 3, 6) for j in (0, 3, 6)]
        return all(len(set(x)) == len(x) for x in (*row, *col, *pal))
"""
'''-----------------------------------itertools-------------------------------------'''
# 这是python自带的迭代器库，有很多实用的，与遍历，迭代相关的函数
'''-----------------------------------permutations排列-------------------------------------'''
from itertools import permutations, combinations, groupby

print(list(permutations('ABCD', 2)))
'''-----------------------------------combinations组合-------------------------------------'''
print(list(combinations('ABCD', 2)))
'''-----------------------------------groupby合并-------------------------------------'''
print([k for k, g in groupby('AAAABBBCCDAABBB')])
# ['A', 'B', 'C', 'D', 'A', 'B']

print([list(g) for k, g in groupby('AAAABBBCCD')])
# [['A', 'A', 'A', 'A'], ['B', 'B', 'B'], ['C', 'C'], ['D']]
'''-----------------------------------functools-------------------------------------'''
"""
这个库里有很多高阶的函数，包括前面介绍到的cmp_to_key以及reduce,但是比较逆天的有lru_cache，
即least recently used cache,这个LRU Cache是一个常见的面试题，通常用hashmap和双向链表来实现，python居然内置了
用法即直接作为decorator 装饰在要cache的函数上，以变量值为key存储，当反复调用时直接返回计算过的值

https://leetcode.com/problems/stone-game-ii/discuss/345230/Python-DP-Solution

    def stoneGameII(self, A: List[int]) -> int:
        N = len(A)
        for i in range(N - 2, -1, -1):
            A[i] += A[i + 1]
        from functools import lru_cache
        @lru_cache(None)
        def dp(i, m):
            if i + 2 * m >= N: return A[i]
            return A[i] - min(dp(i + x, max(m, x)) for x in range(1, 2 * m + 1))
        return dp(0, 1)
"""
'''-----------------------------------LeetCode-------------------------------------'''
'''-----------------------------------1 Two Sum-------------------------------------'''
"""
Given an array of integers, return indices of the two numbers such that they add up to a specific target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

Example:

Given nums = [2, 7, 11, 15], target = 9,

Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].
"""
# class Solution1:
#     def twoSum1(self, nums: List[int], target: int) -> List[int]:
#         for first_index, num in enumerate(nums):
#             first = num
#             second = target - first
#             for second_index in range(first_index + 1, len(nums)):
#                 if nums[second_index] == second:
#                     return [first_index, second_index]
#                 
#     def twoSum2(self, nums: List[int], target: int) -> List[int]:
#         for first_index, num in enumerate(nums):
#             first = num
#             second = target - first
#             res = nums[first_index + 1:]
#             if second in res:
#                 return [first_index, first_index + 1 + res.index(second)]
# 
#     def twoSum3(self, nums: List[int], target: int) -> List[int]:
#         dict = {}
#         for i in range(len(nums)):
#             if target - nums[i] not in dict:
#                 dict[nums[i]] = i
#             else:
#                 return [dict[target - nums[i]], i]

'''-----------------------------------Test-------------------------------------'''
# nums = [6,5,4,3,2,1]
# print(nums.index(6))
'''-----------------------------------2 Add Two Numbers-------------------------------------'''
"""
You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order
 and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

Example:

Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
Explanation: 342 + 465 = 807.
"""

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
#
#
# class Solution2:
#     def addTwoNumbers2(self, l1: ListNode, l2: ListNode) -> ListNode:
#         output = ListNode(0)    #初始化一个node, value = 0
#         ans = output
#         sum = 0
#         while True:
#             if l1 != None:
#                 sum += l1.val
#                 l1 = l1.next
#             if l2 != None:
#                 sum += l2.val
#                 l2 = l2.next
#             ans.val = sum % 10 # not carry bit
#             tmp_1 = sum // 10  # carry bit
#             sum = tmp_1
#
#             if l1 == None and l2 == None and sum == 0:
#                 break
#
#             ans.next = ListNode(0) #指向下一个初始化的新的node  value = 0
#             ans = ans.next
#
#         return output
'''-----------------------------------3 Longest Substring Without Repeating Characters-------------------------------------'''

#
# class Solution3:
#     def lengthOfLongestSubstring(self, s: str) -> int:
#         d = {}
#         start = -1
#         max = 0
#         for i in range(len(s)):
#             if s[i] in d and d[s[i]] > start:
"""
思路理解：最长不重复子串在两个重复的字母之间选定，或者本身就是最大的如果没有重复字符
更新start的值的时候要保留之前的重复值，这样可以保证最大
更新字典的value值时，要用最新的index
"""
#                 start = d[s[i]]  # update start, don't use i, because you wanna keep ealier one
#                 d[s[i]] = i  # update key's value if repeat
#
#             else:
#                 d[s[i]] = i
#                 if i - start > max:
#                     max = i - start
#                 else:
#                     pass
#         return max
'''-----------------------------------7 Reverse Integer-------------------------------------'''
"""
Given a 32-bit signed integer, reverse digits of an integer.

Example 1:

Input: 123
Output: 321
Example 2:

Input: -123
Output: -321
Example 3:

Input: 120
Output: 21
Note:
Assume we are dealing with an environment which could only store integers within the 32-bit signed integer range: [−231,  231 − 1]. 
For the purpose of this problem, assume that your function returns 0 when the reversed integer overflows.
解题思路： 连续的整除，和连续的取余，判断界限和正负
"""
# class Solution7:
#     def reverse(self, x: int) -> int:
#         num = 0
#         a = abs(x)
#         while a != 0:
#             num = num * 10 + (a % 10)
#             a //= 10
#         if num > 2 ** 31 - 1 or num < -2 ** 31:
#             return 0
#         if x < 0:
#             return -1 * num
#         else:
#             return num
'''-----------------------------------9 Palindrome Number-------------------------------------'''
"""
Determine whether an integer is a palindrome. An integer is a palindrome when it reads the same backward as forward.

Example 1:

Input: 121
Output: true
Example 2:

Input: -121
Output: false
Explanation: From left to right, it reads -121. From right to left, it becomes 121-. Therefore it is not a palindrome.
Example 3:

Input: 10
Output: false
Explanation: Reads 01 from right to left. Therefore it is not a palindrome.
"""

# class Solution:
#     def isPalindrome(self, x: int) -> bool:
#         if x < 0:
#             return False
#         else:
#             num = 0
#             a = x
#             while a != 0:
#                 num = num * 10 + (a % 10)
#                 a //= 10
#             if num == x:
#                 return True
#             else:
#                 return False
'''-----------------------------------13 Roman to Integer-------------------------------------'''
"""
Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.

Symbol       Value
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
For example, two is written as II in Roman numeral, just two one's added together. Twelve is written as, XII, which is 
simply X + II. The number twenty seven is written as XXVII, which is XX + V + II.

Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII.
 Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle 
 applies to the number nine, which is written as IX. There are six instances where subtraction is used:

I can be placed before V (5) and X (10) to make 4 and 9. 
X can be placed before L (50) and C (100) to make 40 and 90. 
C can be placed before D (500) and M (1000) to make 400 and 900.
Given a roman numeral, convert it to an integer. Input is guaranteed to be within the range from 1 to 3999.
解题思路：如果后面的一直比前面的小，，就一直加就行了，一旦碰到后面的比前面的大，就要减去2次，也可以按照长度分类
"""

# class Solution13:
#     def romanToInt13(self, s: str) -> int:
#         dict = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
#         num = 0
#         if len(s) == 1:
#             return dict[s[0]]
#         else:
#             for i in range(len(s)):
#                 if i > 1:
#                     if dict[s[i - 1]] < dict[s[i]]:
#                         num = num - 2 * dict[s[i - 1]] + dict[s[i]]
#                     else:
#                         num = num + dict[s[i]]
#                 elif i > 0:
#                     if dict[s[i - 1]] < dict[s[i]]:
#                         num = num + dict[s[i]]
#                     else:
#                         num = num + dict[s[i]]
#
#                 else:
#                     if dict[s[i]] < dict[s[i + 1]]:
#                         num = num + -1 * dict[s[i]]
#                     else:
#                         num = num + dict[s[i]]
#         return num
#
#     def romanToInt(self, s: str) -> int:
#         dict = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
#         num = 0
#         for i in range(len(s)):
#             if i > 0 and dict[s[i]] > dict[s[i - 1]]:
#                 num = num + dict[s[i]] - 2 * dict[s[i - 1]]
#             else:
#                 num = num + dict[s[i]]
#         return num
'''-----------------------------------14 Longest Common Prefix-------------------------------------'''
"""
Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string "".

Example 1:

Input: ["flower","flow","flight"]
Output: "fl"
Example 2:

Input: ["dog","racecar","car"]
Output: ""
Explanation: There is no common prefix among the input strings.
Note:

All given inputs are in lowercase letters a-z.
"""


# class Solution14:
#     def longestCommonPrefix(self, strs: List[str]) -> str:
#         prefix = ""
#         if not strs:
#             return prefix
#         for i in range(len(strs[0])):
#             for string in strs[1:]:
#                 if i + 1 > len(string) or strs[0][i] != string[i]:
#                     return string[:i]
#         return strs[0]



