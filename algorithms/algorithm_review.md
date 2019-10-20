##Data Structure

### Satck and Queue

**互相转化**

两个stack实现queue，或者两个queue实现stack，要实现pop，插入。

到过去再到回来

**Min stack**

two  stack （优化 小于等于差 大雨不差）

**括号类问题**

一般用stack O(n)

**运算类问题**

stack O(n)

**单调斩**

性质：维持站的值单调递减或者单调递增

维持一个顺序又维持O（n) monoto stack

2 3 2 1 4 找下个比他大的第一个位置- 单调递减站

1 4 4 4 -1

Stack[2 3 2 2 1]

### Priority Queue/Heap

**实现方法**

swim sink

arraylist 来实现

merge K sorted list -> 动态的知道n个元素的最小值（怎么写）-》 用pq来找n个元素的最小值

看数据结构怎么implement

static type，dynamic type

**和Treeset treemap区别 维持了关系O(lgn)**

### HashSet and HashMap

**TreeSet TressMap HashSet HashMap**

java 实现eqaul和hashcode

**Two Sum** HashMap

**判断重复？？？？？** O(1) 

**连续性**

判断一个无序数组中最长的连续串多长，Hashset可以随机get一个数，往小往大get

判断

###String

Subtring Subarray Subsequence

**String+判断重复**

anagram

**Palindrom, Di 增**

DP，Stack

string中最长最短：dp 或者用sort维持order

**647 Palindrom subString**

对于i，和i+1前后extending

**Calcutor**

recursion + stack

KMP O(m+n) 字符串搜字符串

### List(Array) and Matrix

矩阵寻找

从左向右sort好的 从上到下sort好 找一个数从右上角开始找

找第K大的数  Binary Search

可以通过0-n-1index关系利用array操作

Split 从左往右走当前最大值等于index的话 cnt+1 0-n-769

wiggle sort 数组： 判断满不满足关系不满足就sort一下就好

**积分图和前缀和**

把结果作为value或者key存，index作为另一个

560 hashmap（前缀和，前缀和出现的次数）

### linkedlist

反转 206 while recur

 merge two linked list

sort list 1. 2. 快慢指针找中点 merge sort

### Tree

求树的高度（one line code recursion） 数是否平衡

最长路径 更新的global值且热突然不同的值

判断两个树是否相同，判断自己是不是对称（把前一个改一改）

**BST**

1. trim a bst，保留[a，b]， recursion 

2. 利用整体的关系 把所有的比自己本身大的val相加

3.  树上的two sum 转化为arraylist （没有别的方法）

99

### Trie 26叉树

### 线段树 online

### Graph

表示方法1. 2d array 连续空间 很不好申请，提前知道有几个节点

2. hashmap of hashset

1.dfs bfs

2. 染色法 
3. topological sort(判断条件入读为1

Dikstra : PQ bfs 中的queue换成pq

### LRU 146 背

 O(1)  online 求median，两个PQ，一个最大PQ，一个最小PQ

###线程安全

写一个算法或者algorithm怎么使得 某一部分线程安全

read write safe



##ALgorithms

###Greedy

1.interval

### Two pointers

1. 对一个array的双指针，对两个1-darray的双指针 

2. 异向双指针，完全平方和

3. 快慢指针

### Binary Search

自己看

### 排序

1. merge sort O(nlgn)

2. quick sort O(nlgn) / O($n^2$) - > quick selection -> 第k大的元素 average O(n)
3. 桶排序 O(n) 知道最大的和最小的->hashmap去group.

非online的order需要排序

### Search BFS DFS

backtracking、 数独、九皇后

### DP

###Divide Conquer

master theory 

给表达式加括号 241

### 数学

1. i式筛法  求1-n之间的所有素数 i从2开始依次删掉2的<n所有倍数（右界限 sqrt(N))
2. 两个数最大公因数,最小公倍数： 

```java
//最大公因数
int gcd(int a, int b) {
    return b == 0 ? a : gcd(b, a% b);
}

// 最小公倍数
int lcm(int a, int b) {
    return a * b / gcd(a, b);
}
```

3. 在O(n)的时间内找到中位数： quick selection

### bit operation

x ^ 0s = x      x & 0s = 0      x | 0s = x

x ^ 1s = ~x     x & 1s = x      x | 1s = 1s

x ^ x = 0       x & x = x       x | x = x

