## Data_Structure

###常见的interface: List, Set, Map

**List-> ArrayList LinkedList**

ArrayList 没有固定的size，可以实现 random access, because arraylist is stored in a continious memory, we can access to any element by increasing the address of the first element.

LinkedList: java 中是doubled linked list, 在head, tail的增删操作都是O(1)的。

**Set-> HashSet TreeSet**

HashSet 存的是key，通过hash function 转化成hashcode。可以实现O(1)的增添、删除、查找。

TreeSet 的key维持了一定的order，是由红黑树实现的，是一种自平衡二叉树。增添、删除、查找都是O(lgn)的，所以有时会把array的元素放入Treeset，然后遍历输出就是有序的。(Time Complexity: O(nlgn))

**Map-> HashMap TreeMap**

Map 相比于Set来说存的是(key value) pair。别的没什么区别。

**Queue**

是个abstract class，需要用linked list 实现。需要first in first out。头部尾部操作都是O(1),所以一般用linkedlist实现。在头部删除，尾部增加的题要想到用queue。

Queue<T> queue = new LinkedList<>();

如果看到有的人用的是arraylist，arraylist是不能在O(1)时间头部删除的，可能并不需要删除，只要一直往后面加。

**Stack**

是已经被实现的类，java中由vector实现（vector和arraylist基本相同）。(stack只需要在尾部进行操作，所以用arraylist就足以，也可以直接用stack)。

Stack<T> stack = new Stack<>();

入stack: add, push

出stack: pop

**Deque**

可以实现两头的O(1)增删，但是没办法在中间删除。一般用ArrayDeque或者LinkedList实现。

**BST(Binary Search Tree)**

left<root<right。高度最坏为n。查找、增添、删除 O(n)。 

删除，找到需要删除的点是O(n), 然后把该节点删除，把左右子结点随便拿一个上来。（仍然满足原来的条件）（高度没有要求）

增添，同样找到需要添加的位置是O(n),最后找到的一定是个null,虽然这不是唯一可以插入的位置。

对BST做inorder travesal, we could get the ordered array。

**TreeMap, TreeSet VS Heap**

TreeMap,TreeSet 是由Balanced BST(Binary Search Tree)(red black tree)实现的。不仅可以得到最大值，最小值，还可以得到大于某个数所有值，小于某个数所有值，时间复杂度都是O(logn)。

Heap 的性质是左右子结点都比root小(最小堆)，或者都比root大(最大堆)。可以在O(1)时间在找到最小值或者最大值，如果要增添数进去，或者pop顶点(min,max)出去，都是O(logn)。需要swim或者sink。PriorityQueue是由heap实现能够找到最小值的一种结构，也可以override compare使得其实现找到最大值。

### Satck and Queue

**Stack and Queue互相转化**

两个stack实现queue，或者两个queue实现stack，要实现pop，pushO(1)操作。

到过去再到回来

**Min stack**

two  stack （优化 新的元素小于等于min stack peek 则push到min stack中,  大于不push进去）

**括号类问题**

一般用stack O(n)

**运算类问题**

stack O(n)

**单调栈**

性质：维持栈内的值单调递减或者单调递增

维持一个顺序又维持O(n) ->monoto stack

example: 找下个比他大的第一个位置- 单调递减

input: 2 3 2 1 4

output: 1 4 4 4 -1(index)

用一个单调栈，栈里边存index，遍历一遍input，遇到比栈顶的元素(index)相对应的位置元素小的，加入栈，否则，把stack中元素pop出来，pop到栈中元素比当前元素大于等于为止，在result的相应位置上写上新的元素的index。整个过程一直在维护单调递减栈。

单调递减栈 peek是最小的那个。

### Priority Queue vs Heap

**实现方法**

swim sink

arraylist 来实现 (根据index找到parent和child)

merge K sorted list -> 动态的知道n个元素的最小值（怎么写）-> 用pq来找n个元素的最小值

看数据结构怎么implement

static type，dynamic type

**和Treeset treemap区别 维持了关系O(lgn)**

### HashSet and HashMap

**TreeSet TreeMap HashSet HashMap**

java 实现eqaul和hashcode

**Two Sum** HashMap

**连续性**

判断一个无序数组中最长的连续串多长，Hashset可以随机get一个数，往小往大get

### String

Subtring Subarray Subsequence

**String+判断重复**

anagram

**Palindrom, 递增**

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

### Tree

求树的高度（one line code recursion） 数是否平衡

最长路径 更新的global值且热突然不同的值

判断两个树是否相同，判断自己是不是对称（把前一个改一改）

**BST**

1. trim a bst，保留[a，b]， recursion 
2. 利用整体的关系 把所有的比自己本身大的val相加
3. 树上的two sum 转化为arraylist （没有别的方法）

leetcode 99

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

### Linked List

基本就三个考点：

1. Merge two linked lists
2. Reverse linked list : 206 while or recur
3. Slow fast pointers to find the middle

merge two linked list

sort list  快慢指针找中点 merge sort

###树的操作

一般会问到关于root，或者从root出发的。基本是用recursion，traversal，dfs，bfs。一般会问到关于root，或者从root出发的。

如果说任意一个结点（比如一个结点不一定是root到任意子结点的最长路径），因为不能确定parent结点是谁，所以需要把树转化成图进行操作

### Map Iterator

```java
// 不改变map时
// using for-each loop for iteration over Map.entrySet() 
for (Map.Entry<String,String> entry : gfg.entrySet())  
  System.out.println("Key = " + entry.getKey() + 
                     ", Value = " + entry.getValue());

// using keySet() for iteration over keys 
for (String name : gfg.keySet())  
  System.out.println("key: " + name); 


// using values() for iteration over keys 
for (String url : gfg.values())  
  System.out.println("value: " + url); 

// 需要改变map的话用iterator
// using iterators 
Iterator<Map.Entry<String, String>> itr = gfg.entrySet().iterator(); 

while(itr.hasNext()) 
{ 
  Map.Entry<String, String> entry = itr.next(); 
  System.out.println("Key = " + entry.getKey() +  
                     ", Value = " + entry.getValue()); 
} 
```

### Priority Queue Override

```java
Comparator<String> stringLengthComparator = new Comparator<String>() {
            @Override
            public int compare(String s1, String s2) {
                return s1.length() - s2.length();
            }
        };

        /*
        The above Comparator can also be created using lambda expression like this =>
        Comparator<String> stringLengthComparator = (s1, s2) -> {
            return s1.length() - s2.length();
        };

        Which can be shortened even further like this =>
        Comparator<String> stringLengthComparator = Comparator.comparingInt(String::length);
        */

        // Create a Priority Queue with a custom Comparator
        PriorityQueue<String> namePriorityQueue = new PriorityQueue<>(stringLengthComparator);
```



###Thread Safe

ArrayList和Vector有什么区别？HashMap和HashTable有什么区别？StringBuilder和StringBuffer有什么区别？这些都是Java面试中常见的基础问题。面对这样的问题，回答是：ArrayList是非线程安全的，Vector是线程安全的；HashMap是非线程安全的，HashTable是线程安全的；StringBuilder是非线程安全的，StringBuffer是线程安全的。

**1）synchronized**

- Java提供这个关键字，为防止资源冲突提供的内置支持。当任务执行到被synchronized保护的代码片段的时候，它检查锁是否可用，然后获取锁，执行代码，释放锁。
- 常用这个关键字可以修饰成员方法和代码块

**2）读写锁**

我们对数据的操作无非两种：“读”和“写”，试想一个这样的情景，当十个线程同时读取某个数据时，这个操作应不应该加同步。答案是没必要的。只有以下两种情况需要加同步：

- 这十个线程对这个公共数据既有读又有写
- 这十个线程对公共数据进行写操作
- **以上两点归结起来就一点就是有对数据进行改变的操作就需要同步**

java5提供了读写锁

这种锁支持多线程读操作不互斥，多线程读写互斥，多线程写写互斥。

References:

https://wiki.jikexueyuan.com/project/java-concurrent/read-write-locks-in-java.html

https://blog.csdn.net/u011877584/article/details/78339128