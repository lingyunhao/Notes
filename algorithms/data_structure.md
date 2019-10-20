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

### Linked List

基本就三个考点：

1. Merge two linked lists
2. Reverse linked list
3. Slow fast pointers to find the middle

###树的操作

一般会问到关于root，或者从root出发的。基本是用recursion，traversal，dfs，bfs。一般会问到关于root，或者从root出发的。

如果说任意一个结点（比如一个结点不一定是root到任意子结点的最长路径），因为不能确定parent结点是谁，所以需要把树转化成图进行操作

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

https://wiki.jikexueyuan.com/project/java-concurrent/read-write-locks-in-java.html

https://blog.csdn.net/u011877584/article/details/78339128