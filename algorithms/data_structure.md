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