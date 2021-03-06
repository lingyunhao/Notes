## Java 集合框架底层数据结构总结

### Collection

1. List
   * ArrayList: Object 数组
   * Vector: Object 数组
   * LinkedList：双向链表

2. Set
   * HashSet：无序，无重复，基于HashMap实现，底层使用HashMap保存元素
   * LinkedHashSet: 继承HashSet,内部通过LinkedHashMap实现
   * TreeSet：有序，无重复，红黑树（自平衡排序二叉树）

3. Map
   * HashMap
   * LinkedHashMap：实现了访问顺序
   * HashTable
   * TreeMap：红黑树

## Java 集合框架常见问题

### List 的遍历方式选择

* 实现了RandomAccess 接口的list，优先选择普通for循环，其次for each
* 未实现 RandomAccess 接口的list，优先选择iterator遍历 (for each遍历底层也是通过iterator实现的)
* 大size的数据，不要使用普通 for 循环

### 双向链表和双向循环链表

* 双向链表：prev指向前一个节点，next指向后一个节点
* 双向循环链表：双向链表的head 的 prev指向tail，tail的next指向head

### ArrayList VS Vector

Vector 所有方法都是同步的，可以由两个线程安全的访问同一个Vector对象。**但是一个线程访问Vector的话，代码要在同步操作上耗费大量的时间。**

ArrayList 不同步，不需要保证线程安全时使用ArrayList。

### HashMap VS HashTable

1. HashMap 线程非安全，HashTable 线程安全(synchronized修饰) (需要线程安全的话可以使用 ConcurrentHashMap)
2. HashMap 效率高一点(不需要线程安全)。HashTable 基本被淘汰（不要使用)。
3. HashMap null可以作为key，但只有一个，value也可以是null；HashTable 不可以put null key。
4. 初始容量和每次扩充容量大小的不同。①创建时如果不指定容量初始值，Hashtable 默认的初始大小为11，之后每次扩充，容量变为原来的2n+1。HashMap 默认的初始化大小为16。之后每次扩充，容量变为原来的2倍。②创建时如果给定了容量初始值，那么 Hashtable 会直接使用你给定的大小，而 HashMap 会将其扩充为2的幂次方大小，
5. **底层数据结构：** JDK1.8 以后的 HashMap 在解决哈希冲突时有了较大的变化，当链表长度大于阈值（默认为8）时，将链表转化为红黑树，以减少搜索时间。Hashtable 没有这样的机制。

### HashSet 如何检查重复

当你把对象加入`HashSet`时，HashSet会先计算对象的`hashcode`值来判断对象加入的位置，同时也会与其他加入的对象的hashcode值作比较，如果没有相符的hashcode，HashSet会假设对象没有重复出现。但是如果发现有相同hashcode值的对象，这时会调用`equals（）`方法来检查hashcode相等的对象是否真的相同。如果两者相同，HashSet就不会让加入操作成功。（摘自我的Java启蒙书《Head fist java》第二版）

当你把对象加入`HashSet`时，HashSet会先计算对象的`hashcode`值来判断对象加入的位置，同时也会与其他加入的对象的hashcode值作比较，如果没有相符的hashcode，HashSet会假设对象没有重复出现。但是如果发现有相同hashcode值的对象，这时会调用`equals（）`方法来检查hashcode相等的对象是否真的相同。如果两者相同，HashSet就不会让加入操作成功。（摘自我的Java启蒙书《Head fist java》第二版）

**hashCode（）与equals（）的相关规定：**

1. 如果两个对象相等，则hashcode一定也是相同的
2. 两个对象相等,对两个equals方法返回true
3. 两个对象有相同的hashcode值，它们也不一定是相等的
4. 综上，equals方法被覆盖过，则hashCode方法也必须被覆盖
5. hashCode()的默认行为是对堆上的对象产生独特值。如果没有重写hashCode()，则该class的两个对象无论如何都不会相等（即使这两个对象指向相同的数据）。

**==与equals的区别**

1. ==是判断两个变量或实例是不是指向同一个内存空间 equals是判断两个变量或实例所指向的内存空间的值是不是相同
2. ==是指对内存地址进行比较 equals()是对字符串的内容进行比较
3. ==指引用是否相同 equals()指的是值是否相同

### HashMap 长度是2的幂次方

为了能让 HashMap 存取高效，尽量较少碰撞，也就是要尽量把数据分配均匀。我们上面也讲到了过了，Hash 值的范围值-2147483648到2147483647，前后加起来大概40亿的映射空间，只要哈希函数映射得比较均匀松散，一般应用是很难出现碰撞的。但问题是一个40亿长度的数组，内存是放不下的。所以这个散列值是不能直接拿来用的。用之前还要先做对数组的长度取模运算，得到的余数才能用来要存放的位置也就是对应的数组下标。这个数组下标的计算方法是“ `(n - 1) & hash`”。（n代表数组长度）。这也就解释了 HashMap 的长度为什么是2的幂次方。

**这个算法应该如何设计呢？**

我们首先可能会想到采用%取余的操作来实现。但是，重点来了：**“取余(%)操作中如果除数是2的幂次则等价于与其除数减一的与(&)操作（也就是说 hash%length==hash&(length-1)的前提是 length 是2的 n 次方；）。”** 并且 **采用二进制位操作 &，相对于%能够提高运算效率，这就解释了 HashMap 的长度为什么是2的幂次方。**

### ConcurrentHashMap VS HashTable

ConcurrentHashMap 和 HashTable 的底层数据结构类似，区别主要是实现线程安全的方式不同。

* **底层数据结构**：JDK1.7 的ConcurrentHashMap底层采用分段的数组+链表实现，JDK1.8 采用数组+链表/红黑二叉树。HashTable 的底层结构是数组+链表。

* **实现线程安全**：

  * **在JDK1.7的时候，ConcurrentHashMap（分段锁）** 对整个桶数组进行了分割分段(Segment)，每一把锁只锁容器其中一部分数据，多线程访问容器里不同数据段的数据，就不会存在锁竞争，提高并发访问率。一个 ConcurrentHashMap 里包含一个 Segment 数组。Segment 的结构和HashMap类似，是一种数组和链表结构，一个 Segment 包含一个 HashEntry 数组，每个 HashEntry 是一个链表结构的元素，每个 Segment 守护着一个HashEntry数组里的元素，当对 HashEntry 数组的数据进行修改时，必须首先获得对应的 Segment的锁。

     **到了 JDK1.8 的时候已经摒弃了Segment的概念，而是直接用 Node 数组+链表+红黑树的数据结构来实现，并发控制使用 synchronized 和 CAS 来操作。（JDK1.6以后 对 synchronized锁做了很多优化）** 整个看起来就像是优化过且线程安全的 HashMap，虽然在JDK1.8中还能看到 Segment 的数据结构，但是已经简化了属性，只是为了兼容旧版本。Java 8在链表长度超过一定阈值（8）时将链表（寻址时间复杂度为O(N)）转换为红黑树（寻址时间复杂度为O(log(N))）

    synchronized只锁定当前链表或红黑二叉树的首节点，这样只要hash不冲突，就不会产生并发，效率又提升N倍。

  * **Hashtable(同一把锁)** :使用 synchronized 来保证线程安全，效率非常低下。当一个线程访问同步方法时，其他线程也访问同步方法，可能会进入阻塞或轮询状态，如使用 put 添加元素，另一个线程不能使用 put 添加元素，也不能使用 get，竞争会越来越激烈效率越低。

### Comparable VS Comparator

- comparable接口出自 java.lang 包，通过`compareTo(Object obj)`方法排序
- comparator接口出自 java.util 包，通过`compare(Object obj1, Object obj2)`方法排序

当我们需要对一个集合使用自定义排序时， 我们要重写`compareTo()` `compare()`方法。或者当我们需要对一个集合实现两种排序方式时，比如一个Student中的Score和Name分别采用一种排序方法来排序，我们可以重写`compareTo()`方法和使用自制的Comparator方法或者以两个Comparator来实现歌名排序和歌星名排序，第二种代表我们只能使用两个参数版的 `Collections.sort()`。

