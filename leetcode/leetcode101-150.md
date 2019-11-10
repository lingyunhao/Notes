### 102. Binary Tree Order Level Traversal

**Example:**

Given binary tree `[3,9,20,null,null,15,7]`

```
    3
   / \
  9  20
    /  \
   15   7
```

return its level order traversal as:

```
[
  [3],
  [9,20],
  [15,7]
]
```

**Solution:**

```java
public List<List<Integer>> levelOrder(TreeNode root) {
    List<List<Integer>> results = new ArrayList<>();
    if(root == null) return results;

    Queue<TreeNode> queue = new LinkedList<>();
    queue.offer(root);

    while(!queue.isEmpty()) {
        int size = queue.size();
        List<Integer> level = new ArrayList<>();
        for(int i=0; i<size; i++) {
            TreeNode node = queue.poll();
            level.add(node.val);
            if(node.left != null) queue.offer(node.left);
            if(node.right != null) queue.offer(node.right);
        }
        results.add(level);
    }
    return results;
}
```



### 107. Binary Tree Level Order Traversal II

**Example:**

Given binary tree `[3,9,20,null,null,15,7]`,

```
    3
   / \
  9  20
    /  \
   15   7
```

return its bottom-up level order traversal as:

```
[
  [15,7],
  [9,20],
  [3]
]
```

**Solution:**

```java
public List<List<Integer>> levelOrderBottom(TreeNode root) {
    List<List<Integer>> results = new ArrayList<>();
    if(root == null) return results;
    Queue<TreeNode> queue = new LinkedList<>();
    queue.offer(root);
    while(!queue.isEmpty()) {
        int size = queue.size();
        List<Integer> level = new ArrayList<>();
        for(int i=0; i<size; i++) {
            TreeNode node = queue.poll();
            if(node.left != null) queue.add(node.left);
            if(node.right != null) queue.add(node.right);
            level.add(node.val);
        }
        results.add(level);
    }
    Collections.reverse(results);
    return results;
}
```



### 119. Pascal's Triangle II

Given a non-negative index *k* where *k* ≤ 33, return the *k*th index row of the Pascal's triangle.

Note that the row index starts from 0. Could you optimize your algorithm to use only *O*(*k*) extra space?

![img](https://upload.wikimedia.org/wikipedia/commons/0/0d/PascalTriangleAnimated2.gif)
In Pascal's triangle, each number is the sum of the two numbers directly above it.

**Example:**

```
Input: 3
Output: [1,3,3,1]
```

**Solution:**

本题是二维的数组，需要两个for loop，需要压缩extra space的题，一般需要在当前的答案上进行修改，为此一般需要两个额外变量：

1. 存当前的状态 (get(i))
2. 改变当前的状态 (set(i))
3. prev = cur

```java
public List<Integer> getRow(int rowIndex) {
    List<Integer> list = new ArrayList<>();
    list.add(1);
    if (rowIndex == 0) return list;
    int prev = 1;
    for (int i = 1; i <= rowIndex; i++) {
        for (int j = 1; j <= i; j++) {
            if (j == i) {
                list.add(1);
            } else {
                int cur = list.get(j);
                list.set(j, prev + cur);
                prev = cur;
            }
        }
    }
    return list;
}
```



### 120. Triangle

Given a triangle, find the minimum path sum from top to bottom. Each step you may move to adjacent numbers on the row below.

**Example:**

```
[
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]
```

The minimum path sum from top to bottom is `11` (i.e., **2** + **3** + **5**+ **1** = 11).

**Solution 1:**

DP bottom up with O($n^2$) time complexity and exrta O($n^2$) space.

自底向上的DP, 开一个NN的二维数组, 存的是从(i,j)出发走到最底层的最小路径，先初始化最后一层为其本身，两层for循环遍历前n-1层，每个节点的值depend on (i+1,j) 和 (i+1,j+1) 两个节点，取其最小值再加上本身即为从(i,j)出发到bottom的最短路径，直到求到(0,0)为止， return(0,0) (从(0,0)出发到bottom 的最小值)。

```java
public int minimumTotal(List<List<Integer>> triangle) {
    int n = triangle.size();

    // Record the minimum path from (i,j) to the bottom
    int dp[][] = new int[n][n];

    // Initialize the bottom
    for(int i = 0; i < n; ++i) {
        dp[n-1][i] = triangle.get(n-1).get(i); 
    }

    // DP function
    for(int i = n - 2; i >= 0; --i) {
        for(int j = 0; j <= i; ++j) {
            dp[i][j] = Math.min(dp[i+1][j], dp[i+1][j+1]) + triangle.get(i).get(j);
        }
    }

    return dp[0][0];
}
```

初始化也可以放在两层赋值for循环中，加个if语句即可。

```java
public int minimumTotal(List<List<Integer>> triangle) {
    int n = triangle.size();

    // Record the minimum path from (i,j) to the bottom
    int dp[][] = new int[n][n];

    // DP function
    for(int i = n - 1; i >= 0; --i) {
        for(int j = 0; j <= i; ++j) {
            // Initialize
            if(i == n - 1) {
                dp[i][j] = triangle.get(i).get(j);
                continue;
            }
            dp[i][j] = Math.min(dp[i+1][j], dp[i+1][j+1]) + triangle.get(i).get(j);
        }
    }

    return dp[0][0];
}
```

**Solution 2:**

DP bottom up with O($n^2$) time complexity and exrta O($n^2$) space.

与Solution 1不同的是，dp(i,j) represents the minimum path from (0,0) to (i,j). 赋值时需要取两个前继节点的最小值取其本身，三角形左边右边只有一个前继节点，需要对三角形左边(i,0)右边(i,i)初始化为其唯一的前继节点+其本身。先初始化顶点，然后左边后边，初始化可以在外边也可以在两层for循环内加if语句完成。最后在最底层打擂台得到状态矩阵最底层的最小值。

```java
public int minimumTotal(List<List<Integer>> triangle) {
    int n = triangle.size();

    // Record the minimum path from (i,j) to the bottom
    int dp[][] = new int[n][n];

    // Initialize
    dp[0][0] = triangle.get(0).get(0);
    for(int i = 1; i < n; ++i) {
        dp[i][0] = dp[i-1][0] + triangle.get(i).get(0);
        dp[i][i] = dp[i-1][i-1] + triangle.get(i).get(i);
    }

    // DP function
    for(int i = 1; i < n; ++i) {
        for(int j = 1; j < i; ++j) {
            dp[i][j] = Math.min(dp[i-1][j-1], dp[i-1][j]) + triangle.get(i).get(j);
        }
    }

    int result = Integer.MAX_VALUE;
    for(int i = 0; i < n; ++i) {
        result = Math.min(dp[n-1][i], result);
    }

    return result;
}
```

**Solution 3:**

将上面的算法优化到O(n)extra space. 实际上可以只开一个2 * n的矩阵，只保留计算当前这一层需要的前一层和此层，和n * n算法上并没有什么需别，而开1 * n的矩阵就需要考虑到谁先更新的问题。

**Solution 4：**

DP with no extra space with bottom up.

不开新的矩阵，直接在输入上进行操作，bottom up，不需要初始化最底层。



### 121. Best Time to Buy and Sell Stock

Say you have an array for which the *i*th element is the price of a given stock on day *i*.

**Example:**

```
Input: [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
             Not 7-1 = 6, as selling price needs to be larger than buying price.
```

**Solution:**

保持一个maxDiff 和一个min，遍历数组，min records the minmum number up to the current position, and maxDiff records the max difference up to now.

Note : 1. Should update maxDiff first then min.

2. initiate maxDiff as 0, not Integer.MAX_VALUE to avoid the corner case array is in descending order, [7,5,4,3,1]. In this case, we should return 0, not a negative number.

```java
public int maxProfit(int[] prices) {
    if (prices == null || prices.length == 0 || prices.length == 1) return 0;
    int min = prices[0];
    int maxDiff = 0;

    for (int i = 1; i < prices.length; i++) {
        maxDiff = Math.max(maxDiff, prices[i] - min);
        min = Math.min(min, prices[i]);
    }

    return maxDiff;
}
```

### 130. Surrounded Regions

Given a 2D board containing `'X'` and `'O'` (**the letter O**), capture all regions surrounded by `'X'`.

A region is captured by flipping all `'O'`s into `'X'`s in that surrounded region.

**Example:**

```
X X X X
X O O X
X X O X
X O X X
```

**Solution:**

先用dfs把最外围的联通量改成另一个字母'T',然后把里边的'O'全改为'X',最后把'T'全部改回来。

```java
public void solve(char[][] board) {
    if (board == null || board.length == 0) return;
    int m = board.length, n = board[0].length;
    for (int i = 0; i < m; ++i) {
        dfs(board, i, 0, m, n);
        dfs(board, i, n-1, m, n);
    }
    for (int i = 0; i < n; ++i) {
        dfs(board, 0, i, m, n);
        dfs(board, m-1, i, m, n);
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (board[i][j] == 'O') board[i][j] = 'X';
            if (board[i][j] == 'T') board[i][j] = 'O';
        }
    }
}

public void dfs(char[][] board, int r, int c, int m, int n) {
    if (r < 0 || r >= m || c < 0 || c >= n ||  board[r][c] != 'O') return;
    board[r][c] = 'T';
    dfs(board, r+1, c, m, n);
    dfs(board, r-1, c, m, n);
    dfs(board, r, c+1, m, n);
    dfs(board, r, c-1, m, n);
}
```

### 139. Word Break

Given a **non-empty** string *s* and a dictionary *wordDict* containing a list of **non-empty** words, determine if *s* can be segmented into a space-separated sequence of one or more dictionary words.

**Solution: dp**

```java
public boolean wordBreak(String s, List<String> wordDict) {
    int n = s.length();
    boolean[] dp = new boolean[n+1];
    dp[0] = true;
    for (int i = 1; i <= n; ++i) {
        for (String word : wordDict) {
            int len = word.length();
            if (i < len) continue;
            if (s.substring(i - len, i).equals(word)) dp[i] = dp[i] || dp[i-len];
        }
    }
    return dp[n];
}
```

### 146. LRU Cache

Design and implement a data structure for [Least Recently Used (LRU) cache](https://en.wikipedia.org/wiki/Cache_replacement_policies#LRU). It should support the following operations: `get` and `put`.

`get(key)` - Get the value (will always be positive) of the key if the key exists in the cache, otherwise return -1.
`put(key, value)` - Set or insert the value if the key is not already present. When the cache reached its capacity, it should invalidate the least recently used item before inserting a new item.

The cache is initialized with a **positive** capacity.

**Follow up:**
Could you do both operations in **O(1)** time complexity?

**Example:**

```
LRUCache cache = new LRUCache( 2 /* capacity */ );

cache.put(1, 1);
cache.put(2, 2);
cache.get(1);       // returns 1
cache.put(3, 3);    // evicts key 2
cache.get(2);       // returns -1 (not found)
cache.put(4, 4);    // evicts key 1
cache.get(1);       // returns -1 (not found)
cache.get(3);       // returns 3
cache.get(4);       // returns 4
```

#### Approach 1: Ordered dictionary

**Intuition**

We're asked to implement [the structure](https://en.wikipedia.org/wiki/Cache_replacement_policies#LRU) which provides the following operations in \mathcal{O}(1)O(1) time :

- Get the key / Check if the key exists
- Put the key
- Delete the first added key

The first two operations in \mathcal{O}(1)O(1) time are provided by the standard hashmap, and the last one - by linked list.

> There is a structure called *ordered dictionary*, it combines behind both hashmap and linked list. In Python this structure is called [*OrderedDict*](https://docs.python.org/3/library/collections.html#collections.OrderedDict) and in Java [*LinkedHashMap*](https://docs.oracle.com/javase/8/docs/api/java/util/LinkedHashMap.html).

```java
class LRUCache extends LinkedHashMap<Integer, Integer>{
    private int capacity;
    
    public LRUCache(int capacity) {
        super(capacity, 0.75F, true);
        this.capacity = capacity;
    }

    public int get(int key) {
        return super.getOrDefault(key, -1);
    }

    public void put(int key, int value) {
        super.put(key, value);
    }

    @Override
    protected boolean removeEldestEntry(Map.Entry<Integer, Integer> eldest) {
        return size() > capacity; 
    }
}
```

#### Approach 2: Hashmap + DoubleLinkedList

**Intuition**

This Java solution is an extended version of the [the article published on the Discuss forum](https://leetcode.com/problems/lru-cache/discuss/45911/Java-Hashtable-%2B-Double-linked-list-(with-a-touch-of-pseudo-nodes)).

The problem can be solved with a hashmap that keeps track of the keys and its values in the double linked list. That results in \mathcal{O}(1)O(1) time for `put` and `get` operations and allows to remove the first added node in \mathcal{O}(1)O(1) time as well.

![compute](https://leetcode.com/problems/lru-cache/Figures/146/structure.png)

One advantage of *double* linked list is that the node can remove itself without other reference. In addition, it takes constant time to add and remove nodes from the head or tail.

One particularity about the double linked list implemented here is that there are *pseudo head* and *pseudo tail* to mark the boundary, so that we don't need to check the `null` node during the update.

![compute](https://leetcode.com/problems/lru-cache/Figures/146/new_node.png)

**Implementation**

```java
public class LRUCache {

  class DLinkedNode {
    int key;
    int value;
    DLinkedNode prev;
    DLinkedNode next;
  }

  private void addNode(DLinkedNode node) {
    /**
     * Always add the new node right after head.
     */
    node.prev = head;
    node.next = head.next;

    head.next.prev = node;
    head.next = node;
  }

  private void removeNode(DLinkedNode node){
    /**
     * Remove an existing node from the linked list.
     */
    DLinkedNode prev = node.prev;
    DLinkedNode next = node.next;

    prev.next = next;
    next.prev = prev;
  }

  private void moveToHead(DLinkedNode node){
    /**
     * Move certain node in between to the head.
     */
    removeNode(node);
    addNode(node);
  }

  private DLinkedNode popTail() {
    /**
     * Pop the current tail.
     */
    DLinkedNode res = tail.prev;
    removeNode(res);
    return res;
  }

  private Map<Integer, DLinkedNode> cache = new HashMap<>();
  private int size;
  private int capacity;
  private DLinkedNode head, tail;

  public LRUCache(int capacity) {
    this.size = 0;
    this.capacity = capacity;

    head = new DLinkedNode();
    // head.prev = null;

    tail = new DLinkedNode();
    // tail.next = null;

    head.next = tail;
    tail.prev = head;
  }

  public int get(int key) {
    DLinkedNode node = cache.get(key);
    if (node == null) return -1;

    // move the accessed node to the head;
    moveToHead(node);

    return node.value;
  }

  public void put(int key, int value) {
    DLinkedNode node = cache.get(key);

    if(node == null) {
      DLinkedNode newNode = new DLinkedNode();
      newNode.key = key;
      newNode.value = value;

      cache.put(key, newNode);
      addNode(newNode);

      ++size;

      if(size > capacity) {
        // pop the tail
        DLinkedNode tail = popTail();
        cache.remove(tail.key);
        --size;
      }
    } else {
      // update the value.
      node.value = value;
      moveToHead(node);
    }
  }
}
```

### 