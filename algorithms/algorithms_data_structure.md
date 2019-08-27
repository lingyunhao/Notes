# Algorithms

### Time Complexity

O(1) 极少

O(logn) 几乎都是binary search

O($\sqrt{n}$) 几乎都是分解质因数，少见

O(n) 高频（遍历一个规模为n的数据，对每个数据进行O(1)的操作）

O(nlogn) 一般都可能要排序 （遍历一个规模为n的数据，对每个数据进行O(logn)的操作）

O($n^2$) 枚举、数组、动态规划 （遍历一个规模为n的数据，对每个数据进行O(n)的操作）

O($n^3$) 枚举、数组、动态规划

O($2^n$) 与组合有关的搜索

O(n!) 与排列有关的搜索

### Binary Search

**Time Complexity : O(logn)**  

Binary Search 是通过O(1)的时间，将规模为n的问题变为规模为n/2的问题。例如通过一个if判断去掉一半的不可能的答案。 这类算法的时间复杂度是O(logn)。

T(n) = T(n/2) + O(1) = (T(n/4) + O(1)) + O(1) = T(8/n) + 3*O(1) = … = T(n/n) + logn * O(1) = T(1) + O(logn) = O(logn) (省略了以2为底，底数都可以提到log外面作为系数，所以都一样)

若面试中用了O(n)的解法，仍然需要优化就很有可能是二分。比O(n)更好的解法就是O(logn)。根据时间复杂度倒推算法。

**Binary Search 的三种境界**

1. Given a sorted integer array - noms, and an integer - target, find any/first/last postion of the target
2. 给一个数组满足前半段和后半段在某一点不同，找第一个或者最后一个满足某种条件的位置 OOOOOXXXXXXXX， 这种情况判断结果只有两种，是O或者不是O，最后出了循环再去判断一下left，right指针，return相应值去避免corner case
3. 可能无法找到某个条件使得前半段和后半段不同，但二分的本质是每次去掉无解的一半保留有解的一半

**二分模版：**

left + 1 < right  : 最终结束循环时，left 和 right 相邻且在 [0,nums.length-1]区间内，这种情况下，left，right =mid 和 left = mid +1，right = mid - 1是一样的。求first，last positoin时做相应的改变即可。 比如求last position 时，当 nums[mid] == target时，left 不能 mid+1，否则后面一旦没有nums[mid] == target 就可能错过了这个值，所以left = mid, 因为 mid = left + (right - left) / 2, 是左倾的，可能算出来的mid 仍然等于left，就造成了死循环。用left+1<right就会避免死循环。 

while循环去逼近target，在循环外面判断left，right是否等于target。

```java
// normal 
public int search(int[] nums, int target) {
    if (nums == null || nums.length == 0) return -1;
    int left = 0, right = nums.length - 1; // 左闭右闭
    while (left + 1 < right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) {
            right = mid;
        } else if (nums[mid] > target) {
            right = mid;
        } else {
            left = mid;
        }
    }
    if (nums[left] == target) return left;
    if (nums[right] == target) return right;
    return -1;
}
```

```java
// find the first position of target, if doesn't exist return -1
public int first_postion(int[] nums, int target) {
    if (nums == null || nums.length == 0) return -1;
    int left = 0, right = nums.length - 1; 
    while (left + 1 < right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) {
            right = mid;
        } else if (nums[mid] > target) {
            right = mid;
        } else {
            left = mid;
        }
    }
    if (nums[left] == target) return left;
    if (nums[right] == target) return right;
    return -1;
}
```

```java
// find the last position of target, if doesn't exist return -1
public int last_position(int[] nums, int target) {
    if (nums == null || nums.length == 0) return -1;
    int left = 0, right = nums.length - 1;
    while (left + 1 < right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) {
            left = mid;
        } else if (nums[mid] > target) {
            right = mid;
        } else {
            left = mid;
        }
    }
    if (nums[right] == target) return right;
    if (nums[left] == target) return left;
    return -1;
}
```



**Examples:**

**704. Binary Search**

Given a **sorted** (in ascending order) integer array `nums` of `n`elements and a `target` value, write a function to search `target` in `nums`. If `target` exists, then return its index, otherwise return `-1`.

**Solution**:

（二分模版）使用while循环去逼近target，循环中没有return，将target范围缩小在left，right两个范围内。出了循环之后再进行判断，本题没有重复所以先判断left，right都可以。注意在循环中nums[mid] == target的情况必须把left或者right置为mid，不能mid+1/mid-1，否则就会miss掉这个答案。而实际上对于left+1<right的判断条件，把left，right置为mid-1，mid+1和mid是完全没有区别的。

```java
public int search(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    while(left + 1 < right) {
        int mid = left + (right - left)/2;
        if(nums[mid] > target) {
            right = mid;
        } else {
            left = mid;
        }
    }
    if(nums[left] == target) return left;
    if(nums[right] == target) return right;
    return -1;
}
```



**34. Find First and Last Position of Element in Sorted Array**

Given an array of integers `nums` sorted in ascending order, find the starting and ending position of a given `target` value.

Your algorithm's runtime complexity must be in the order of *O*(log *n*).

If the target is not found in the array, return `[-1, -1]`.

**Example 1:**

```
Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]
```

**Example 2:**

```
Input: nums = [5,7,7,8,8,10], target = 6
Output: [-1,-1]
```

**Solution:**

分别找first，last position，找不到返回-1。

```java
public int[] searchRange(int[] nums, int target) {
    if (nums == null || nums.length == 0) return new int[]{-1, -1};
    int first = first_position(nums, target);
    int last = last_position(nums, target);
    return new int[]{first, last};
}

private int first_position(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    while (left + 1 < right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] >= target) {
            right = mid;
        } else {
            left = mid;
        }
    }
    if (nums[left]  == target) return left;
    if (nums[right] == target) return right;
    return -1;
}

private int last_position(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    while (left + 1 < right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] > target) {
            right = mid;
        } else {
            left = mid;
        }
    }
    if (nums[right] == target) return right;
    if (nums[left]  == target) return left;
    return -1;
}
```



**278. First Bad Version**

**Solution:**

Binary Search 之境界二，OOOOOXXXXX find the first bad version. While 循环去逼近first bad version, 正常情况下应该left和right应该定位到中间的OX上，一般情况下应该返回right，但是若第一版就是bad，就需要返回left，因为left，right达到了左边的极限left=1，right=2。

```java
public int firstBadVersion(int n) {
    int left = 1, right = n;
    while (left + 1 < right) {
        int mid = left + (right - left) / 2;
        if (isBadVersion(mid)) {
            right = mid;
        } else {
            left = mid;
        }
    }
    if(isBadVersion(left)) return left;
    return right;
}
```



**153. Find Minimum in Rotated Sorted Array**

Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.Find the minimum element. You may assume no duplicate exists in the array.

**Example :**

```
Input: [3,4,5,1,2] 
Output: 1
```

**Solution:**

Binary Search 之境界二，OOOOXXXXXX。

Find the first element which is less than or equal to the last number. 不会出现nums[mid] = nums[nums.length -1] 的情况，因为left,right最大的可能是length-2，length-1。 正常情况下，应该返回的是right,但是会出现 nums = [1,2,3,4], left 和right会停在1,2(index 0,1) 的位置，此时应该返回left, 而[5,4,3,2,1]的话，left,right会停在length-2,length-1的地方，此时返回right是正确的。

```java
public int findMin(int[] nums) {
    if (nums == null || nums.length == 0) return -1;
    int left = 0, right = nums.length-1;
    while (left + 1 < right) {
        int mid = left + (right - left) / 2;
        if(nums[mid] <= nums[nums.length - 1]) {
            right = mid;
        } else {
            left = mid;
        }
    }
    return Math.min(nums[left],nums[right]);
    // or
    // if (left == 0 && nums[left] < nums[right]) return nums[left];
    // return nums[right];
}
```



**852. Peak Index in a Mountain Array**

Binary Search 之境界二，find the last element which is bigger the previous one. 考虑两个边界条件,[0210],最后停在[2,1]，返回left正确。[3,4,5,1] 最后停在[5,1] 返回left正确。

```java
public int peakIndexInMountainArray(int[] A) {
    int left = 1, right = A.length - 1, mid;
    while(left + 1 < right) {
        mid = left + (right - left) / 2;
        if (A[mid] > A[mid-1]) {
            left = mid;
        } else {
            right = mid;
        }
    }
    return left;
}
```

**Smallest Rectangle Enclosing Black Pixels**

**Search In a Big Sorted Array**

**33. Search In Rotated Sorted Array**

**Find Peak Element**



### Breadth First Search

BFS in Binary Tree

BFS in Graph -> Topological sorting

BFS in board

**使用BFS的cases**

Traverse in graph(Tree is one kind of graph)

* Level order traversal(层级遍历)
* Connected component
* Topological sorting

Shortest path in simple graph(仅限每条边长度都为1，且没有方向)

最短路径：BFS, dp

最长路径：DFS, dp

BFS: Queue (stack 也可以，但顺序是反的，没人用的)

DFS: Stack

能用BFS写的尽量不要用DFS，non-recursion的DFS不好写，recursion会造成stack overflow。

**模板：**

BFS写法几乎都一样(参考102)：

1. 创建一个队列，把起始节点都放到里面去
2. while队列不空，处理队列中的节点并扩展出新的节点

如果不需要分层，则只需要一个循环

**Binary Tree Serialization:**

将内存中结构化的数据变成String的过程。

Seriazation: Object -> String

Deserialization: String -> Object 

**BFS in Graph VS BFS in Tree:**

图中存在环，意味着有可能有节点要重复进入队列，解决办法是用hashset或者hashmap记录是否在队列中。

**图的表示方法：**

1. Map<Integer, Set<Integer>>
2. Node class中加neighbours

**拓扑排序：**

Given an directed graph, a topological order of the graph nodes is defined as follow:

- For each directed edge `A -> B` in graph, A must before B in the order list.
- The first node in the order can be any node in the graph with no nodes direct to it.

Find any topological order for the given graph.

You can assume that there is at least one topological order in the graph.

**Example：**

For graph as follow:

![图片](https://media-cdn.jiuzhang.com/markdown/images/8/6/91cf07d2-b7ea-11e9-bb77-0242ac110002.jpg)

The topological order can be:

```
[0, 1, 2, 3, 4, 5]
[0, 2, 3, 1, 5, 4]
...
```



**Examples:**

**102. Binary Tree Level Order Traversal**

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

**297. Serialize and Deserialize Binary Tree**

**Example:** 

```
You may serialize the following tree:

    1
   / \
  2   3
     / \
    4   5

as "[1,2,3,null,null,4,5]"
```

**Solution:**

serialize:

1. 开一个arraylist，把root丢进去，BFS用一层循环把每层节点都丢进去，判断条件是i<queue.size(), queue的size每层是变化的。依次把所有节点的左右节点丢进去，遇到null跳出，直到不再丢进去，同时也运行到最后一个节点。

2. 使用while循环把最后一层尾部的null全部去掉

3. 把TreeNode的queue(arraylist) 中的节点的val code成一串字符串

Deserialize：

把String按照","split成String array，建立 一个Arraylist去存所有TreeNode，index为当前进行到哪个node，用isLeftNode去判断左右子节点。

```java
// Encodes a tree to a single string.
public String serialize(TreeNode root) {
    if (root == null) return "[]";

    List<TreeNode> queue = new ArrayList<TreeNode>();
    queue.add(root);

    for (int i = 0; i < queue.size(); i++) {
        TreeNode node = queue.get(i);
        if (node == null) continue;
        queue.add(node.left);
        queue.add(node.right);
    }

    while (queue.get(queue.size() - 1) == null) {
        queue.remove(queue.size() - 1);
    }

    StringBuilder sb = new StringBuilder();
    sb.append("[");
    sb.append(queue.get(0).val);
    for (int i = 1; i < queue.size(); i++) {
        if (queue.get(i) == null) {
            sb.append(",null");
        } else {
            sb.append(",");
            sb.append(queue.get(i).val);
        }
    }
    sb.append("]");

    return sb.toString();
}

// Decodes your encoded data to tree.
public TreeNode deserialize(String data) {
    if (data.equals("[]")) return null;

    String[] vals = data.substring(1, data.length()-1).split(",");

    List<TreeNode> queue = new ArrayList<TreeNode>();
    TreeNode root = new TreeNode(Integer.parseInt(vals[0]));
    queue.add(root);

    int index = 0;
    boolean isLeftNode = true;
    for (int i = 1; i < vals.length; i++) {
        if (!vals[i].equals("null")) {
            TreeNode node = new TreeNode(Integer.parseInt(vals[i]));
            if (isLeftNode) {
                queue.get(index).left = node;
            } else {
                queue.get(index).right = node;
            }
            queue.add(node);
        }

        if (!isLeftNode) {
            index++;
        }

        isLeftNode = !isLeftNode;
    }
    return root;
}
```

**261. Graph Valid Tree**

Given `n` nodes labeled from `0` to `n - 1` and a list of `undirected` edges (each edge is a pair of nodes), write a function to check whether these edges make up a valid tree.

You can assume that no duplicate edges will appear in edges. Since all edges are `undirected`, `[0, 1]` is the same as `[1, 0]` and thus will not appear together in edges.

**Example:**

**Example 1:**

```
Input: n = 5 edges = [[0, 1], [0, 2], [0, 3], [1, 4]]
Output: true.
```

**Example 2:**

```
Input: n = 5 edges = [[0, 1], [1, 2], [2, 3], [1, 3], [1, 4]]
Output: false.
```

**Solution：**

Graph is a tree if and only if 1. There are n-1 edges. 2. n nodes are connected

用基本数据结构表示图的方法：Map<Integer, Set<Integer>>, key为node index，value为与该node相连的node的index构成的set。

```java
    public boolean validTree(int n, int[][] edges) {
        if (n == 0) {
            return false;
        }
        
        if (edges.length != n - 1) {
            return false;
        }
        
        Map<Integer, Set<Integer>> graph = initializeGraph(n, edges);
        
        // bfs
        Queue<Integer> queue = new LinkedList<>();
        Set<Integer> hash = new HashSet<>();
        
        queue.offer(0);
        hash.add(0);
        while (!queue.isEmpty()) {
            int node = queue.poll();
            for (Integer neighbor : graph.get(node)) {
                if (hash.contains(neighbor)) {
                    continue;
                }
                hash.add(neighbor);
                queue.offer(neighbor);
            }
        }
        
        return (hash.size() == n);
    }
    
    private Map<Integer, Set<Integer>> initializeGraph(int n, int[][] edges) {
        Map<Integer, Set<Integer>> graph = new HashMap<>();
        for (int i = 0; i < n; i++) {
            graph.put(i, new HashSet<Integer>());
        }
        
        for (int i = 0; i < edges.length; i++) {
            int u = edges[i][0];
            int v = edges[i][1];
            graph.get(u).add(v);
            graph.get(v).add(u);
        }
        
        return graph;
    }
```



### Dynamic Programming





