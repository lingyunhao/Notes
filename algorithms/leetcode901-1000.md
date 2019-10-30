### 939. Minimum Area Rectangle

Given a set of points in the xy-plane, determine the minimum area of a rectangle formed from these points, with sides parallel to the x and y axes.If there isn't any rectangle, return 0.

**Solution:**

把所有点按照x group到一起，HashMap,key->x, value->list of y。把list of y全部sort一遍。遍历两个column，去找相同的y值，有两对的话就可以组成一个长方形。

Note：

1. Integer的相等要用equals
2. 用一个x list去存x，当作遍历HashMap时的index。防止重复两次遍历一对x。有了index就可以用j+1，防止重复之前遍历过的。
3. 一些加速的小技巧

```java
public int minAreaRect(int[][] points) {
    Map<Integer, List<Integer>> map = new HashMap<>();
    List<Integer> x = new ArrayList<>();
    for (int[] p : points) {
        if (map.containsKey(p[0])) {
            map.get(p[0]).add(p[1]);
        } else {
            x.add(p[0]);
            map.put(p[0], new ArrayList<>());
            map.get(p[0]).add(p[1]);
        }
    }
    for (Map.Entry<Integer, List<Integer>> entry : map.entrySet()) {
        Collections.sort(entry.getValue());
    }
    int lasty = 0;
    int minArea = Integer.MAX_VALUE;
    boolean last = false;
    for (int i = 0; i < x.size(); ++i) {
        if (map.get(x.get(i)).size() <= 1) continue;
        for (int j = i + 1; j < x.size(); ++j) {
            if (map.get(x.get(j)).size() <= 1) continue;
            int m = 0, n = 0;
            List<Integer> col1 = map.get(x.get(i));
            List<Integer> col2 = map.get(x.get(j));
            while (m < col1.size() && n < col2.size()) {
                if (col1.get(m) < col2.get(n)) {
                    ++m;
                } else if (col1.get(m) > col2.get(n)) {
                    ++n;
                } else {
                     if (!last) {
                        last = true;
                    } else {
                        minArea = Math.min(minArea, Math.abs((x.get(i)-x.get(j)) * (col1.get(m) - lasty)));
                    }
                    lasty = col1.get(m);
                    ++m;
                    ++n;
                }
            }
            last = false;
        }
    }
    return minArea < Integer.MAX_VALUE ? minArea : 0;
}
```

### 947. Most Stones Removed with Same Row or Column

On a 2D plane, we place stones at some integer coordinate points.  Each coordinate point may have at most one stone.

Now, a *move* consists of removing a stone that shares a column or row with another stone on the grid.

What is the largest possible number of moves we can make?

**Solution:**

Number of islands 变种。求component个数，用结点个数减去conponent个数。dfs把与当前的点相连的所有点都标为visited。

```java
public int removeStones(int[][] stones) {
    Map<Integer, Set<Integer>> map = new HashMap<>();
    int n = stones.length;
    for (int i = 0; i < n; ++i) {
        Set<Integer> set = new HashSet<>();
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            if (stones[i][0] == stones[j][0] || stones[i][1] == stones[j][1]) {
                set.add(j);
            }
        }
        map.put(i, set);
    }
    boolean[] visited = new boolean[n];
    int res = 0;
    for (int i = 0; i < stones.length; ++i) {
        if (!visited[i]) {
            dfs(i, map, visited);
            res++;
        }
    }
    return n - res;
}

private void dfs(int index, Map<Integer, Set<Integer>> map, boolean[] visited) { 
    if (visited[index]) return;
    visited[index] = true;
    for (int i : map.get(index)) {
        dfs(i, map, visited);
    }
}
```

### 951. Flip Equivalent Binary Trees

For a binary tree T, we can define a flip operation as follows: choose any node, and swap the left and right child subtrees.

A binary tree X is *flip equivalent* to a binary tree Y if and only if we can make X equal to Y after some number of flip operations.

Write a function that determines whether two binary trees are *flip equivalent*.  The trees are given by root nodes `root1` and `root2`.

**Example 1:**

```
Input: root1 = [1,2,3,4,5,6,null,null,null,7,8], root2 = [1,3,2,null,6,4,5,null,null,null,null,8,7]
Output: true
Explanation: We flipped at nodes with values 1, 3, and 5.
```

 **Solution:**

递推：recursion。（也可以理解为是个dfs，dfs(recursion写法）本质上就是在用recursion去搜索）。

主要要把recursion单独当作一种方法，不要和dfs混淆，只是dfs经常用recursion来写，因为快且方便。

**Recursion 的要素**

- 递归的定义（递归函数求的是什么，完成了什么功能，类似dp[i]表示什么）
- 递归的拆解 （这次递归和之前的递归有什么关系，在本次递归调用递归传参，return等等，类似dp fucntion）
- 递归的出口 （什么时候可以return）

**写recursion的时候，assume对于当前的node是正确的，那么对于所有的node一定正确。**

**tree等需要recursion结构 判断identical类问题四步：**

1. 两个root皆为null return true （出口）

2. 两个root其中一个为null return false （出口）

3. 两个root的值不相等 return false （出口）

4. 根据题意，判断左右子树调用问题传参（递归的拆解）

   本题的recursion就是判断当前节点正确，且左右子树正确

```java
public boolean flipEquiv(TreeNode root1, TreeNode root2) {
    if (root1 == null && root2 == null) return true;
    if (root1 == null || root2 == null) return false;
    if (root1.val != root2.val) return false;
    return (flipEquiv(root1.left, root2.left) && flipEquiv(root1.right, root2.right)) || (flipEquiv(root1.left, root2.right) && flipEquiv(root1.right, root2.left));
}
```

### 963. Minimum Area Rectangle II

Given a set of points in the xy-plane, determine the minimum area of **any** rectangle formed from these points, with sides **not necessarily parallel** to the x and y axes.

If there isn't any rectangle, return 0.

**Solution:**

O($n^3$)遍历。先把所有的点加入set，取三个点，判断满足条件的第四个点在不在set种。

判断四个点p1, p2, p3, p4是rectangle(假设p1,p2是对角)：

1. p1.x + p2.x = p3.x + p4.x && p1.y + p2.y = p3.y + p4.y—> 保证了平行（对角线中点是同一个点）
2. (p1.x-p3.x) * (p2.x-p3.x) + (p1.y-p3.y) * (p2.y-p3.y) == 0 保证了垂直

```java
import java.awt.Point;

class Solution {
    
    public double minAreaFreeRect(int[][] points) {
        if (points == null  || points.length <= 3) return 0;
        int n = points.length;
        Point[] A = new Point[n];
        Set<Point> set = new HashSet<>();
        for (int i = 0; i < n; ++i) {
            A[i] = new Point(points[i][0], points[i][1]);
            set.add(A[i]);
        }
        
        double res = Double.MAX_VALUE;
        for (int i = 0; i < n; ++i) {
            for (int j = i+1; j < n; ++j) {
                for (int m = j + 1; m < n; ++m) {
                    Point candidate1 = new Point(A[i].x + A[j].x - A[m].x, A[i].y + A[j].y - A[m].y);   
                    double area;
                    if (checkRec(A[i], A[j], A[m], candidate1, set)) {
                        area = A[i].distance(A[m]) * A[j].distance(A[m]);
                        res = Math.min(res, area);
                    }  
                    Point candidate2 = new Point(A[i].x + A[m].x - A[j].x, A[i].y + A[m].y - A[j].y);
                    if (checkRec(A[i], A[m], A[j], candidate2, set)) {
                        area = A[i].distance(A[j]) * A[m].distance(A[j]);
                        res = Math.min(res, area);
                    }
                    Point candidate3 = new Point(A[j].x + A[m].x - A[i].x, A[j].y + A[m].y - A[i].y);
                    if (checkRec(A[j], A[m], A[i], candidate3, set)) {
                        area = A[j].distance(A[i]) * A[m].distance(A[i]);
                        res = Math.min(res, area);
                    }
                }
            }
        }
        return res < Double.MAX_VALUE ? res : 0;
    }
    
    private boolean checkRec(Point p1, Point p2, Point p3, Point p4, Set<Point> set) {
        if (!set.contains(p4)) return false;
        if ((p1.x - p3.x) * (p2.x - p3.x) + (p1.y - p3.y) * (p2.y - p3.y) == 0) return true;
        return false;
    } 
}
```

### 981. Time Based Key-Value Store

Create a timebased key-value store class `TimeMap`, that supports two operations.

\1. `set(string key, string value, int timestamp)`

- Stores the `key` and `value`, along with the given `timestamp`.

\2. `get(string key, int timestamp)`

- Returns a value such that `set(key, value, timestamp_prev)` was called previously, with `timestamp_prev <= timestamp`.
- If there are multiple such values, it returns the one with the largest `timestamp_prev`.
- If there are no values, it returns the empty string (`""`).

 **Solution:**

数据结构可以用一个HashMap<key, List<Pair<timestamp,String>>> 这样set是O(1), get正常情况是O(n)（暴力搜一遍）。或者set,get都是O(lgn)。用TreeMap，treemap的get和put都是O(lgn)，所以这里的set,get都是O(lgn)。

```java
class TimeMap {
    /** Initialize your data structure here. */
    private Map<String, TreeMap<Integer, String>> map;
    public TimeMap() {
        map = new HashMap();
    }
    
    public void set(String key, String value, int timestamp) {
        if (!map.containsKey(key)) {
            map.put(key, new TreeMap<>());
        } 
        map.get(key).put(timestamp, value);
        
    }
    
    public String get(String key, int timestamp) {
        if (map.containsKey(key)) {
            Integer t = map.get(key).floorKey(timestamp);
            return t == null ? "" : map.get(key).get(t);
        } else {
            return "";
        }
    }
}
```

### 987. Vertical Order Traversal of a Binary Tree

Given a binary tree, return the *vertical order* traversal of its nodes values.

For each node at position `(X, Y)`, its left and right children respectively will be at positions `(X-1, Y-1)` and `(X+1, Y-1)`.

Running a vertical line from `X = -infinity` to `X = +infinity`, whenever the vertical line touches some nodes, we report the values of the nodes in order from top to bottom (decreasing `Y` coordinates).

If two nodes have the same position, then the value of the node that is reported first is the value that is smaller.

Return an list of non-empty reports in order of `X` coordinate.  Every report will have a list of values of nodes.

**Example 1:**

![img](https://assets.leetcode.com/uploads/2019/01/31/1236_example_1.PNG)

```
Input: [3,9,20,null,null,15,7]
Output: [[9],[3,15],[20],[7]]
Explanation: 
Without loss of generality, we can assume the root node is at position (0, 0):
Then, the node with value 9 occurs at position (-1, -1);
The nodes with values 3 and 15 occur at positions (0, 0) and (0, -2);
The node with value 20 occurs at position (1, -1);
The node with value 7 occurs at position (2, -2).
```

**Soluiton:**

用某种搜索或者traversal去遍历每个node，把其位置和val放进去。

BFS + sort

```java
public List<List<Integer>> verticalTraversal(TreeNode root) {
    if (root == null) return null;
    Queue<TreeNode> queue = new LinkedList<>();
    List<List<Integer>> result = new ArrayList<>();
    //int[0]:x int[1]:y int[2]:val
    List<int[]> infos = new ArrayList<>();
    queue.offer(root);
    infos.add(new int[]{0,0,root.val});
    int index = -1;
    while (!queue.isEmpty()) {
        TreeNode node = queue.poll();
        index++;
        int cur_x = infos.get(index)[0];
        int cur_y = infos.get(index)[1];
        if (node.left != null) {
            queue.offer(node.left);
            infos.add(new int[]{cur_x - 1, cur_y - 1, node.left.val});
        }
        if (node.right != null) {
            queue.offer(node.right);
            infos.add(new int[]{cur_x + 1, cur_y - 1, node.right.val});
        }
    }
    Collections.sort(infos, new Comparator<int[]>() {
        public int compare(int[] a, int[] b) {
            int d = a[0]-b[0];
            if (d == 0) {
                d = b[1] - a[1];
                if (d == 0) {
                    d = a[2] - b[2];
                }
            }
            return d;
        }
    });
    int x = Integer.MAX_VALUE;
    List<Integer> col = null;
    for (int[] each : infos) {
        if (each[0] != x) {
            if (col != null) {
                result.add(col);
            }
            col = new ArrayList<>();
        }
        col.add(each[2]);
        x = each[0];
    }
    result.add(col);
    return result;
}
```

### 