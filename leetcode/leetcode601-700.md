### 605. Can Place Flowers

Suppose you have a long flowerbed in which some of the plots are planted and some are not. However, flowers cannot be planted in adjacent plots - they would compete for water and both would die.

Given a flowerbed (represented as an array containing 0 and 1, where 0 means empty and 1 means not empty), and a number **n**, return if **n** new flowers can be planted in it without violating the no-adjacent-flowers rule.

**Example 1:**

```
Input: flowerbed = [1,0,0,0,1], n = 1
Output: True
```

**Example 2:**

```
Input: flowerbed = [1,0,0,0,1], n = 2
Output: False
```

**Solution: **

判断已给的flowerbed最多能种多少花，判断是否大于等于parameter2。或者可以每次判断cnt是否大于n, 满足则直接return。本题主要是要考虑第一个item和最后一个item，因为他们一个没有prev，一个没有next，可以把这两个单独拿出来讨论，但讨论的时候一定要考虑flowerbed的长度是否等于1的情况，或者我们可以把第一个和最后一个同化到中间那些。即，把flowerbed[0]的prev设为0，flowbed[length-1]的next设为0。

```java
public boolean canPlaceFlowers(int[] flowerbed, int n) {
    int cnt = 0, len = flowerbed.length;

    for (int i = 0; i < len; i++) {
        int pre = i == 0 ? 0 : flowerbed[i-1];
        int next = i == len - 1 ? 0 : flowerbed[i+1];
        if (pre == 0 && next == 0 && flowerbed[i] == 0) {
            cnt++;
            flowerbed[i] = 1;
        }
    }
    return cnt >= n;
}
```



### 646. Maximum Length of Pair Chain

You are given `n` pairs of numbers. In every pair, the first number is always smaller than the second number.

Now, we define a pair `(c, d)` can follow another pair `(a, b)`if and only if `b < c`. Chain of pairs can be formed in this fashion.

**Example:**

```
Input: [[1,2], [2,3], [3,4]]
Output: 2
Explanation: The longest chain is [1,2] -> [3,4]
```

**Solution:**

根据pair第一个元素进行排序，找由第二个元素组成的最长递增子序列。同problem 300。dp数组要初始化为1。

```java
public int findLongestChain(int[][] pairs) {
    if (pairs == null || pairs.length == 0) return 0;
    Arrays.sort(pairs, (a,b) -> (a[0] - b[0]));
    int[] dp = new int[pairs.length];
    Arrays.fill(dp, 1);

    for (int i = 1; i < pairs.length; ++i) {
        for (int j = 0; j < i; ++j) {
            if (pairs[j][1] < pairs[i][0]) {
                dp[i] = Math.max(dp[j]+1, dp[i]);
            }
        }
    }

    int result = 0;
    for (int i = 0; i < dp.length; ++i) {
        result = Math.max(dp[i], result);
    }
    return result;
}
```

### 658. Find K Closest Elements

Given a sorted array, two integers `k` and `x`, find the `k` closest elements to `x` in the array. The result should also be sorted in ascending order. If there is a tie, the smaller elements are always preferred.

**Example 1:**

```
Input: [1,2,3,4,5], k=4, x=3
Output: [1,2,3,4]
```

**Solution:**

Binary search + two pointers  O(logn + k)

```java
public List<Integer> findClosestElements(int[] arr, int k, int x) {
    List<Integer> res = new ArrayList<>();
    if (arr == null || arr.length == 0) return res;
    int left = findLastEqualOrSmaller(arr, x);
    int right = left + 1;
    for (int i = 0; i < k; i++) {
        // 相同的差要取小的那个 所以后边是小于等于
        if (right >= arr.length || (left >= 0 && (Math.abs(arr[left] - x) <= Math.abs(arr[right] - x)))) {
            left--;
        } else {
            right++;
        }
    }
    for (int i = left + 1; i < right; i++) {
        res.add(arr[i]);
    }
    return res;
}
public int findLastEqualOrSmaller(int[] array, int x) {
    int start = 0;
    int end = array.length - 1;
    int mid;
    while (start + 1 < end) {
        mid = start + (end - start) / 2;
        if (array[mid] == x) {
            start = mid;
        } else if (array[mid] < x) {
            start = mid;
        } else {
            end = mid;
        }
    }
    if (array[end] <= x) {
        return end;
    } else {
        return start;
    }
}
```

### 659. Split Array into Consecutive Subsequences

Given an array `nums` sorted in ascending order, return `true` if and only if you can split it into 1 or more subsequences such that each subsequence consists of consecutive integers and has length at least 3.

**Example 1:**

```
Input: [1,2,3,3,4,5]
Output: True
Explanation:
You can split them into two consecutive subsequences : 
1, 2, 3
3, 4, 5
```

**Soluiton:**

Greedy. 先把所有count存下，对于每一个num，去看能否把它加到当前存在的chain中，不能的话就propose一个新的chain(x,x+1,x+2)。tails[i]表示的是目前有一个chain以i-1结尾。 比如说1，2，3，3，4，5. 处理1的时候，把2，3的frequcny都-1，那么2就get不到，3的话不能加到当前的chain所以propose一个新的chain，把，4，5加进去。如果这两个条件都不能满足返回false。

#### Opening and Closing Events

We can think of the problem as drawing intervals on a number line. This gives us the idea of opening and closing events.

To illustrate this concept, say we have `nums = [10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13]`, with no `9`s and no `14`s. We must have two sequences start at 10, two sequences start at 11, and 3 sequences end at 12.

In general, when considering a chain of consecutive integers `x`, we must have `C = count[x+1] - count[x]` sequences start at `x+1` when `C > 0`, and `-C` sequences end at `x` if `C < 0`. Even if there are more endpoints on the intervals we draw, there must be at least this many endpoints.

With the above example, `count[11] - count[10] = 2` and `count[13] - count[12] = -3` show that two sequences start at `11`, and three sequences end at `12`.

Also, if for example we know some sequences must start at time `1` and `4` and some sequences end at `5` and `7`, to maximize the smallest length sequence, we should pair the events together in the order they occur: ie., `1` with `5` and `4` with `7`.

Call a *chain* a sequence of 3 or more consecutive numbers.

Considering numbers `x` from left to right, if `x` can be added to a current chain, it's at least as good to add `x` to that chain first, rather than to start a new chain.

Why? If we started with numbers `x` and greater from the beginning, the shorter chains starting from `x` could be concatenated with the chains ending before `x`, possibly helping us if there was a "chain" from `x` that was only length 1 or 2.

**Algorithm**

Say we have a count of each number, and let `tails[x]` be the number of chains ending right before `x`.

Now let's process each number. If there's a chain ending before `x`, then add it to that chain. Otherwise, if we can start a new chain, do so.

```java
class Solution {
    public boolean isPossible(int[] nums) {
        Counter count = new Counter();
        Counter tails = new Counter();
        for (int x: nums) count.add(x, 1);

        for (int x: nums) {
            if (count.get(x) == 0) {
                continue;
            } else if (tails.get(x) > 0) {
                tails.add(x, -1);
                tails.add(x+1, 1);
            } else if (count.get(x+1) > 0 && count.get(x+2) > 0) {
                count.add(x+1, -1);
                count.add(x+2, -1);
                tails.add(x+3, 1);
            } else {
                return false;
            }
            count.add(x, -1);
        }
        return true;
    }
}
class Counter extends HashMap<Integer, Integer> {
    public int get(int k) {
        return containsKey(k) ? super.get(k) : 0;
    }
    
    public void add(int k, int v) {
        put(k, get(k) + v);
    }
}
```

### 684. Redundant Connection

In this problem, a tree is an **undirected** graph that is connected and has no cycles.

The given input is a graph that started as a tree with N nodes (with distinct values 1, 2, ..., N), with one additional edge added. The added edge has two different vertices chosen from 1 to N, and was not an edge that already existed.

The resulting graph is given as a 2D-array of `edges`. Each element of `edges` is a pair `[u, v]` with `u < v`, that represents an **undirected** edge connecting nodes `u` and `v`.

Return an edge that can be removed so that the resulting graph is a tree of N nodes. If there are multiple answers, return the answer that occurs last in the given 2D-array. The answer edge `[u, v]` should be in the same format, with `u < v`.

**Example 1:**

```
Input: [[1,2], [1,3], [2,3]]
Output: [2,3]
Explanation: The given undirected graph will be like this:
  1
 / \
2 - 3
```

**Solution:**

```java
Set<Integer> seen = new HashSet();
int MAX_EDGE_VAL = 1000;

public int[] findRedundantConnection(int[][] edges) {
    ArrayList<Integer>[] graph = new ArrayList[MAX_EDGE_VAL + 1];
    for (int i = 0; i <= MAX_EDGE_VAL; i++) {
        graph[i] = new ArrayList();
    }

    for (int[] edge: edges) {
        seen.clear();
        if (!graph[edge[0]].isEmpty() && !graph[edge[1]].isEmpty() &&
                dfs(graph, edge[0], edge[1])) {
            return edge;
        }
        graph[edge[0]].add(edge[1]);
        graph[edge[1]].add(edge[0]);
    }
    throw new AssertionError();
}
public boolean dfs(ArrayList<Integer>[] graph, int source, int target) {
    if (!seen.contains(source)) {
        seen.add(source);
        if (source == target) return true;
        for (int nei: graph[source]) {
            if (dfs(graph, nei, target)) return true;
        }
    }
    return false;
}
```

### 686. Repeated String Match

Given two strings A and B, find the minimum number of times A has to be repeated such that B is a substring of it. If no such solution, return -1.

For example, with A = "abcd" and B = "cdabcdab".

Return 3, because by repeating A three times (“abcdabcdabcd”), B is a substring of it; and B is not a substring of A repeated two times ("abcdabcd").

**Solution:**

indexOf 函数， time complexity：O(M*(M+N))

```java
public int repeatedStringMatch(String A, String B) {
    int m = A.length(), n = B.length();
    int cnt = 1;
    StringBuilder sb = new StringBuilder(A);
    while (sb.length() < n) {
        sb.append(A);
        ++cnt;
    }
    if (sb.indexOf(B) >= 0) return cnt;
    if (sb.append(A).indexOf(B) >= 0) return cnt+1;
    return -1;
}
```

### 695. Max Area of Island

Given a non-empty 2D array `grid` of 0's and 1's, an **island** is a group of `1`'s (representing land) connected 4-directionally (horizontal or vertical.) You may assume all four edges of the grid are surrounded by water.求最大的联通面积。

**Solution:**

dfs recursion函数求的是，与当前元素向四个方向所联通的面积和，每个area在四个方向都加了一次。实际上是先调用dfs，然后判断满不满足条件需要return(函数的出口，当前函数return，把调用当前函数的函数pop出栈), 写的时候，是在dfs函数第一行进行判断。注意每次调用dfs把当前元素置为1，下一个方向上的元素又回头把自己加上，每个元素最终只进行了一次dfs。不需要改回来，改回来的话，相当于一个联通面积中每个元素都进行了一次dfs，求出了一个相等的area返回到主函数中进行打擂台。dfs是一种search，原则上每个元素遍历一次，有些题目必须要改回来，这种叫backtracking, 例如排列组合的题目。

```java
private int[][] direction = {{0,1}, {0,-1}, {1,0}, {-1,0}}; 
public int maxAreaOfIsland(int[][] grid) {
    if (grid == null || grid.length == 0) {
        return 0;
    }
    int m = grid.length;
    int n = grid[0].length;
    int maxArea = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            maxArea = Math.max(maxArea, dfs(grid, i, j, m, n));
        }
    }
    return maxArea;   
}

private int dfs(int[][] grid, int r, int c, int m, int n) {
    if (r < 0 || r >= m || c < 0 || c >= n || grid[r][c] == 0) {
        return 0;
    }
    grid[r][c] = 0;
    int area = 1;
    for (int[] d : direction) {
        area += dfs(grid, r+d[0], c+d[1], m, n);
    }
    return area;
}
```

