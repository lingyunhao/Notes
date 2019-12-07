### 304. Range Sum Query 2D - Immutable

Given a 2D matrix *matrix*, find the sum of the elements inside the rectangle defined by its upper left corner (*row*1, *col*1) and lower right corner (*row*2, *col*2).

![Range Sum Query 2D](https://leetcode.com/static/images/courses/range_sum_query_2d.png)
The above rectangle (with the red border) is defined by (row1, col1) = **(2, 1)** and (row2, col2) = **(4, 3)**, which contains sum = **8**.

**Example:**

```
Given matrix = [
  [3, 0, 1, 4, 2],
  [5, 6, 3, 2, 1],
  [1, 2, 0, 1, 5],
  [4, 1, 0, 1, 7],
  [1, 0, 3, 0, 5]
]

sumRegion(2, 1, 4, 3) -> 8
sumRegion(1, 1, 2, 2) -> 11
sumRegion(1, 2, 2, 4) -> 12
```

**Solution:**

dp、积分图。初始化的时候用dp存从(0,0)到(i,j)的和，sumRegion的时候就用递推公式。如果要改变matrix的话，得重新求dp。

```java
class NumMatrix {
    private int[][] dp;
    public NumMatrix(int[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) return;
        int m = matrix.length, n = matrix[0].length;
        dp = new int[m][n];
        // initialize
        dp[0][0] = matrix[0][0];
        for (int i = 1; i < n; ++i) {
            dp[0][i] = dp[0][i-1] + matrix[0][i];
        }
        for (int i = 1; i < m; ++i) {
            dp[i][0] = dp[i-1][0] + matrix[i][0];
        }
        for (int i = 1; i < m; ++i) {
            for (int j = 1; j < n; ++j) {
                // 递推公式
                dp[i][j] = dp[i-1][j] + dp[i][j-1] - dp[i-1][j-1] + matrix[i][j];
            }
        }
    }
    
    public int sumRegion(int row1, int col1, int row2, int col2) {
        int res;
        if (col1 == 0 && row1 == 0) {
            res = dp[row2][col2];
        } else if (col1 == 0) {
            res = dp[row2][col2] - dp[row1-1][col2];
        } else if (row1 == 0) {
            res = dp[row2][col2] - dp[row2][col1-1];
        } else {
            res = dp[row2][col2] - dp[row2][col1-1] - dp[row1-1][col2] + dp[row1-1][col1-1];
        }
        return res;
    }
}
```

### 329. Longest Increasing Path in a Matrix

Given an integer matrix, find the length of the longest increasing path.

From each cell, you can either move to four directions: left, right, up or down. You may NOT move diagonally or move outside of the boundary (i.e. wrap-around is not allowed).

**Example 1:**

```
Input: nums = 
[
  [9,9,4],
  [6,6,8],
  [2,1,1]
] 
Output: 4 
Explanation: The longest increasing path is [1, 2, 6, 9].
```

**Solution:**

Topological sort

```java
// Topological Sort Based Solution
// An Alternative Solution
public class Solution {
    private static final int[][] dir = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
    private int m, n;
    public int longestIncreasingPath(int[][] grid) {
        int m = grid.length;
        if (m == 0) return 0;
        int n = grid[0].length;
        // padding the matrix with zero as boundaries
        // assuming all positive integer, otherwise use INT_MIN as boundaries
        int[][] matrix = new int[m + 2][n + 2];
        for (int i = 0; i < m; ++i)
            System.arraycopy(grid[i], 0, matrix[i + 1], 1, n);

        // calculate outdegrees
        int[][] outdegree = new int[m + 2][n + 2];
        for (int i = 1; i <= m; ++i)
            for (int j = 1; j <= n; ++j)
                for (int[] d: dir)
                    if (matrix[i][j] < matrix[i + d[0]][j + d[1]])
                        outdegree[i][j]++;

        // find leaves who have zero out degree as the initial level
        n += 2;
        m += 2;
        List<int[]> leaves = new ArrayList<>();
        for (int i = 1; i < m - 1; ++i)
            for (int j = 1; j < n - 1; ++j)
                if (outdegree[i][j] == 0) leaves.add(new int[]{i, j});

        // remove leaves level by level in topological order
        int height = 0;
        while (!leaves.isEmpty()) {
            height++;
            List<int[]> newLeaves = new ArrayList<>();
            for (int[] node : leaves) {
                for (int[] d:dir) {
                    int x = node[0] + d[0], y = node[1] + d[1];
                    if (matrix[node[0]][node[1]] > matrix[x][y])
                        if (--outdegree[x][y] == 0)
                            newLeaves.add(new int[]{x, y});
                }
            }
            leaves = newLeaves;
        }
        return height;
    }
}
```

### 334. Increasing Triplet Subsequence

Given an unsorted array return whether an increasing subsequence of length 3 exists or not in the array.

**Solution:**

用两个Integer来记录，到当前为止，最小的值，和第二小的值，如果每次判断既大于最小值又大于第二小的值，则存在(也就是else)，注意判断条件的顺序，先判断是否小于等于最小值，再判断是否小于等于第二小的值。都不是则存在。这种方法也可以写increasing quatre subsequence，5，6，7，8constant个。 

注意等于号，else一定要是比两个buffer都大才行。

```java
public boolean increasingTriplet(int[] nums) {
    if (nums == null || nums.length < 3) return false;
    int min = Integer.MAX_VALUE, sec = Integer.MAX_VALUE;
    for (int i = 0; i < nums.length; i++) {
        if (nums[i] <= min) {
            min = nums[i];
        } else if (nums[i] <= sec) {
            sec = nums[i];
        } else {
            return true;
        }
    }
    return false;
}
```

### 343. Integer Break

Given a positive integer *n*, break it into the sum of **at least** two positive integers and maximize the product of those integers. Return the maximum product you can get.

**Solution：**

```java
public int integerBreak(int n) {
    int[] dp = new int[n+1];
    dp[1] = 1;
    for (int i = 2; i <= n; ++i) {
        for (int j = 1; j <= i - 1; ++j) {
            dp[i] = Math.max(dp[i], Math.max(j * dp[i-j], j * (i - j)));
        }
    }
    return dp[n];
}
```

### 346. Moving Average from Data Stream

Given a stream of integers and a window size, calculate the moving average of all integers in the sliding window.

**Example:**

```
MovingAverage m = new MovingAverage(3);
m.next(1) = 1
m.next(10) = (1 + 10) / 2
m.next(3) = (1 + 10 + 3) / 3
m.next(5) = (10 + 3 + 5) / 3
```

**Solution:**

是一个sliding window，尾进头出，(FIFO) 用queue。（头部删除尾部增加都是O（1））

单独存一个sum使得next O(1)时间。

**这种考OOD的需要注意：**

1. 别忘了数据结构在外面Initialize
2. 别忘了可以structor去初始化数据结构
3. 非常注重Time Complexity, 不可以用arraylist不支持头部O(1)删除。

```java
class MovingAverage {
    /** Initialize your data structure here. */
    private Queue<Integer> queue = new LinkedList<>();
    private int sum = 0;
    private int capacity;
    public MovingAverage(int size) {
        this.capacity = size;
    }
    
    public double next(int val) {
        if (queue.size() < capacity) {
            queue.offer(val);
            sum += val;
        } else {
            sum -= queue.poll();
            queue.add(val);
            sum += val;
        }
        return ((double)sum)/queue.size();
    }
}
```

### 359. Logger Rate Limiter

Design a logger system that receive stream of messages along with its timestamps, each message should be printed if and only if it is **not printed in the last 10 seconds**.

Given a message and a timestamp (in seconds granularity), return true if the message should be printed in the given timestamp, otherwise returns false.

It is possible that several messages arrive roughly at the same time.

**Example:**

```
Logger logger = new Logger();

// logging string "foo" at timestamp 1
logger.shouldPrintMessage(1, "foo"); returns true; 

// logging string "bar" at timestamp 2
logger.shouldPrintMessage(2,"bar"); returns true;

// logging string "foo" at timestamp 3
logger.shouldPrintMessage(3,"foo"); returns false;

// logging string "bar" at timestamp 8
logger.shouldPrintMessage(8,"bar"); returns false;

// logging string "foo" at timestamp 10
logger.shouldPrintMessage(10,"foo"); returns false;

// logging string "foo" at timestamp 11
logger.shouldPrintMessage(11,"foo"); returns true;
```

**Solution:**

重要的是理解题意，前10秒中么有打印过则可以打印。用map存上次打印的timestamp + 10.

```java
class Logger {
    /** Initialize your data structure here. */
    Map<String, Integer> map;
    public Logger() {
        map = new HashMap<>();
    }
    /** Returns true if the message should be printed in the given timestamp, otherwise returns false.
        If this method returns false, the message will not be printed.
        The timestamp is in seconds granularity. */
    public boolean shouldPrintMessage(int timestamp, String message) {
        if (!map.containsKey(message)) {
            map.put(message, timestamp + 10);
        } else {
            if (timestamp >= map.get(message)) {
                map.put(message, timestamp + 10);
                return true;
            } else {
                return false;
            }
        }
        return true;
    }
}
```

### 363.Max Sum of Rectangle No Larger Than K

Given a non-empty 2D matrix *matrix* and an integer *k*, find the max sum of a rectangle in the *matrix* such that its sum is no larger than *k*.

**Solution:**

DP + Naive brute force：store the sum of rec(0,0) - (i,j) in an array dp. Then traverse all of the possible rectangles maintain a maxArea which is less than k, if we find k just return k, if not, update the maxArea so far.

Note: 1. 用dp[m+1] [n+1]这样不用初始化第一列和第一行，因为求和，所以第一列第一行是0即可。 不仅是在求dp时不用考虑，在求任意的矩形面积时都不用考虑。

2. 四个点组成一个矩形，所以for循环应该有四层就可以遍历所有矩形。

```java
public int maxSumSubmatrix(int[][] matrix, int k) {
    if (matrix == null || matrix.length == 0 || matrix[0].length == 0) return 0;
    int m = matrix.length, n = matrix[0].length;
    int[][] dp = new int[m+1][n+1];
    for (int i = 1; i < m+1; ++i) {
        for (int j = 1; j < n+1; ++j) {
            dp[i][j] = dp[i-1][j] + dp[i][j-1] - dp[i-1][j-1] + matrix[i-1][j-1];
        }
    }

    int maxAreaK = Integer.MIN_VALUE;
    for (int i = 1; i < m+1; ++i) {
        for (int j = 1; j < n+1; ++j) {
            for (int p = 1; p <= i; ++p) {
                for (int q = 1; q <=j; ++q) {
                    int sum = dp[i][j] - dp[p-1][j] - dp[i][q-1] + dp[p-1][q-1];
                    if (sum == k) return k;
                    if (sum < k && sum > maxAreaK) {
                        maxAreaK = sum;
                    }
                }
            }
        }
    }
    return maxAreaK;
}
```

### 369. Plus One Linked List

Given a non-negative integer represented as **non-empty** a singly linked list of digits, plus one to the integer.

You may assume the integer do not contain any leading zero, except the number 0 itself.

The digits are stored such that the most significant digit is at the head of the list.

**Example :**

```
Input: [1,2,3]
Output: [1,2,4]
```

**Solution:**

不可以遍历list得到int num然后num+1，再变回去，int会溢出。

普遍答案是用recursion写，这样就不用reverse，直接处理最后的个位，然后一路带着carry返回。recursion函数表示的是，已经处理好的上一位(prev)返回了其进位，然后把自己的node改成相应的val，再返回前一位自己有没有进位。

```java
public ListNode plusOne(ListNode head) {
    int carry = add(head);
    if (carry == 1) {
        ListNode carryNode = new ListNode(1);
        carryNode.next = head;
        return carryNode;
    }
    return head;

}

//recursion定义：改变当前node的val（前一位的进位+node.val)，返回这一位有没有进位
int add(ListNode node) {
    // recursion 出口
    if (node.next == null) {
        int sum = node.val + 1;
        node.val = sum % 10;
        return sum / 10;
    }
    // 递推关系 与下一位的关系
    int nextResult = add(node.next);
    int sum = nextResult + node.val;
    node.val = sum % 10;
    return sum /10; 
}
```

### 371. Sum of Two Integers

Calculate the sum of two integers *a* and *b*, but you are **not allowed** to use the operator `+` and `-`. 

**Solution:**

对数字做运算，除了四则运算之外只能用位运算。

1. 不考虑进位，对每一位相加，相当于异或。
2. 考虑进位，只有1与1时会产生进位，相当于与。
3. 把上面两个结果想加，直到再也没有进位位置。本题也可以用while loop 判断条件是 b!=0

```java
public int getSum(int a, int b) {
    if (b == 0) return a;

    int sum = a ^ b;
    int carry = (a & b) << 1;

    return getSum(sum, carry);
}
```

### 376. Wiggle Subsequence

**Example 1:**

```
Input: [1,7,4,9,2,5]
Output: 6
Explanation: The entire sequence is a wiggle sequence.
```

**Example 2:**

```
Input: [1,17,5,10,13,15,10,5,16,8]
Output: 7
Explanation: There are several subsequences that achieve this length. One is [1,17,10,13,10,16,8].
```

**Solution:**

利用两个变量去记录上一个down,up 的位置。

```java
public int wiggleMaxLength(int[] nums) {
    if (nums == null || nums.length == 0) {
        return 0;
    }

    int up = 1, down = 1;
    for (int i = 1; i < nums.length; ++i) {
        if (nums[i] > nums[i-1]) {
            up = down + 1;
        } else if (nums[i] < nums[i-1]) {
            down = up + 1;
        }
    }
    return Math.max(down, up);
}
```

### 399. Evaluate Division

Equations are given in the format `A / B = k`, where `A` and `B` are variables represented as strings, and `k` is a real number (floating point number). Given some queries, return the answers. If the answer does not exist, return `-1.0`.

**Example:**
Given `a / b = 2.0, b / c = 3.0.`
queries are: `a / c = ?, b / a = ?, a / e = ?, a / a = ?, x / x = ? .`
return `[6.0, 0.5, -1.0, 1.0, -1.0 ].`

The input is: `vector<pair<string, string>> equations, vector<double>& values, vector<pair<string, string>> queries `, where `equations.size() == values.size()`, and the values are positive. This represents the equations. Return `vector<double>`.

According to the example above:

```
equations = [ ["a", "b"], ["b", "c"] ],
values = [2.0, 3.0],
queries = [ ["a", "c"], ["b", "a"], ["a", "e"], ["a", "a"], ["x", "x"] ]. 
```

**Solution:**

题目描述有图的关系，是一种搜索题。dfs + graph. 。 

```java
public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
    HashMap<String, HashMap<String, Double>> map = new HashMap<>();
    double[] results = new double[queries.size()];
    for(int i = 0; i < equations.size(); i++)
    {
        map.computeIfAbsent(equations.get(i).get(0), value -> new HashMap<>()).put(equations.get(i).get(1), values[i]);         
        map.computeIfAbsent(equations.get(i).get(1), value -> new HashMap<>()).put(equations.get(i).get(0), 1.0/values[i]);             
    }

    int i = 0;
    for(List<String> query : queries)
    {
        String param1 = query.get(0);
        String param2 = query.get(1);
        if (!(map.containsKey(param1) && map.containsKey(param2)))
            results[i++] = -1.0;
        else
        {
            results[i] = dfs(map, param1, param2, 1.0, new HashSet<>());
            if(results[i] == Double.MAX_VALUE)
                results[i] = -1.0;
            else
            {
                map.get(param1).put(param2, results[i]);
                map.get(param2).put(param1, 1.0/results[i]);
            }
            i++;
        }
    }

    return results;
}

// dfs求的是 src/dest的结果
private double dfs(HashMap<String, HashMap<String, Double>> map, String src, String dest, double currProd, Set<String> visited)
{
    if(src.equals(dest))
        return currProd;
    visited.add(src);
    if(map.get(src).containsKey(dest))
        return currProd * map.get(src).get(dest);
    Map<String, Double> entry = map.get(src);
    double res = Double.MAX_VALUE;
    for(Map.Entry<String, Double> e : entry.entrySet())
    {
        if(!visited.contains(e.getKey()))
            res = Math.min(dfs(map, e.getKey(), dest, currProd * e.getValue(), visited), res);
    }
    visited.remove(src);
    return res;
}
```

### 