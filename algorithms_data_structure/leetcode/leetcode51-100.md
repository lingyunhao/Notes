### 53. Maximum Subarray

Given an integer array `nums`, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

**Example:**

```
Input: [-2,1,-3,4,-1,2,1,-5,4],
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.
```

**Follow up:**

If you have figured out the O(*n*) solution, try coding another solution using the divide and conquer approach, which is more subtle.

**Solution:**

dp[i] 表示以nums[i]结尾的最大subarray，要么就是与dp[i-1]连起来，要么就是自己。可以用sum代替dp[i-1], 压缩空间到一维。 核心：1. dp[i] = Math.max(nums[i], nums[i] + dp[i-1]) 等同于 sum = Math.max(nums[i], nums[i] + sum).

2.遍历完dp需要打擂台获取最大值，同时这一步也可以在遍历时候进行。

```java
// extra O(n) space
public int maxSubArray(int[] nums) {
    if (nums == null || nums.length == 0) return 0;
    int[] dp = new int[nums.length];
    dp[0] = nums[0];
    int maxSum = dp[0];
    for (int i = 1; i < nums.length; i++) {
        dp[i] = Math.max(nums[i], nums[i] + dp[i-1]);
        maxSum = Math.max(dp[i], maxSum);
    }
    return maxSum;
}

// extra O(1) space
public int maxSubArray(int[] nums) {
    if (nums == null || nums.length == 0) return 0;
    int sum = nums[0];
    int maxSum = sum;
    for (int i = 1; i < nums.length; i++) {
        sum = Math.max(nums[i], nums[i] + sum);
        maxSum = Math.max(maxSum, sum);
    }

    return maxSum;
}
```

### 54. Spiral Matrix

Given a matrix of *m* x *n* elements (*m* rows, *n* columns), return all elements of the matrix in spiral order.

**Example 1:**

```
Input:
[
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]
Output: [1,2,3,6,9,8,7,4,5]
```

**Solution:**

```java
public List<Integer> spiralOrder(int[][] matrix) {
    List ans = new ArrayList();
    if (matrix.length == 0) return ans;
    int R = matrix.length, C = matrix[0].length;
    boolean[][] seen = new boolean[R][C];
    int[] dr = {0, 1, 0, -1};
    int[] dc = {1, 0, -1, 0};
    int r = 0, c = 0, di = 0;
    // for 循环的长度 = result 的长度，num of elements m * n
    for (int i = 0; i < R * C; i++) {
        ans.add(matrix[r][c]);
        seen[r][c] = true;
        int cr = r + dr[di];
        int cc = c + dc[di];
        if (0 <= cr && cr < R && 0 <= cc && cc < C && !seen[cr][cc]){
            r = cr;
            c = cc;
        } else {
            di = (di + 1) % 4;
            r += dr[di];
            c += dc[di];
        }
    }
    return ans;
}
```

### 55. Jump Game

Given an array of non-negative integers, you are initially positioned at the first index of the array.

Each element in the array represents your maximum jump length at that position.

Determine if you are able to reach the last index.

**Example 1:**

```
Input: [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
```

**Solution:**

**Greedy O(n)/ DP(bottom up)/(top down) O($n^2$) / dfs O($2^n$)**

DP 可以判断能不能跳到任意一个点。greedy只是判断了最后位置是不是good。

从后向前贪心，如果能从i跳到lastPos 则跳，判断最终位置是不是0。

Once we have our code in the bottom-up state, we can make one final, important observation. From a given position, when we try to see if we can jump to a *GOOD* position, we only ever use one - the first one (see the break statement). In other words, the left-most one. If we keep track of this left-most *GOOD* position as a separate variable, we can avoid searching for it in the array. Not only that, but we can stop using the array altogether.

Iterating right-to-left, for each position we check if there is a potential jump that reaches a *GOOD* index (`currPosition + nums[currPosition] >= leftmostGoodIndex`). If we can reach a *GOOD* index, then our position is itself *GOOD*. Also, this new *GOOD* position will be the new leftmost *GOOD* index. Iteration continues until the beginning of the array. If first position is a *GOOD* index then we can reach the last index from the first position.

To illustrate this scenario, we will use the diagram below, for input array `nums = [9, 4, 2, 1, 0, 2, 0]`. We write **G** for *GOOD*, **B** for *BAD* and **U** for *UNKNOWN*. Let's assume we have iterated all the way to position 0 and we need to decide if index 0 is *GOOD*. Since index 1 was determined to be *GOOD*, it is enough to jump there and then be sure we can eventually reach index 6. It does not matter that `nums[0]` is big enough to jump all the way to the last index. All we need is **one** way.

| Index |  0   |  1   |  2   |  3   |  4   |  5   |  6   |
| :---: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| nums  |  9   |  4   |  2   |  1   |  0   |  2   |  0   |
| memo  |  U   |  G   |  B   |  B   |  B   |  G   |  G   |

```java
public boolean canJump(int[] nums) {
    int lastPos = nums.length - 1;
    for (int i = nums.length - 1; i >= 0; --i) {
        if (i + nums[i] >= lastPos) {
            lastPos = i;
        }
    }
    return lastPos == 0;
}
```

### 56. Merge Intervals

Given a collection of intervals, merge all overlapping intervals.

**Example:**

```
Input: [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].
```

**Solution:**

对intervals按照第一个元素排序，然后遍历整个数组add到list中。

Note: 1. 背二维数组按照每行第一元素排序的写法，comparator

2. 返回值是int$[ ][ ]$但是返回值的长度不一定，先创建一个int[]的ArrayList，最后再加到int$[ ][ ]$中
3. 注意要用一个index存res走到了哪里，注意只有else的时候index才++，表明res的前一个已经成为过去式，下次考虑的就是else这次刚加进去的interval了。
4. 排序的时间复杂度是O(nlog(n)), 此solution复杂度也是O (nlog(n)).

```java
public int[][] merge(int[][] intervals) {
    if (intervals.length <= 1) return intervals;
    Arrays.sort(intervals, new Comparator<int[]>(){
        public int compare(int[] a, int[] b) {
            return a[0] - b[0];
        }
    });

    List<int[]> res = new ArrayList<>();
    res.add(intervals[0]);
    int index = 1;

    for (int i = 1; i < intervals.length; i++) {
        if (res.get(index-1)[1] >= intervals[i][1]) {
            continue;
        } else if (res.get(index-1)[1] >= intervals[i][0]) {
            res.set(index-1, new int[]{res.get(index-1)[0], intervals[i][1]});
        } else {
            index++;
            res.add(new int[]{intervals[i][0], intervals[i][1]});
        }
    }

    int[][] ret = new int[index][2];

    for (int i = 0; i < index; i++) {
        ret[i][0] = res.get(i)[0];
        ret[i][1] = res.get(i)[1];
    }
    return ret;
}
```

### 62. Unique Paths

A robot is located at the top-left corner of a *m* x *n* grid (marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

How many possible unique paths are there?

![img](https://assets.leetcode.com/uploads/2018/10/22/robot_maze.png)
Above is a 7 x 3 grid. How many possible unique paths are there?

**Note:** *m* and *n* will be at most 100.

**Solution:**

求方案总数。初始化第一行第一列全为1，dp[i] [j] = dp[i] [j-1] + dp[i-1] [j], return dp[m-1] [n-1]

求方案总数的问题要把上一步可能在的位置相加，求最短路径是取上一步可能在的位置的最小值。

```java
public int uniquePaths(int m, int n) {
    if (m == 0 || n == 0) return 1;
    int[][] dp = new int[1][n];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (i == 0 || j == 0) {
                dp[0][j] = 1;
                continue;
            }
            dp[0][j] = dp[0][j-1] + dp[0][j];
        }
    }
    return dp[0][n-1];
}
```



### 63. Unique Paths II

```
Input:
[
  [0,0,0],
  [0,1,0],
  [0,0,0]
]
Output: 2
Explanation:
There is one obstacle in the middle of the 3x3 grid above.
There are two ways to reach the bottom-right corner:
1. Right -> Right -> Down -> Down
2. Down -> Down -> Right -> Right
```

**Solution：** 

先初始化，顶点以及第一行第一列。for循环中加if判断是否有障碍物。

```java
public int uniquePathsWithObstacles(int[][] obstacleGrid) {
    int m = obstacleGrid.length;
    int n = obstacleGrid[0].length;
    if (m == 0 || n == 0) return 1;
    if (obstacleGrid[0][0] == 1) return 0;
   
    int[][] dp = new int[m][n];

    dp[0][0] = 1;

    int column = 1;
    while (column < n && obstacleGrid[0][column] != 1) {
        dp[0][column] = 1;
        column++;
    }
    while (column < n) {
        dp[0][column] = 0;
        column++;
    }

    int row = 1;
    while (row < m && obstacleGrid[row][0] != 1) {
        dp[row][0] = 1;
        row++;
    }
    while (row < m) {
        dp[row][0] = 0;
        row++;
    }

    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            if (obstacleGrid[i][j] == 0) {
                dp[i][j] = dp[i-1][j] + dp[i][j-1];
            } else {
                dp[i][j] = 0;
            }
        }
    }
    return dp[m-1][n-1];
}
```



### 64. Minimum Path Sum

**Example:**

```
Input:
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
Output: 7
Explanation: Because the path 1→3→1→1→1 minimizes the sum.
```

**Solution 1:**

O($m*n$) extra space. 

state: dp[i] [j] 表示从起点到当前位置的最短路径

function: dp[i] [j] = min(dp[i-1] [j], dp[i] [j-1]) + grid[i] [j]

initialize: 第0行，第0列为前一个数加上其本身，实际上就是从0到给该位置的和

answer: dp[m-1] [n-1]

能用dp做的题一定不存在循环依赖，即怎么走都走不出一个环，例如本题只能向下或者向右走，如果四个方向都能走的话，用BFS。规定了只能向右向下走的话，BFS只能解决耗费相同，而dp可以解决耗费不同。

初始化一个二维数组的话先初始化它的第零行第零列。本题的初始化包含在for循环中用if判断了，也可以写在外面先初始化然后再两层for循环。

```java
public int minPathSum(int[][] grid) {
    int m = grid.length;
    int n = grid[0].length;
    int[][] dp = new int [m][n];

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if(i == 0 && j == 0) {
                dp[i][j] = grid[0][0];
            } else if (i == 0) {
                dp[i][j] = dp[i][j-1] + grid[i][j];
            } else if (j == 0) {
                dp[i][j] = dp[i-1][j] + grid[i][j];
            } else {
                dp[i][j] = Math.min(dp[i-1][j], dp[i][j-1]) + grid[i][j];
            }
        }
    }
    return dp[m-1][n-1];
}
```

**Solution 2:**

O(n) extra space. dp[2] [n] 滚动数组，设置pre, cur 两个index变量，处理cur，把pre当作上一层，下次循环时把cur赋给pre。

```java
public int minPathSum(int[][] grid) {
    int m = grid.length;
    int n = grid[0].length;
    int[][] dp = new int [2][n];
    int pre = 0, cur = 0;
    for (int i = 0; i < m; i++) {
        pre = cur;
        cur = 1 - cur;
        for (int j = 0; j < n; j++) {
            if(i == 0 && j == 0) {
                dp[cur][j] = grid[0][0];
            } else if (i == 0) {
                dp[cur][j] = dp[cur][j-1] + grid[i][j];
            } else if (j == 0) {
                dp[cur][j] = dp[pre][j] + grid[i][j];
            } else {
                dp[cur][j] = Math.min(dp[pre][j], dp[cur][j-1]) + grid[i][j];
            }
        }
    }
    return dp[cur][n-1];
}
```

**Solution 3:**

O(n) extra space. dp[1] [n] 因为本题向下向右所以当前状态依赖左边和上一层，那么可以不断更新当前层即可。 

```java
public int minPathSum(int[][] grid) {
    int m = grid.length;
    int n = grid[0].length;
    int[][] dp = new int [1][n];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if(i == 0 && j == 0) {
                dp[0][j] = grid[0][0];
            } else if (i == 0) {
                dp[0][j] = dp[0][j-1] + grid[i][j];
            } else if (j == 0) {
                dp[0][j] = dp[0][j] + grid[i][j];
            } else {
                dp[0][j] = Math.min(dp[0][j], dp[0][j-1]) + grid[i][j];
            }
        }
    }
    return dp[0][n-1];
}
```



### 70. Climbing Stairs

You are climbing a stair case. It takes *n* steps to reach to the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

**Note:** Given *n* will be a positive integer.

**Example:**

```
Input: 3
Output: 3
Explanation: There are three ways to climb to the top.
1. 1 step + 1 step + 1 step
2. 1 step + 2 steps
3. 2 steps + 1 step
```

**Solution:**

dp[i] presents the number of ways to climb to the ith floor. dp[i] = dp[i-1] + dp[i-2]. dp[i] is only related to dp[i-1] and dp[i-2] which is finite, we could use two variables to store and update dp[i-1] and dp[i-2]. 需要每次先用一个cur存两个变量的和, 再更新两个变量。

```java
public int climbStairs(int n) {
    if(n<=2) return n;
    int pre1 = 1, pre2 = 2;
    for(int i = 3; i <= n; ++i) {
        int cur = pre1 + pre2;
        pre1 = pre2;
        pre2 = cur;
    }
    return pre2;
}
```



### 74. Search in a 2D Array

Write an efficient algorithm that searches for a value in an *m* x *n*matrix. This matrix has the following properties:

- Integers in each row are sorted from left to right.
- The first integer of each row is greater than the last integer of the previous row.

**Example:**

```
Input:
matrix = [
  [1,   3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 50]
]
target = 3
Output: true
```

**Solution:**

整个数组按行展开是sorted,在有序数组中search->binary search，left = 0， wight = 元素个数 -1， 将mid进行division和mod操作后转化成二维数组的坐标。本题没有重复也不求first，last，所以就使用了 left <= right 为判断条件（注意等于号不可丢保证每个元素都进行过判断，并且出循环后，left = right + 1），直接在while循环中遇到等于就返回。（其他非简单情况用left + 1 < right) 的模版。

```java
public boolean searchMatrix(int[][] matrix, int target) {
    if (matrix ==  null || matrix.length == 0) return false;
    int m = matrix.length, n = matrix[0].length;
    int left = 0, right = m * n - 1;
    int mid,r,c;
    while (left <= right) {
        mid = left + (right - left) / 2;
        r = mid / n;
        c = mid % n;
        if (matrix[r][c] == target) {
            return true;
        } else if (matrix[r][c] > target) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return false;
}
```

### **79. Word Search**

Given a 2D board and a word, find if the word exists in the grid.

The word can be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once.

**Example:**

```
board =
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]
Given word = "ABCCED", return true.
Given word = "SEE", return true.
Given word = "ABCB", return false.
```

**Solution:**

```java
private int m,n;
private int[][] directions = {{0,1},{0,-1},{1,0},{-1,0}};
public boolean exist(char[][] board, String word) {
    if(word == null || word.length() == 0) {
        return true;
    }
    if(board == null || board.length == 0 || board[0].length == 0) {
        return false;
    }
    m = board.length;
    n = board[0].length;
    boolean[][] visited = new boolean[m][n];
    for(int i=0; i<m; i++) {
        for(int j=0; j<n; j++) {
            if (backtracking(0, i, j, visited, board, word)) {
                return true;
            }
        }
    }
    return false;
}

private boolean backtracking(int curLen, int r, int c, boolean[][] visited, final char[][] board, String word) {
    if (curLen == word.length()) {
        return true;
    }
    if (r<0 || r>=m || c<0 || c>=n 
        || board[r][c] != word.charAt(curLen) || visited[r][c]) {
        return false;
    }
    visited[r][c] = true;

    for(int[] d: directions) {
        if(backtracking(curLen+1, r+d[0], c+d[1], visited, board, word)){
            return true;
        }
    }
    visited[r][c] = false;

    return false;
}
```

### 85. Maximal Rectangle

Given a 2D binary matrix filled with 0's and 1's, find the largest rectangle containing only 1's and return its area.

**Example:**

```
Input:
[
  ["1","0","1","0","0"],
  ["1","0","1","1","1"],
  ["1","1","1","1","1"],
  ["1","0","0","1","0"]
]
Output: 6
```

**Solution:**

Trivially we can enumerate every possible rectangle. This is done by iterating over all possible combinations of coordinates `(x1, y1)` and `(x2, y2)` and letting them define a rectangle with the coordinates being opposite corners. BruteForce的Time Complexity 是O($n^3$*$m^3$)  = O($n^2$$m^2$)(*find a rectangle) O(nm) (compute the sum)。

We can compute the maximum width of a rectangle that ends at a given coordinate in constant time. We do this by keeping track of the number of consecutive ones each square in each row. As we iterate over each row we update the maximum possible width at that point. This is done using `row[i] = row[i - 1] + 1 if row[i] == '1'`.

Once we know the maximum widths for each point above a given point, we can compute the maximum rectangle with the lower right corner at that point in linear time. As we iterate up the column, we know that the maximal width of a rectangle spanning from the original point to the current point is the running minimum of each maximal width we have encountered.

We define:

maxWidth = min(maxWidth, widthHere)*m**a**x**W**i**d**t**h*=*m**i**n*(*m**a**x**W**i**d**t**h*,*w**i**d**t**h**H**e**r**e*)

curArea = maxWidth * (currentRow - originalRow + 1)*c**u**r**A**r**e**a*=*m**a**x**W**i**d**t**h*∗(*c**u**r**r**e**n**t**R**o**w*−*o**r**i**g**i**n**a**l**R**o**w*+1)

maxArea = max(maxArea, curArea)*m**a**x**A**r**e**a*=*m**a**x*(*m**a**x**A**r**e**a*,*c**u**r**A**r**e**a*)

先累积求一行的最大宽度，然后以当前的格子为右下角，向上遍历列，找到以当前格子为右下角的最大面积。为0什么都不做。

```java
public int maximalRectangle(char[][] matrix) {
    if (matrix.length == 0) return 0;
    int m = matrix.length, n = matrix[0].length;
    int[][] dp = new int[m][n];
    int maxArea = 0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (matrix[i][j] == '1') {
                // compute the maximum width and update dp with it
                dp[i][j] = j == 0 ? 1 : dp[i][j-1] + 1;
            }
            int width = dp[i][j];
            // compute the maximum area rectangle with a lower right corner at [i, j]
            for (int k = i; k >= 0; k--) {
                width = Math.min(width, dp[k][j]);
                maxArea = Math.max(maxArea, width * ((i-k) + 1));
            }
        }
    }
    return maxArea;
}
```

### 91. Decode Ways

A message containing letters from `A-Z` is being encoded to numbers using the following mapping:

```
'A' -> 1
'B' -> 2
...
'Z' -> 26
```

```
Input: "12"
Output: 2
Explanation: It could be decoded as "AB" (1 2) or "L" (12).
```

**Solution:**

这里开了n+1，所以i 和 i-1对应

dp[i] 表示一charAt(i-1)结尾

```java
public int numDecodings(String s) {
    int n = s.length();
    if (n == 0) return 0;
    int prev = s.charAt(0) - '0';
    if (prev == 0) return 0;
    if (n == 1) return 1;
    int[] dp = new int[n+1];
    Arrays.fill(dp, 1);
    for (int i = 2; i <= n; ++i) {
        int cur = s.charAt(i-1) - '0';
        // 不可能decode成功的情况 0不能单独decode 必须和前一位组合
        if ((prev == 0 || prev > 2) && cur == 0) return 0;
        if (prev == 1 || prev == 2 && cur < 7) {
            // .... 2 6   = (....) (26) + (.... 2) (6)
            if (cur != 0) dp[i] = dp[i-2] + dp[i-1];
            else dp[i] = dp[i-2];
        }
        else dp[i] = dp[i-1];
        prev = cur;
    }
    return dp[n];
}
```

### 100. Same Tree

Given two binary trees, write a function to check if they are the same or not.

**Solution:**

Recursion : 当前节点，当前节点的left，当前节点的right 全部相同。

```java
public boolean isSameTree(TreeNode p, TreeNode q) {
    if (p == null && q == null) return true;
    if (p == null || q == null) return false;
    return (p.val == q.val) && isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
}
```

