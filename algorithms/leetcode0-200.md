### 2. Add Two Numbers

You are given two **non-empty** linked lists representing two non-negative integers. The digits are stored in **reverse order** and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

**Example:**

```
Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
Explanation: 342 + 465 = 807.
```

**Solution:**

The reverseness of linked list avoid the problem of alignment. 并不需要把数字reverse回来加起来后再reverse回去。倒着加就好了，相当于直接从各位开始加起。记录carry，最后需要处理一下carry。

用一个dummy node，和cur node，最后返回dummy.next。

```java
public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
    int carry = 0;
    ListNode dummy_head = new ListNode(0);
    ListNode cur = dummy_head;
    while (l1 != null || l2 != null) {
        if (l1 != null) {
            carry += l1.val;
            l1 = l1.next;
        }
        if (l2 != null) {
            carry += l2.val;
            l2 = l2.next;
        }
        cur.next = new ListNode(carry % 10);
        carry /= 10;
        cur = cur.next;
    }
    if (carry != 0) {
        cur.next = new ListNode(carry);
        cur = cur.next;
    }
    cur.next = null;
    return dummy_head.next;
}
```

### 13. Roman to Integer

Roman numerals are represented by seven different symbols: `I`, `V`, `X`, `L`, `C`, `D` and `M`.

```
Symbol       Value
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
```

For example, two is written as `II` in Roman numeral, just two one's added together. Twelve is written as, `XII`, which is simply `X` + `II`. The number twenty seven is written as `XXVII`, which is `XX` + `V` + `II`.

Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not `IIII`. Instead, the number four is written as `IV`. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as `IX`. There are six instances where subtraction is used:

**Example:**

```
Input: "III" Output: 3
Input: "IX" Output: 9
Input: "LVIII" Output: 58 Explanation: L = 50, V= 5, III = 3.
Input: "MCMXCIV" Output: 1994 Explanation: M = 1000, CM = 900, XC = 90 and IV = 4.
```

**Solution:**

Roman数字的规则：

1. 把Stirng的所有Character代表的数值进行加或者减的操作

2. 从左向右是从大到小的顺序

3. 每一位要看和后一位的关系决定自己是加还是减

4. 不会出现 “IIX”， 意味着，减的情况不会重复超过两位去判断，只需要判断其后一位就好了

   从后往前遍历，如果比前一位小则减，如果大于等于则加。

```java
public int romanToInt(String s) {
    Map<Character, Integer> map = new HashMap<Character, Integer>() {
        {
            put('I', 1); put('V', 5); put('X', 10); put('L', 50); put('C', 100); put('D', 500); put('M', 1000);
        }
    };
    if (s == null || s.length() == 0) return 0;
    int n = s.length();
    int cur = map.get(s.charAt(n-1)), post = cur, res = cur;
    for (int i = n - 2; i >= 0; i--) {
        cur = map.get(s.charAt(i));
        if (cur < post) res -= cur; 
        else res += cur;
        post = cur;
    }
    return res;
}
```



### 26. Remove Duplicates from Sorted Array

**Example:**

```
Given nums = [1,1,2],

Your function should return length = 2, with the first two elements of nums being 1 and 2 respectively.

It doesn't matter what you leave beyond the returned length.
```

**Solution:**

双指针，

### 34. Find First and Last Position of Element in Sorted Array

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



### 153. Find Minimum in Rotated Sorted Array

Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.Find the minimum element. You may assume no duplicate exists in the array.

**Example :**

```
Input: [3,4,5,1,2] 
Output: 1
```

**Solution:**

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



### 155. Min Stack

Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

- push(x) -- Push element x onto stack.
- pop() -- Removes the element on top of the stack.
- top() -- Get the top element.
- getMin() -- Retrieve the minimum element in the stack.

**Solution:**

首先理解min stack 定义，首先这是个stack，其次他是个可以在O(1)时间内拿出最小值的stack。pop只是pop栈顶的值并不是pop出最小值。

用两个stack来写，一个stack保存入栈、出栈的顺序，另一个来保存最小值。两个stack的大小是一样的。解决了万一把最小值pop出来，仍然可以在O(1)时间拿到最小值。

```java
class MinStack {
    /** initialize your data structure here. */
    Stack<Integer> minStack;
    Stack<Integer> dataStack;
    public MinStack() {
        minStack = new Stack<>();
        dataStack = new Stack<>();
    }
    public void push(int x) {
        dataStack.push(x);
        if (minStack.isEmpty() || x < minStack.peek()) {
            minStack.push(x);
        } else {
            minStack.push(minStack.peek());
        }
    }
    public void pop() {
        dataStack.pop();
        minStack.pop();
    }
    public int top() {
         return dataStack.peek();
    }
    public int getMin() {
        return minStack.peek();
    }
}
```



### 189. Rotate Array

Given an array, rotate the array to the right by *k* steps, where *k* is non-negative.

**Example 1:**

```
Input: [1,2,3,4,5,6,7] and k = 3
Output: [5,6,7,1,2,3,4]
```

**Solution: **

三步翻转，分别翻转前 length - k 个，再翻转后边k个，最后翻转整个数组。

NOTE: 1.  注意要对k取mode, 不然会out of index

​	        2. 注意传参

```java
public void rotate(int[] nums, int k) {
    if (nums == null || nums.length == 0) return;
    k = k % nums.length;
    reverse(nums, 0, nums.length - k - 1);
    reverse(nums, nums.length - k, nums.length - 1);
    reverse(nums, 0, nums.length - 1);
}

private void reverse(int[] nums, int start, int end) {
    while (start < end) {
        int tmp = nums[start];
        nums[start] = nums[end];
        nums[end] = tmp;
        start++;
        end--;
    }
}
```



### 191. Number of 1 Bits

Write a function that takes an unsigned integer and return the number of '1' bits it has (also known as the [Hamming weight](http://en.wikipedia.org/wiki/Hamming_weight)).

**Solution:**

按位& then 右移。java中没有unsigned datatype. To solve this, we could limit the time of shift <= 32, or use >>> .  >> : use the sign bit to fill the trailing positions after shift. >>> : unsigned right shift.

```java
public int hammingWeight(int n) {
    int res = 0;
    while (n != 0 ) {
        res += n & 1;
        n >>>= 1;
    }
    return res;
}
```



### 198. House Robber

抢劫一排住户，但不能抢劫相邻的住户，求最大抢劫量。

dp[i] 表示抢劫到第i个时，最大的抢劫量。dp[i] = max(dp[i-1], nums[i] + dp[i-2])。 一维的数组时，可以压缩成相邻的几个变量。注意不能给pre2赋值为nums[1]。

```java
public int rob(int[] nums) {
    if (nums == null || nums.length == 0) return 0;
    if (nums.length == 1) return nums[0];
    int pre1 = 0, pre2 = 0, cur = 0;

    for (int i = 0; i < nums.length; ++i) {
        cur = Math.max(pre2, nums[i] + pre1);
        pre1 = pre2;
        pre2 = cur;
    }

    return pre2;
}
```

### 200. Number Of Islands

**Example 1:**

```
Input:
11110
11010
11000
00000
Output: 1
```

**Example 2:**

```
Input:
11000
11000
00100
00011
Output: 3
```

**Solution:**

本题同max area of island类似，矩阵可以看为一个有向图，在主函数中调用dfs时先要判断是否为0，在dfs function中去搜索四个方向，把当前元素以及四个方向的land改为water，以后不会再遍历到这个元素。所以两层for循环，对每个land，及其联通的land标记为0，整个记为一个island。

```java
private int m,n;
private int[][] directions = {{0,1},{0,-1},{1,0},{-1,0}};
public int numIslands(char[][] grid) {
    if(grid == null || grid.length == 0) {
        return 0;
    }
    m = grid.length;
    n = grid[0].length;
    int numberOfIsland = 0;
    for(int i=0; i<m; i++) {
        for(int j=0; j<n; j++) {
            if(grid[i][j] != '0') {
                dfs(grid, i, j);
                numberOfIsland++;
            }
        }
    }
    return numberOfIsland;
}

private void dfs(char[][] grid, int i, int j) {
    if(i<0 || i>=m || j<0 || j>=n || grid[i][j] == '0') {
        return;
    }
    grid[i][j] = '0';
    for(int[] d : directions) {
        dfs(grid, i+d[0], j+d[1]);
    }
}
```

