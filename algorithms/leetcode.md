## LeetCode Problems

#### 70. Climbing Stairs

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

#### 120. Triangle

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

#### 283. Move Zeroes

Given an array `nums`, write a function to move all `0`'s to the end of it while maintaining the relative order of the non-zero elements.

**Example:**

```
Input: [0,1,0,3,12]
Output: [1,3,12,0,0]
```

**Solution:**

同向双指针，对j进行for循环，每次遇到非零数把nums[j]赋给nums[i]，并且i++。循环结束后，i记录的是非零数的个数，此时把数组中剩下的数全赋值0即可。

```java
public void moveZeroes(int[] nums) {
    if(nums == null || nums.length == 0) return;
    int i = 0;
    for(int j = 0; j < nums.length; ++j) {
        if(nums[j] != 0) {
            nums[i++] = nums[j];
        }
    }
    while(i < nums.length) {
        nums[i] = 0;
        i++;
    }
}
```

#### 300. Longest Increasing Subsequence

Given an unsorted array of integers, find the length of longest increasing subsequence.

**Example:**

```
Input: [10,9,2,5,3,7,101,18]
Output: 4 
Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4. 
```

**Note:**

- There may be more than one LIS combination, it is only necessary for you to return the length.
- Your algorithm should run in O(*n2*) complexity.

**Solution 1:**

DP with time compelexity O($n^2$).

dp[i] stores the longest increasing subsequence ending with nums[i]. dp[i] = max(dp[j]) +1 | j<i and nums[j] < nums[i]. 本题的初始化需要将dp数组全置为1，在循环中初始化，如果i前面没有比自己小的，则dp[i]为1。最后的结果可能是以任意一个位置结尾的，需要对dp打擂台求最大值。

```java
public int lengthOfLIS(int[] nums) {
    int n = nums.length;
    int[] dp = new int[n];

    for(int i=0; i<n; ++i) {
        //Initialize
        int max = 1;
        for(int j=0; j<i; ++j) {
            if(nums[i] > nums[j]) {
                max = Math.max(dp[j] + 1, max);
            }
        }
        dp[i] = max;
    }

    int result = 0;
    for(int i = 0; i < n; ++i) {
        result = Math.max(dp[i], result);
    }
    return result;
}
```



