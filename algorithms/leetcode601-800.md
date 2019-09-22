##LeetCode Problems 601-800

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



### 704. Binary Search

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



### 796. Rotate String

```java
public boolean rotateString(String A, String B) {
    return A.length() == B.length() && (A+A).contains(B);
}
```

