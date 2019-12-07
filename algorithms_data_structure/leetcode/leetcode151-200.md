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

### 171. Excel Sheet Column Number

Given a column title as appear in an Excel sheet, return its corresponding column number.

For example:

```
    A -> 1
    B -> 2
    C -> 3
    ...
    Z -> 26
    AA -> 27
    AB -> 28 
    ...
```

**Solution:**

把用A-Z表示的26进制转换成十进制。注意进制转换公式：

26 -> 10: 1. 从头开始遍历

2. res = res * (现在的进制) + 当前这位对应的目标进制的值

```java
public int titleToNumber(String s) {
    if (s == null || s.length() == 0) return -1;
    int res = 0;
    for (int i = 0; i < s.length(); ++i) {
        char c = s.charAt(i);
        res = res * 26 + (c - 'A' + 1);
    }
    return res;
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

