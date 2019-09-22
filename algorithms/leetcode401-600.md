##LeetCode Problems 401-600

### 413. Arithmetic Slices

A sequence of number is called arithmetic if it consists of at least three elements and if the difference between any two consecutive elements is the same.

**Example:**

```
A = [1, 2, 3, 4]

return: 3, for 3 arithmetic slices in A: [1, 2, 3], [2, 3, 4] and [1, 2, 3, 4] itself.
```

**Solution:**

求方案总数，dp[i] 表示的是以A[i]结尾的number of arithmetic slices，如果dp[i] - dp[i-1] = dp[i-1] - dp[i-2], 那么以A[i-1]结尾的所有等差递增序列在A[i]也成立，不过是长度+1，除此之外，会多一个子序列dp[i-2],dp[i-1],dp[i]。而如果上面等式不成立的话，将没有以A[i]结尾的arithmetic slices，所以dp[i]为0。在初始化dp时已经定义为0。所以dp[i] = dp[i-1] + 1。

Note: 本题求的是所有arithmatic slices，也就是说以任何一个A[i]结尾的arithmetic slices的个数之和，所以最后要遍历dp求和,因此本题也不能压缩dp数组到几个变量。

```java
public int numberOfArithmeticSlices(int[] A) {
    if (A == null || A.length <= 2) return 0;
    int n = A.length;
    int[] dp = new int[n];

    for(int i = 2; i < n; i++) {
        if (A[i] - A[i-1] == A[i-1] - A[i-2]) {
            dp[i] = dp[i-1] + 1;
        }
    }
    int total = 0;
    for(int each : dp) {
        total += each;
    }
    return total;
}
```



### 547. Friend Circles

There are **N** students in a class. Some of them are friends, while some are not. Their friendship is transitive in nature. For example, if A is a **direct** friend of B, and B is a **direct** friend of C, then A is an **indirect** friend of C. And we defined a friend circle is a group of students who are direct or indirect friends.

Given a **N\*N** matrix **M** representing the friend relationship between students in the class. If M[i][j] = 1, then the ith and jthstudents are **direct** friends with each other, otherwise not. And you have to output the total number of friend circles among all the students.

**Example 1:**

```
Input: 
[[1,1,0],
 [1,1,0],
 [0,0,1]]
Output: 2
Explanation:The 0th and 1st students are direct friends, so they are in a friend circle. 
The 2nd student himself is in a friend circle. So return 2.
```

**Example 2:**

```
Input: 
[[1,1,0],
 [1,1,1],
 [0,1,1]]
Output: 1
Explanation:The 0th and 1st students are direct friends, the 1st and 2nd students are direct friends, 
so the 0th and 2nd students are indirect friends. All of them are in the same friend circle, so return 1.
```

**Solution:**

好友关系可以看成一个无向图，本题要搞明白和island题之间的不同，

- 本题中所有的可能的情况（元素）是n个人，而不是input M中的每个元素，而对于每个元素，他可能和其他任意每个元素相连，而对于island中，每个元素只可能与上下所有四个方向的元素相连。
- 在主函数中要遍历所有元素，在dfs函数中要通过所有可能出现的边去处理下一层元素。
- 本题因为可能出现的边为剩下n个人，要先通过判断是不是好友再去调用dfs（入栈）不然先调用，再在dfs第一行写return条件的话，容易stackoverflow
- 这种类型的题目要搞清楚node和edge，主函数对每个node调用dfs，dfs对当前node可能有edge相连的所有node再做dfs，本题相当于一维的元素，每个人可能与剩下的人相连，而这种相连关系用一个矩阵表示出来了而已。
- 本题的dfs表示去标记与当前的人是好友的其他所有人。

```java
public int findCircleNum(int[][] M) {
    if (M == null || M.length == 0) {
        return 0;
    }

    int n = M.length;
    boolean[] visited = new boolean[n];

    int circle = 0;
    for(int i = 0; i < n; i++) {
        if (!visited[i]) {
            dfs(M, i, n, visited);
            circle++;
        }
    }
    return circle;
}

private void dfs(int[][] M, int i, int n, boolean[] visited) {
    visited[i] = true;
    for(int k = 0; k < n; k++) {
        if(M[i][k] == 1 && !visited[k]) {
            dfs(M, k, n, visited);
        }
    }
}
```

