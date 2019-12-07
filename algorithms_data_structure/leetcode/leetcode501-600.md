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

### 551. Student Attendance Record I

You are given a string representing an attendance record for a student. The record only contains the following three characters:

1. **'A'** : Absent.
2. **'L'** : Late.
3. **'P'** : Present.

A student could be rewarded if his attendance record doesn't contain **more than one 'A' (absent)** or **more than two continuous 'L' (late)**.

You need to return whether the student could be rewarded according to his attendance record.

**Example 1:**

```
Input: "PPALLP"
Output: True
```

**Solution:**

首先注意读题，连续的三个L。对于这种tripel的问题，像334一样，可以设置两个变量来存是否起前一个是L和是否前两个是L，然后用两个if else注意顺序来update这两个变量，如果第一个不是true，则更新第一个，否则去看第二个是不是true，两个都是true的话，则满足不reward条件。

另外要注意本题是连续三个L，所以在遇到非L时，需要重置那两个变量。

```java
public boolean checkRecord(String s) {
    int cntA = 0;
    boolean isL = false;
    boolean isSecL = false;
    for (int i = 0; i < s.length(); ++i) {
        char c = s.charAt(i);
        if (c == 'A') {
            isL = false;
            isSecL = false;
            cntA += 1;
        } else if (c == 'L'){
            if (!isL)  isL = true;
            else if (!isSecL) isSecL = true;
            else return false;
        } else {
            isL = false;
            isSecL = false;
        }
        if(cntA > 1) return false;
    }
    return true;
}
```

### **583. Delete Operation for Two Strings**

Given two words *word1* and *word2*, find the minimum number of steps required to make *word1* and *word2* the same, where in each step you can delete one character in either string.

**Example 1:**

```
Input: "sea", "eat"
Output: 2
Explanation: You need one step to make "sea" to "ea" and another step to make "eat" to "ea".
```

**Solution:**

等价于最长公共子序列问题。

```java
public int minDistance(String word1, String word2) {
    int m = word1.length(), n = word2.length();
    int[][] dp = new int[m+1][n+1];
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            dp[i][j] = word1.charAt(i-1) == word2.charAt(j-1) ? 1 + dp[i-1][j-1] : Math.max(dp[i-1][j], dp[i][j-1]);
        }
    }
    return m + n - 2 * dp[m][n];
}
```

### 