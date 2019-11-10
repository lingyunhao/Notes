### 410. Split Array Largest Sum

Given an array which consists of non-negative integers and an integer *m*, you can split the array into *m* non-empty continuous subarrays. Write an algorithm to minimize the largest sum among these *m* subarrays.

**Examples:**

```
Input:
nums = [7,2,5,10,8]
m = 2

Output:
18

Explanation:
There are four ways to split nums into two subarrays.
The best way is to split it into [7,2,5] and [10,8],
where the largest sum among the two subarrays is only 18.
```

**Solution:**

DP

```java
public int splitArray(int[] nums, int m) {
    int n = nums.length;
    int[][] f = new int[n + 1][m + 1];
    int[] sub = new int[n + 1];
    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= m; j++) {
            f[i][j] = Integer.MAX_VALUE;
        }
    }
    for (int i = 0; i < n; i++) {
        sub[i + 1] = sub[i] + nums[i];
    }
    f[0][0] = 0;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            for (int k = 0; k < i; k++) {
                f[i][j] = Math.min(f[i][j], Math.max(f[k][j - 1], sub[i] - sub[k]));
            }
        }
    }
    return f[n][m];        
}
```

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

### 417. Pacific Atlantic Water Flow

Given an `m x n` matrix of non-negative integers representing the height of each unit cell in a continent, the "Pacific ocean" touches the left and top edges of the matrix and the "Atlantic ocean" touches the right and bottom edges.

Water can only flow in four directions (up, down, left, or right) from a cell to another one with height equal or lower.

Find the list of grid coordinates where water can flow to both the Pacific and Atlantic ocean.

**Solution:**

DFS, 起始结点是第0行，n-1行，第0列，n-1列。把和第0行，n-1行，第0列，n-1列联通的node设置为相应的true。相连的定义是从起始结点开始，cur的值<=next的值，相当于水倒着流去推的。用两个boolean分别去记录Atlantic 和 Pacific

```java
int[][] directions = new int[][]{{1,0}, {-1,0}, {0,1}, {0,-1}};
public List<List<Integer>> pacificAtlantic(int[][] matrix) {
    if (matrix == null) return null;
    if (matrix.length == 0) return new ArrayList();
    List<List<Integer>> res = new ArrayList<>();
    int m = matrix.length, n = matrix[0].length;
    boolean[][] reach_p = new boolean[m][n];
    boolean[][] reach_a = new boolean[m][n];
    // 与第0列相连的（相连的定义有海拔限制）可以到达pacific
    // 与最后一列相连的，可以到达atlantic
    for (int i = 0; i < m; ++i) {
        dfs(matrix, reach_p, i, 0, m, n);
        dfs(matrix, reach_a, i, n-1, m, n);
    }
    // 与第0行相连的（相连的定义有海拔限制）可以到达pacific
    // 与最后一行相连的，可以到达atlantic
    for (int i = 0; i < n; ++i) {
        dfs(matrix, reach_p, 0, i, m, n);
        dfs(matrix, reach_a, m-1, i, m, n);
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (reach_p[i][j] && reach_a[i][j]) 
                res.add(new ArrayList(Arrays.asList(i, j)));
        }
    }
    return res;
}

// 将联通区域设置为ture（此时的联通定义中多了个条件就是海拔高度限制
private void dfs(int[][] matrix, boolean[][] can_reach, int r, int c, int m, int n) {
    // 出口
    if (can_reach[r][c]) return;
    can_reach[r][c] = true;
    for (int[] d : directions) {
        int nextR = r + d[0], nextC = c + d[1];
        // 出口
        if (nextR < 0 || nextR >= m || nextC < 0 || nextC >= n || matrix[r][c] > matrix[nextR][nextC]) continue;
        dfs(matrix, can_reach, nextR, nextC, m, n);
    }
}
```

### 445. Add Two Numbers II

You are given two **non-empty** linked lists representing two non-negative integers. The most significant digit comes first and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

**Follow up:**
What if you cannot modify the input lists? In other words, reversing the lists is not allowed.

**Example:**

```
Input: (7 -> 2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 8 -> 0 -> 7
```

**Solution1:**

reverse函数 + add

```java
public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
    return reverse(add(reverse(l1),reverse(l2)));
}

public ListNode reverse(ListNode head)
{
    ListNode cur=head, prev=null, next=null;
    while(cur!=null){
        next = cur.next;
        cur.next = prev;
        prev=cur;
        cur=next;
    }
    return prev;
}

public ListNode add(ListNode l1, ListNode l2)
{
    ListNode dummy = new ListNode(0);
    ListNode prev = dummy;
    int carry=0;
    while(l1!=null||l2!=null)
    {
        int sum=0;
        if(l1!=null)
        {
            sum+=l1.val;
            l1=l1.next;
        }
        if(l2!=null)
        {
            sum+=l2.val;
            l2=l2.next;
        }
        sum += carry;
        carry = sum/10;
        sum = sum%10;
        prev.next = new ListNode(sum);
        prev = prev.next;
    }
    // 最高位有进位别忘了！！！！
    if(carry !=0)
        prev.next= new ListNode(1);
    return dummy.next;   
}
```

**Solution2:**

借助数据结构来reverse一个linkedlist。stack

```java
public ListNode addTwoNumbers(ListNode l1, ListNode l2) {

    Stack<Integer> stack1 = new Stack<>();
    Stack<Integer> stack2 = new Stack<>();
    while(l1 != null) {
        stack1.push(l1.val);
        l1 = l1.next;
    }
    while(l2 != null) {
        stack2.push(l2.val);
        l2 = l2.next;
    }
    ListNode resultNode = null;
    int carry = 0;
    // 别忘了最高位的carry
    while(!stack1.isEmpty() || !stack2.isEmpty() || carry>0) {
        if(stack1.isEmpty()) {
            stack1.push(0);
        }
        if(stack2.isEmpty()) {
            stack2.push(0);
        }
        int value = stack1.pop() + stack2.pop() + carry;
        carry = value / 10;
        value = value % 10;
        resultNode = insertAtHead(resultNode, value);
    }
    return resultNode;
}

public ListNode insertAtHead(ListNode node, int value){
    ListNode newNode = new ListNode(value);
    newNode.next = node;
    return newNode;

}
```

### 482. License Key Formatting

You are given a license key represented as a string S which consists only alphanumeric character and dashes. The string is separated into N+1 groups by N dashes.

Given a number K, we would want to reformat the strings such that each group contains *exactly* K characters, except for the first group which could be shorter than K, but still must contain at least one character. Furthermore, there must be a dash inserted between two groups and all lowercase letters should be converted to uppercase.

**Example 1:**

```
Input: S = "5F3Z-2e-9-w", K = 4

Output: "5F3Z-2E9W"

Explanation: The string S has been split into two parts, each part has 4 characters.
Note that the two extra dashes are not needed and can be removed.
```

**Solution:**

解法不难，但是很多小细节要注意。 1. 什么时候该加 '-' 2. char is primitive time, can not be referenced by ., Character can. 注意char变大小写的写法。 3. 注意firstGroup可能没有的情况，就是说刚好整除。

```java
public String licenseKeyFormatting(String S, int K) {
    if (S == null || S.length() == 0) return S;
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < S.length(); ++i) {
        if (S.charAt(i) != '-') {
            sb.append(Character.toUpperCase(S.charAt(i)));
        }
    }

    String sWithoutDash = sb.toString();
    int n = sWithoutDash.length();
    int group = n/K;
    int firstGroup = n%K;

    StringBuilder res = new StringBuilder();
    if (firstGroup > 0) group++;
    int index = 0;
    for (int i = 0; i < group; ++i) {
        if (i == 0 && firstGroup > 0) {
            while(firstGroup > 0) {
                res.append(sWithoutDash.charAt(index++));
                --firstGroup;
            }
            if (i != group - 1) {
                res.append('-');
            }
        } else {
            for (int j = 0; j < K; ++j) {
                res.append(sWithoutDash.charAt(index++));
            }
            if (i != group - 1) {
                res.append('-');
            }
        }
    }
    return res.toString();
}
```

### 