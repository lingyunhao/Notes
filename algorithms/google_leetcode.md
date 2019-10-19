## Google tag

### 3. Longest Substring Without Repeating Characters

Given a string, find the length of the **longest substring** without repeating characters.

**Example 1:**

```
Input: "abcabcbb"
Output: 3 
Explanation: The answer is "abc", with the length of 3. 
```

**Solution:**

Sliding window, 用双指针i,j去指向不重复的字符串，window的两端。

```java
public int lengthOfLongestSubstring(String s) {
    int n = s.length(), ans = 0;
    int[] index = new int[128];
    Arrays.fill(index, -1);
    int i = 0;
    //sliding window [i, j]
    for (int j = 0; j < n; ++j) {
        // 如果上次出现的s.charAt(j)在当前的sliding window中，则表明重复
        if (index[s.charAt(j)] < i) {
            ans = Math.max(ans, j - i + 1);
        } else {
            i = index[s.charAt(j)] + 1;
        }
        index[s.charAt(j)] = j;
    }
    return ans;
}
```

### 5. Longest Palindromic Substring

Given a string **s**, find the longest palindromic substring in **s**. You may assume that the maximum length of **s** is 1000.

**solution:**

注意初始化（没看懂）、还有两个forloop怎么写。

To improve over the brute force solution, we first observe how we can avoid unnecessary re-computation while validating palindromes. Consider the case "ababa". If we already knew that "bab" is a palindrome, it is obvious that "ababa" must be a palindrome since the two left and right end letters are the same.

We define P(i,j)*P*(*i*,*j*) as following:

P(i,j) = \begin{cases} \text{true,} &\quad\text{if the substring } S_i \dots S_j \text{ is a palindrome}\\ \text{false,} &\quad\text{otherwise.} \end{cases}*P*(*i*,*j*)={true,false,if the substring *S**i*…*S**j* is a palindromeotherwise.

Therefore,

P(i, j) = ( P(i+1, j-1) \text{ and } S_i == S_j )*P*(*i*,*j*)=(*P*(*i*+1,*j*−1) and *S**i*==*S**j*)

The base cases are:

P(i, i) = true*P*(*i*,*i*)=*t**r**u**e*

P(i, i+1) = ( S_i == S_{i+1} )*P*(*i*,*i*+1)=(*S**i*==*S**i*+1)

This yields a straight forward DP solution, which we first initialize the one and two letters palindromes, and work our way up finding all three letters palindromes, and so on...

```java
public String longestPalindrome(String s) {
    if (s == null || s.length() == 0) return s;
    int n = s.length();
    char[] chars = s.toCharArray();
    boolean[][] dp = new boolean[n][n];
    int row = 0, col = 0;
    for (int i = n - 1; i >= 0; --i) {
        for (int j = i; j < n; ++j) {
            // 初始化
            if (j == i || (chars[i] == chars[j] && j-i <= 1)) {
                dp[i][j] = true;
            } else {
                //通项公式
                dp[i][j] = dp[i+1][j-1] && chars[i] == chars[j];
            }
            if (dp[i][j] && (j - i > col - row)) {
                col = j;
                row = i;
            }
        }
    }
    return s.substring(row, col+1);
}
```

###46. Permutations

Given a collection of **distinct** integers, return all possible permutations.

不含相同元素排列

**Example:**

```
Input: [1,2,3]
Output:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]
```

**Solution:**

背

```java
public List<List<Integer>> permute(int[] nums) {
    List<List<Integer>> permutes = new ArrayList<>();
    List<Integer> permuteList = new ArrayList<>();
    boolean[] visited = new boolean[nums.length];
    backtracking(permutes, permuteList, visited, nums);
    return permutes;
}

private void backtracking(List<List<Integer>> permutes, List<Integer> permuteList, boolean[] visited, int[] nums) {
    if(permuteList.size() == visited.length) {
        permutes.add(new ArrayList<>(permuteList));
        return;
    }
    for(int i=0; i<visited.length; i++) {
        if(visited[i]) continue;
        visited[i] = true;
        permuteList.add(nums[i]);
        backtracking(permutes, permuteList, visited, nums);
        permuteList.remove(permuteList.size()-1);
        visited[i] = false;
    }
}
```

###47. Permutations II

Given a collection of numbers that might contain duplicates, return all possible unique permutations.

含有相同元素排列

**Example:**

```
Input: [1,1,2]
Output:
[
  [1,1,2],
  [1,2,1],
  [2,1,1]
]
```

**Solution:**

数组元素可能含有相同的元素，进行排列时就有可能出现重复的排列，要求重复的排列只返回一个。

先sort然后多判断了一下

```java
public List<List<Integer>> permuteUnique(int[] nums) {
    List<List<Integer>> permutes = new ArrayList<>();
    List<Integer> permuteList = new ArrayList<>();
    Arrays.sort(nums);
    boolean[] visited = new boolean[nums.length];
    backtracking(permutes, permuteList, visited, nums);
    return permutes;
}

private void backtracking(List<List<Integer>> permutes, List<Integer> permuteList, boolean[] visited, int[] nums) {
    if(permuteList.size() == nums.length) {
        permutes.add(new ArrayList<>(permuteList));
        return;
    }

    for(int i=0; i<visited.length; i++) {
        //排除重复
        if(i!=0 && nums[i] == nums[i-1] && !visited[i-1]) {
            continue;
        }
        if(visited[i]) continue;
        visited[i] = true;
        permuteList.add(nums[i]);
        backtracking(permutes, permuteList, visited, nums);
        permuteList.remove(permuteList.size()-1);
        visited[i] = false;
    }
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

###**79. Word Search**

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
            if (backtracking(0, i,j,visited, board, word)) {
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

### 130. Surrounded Regions

Given a 2D board containing `'X'` and `'O'` (**the letter O**), capture all regions surrounded by `'X'`.

A region is captured by flipping all `'O'`s into `'X'`s in that surrounded region.

**Example:**

```
X X X X
X O O X
X X O X
X O X X
```

**Solution:**

先用dfs把最外围的联通量改成另一个字母'T',然后把里边的'O'全改为'X',最后把'T'全部改回来。

```java
public void solve(char[][] board) {
    if (board == null || board.length == 0) return;
    int m = board.length, n = board[0].length;
    for (int i = 0; i < m; ++i) {
        dfs(board, i, 0, m, n);
        dfs(board, i, n-1, m, n);
    }
    for (int i = 0; i < n; ++i) {
        dfs(board, 0, i, m, n);
        dfs(board, m-1, i, m, n);
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (board[i][j] == 'O') board[i][j] = 'X';
            if (board[i][j] == 'T') board[i][j] = 'O';
        }
    }
}

public void dfs(char[][] board, int r, int c, int m, int n) {
    if (r < 0 || r >= m || c < 0 || c >= n ||  board[r][c] != 'O') return;
    board[r][c] = 'T';
    dfs(board, r+1, c, m, n);
    dfs(board, r-1, c, m, n);
    dfs(board, r, c+1, m, n);
    dfs(board, r, c-1, m, n);
}
```

### 139. Word Break

Given a **non-empty** string *s* and a dictionary *wordDict* containing a list of **non-empty** words, determine if *s* can be segmented into a space-separated sequence of one or more dictionary words.

**Solution: dp**

```java
public boolean wordBreak(String s, List<String> wordDict) {
    int n = s.length();
    boolean[] dp = new boolean[n+1];
    dp[0] = true;
    for (int i = 1; i <= n; ++i) {
        for (String word : wordDict) {
            int len = word.length();
            if (i < len) continue;
            if (s.substring(i - len, i).equals(word)) dp[i] = dp[i] || dp[i-len];
        }
    }
    return dp[n];
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

### 205.Isomorphic Strings

Given two strings **s** and **t**, determine if they are isomorphic.

Two strings are isomorphic if the characters in **s** can be replaced to get **t**.

All occurrences of a character must be replaced with another character while preserving the order of characters. **No two characters may map to the same character but a character may map to itself.**

**Example 1:**

```
Input: s = "egg", t = "add"
Output: true
```

**Example 2:**

```
Input: s = "foo", t = "bar"
Output: false
```

**Solution:**

注意本题S中的两个character不能map到同一个character，用两个map分别存character和上一次出现的位置。所以讨论false的情况，1. map中一个存在另一个不存在 2. 两个都存在但是位置不相同。Integer的比较应该用equals而不是==，他是个object。

```java
public boolean isIsomorphic(String s, String t) {
    if (s.length() != t.length()) {
        return false;
    }
    int n = s.length();
    Map<Character, Integer> map1 = new HashMap<Character, Integer>();
    Map<Character, Integer> map2 = new HashMap<Character, Integer>();
    for (int i = 0; i < n; i++) {
        char cs = s.charAt(i);
        char ct = t.charAt(i);
        if ((map1.containsKey(cs) && !map2.containsKey(ct)) || (!map1.containsKey(cs) && map2.containsKey(ct))) {
            return false;
        } else if (map1.containsKey(cs) && map2.containsKey(ct) && !map1.get(cs).equals(map2.get(ct))) {
            return false;
        }
        map1.put(cs,i);
        map2.put(ct,i);
    }
    return true;
}
```

或者用数组来代替map，index是char的值，value是string中的index。

```c++
bool isIsomorphic(string s, string t) {
    vector<int> s_first_index (256, 0), t_first_index (256, 0);
    for (int i = 0; i < s.length(); ++i) {
        if (s_first_index[s[i]] != t_first_index[t[i]]) return false;
        s_first_index[s[i]] = i + 1;
        t_first_index[t[i]] = i + 1;
    }
    return true;
}
```

### 208. Implement Trie (Prefix Tree)

Trie, prefix tree, 用于判断一个字符串是否存在(search)或者字符串有某种前缀(startwith),search 和 prefix 的区别就是最后遍历到字符串末尾时候一个return node.isLeaf, 一个直接return true。

```java
class Trie {
    /** Initialize your data structure here. */
    private class Node {
        Node[] children = new Node[26];
        boolean isLeaf;
    }
    private Node root;
    // Constructor
    public Trie() {
        root = new Node();
    }
    /** Inserts a word into the trie. */
    public void insert(String word) {
        insert(word, root);
    }
    private void insert(String word, Node node) {
        Node tmp = node;
        for (int i = 0; i < word.length(); ++i) {
            int index = charToIndex(word.charAt(i));
            if (tmp.children[index] == null) {
                tmp.children[index] = new Node();
            }
            tmp = tmp.children[index];
        }
        tmp.isLeaf = true;
    }
    /** Returns if the word is in the trie. */
    public boolean search(String word) {
        return search(word, root);
    }
    private boolean search(String word, Node node) {
        Node tmp = node;
        for (int i = 0; i < word.length(); ++i) {
            int index = charToIndex(word.charAt(i));
            if (tmp.children[index] == null) {
                return false;
            } else { 
                tmp = tmp.children[index];
            }
        }
        return tmp.isLeaf;
    }
    /** Returns if there is any word in the trie that starts with the given prefix. */
    public boolean startsWith(String prefix) {
        return startsWith(prefix, root);
    }
    private boolean startsWith(String prefix, Node node) {
        Node tmp = node;
        for (int i = 0; i < prefix.length(); ++i) {
            int index = charToIndex(prefix.charAt(i));
            if (tmp.children[index] == null) {
                return false;
            } else {
                tmp = tmp.children[index];
            }
        }
        return true;
    }
    private int charToIndex(char c) {
        return c - 'a';
    }
}
```

### 213. House Robber II

You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. All houses at this place are **arranged in a circle.** That means the first house is the neighbor of the last one. Meanwhile, adjacent houses have security system connected and **it will automatically contact the police if two adjacent houses were broken into on the same night**.

Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight **without alerting the police**.

**Solution:**

cycle的情况分是否rob第一个。用两个dp，dp1的通项公式有点不懂。

```java
public int rob(int[] nums) {
    if (nums == null || nums.length == 0) return 0;
    if (nums.length == 1) return nums[0];
    int n = nums.length;
    int[] dp1 = new int[nums.length];
    int[] dp2 = new int[nums.length];

    // rob nums[0]
    dp1[1] =  nums[0];
    dp2[1] = nums[1];
    for (int i = 1; i < n - 1; ++i) {
        dp1[i+1] = Math.max(dp1[i], dp1[i-1] + nums[i]);
        dp2[i+1] = Math.max(dp2[i], dp2[i-1] + nums[i+1]);
    }
    return Math.max(dp1[n-1], dp2[n-1]);
}
```

### 221. Maximal Square

Given a 2D binary matrix filled with 0's and 1's, find the largest square containing only 1's and return its area.

**Example:**

```
Input: 
1 0 1 0 0
1 0 1 1 1
1 1 1 1 1
1 0 0 1 0
Output: 4
```

**Solution:**

dp。 dp$[i][j]$  表示以matrix$[i][j]$为右下角的最大正方形的边长。 dp$[i][j]$  = min( dp$[i-1][j]$  , dp$[i-1][j-1]$  , dp$[i][j-1]$  )+1。 上面，左面，左上角的最小值保证了能取到的最大的正方形。初始化第0行，第0列和matrix本身相同。在赋值同时打擂台获得最大值（边长）。

初始化的问题也可以一开始开一个m+1,n+1的dp，直接从i=1,j=1开始,这样相当于在原来的matrix加了全为零的第零行和全为零的第零列。

```java
public int maximalSquare(char[][] matrix) {
    if (matrix == null || matrix.length == 0) return 0;
    int m = matrix.length, n = matrix[0].length;
    int[][] dp = new int[m][n];
    int max = Integer.MIN_VALUE;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (i == 0 || j == 0) {
                if (matrix[i][j] == '1') {
                    dp[i][j] = 1;
                } else {
                    dp[i][j] = 0;
                }
            } else {
                if (matrix[i][j] == '1') {
                    dp[i][j] = Math.min(Math.min(dp[i-1][j], dp[i-1][j-1]), dp[i][j-1]) + 1; 
                }
            }
            max = Math.max(max, dp[i][j]);
        }
    }
    return max * max;
}
```

### 222. Count Complete Tree Nodes

Given a **complete** binary tree, count the number of nodes.
In a complete binary tree every level, except possibly the last, is completely filled, and all nodes in the last level are as far left as possible. It can have between 1 and 2h nodes inclusive at the last level h.

**Solution:**

cnt表示最后一层的个数，别忘了循环只进行了h-1次，最后一层一定到了最后一层的根节点上，但是没有进行判断把这个节点加进去，所以最后要判断最后节点，如果此节点不为null要+1.

Math.pow(2, h-1)是前h-1层的结点个数之和。

求高度是O(lgn)的复杂度，在一个h-1的for循环中，又求高度，所以是O(lgn * lgn)的复杂度， 比O(n)小很多。 可以用搜索这样每个节点遍历一遍，时间复杂度为O(n)。

```java
private int countHeight(TreeNode root) {
    return (root == null) ? 0 : countHeight(root.left) + 1;
}
public int countNodes(TreeNode root) {
    if (root == null) return 0;
    int h = countHeight(root);
    int cnt = 0;
    for (int i = h - 1; i > 0; --i) {
        if (countHeight(root.right) == i) {
            cnt += (int)Math.pow(2, i-1);
            root = root.right;
        } else {
            root = root.left;
        }
    }
    if (root != null) cnt++;
    return (int)Math.pow(2, h-1) - 1 + cnt;
}
```

### 241. Different Ways to Add Parentheses

Given a string of numbers and operators, return all possible results from computing all the different possible ways to group numbers and operators. The valid operators are `+`, `-` and `*`.

**Example 1:**

```
Input: "2-1-1"
Output: [0, 2]
Explanation: 
((2-1)-1) = 0 
(2-(1-1)) = 2
```

**Example 2:**

```
Input: "2*3-4*5"
Output: [-34, -14, -10, -10, 10]
Explanation: 
(2*(3-(4*5))) = -34 
((2*3)-(4*5)) = -14 
((2*(3-4))*5) = -10 
(2*((3-4)*5)) = -10 
```

**Solution:**

先把数字和运算符分别存在两个arraylist中。

dp[i] [j] 表示从第i个数字到第j个数字之间各种运算符所有可能出现的结果。dp[i] [j] 则是dp[i] [k] 和dp[k+1] [j]中的所有数字在第k个运算符下的所有的combination。i < k, k < j所以要知道dp[i] [j] 则必须知道dp[i] [k] 和dp[k+1] [j]，则i是从右向左走，j 是从i开始向右走，k是从i到j。

在ops最后加一个+使得数字和运算符的个数相同。

是典型的divide conquer的题，对于一个符号，分别算左边的和右边的然后把左右两边用此富豪连接起来。divide conquer要用recursion写。

```java
public List<Integer> diffWaysToCompute(String input) {
    List<Integer> data = new ArrayList<>();
    List<Character> ops = new ArrayList<>();

    for (int i = 0; i < input.length();) {
        if (input.charAt(i) == '+' || input.charAt(i) == '-' || input.charAt(i) == '*') {
            ops.add(input.charAt(i));
            ++i;
        } else {
            StringBuilder sb = new StringBuilder();
            while (i < input.length() && input.charAt(i) != '+' && input.charAt(i) != '-' && input.charAt(i) != '*') {
                sb.append(input.charAt(i));
                ++i;
            }
            data.add(Integer.valueOf(sb.toString()));
        }
    }

    ops.add('+');
    int size = data.size();

    //array of list 的定义方法
    List<Integer>[][] dp = new List[size][size];
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            dp[i][j] = new ArrayList<>();
        }
    }
    for (int i = size; i >= 0; --i) {
        for (int j = i; j < size; ++j) {
            // 初始化
            if (i == j) { dp[i][j].add(data.get(i)); continue;}
            for (int k = i; k < j; k++) {
                for (int left : dp[i][k]) {
                    for (int right : dp[k+1][j]) {
                        int val = 0;
                        switch (ops.get(k)) {
                            // 别忘了break
                            case '+' : val = left + right; break;
                            case '-' : val = left - right; break;
                            case '*' : val = left * right; break;
                        }
                        dp[i][j].add(val);
                    }
                }
            }
        }
    }
    return dp[0][size-1];
}
```



### 243. Shortest Word Distance

Given a list of words and two words *word1* and *word2*, return the shortest distance between these two words in the list.

**Example:**
Assume that words = `["practice", "makes", "perfect", "coding", "makes"]`.

```
Input: word1 = “coding”, word2 = “practice”
Output: 3
Input: word1 = "makes", word2 = "coding"
Output: 1
```

**Solution:**

注意res那里不能直接return还是要打擂台。

```java
public int shortestDistance(String[] words, String word1, String word2) {
    if (word1.equals(word2)) return 0;
    int index1 = -1, index2 = -1;
    int res = Integer.MAX_VALUE;
    for (int i = 0; i < words.length; ++i) {
        if (words[i].equals(word1)) {
            index1 = i;
        } else if (words[i].equals(word2)) {
            index2 = i;
        }
        if (index1 != -1 && index2 != -1) res = Math.min(res, Math.abs(index1 - index2));
    }
    return res;
}
```

### 247. Strobogrammatic Number II

A strobogrammatic number is a number that looks the same when rotated 180 degrees (looked at upside down).

Find all strobogrammatic numbers that are of length = n.

**Example:**

```
Input:  n = 2
Output: ["11","69","88","96"]
```

**Solution:**

有点像dp的解法，也不典型。分奇偶base case，然后一层一层往外边加。注意初始化和define。背吧。

```java
public List<String> findStrobogrammatic(int n) {
    List<String>[] dp = new List[n + 1];
    for (int i = 0; i <= n; i++) {
        dp[i] = new ArrayList<>();
    }
    dp[0].add("");
    dp[1].add("0");
    dp[1].add("1");
    dp[1].add("8");
    int i = n % 2 == 0 ? 0 : 1;
    while (i != n) {
        for (String s : dp[i]) {
            if (i + 2 != n) {
                dp[i + 2].add("0" + s + "0");
            }
            dp[i + 2].add("1" + s + "1");
            dp[i + 2].add("8" + s + "8");
            dp[i + 2].add("6" + s + "9");
            dp[i + 2].add("9" + s + "6");
        }
        i += 2;
    }
    return dp[n];
}
```

### 253. Meeting Rooms II

Given an array of meeting time intervals consisting of start and end times `[[s1,e1],[s2,e2],...]` (si < ei), find the minimum number of conference rooms required.

**Example 1:**

```
Input: [[0, 30],[5, 10],[15, 20]]
Output: 2
```

**Example 2:**

```
Input: [[7,10],[2,4]]
Output: 1
```

**Solution1:**

实际上是求在某一刻最多的overlape的meeting interval的个数。用一个priorityqueue去存interval的end，遍历intervals，每次pop出比当前start小的元素，说明该会议已经结束，然后把自己的end加进去，queue中的元素，代表有overlap的meeting。去存用一个global max去找哪个瞬间queue的size最大，这个size就是最终答案。

```java
public int minMeetingRooms(int[][] intervals) {
    if (intervals == null || intervals.length == 0) return 0;
    Arrays.sort(intervals, new Comparator<int[]>() {
       public int compare(int[] a, int[] b) {
           return a[0] - b[0];
       } 
    });

    int min = 1;
    PriorityQueue<Integer> queue = new PriorityQueue<>();
    for (int i = 0; i < intervals.length; ++i) {
        if (queue.isEmpty()) {
            queue.add(intervals[i][1]);
        } else {
            while (!queue.isEmpty() && intervals[i][0] >= queue.peek()){
                queue.poll();
            }
            queue.offer(intervals[i][1]);
            min = Math.max(min, queue.size());
        }
    }
    return min;
}
```

**Solution2:**

很tricky的写法。把intervals所有的start end存成，(start, 1) (end, -1) pair然后根据第一个元素sort一遍。用一个cnt去把value相加，哪个时刻cnt最大就是最少的room数。

**进进出出问题。**

```java
class Pair {
    int key;
    int value;
    public Pair(int key, int value) {
        this.key = key;
        this.value = value;
    }
}
class Solution {
    public int minMeetingRooms(int[][] intervals) {
        Pair[] pairs = new Pair[2 * intervals.length];
        int index = 0;
        for (int i = 0; i < intervals.length; i++) {
            pairs[index++] = new Pair(intervals[i][0], 1);
            pairs[index++] = new Pair(intervals[i][1], -1);
        }
        Arrays.sort(pairs, new Comparator<Pair>() {
            public int compare(Pair p1, Pair p2) {
                int d =  p1.key - p2.key;
                if (d == 0) {
                    d = p1.value - p2.value;
                }
                return d;
            }
        });
        int cnt = 0;
        int min = 0;
        for (Pair p : pairs) {
            cnt += p.value;
            min = Math.max(cnt, min);
        }
        return min;
    }
}
```

###**257. Binary Tree Paths**

Given a binary tree, return all root-to-leaf paths.

**Note:** A leaf is a node with no children.

**Example:**

```
Input:

   1
 /   \
2     3
 \
  5

Output: ["1->2->5", "1->3"]

Explanation: All root-to-leaf paths are: 1->2->5, 1->3
```

**Solution:**

Backtracking(dfs),把当前node加入result，如果isleaf则算为一个path

```java
public List<String> binaryTreePaths(TreeNode root) {
    List<String> paths = new ArrayList();
    if(root == null) return paths;
    List<Integer> values = new ArrayList();
    backtracking(root, values, paths);
    return paths;
}
// 将当前的node加到路径 values中，如果当前node为leaf，则把path加到结果中
private void backtracking(TreeNode node, List<Integer> values, List<String> paths) {
    // parent node 不是leaf，但是left right可能有一个为null
    if(node == null) return;
    values.add(node.val);
    if(isLeaf(node)) {
        paths.add(buildPath(values));
    } else {
        backtracking(node.left, values, paths);
        backtracking(node.right, values, paths);
    }
    //backtracking
    values.remove(values.size()-1);
}

private boolean isLeaf(TreeNode node) {
    return node.left == null  && node.right == null;
}

private String buildPath(List<Integer> values) {
    StringBuilder sb = new StringBuilder();
    for(int i=0; i<values.size(); i++) {
        sb.append(values.get(i));
        if(i!=values.size()-1) {
            sb.append("->");
        }
    }
    return sb.toString();
}
```

### 271. Encode and Decode Strings

Design an algorithm to encode **a list of strings** to **a string**. The encoded string is then sent over the network and is decoded back to the original list of strings.

**Solution1:**

Naive solution here is to join strings using delimiters.

> What to use as a delimiter? Each string may contain any possible characters out of 256 valid ascii characters.

Seems like one has to use non-ASCII unichar character, for example `unichr(257)` in Python and `Character.toString((char)257)` in Java (it's character `ā`).

![fig](https://leetcode.com/problems/encode-and-decode-strings/Figures/271/delimiter.png)

Here it's convenient to use two different non-ASCII characters, to distinguish between situations of "empty array" and of "array of empty strings".

```java
public class Codec {
  // Encodes a list of strings to a single string.
  public String encode(List<String> strs) {
    if (strs.size() == 0) return Character.toString((char)258);

    String d = Character.toString((char)257);
    StringBuilder sb = new StringBuilder();
    for(String s: strs) {
      sb.append(s);
      sb.append(d);
    }
    sb.deleteCharAt(sb.length() - 1);
    return sb.toString();
  }

  // Decodes a single string to a list of strings.
  public List<String> decode(String s) {
    String d = Character.toString((char)258);
    if (s.equals(d)) return new ArrayList();

    d = Character.toString((char)257);
    return Arrays.asList(s.split(d, -1));
  }
}
```

**Solution2:**

注意intToString 把String的长度转换成一个4个char的string来表示。StringToInt把4位的char decode成对应的int，理解背住着两种转换方式。

This approach is based on the [encoding used in HTTP v1.1](https://en.wikipedia.org/wiki/Chunked_transfer_encoding). **It doesn't depend on the set of input characters, and hence is more versatile and effective than Approach 1.**

> Data stream is divided into chunks. Each chunk is preceded by its size in bytes.

**Encoding Algorithm**

![fig](https://leetcode.com/problems/encode-and-decode-strings/Figures/271/encodin.png)

- Iterate over the array of chunks, i.e. strings.
  - For each chunk compute its length, and convert that length into 4-bytes string.
  - Append to encoded string :
    - 4-bytes string with information about chunk size in bytes.
    - Chunk itself.
- Return encoded string.

```java
public class Codec {
  // Encodes a list of strings to a single string.
  public String encode(List<String> strs) {
    if (strs.size() == 0) return Character.toString((char)258);

    String d = Character.toString((char)257);
    StringBuilder sb = new StringBuilder();
    for(String s: strs) {
      sb.append(s);
      sb.append(d);
    }
    sb.deleteCharAt(sb.length() - 1);
    return sb.toString();
  }

  // Decodes a single string to a list of strings.
  public List<String> decode(String s) {
    String d = Character.toString((char)258);
    if (s.equals(d)) return new ArrayList();

    d = Character.toString((char)257);
    return Arrays.asList(s.split(d, -1));
  }
}
```

### 279. Perfect Squares

Given a positive integer *n*, find the least number of perfect square numbers (for example, `1, 4, 9, 16, ...`) which sum to *n*.

**Solution1: BFS**

列出所有可能的方式，把squares看成图，走出一条路。

```java
public int numSquares(int n) {
    List<Integer> squares = generateSquares(n);
    Queue<Integer> queue = new LinkedList<>();
    queue.add(n);
    boolean[] marked = new boolean[n+1];
    marked[n] = true;
    int len = 0;
    while(!queue.isEmpty()) {
        int size = queue.size();
        len++;
        while(size-- > 0) {
            int cur = queue.poll();
            for(int square : squares){
              int next = cur - square;
              if(next < 0) break;
              if(next == 0) return len;
              if(marked[next]) continue;
              marked[next] = true;
              queue.add(next);
            }
        }
    }
    return n;
}
private List<Integer> generateSquares(int n) {
    List<Integer> squares = new ArrayList<>();
    int square = 1;
    int diff = 3;
    while(square <= n) {
        squares.add(square);
        square += diff;
        diff += 2;
    }
    return squares;
}
```

**Soluiton2:dp**

dp[n] 表示最少由几个平方数组成。

```java
public int numSquares(int n) {
    if (n <= 0) return 0;
    int[] dp = new int[n+1];
    dp[0] = 0;
    for (int i = 1; i < n+1; ++i) {
        int cnt = Integer.MAX_VALUE;
        for (int j = 1; j * j <= i; ++j) cnt = Math.min(cnt, dp[i-j*j] + 1);
        dp[i] = cnt;
    }
    return dp[n];
}
```

### 299. Bulls and Cows

You are playing the following [Bulls and Cows](https://en.wikipedia.org/wiki/Bulls_and_Cows) game with your friend: You write down a number and ask your friend to guess what the number is. Each time your friend makes a guess, you provide a hint that indicates how many digits in said guess match your secret number exactly in both digit and position (called "bulls") and how many digits match the secret number but locate in the wrong position (called "cows"). Your friend will use successive guesses and hints to eventually derive the secret number.

Write a function to return a hint according to the secret number and friend's guess, use `A` to indicate the bulls and `B` to indicate the cows. 

Please note that both secret number and friend's guess may contain duplicate digits.

**Example 1:**

```
Input: secret = "1807", guess = "7810"

Output: "1A3B"

Explanation: 1 bull and 3 cows. The bull is 8, the cows are 0, 1 and 7.
```

**Example 2:**

```
Input: secret = "1123", guess = "0111"

Output: "1A1B"

Explanation: The 1st 1 in friend's guess is a bull, the 2nd or 3rd 1 is a cow.
```

**Solution:**

用两个长度为10的array来记录两个字符串中，index(0-9)出现且不在正确位置的次数，最后遍历0-9，取min相加。

```java
public String getHint(String secret, String guess) {
    if (secret == null || secret.length() == 0) return secret;
    int n = secret.length();
    int[] s = new int[10];
    int[] g = new int[10];
    int bullCnt = 0, cowCnt = 0;
    for (int i = 0; i < n; ++i) {
        char cs = secret.charAt(i);
        char cg = guess.charAt(i);
        if (cs == cg) {
            System.out.println(cs == cg);
            ++bullCnt;
        } else {
            s[charToIndex(cs)]++;
            g[charToIndex(cg)]++;
        }
    }
    for (int i = 0; i < 10; ++i) {
        cowCnt += Math.min(s[i],g[i]);
    }
    StringBuilder sb = new StringBuilder();
    sb.append(String.valueOf(bullCnt));
    sb.append("A");
    sb.append(String.valueOf(cowCnt));
    sb.append("B");
    return sb.toString();
}
private int charToIndex(char c) {
    return c - '0';
}
```

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

###343. Integer Break

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

###363.Max Sum of Rectangle No Larger Than K

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

###**417. Pacific Atlantic Water Flow**

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

###449. Serialize and Deserialize BST

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

###**583. Delete Operation for Two Strings**

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

### *659. Split Array into Consecutive Subsequences

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

### 702. Search in a Sorted Array of Unknown Size

Given an integer array sorted in ascending order, write a function to search `target` in `nums`.  If `target` exists, then return its index, otherwise return `-1`. **However, the array size is unknown to you**. You may only access the array using an `ArrayReader` interface, where `ArrayReader.get(k)` returns the element of the array at index `k` (0-indexed).

You may assume all integers in the array are less than `10000`, and if you access the array out of bounds, `ArrayReader.get` will return `2147483647`. 

**Solution:**

对于这种sorted实际上是单调递增的去找去判断等问题，binary search。由于右边界未知，所以本题先找右边界然后再binary search。

```java
public int search(ArrayReader reader, int target) {
    if (reader.get(0) == target) return 0;

    // search boundaries
    int left = 0, right = 1;
    while (reader.get(right) < target) {
      left = right;
      right <<= 1;
    }

    // binary search
    int pivot, num;
    while (left <= right) {
      pivot = left + ((right - left) >> 1);
      num = reader.get(pivot);

      if (num == target) return pivot;
      if (num > target) right = pivot - 1;
      else left = pivot + 1;
    }

    // there is no target element
    return -1;
}
```

### 763. Partition Labels

**Example 1:**

```
Input: S = "ababcbacadefegdehijhklij"
Output: [9,7,8]
Explanation:
The partition is "ababcbaca", "defegde", "hijhklij".
This is a partition so that each letter appears in at most one part.
A partition like "ababcbacadefegde", "hijhklij" is incorrect, because it splits S into less parts.
```

**Solution：**

贪心？设定一个start，然后动态

```java
public List<Integer> partitionLabels(String S) {
    int[] lastIndexofChar = new int[26];
    for(int i=0; i<S.length(); i++) {
        lastIndexofChar[S.charAt(i) - 'a'] = i;
    }
    int firstIndex = 0;
    List<Integer> ret = new ArrayList<>();
    while(firstIndex < S.length()){
        int lastIndex = firstIndex;
        // 重复更新lastindex知道partition完毕
        for(int i=firstIndex; i<S.length() && i<=lastIndex; i++) {
            int index = lastIndexofChar[S.charAt(i) - 'a'];
            if(index > lastIndex){
                lastIndex = index;
            }
        }
        ret.add(lastIndex - firstIndex + 1);
        firstIndex = lastIndex + 1;
    }
    return ret;
}
```

### 767. Reorganize String

Given a string `S`, check if the letters can be rearranged so that two characters that are adjacent to each other are not the same.

If possible, output any possible result.  If not possible, return the empty string.

**Example 1:**

```
Input: S = "aab"
Output: "aba"
```

**Example 2:**

```
Input: S = "aaab"
Output: ""
```

**Solution:**

统计所有字母的count, count > (n+1) / 2 则不可能。

用PriorityQueue来存letter 和 count的pair，按照count排序。

每次poll两个字母下来(当前count最多和第二多的字母)，把这两个append上去，（保证了相邻两个不会相同），不能只poll一个。

Time Complexity : O(NlogA) A为字母表的长度，即为26，每次poll，pq都要从剩下的中找到最大的，是logA的复杂度。

注意定义PriorityQueue的排序。

```java
class Pair {
    int count;
    char letter;
    public Pair(int ct, char ch) {
        count = ct;
        letter = ch;
    }
}
class Solution {
    public String reorganizeString(String S) {
        int n = S.length();
        int[] count = new int[26];
        for (char c : S.toCharArray()) count[c - 'a']++;
        PriorityQueue<Pair> pq = new PriorityQueue<Pair>((a,b) -> a.count == b.count ? a.letter - b.letter : b.count - a.count);
        
        for (int i = 0; i < 26; ++i) {
            if (count[i] > 0) {
                if (count[i] > (n+1)/2) return "";
                pq.add(new Pair(count[i], (char)(i + 'a')));
            }
        }
        
        StringBuilder res = new StringBuilder();
        while (pq.size() >= 2) {
            Pair c1 = pq.poll();
            Pair c2 = pq.poll();
            res.append(c1.letter);
            res.append(c2.letter);
            if (--c1.count > 0) pq.add(c1);
            if (--c2.count > 0) pq.add(c2);
        }
        
        if (pq.size() > 0) res.append(pq.poll().letter);
        return res.toString();
    }
}
```

### 708. Insert into a Cyclic Sorted List

Given a node from a cyclic linked list which is sorted in ascending order, write a function to insert a value into the list such that it remains a cyclic sorted list. The given node can be a reference to *any* single node in the list, and may not be necessarily the smallest value in the cyclic list.

If there are multiple suitable places for insertion, you may choose any place to insert the new value. After the insertion, the cyclic list should remain sorted.

If the list is empty (i.e., given node is `null`), you should create a new single cyclic list and return the reference to that single node. Otherwise, you should return the original given node.

**Soution：**

分情况讨论所有可能的情况，用一个cur和next指针去逼近该插入的位置，然后break出循环，插到cur和next中间。

```java
public Node insert(Node head, int insertVal) {
    if (head == null) {
        Node cur = new Node();
        cur.val = insertVal;
        cur.next = cur;
        return cur;
    }
    Node cur = head;
    Node next = head.next;
    while (next != head) {
        // 2->2->2->3->3->3 insert2,3
        // 1->3->3->4 insert 2,3
        // 1->3->4->1(head) insert 1
        if (cur.val <= next.val && insertVal >= cur.val && insertVal <= next.val) break;
        // 3->4->1->3(head) insert 5
        if (cur.val > next.val && insertVal >= cur.val) break;
        // 3->4->1->3->(head) insert 0,1
        if (cur.val > next.val && insertVal <= next.val) break;
        // 1->3->4->1(head) insert 5(while)
        cur = next;
        next = cur.next;
    }
    Node node = new Node(insertVal, next);
    cur.next = node;
    return head;
}
```

### 734. Sentence Similarity

Given two sentences `words1, words2` (each represented as an array of strings), and a list of similar word pairs `pairs`, determine if two sentences are similar.

For example, "great acting skills" and "fine drama talent" are similar, if the similar word pairs are `pairs = [["great", "fine"], ["acting","drama"], ["skills","talent"]]`.

Note that the similarity relation is not transitive. For example, if "great" and "fine" are similar, and "fine" and "good" are similar, "great" and "good" are **not** necessarily similar.

However, similarity is symmetric. For example, "great" and "fine" being similar is the same as "fine" and "great" being similar.

Also, a word is always similar with itself. For example, the sentences `words1 = ["great"], words2 = ["great"], pairs = []` are similar, even though there are no specified similar word pairs.

Finally, sentences can only be similar if they have the same number of words. So a sentence like `words1 = ["great"]` can never be similar to `words2 = ["doubleplus","good"]`.

**Solution:**

放到一个map中。复杂的逻辑关系可以用boolean变量来存。

```java
public boolean areSentencesSimilar(String[] words1, String[] words2, List<List<String>> pairs) {
    if (words1 == null || words2 == null) return false;
    int m = words1.length, n = words2.length;
    if (m != n) return false;
    Map<String, Set<String>> map = new HashMap<>();
    for (int i = 0; i < pairs.size(); ++i) {
        if (!map.containsKey(pairs.get(i).get(0))) {
            Set<String> set = new HashSet<>();
            set.add(pairs.get(i).get(1));
            map.put(pairs.get(i).get(0), set);
        } else {
            map.get(pairs.get(i).get(0)).add(pairs.get(i).get(1));
        }
    }
    for (int i = 0; i < m; ++i) {
        if (words1[i].equals(words2[i])) continue;
        boolean checkFirst = map.containsKey(words1[i]) && map.get(words1[i]).contains(words2[i]);
        boolean checkSecond = map.containsKey(words2[i]) && map.get(words2[i]).contains(words1[i]);
        if (!checkFirst && !checkSecond) return false;
    }
    return true;
}
```

### 742. Closest Leaf in a Binary Tree

Given a binary tree **where every node has a unique value**, and a target key `k`, find the value of the nearest leaf node to target `k` in the tree.

Here, *nearest* to a leaf means the least number of edges travelled on the binary tree to reach any leaf of the tree. Also, a node is called a *leaf* if it has no children.

In the following examples, the input tree is represented in flattened form row by row. The actual `root` tree given will be a TreeNode object.

**Example 1:**

```
Input:
root = [1, 3, 2], k = 1
Diagram of binary tree:
          1
         / \
        3   2

Output: 2 (or 3)

Explanation: Either 2 or 3 is the nearest leaf node to the target of 1.
```

**Example 2:**

```
Input:
root = [1], k = 1
Output: 1

Explanation: The nearest leaf node is the root node itself.
```

**Solution:**

把树转化成一个无向图，用BFS从出发节点k去遍历无向图。找到leaf的最短路径。遇到第一个叶子节点返回其值即为最短。注意root如果一边为null，另一边不是的话不算是一个leaf。return前判断一下即可。

用BFS遍历图的时候并不需要分层所以只需要一个loop就行。

用任何order的travesal recursion都可以创建树。注意区分travesal和dfs，dfs是一种search方法，只要满足深度优先的搜索(注意是搜索而不是处理）就是dfs，这里的traverse任何一个位置都可以说是dfs。dfs经常用recursion来实现。

```java
public int findClosestLeaf(TreeNode root, int k) {
    if (root.left == null && root.right == null) return 1;
    Map<Integer, Set<Integer>> graph = new HashMap();
    traverse(root, null, graph);
    Queue<Integer> queue = new LinkedList<>();
    Set<Integer> visited = new HashSet<>();
    queue.add(k);
    visited.add(k);
    while (!queue.isEmpty()) {
        int cur = queue.poll();
        Set<Integer> set = graph.get(cur);
        if (set.size() <= 1 && cur != root.val) return cur;
        for (int m : set) {
            if (!visited.contains(m)){
                queue.offer(m);
                visited.add(m);
            }
        }
    }
    return -1;
}

private void traverse(TreeNode root, TreeNode parent, Map<Integer, Set<Integer>> graph) {
    if (root == null) return;
    graph.put(root.val, new HashSet<>());
    // 1. traverse(root.left, root, graph);
    if (parent != null) graph.get(root.val).add(parent.val);
    if (root.left != null) graph.get(root.val).add(root.left.val);
    if (root.right != null) graph.get(root.val).add(root.right.val);
    traverse(root.left, root, graph);
    // 2.
    traverse(root.right, root, graph);   
    // 3. traverse(root.left, root, graph);
}
```

### 743. Network Delay Time

There are `N` network nodes, labelled `1` to `N`.

Given `times`, a list of travel times as **directed** edges `times[i] = (u, v, w)`, where `u` is the source node, `v` is the target node, and `w` is the time it takes for a signal to travel from source to target.

Now, we send a signal from a certain node `K`. How long will it take for all nodes to receive the signal? If it is impossible, return `-1`.

**Example 1:**

![img](https://assets.leetcode.com/uploads/2019/05/23/931_example_1.png)

```
Input: times = [[2,1,1],[2,3,1],[3,4,1]], N = 4, K = 2
Output: 2
```

 **Solution:**

给定有向图，找start 到其他所有节点最短路径中的max。

用**Dijkstra Algorithm**, 这个算法是通过为每个顶点 *v* 保留当前为止所找到的从s到v的最短路径来工作的。实现：priorityqueue (+ 类似bfs）

pq中存放的是下一层中可能会是出发节点的node，每次node被当作出发节点就把它放入map中，相当于设置了visited。

```java
public int networkDelayTime(int[][] times, int N, int K) {
    Map<Integer, List<int[]>> graph = new HashMap<>();
    //建立有向图
    for (int[] edge : times) {
        if (!graph.containsKey(edge[0])) {
            graph.put(edge[0], new ArrayList<int[]>());
        }
        graph.get(edge[0]).add(new int[]{edge[1], edge[2]});
    }

    // 存weight 和 node pair
    PriorityQueue<int[]> pq = new PriorityQueue<int[]>((info1,info2)->info1[0]-info2[0]);
    pq.offer(new int[]{0, K});

    //去重 同时记录从start到其他所有节点的距离
    Map<Integer, Integer> dist = new HashMap<>();

    while (!pq.isEmpty()) {
        int[] info = pq.poll();
        int d = info[0], node = info[1];

        if (dist.containsKey(node)) continue;
        dist.put(node, d);
        if (graph.containsKey(node)) {
            for (int[] edge : graph.get(node)) {
                int nei = edge[0], d2 = edge[1];
                if (!dist.containsKey(nei)) {
                    pq.offer(new int[]{d+d2, nei});
                }
            }
        }
    }

    if (dist.size() != N) return -1;
    int ans = 0;
    for (int cand : dist.values()) {
        ans = Math.max(ans, cand);
    }
    return ans;
}
```

### 792. Number of Matching Subsequences

Given string `S` and a dictionary of words `words`, find the number of `words[i]` that is a subsequence of `S`.

```
Example :
Input: 
S = "abcde"
words = ["a", "bb", "acd", "ace"]
Output: 3
Explanation: There are three words in words that are a subsequence of S: "a", "acd", "ace".
```

**Solution:**

Brute force 是可以对每个word去S中找。这样S会被重复words.length遍。Since the length of `S` is large, let's think about ways to iterate through `S` only once, instead of many times as in the brute force solution.

**注意本题是subsequence而不是substring所以要新建一个Node以及index去存当前走到word的哪一位char了**

We can put words into buckets by starting character. If for example we have `words = ['dog', 'cat', 'cop']`, then we can group them `'c' : ('cat', 'cop'), 'd' : ('dog',)`. This groups words by what letter they are currently waiting for. Then, while iterating through letters of `S`, we will move our words through different buckets. 

用一些bucktes（heads）去存以这个字母开头的单词有哪些，然后遍历S时，每一位char，找到相应的bucket，把buket的index向后移一位，如果移到头了说明存在，没有移到头，把这个word加到以下一位char为head的bucket中。

```java
class Solution {
    public int numMatchingSubseq(String S, String[] words) {
        int ans = 0;
        // 注意声明一个arraylist数组的方法
        ArrayList<Node>[] heads = new ArrayList[26];
        for (int i = 0; i < 26; ++i)
            heads[i] = new ArrayList<Node>();

        for (String word: words)
            heads[word.charAt(0) - 'a'].add(new Node(word, 0));

        for (char c: S.toCharArray()) {
            ArrayList<Node> old_bucket = heads[c - 'a'];
            heads[c - 'a'] = new ArrayList<Node>();

            for (Node node: old_bucket) {
                node.index++;
                if (node.index == node.word.length()) {
                    ans++;
                } else {
                    heads[node.word.charAt(node.index) - 'a'].add(node);
                }
            }
            old_bucket.clear();
        }
        return ans;
    }

}

class Node {
    String word;
    int index;
    public Node(String w, int i) {
        word = w;
        index = i;
    }
}
```

### 809. Expressive Words

Sometimes people repeat letters to represent extra feeling, such as "hello" -> "heeellooo", "hi" -> "hiiii".  In these strings like "heeellooo", we have *groups* of adjacent letters that are all the same:  "h", "eee", "ll", "ooo".

For some given string `S`, a query word is *stretchy* if it can be made to be equal to `S` by any number of applications of the following *extension* operation: choose a group consisting of characters `c`, and add some number of characters `c` to the group so that the size of the group is 3 or more.

For example, starting with "hello", we could do an extension on the group "o" to get "hellooo", but we cannot get "helloo" since the group "oo" has size less than 3.  Also, we could do another extension like "ll" -> "lllll" to get "helllllooo".  If `S = "helllllooo"`, then the query word "hello" would be stretchy because of these two extension operations: `query = "hello" -> "hellooo" -> "helllllooo" = S`.

Given a list of query words, return the number of words that are stretchy. 

```
Example:
Input: 
S = "heeellooo"
words = ["hello", "hi", "helo"]
Output: 1
Explanation: 
We can extend "e" and "o" in the word "hello" to get "heeellooo".
We can't extend "helo" to get "heeellooo" because the group "ll" is not size 3 or more.
```

 **Solution1:**

把S按照character依次group存key和count。注意valid的判断条件，一些variables需要重置等等，空间比solution2少，但是solution2更容易理解。

```java
public int expressiveWords(String S, String[] words) {
    List<Integer> counts = new ArrayList<>();
    List<Character> chars = new ArrayList<>();
    int prev = -1;
    for (int i = 0; i < S.length(); i++) {
        if (i == S.length() - 1 || S.charAt(i) != S.charAt(i+1)) {
            chars.add(S.charAt(i));
            counts.add(i - prev);
            prev = i;
        }
    }
    int index = 0;
    int cnt = 0;
    int previous = -1;
    boolean valid = false;
    for (String word : words) {
        for (int i = 0; i < word.length(); i++) {
            if (i == word.length() - 1 || word.charAt(i) != word.charAt(i+1)) {
                if (index >= chars.size()) break;
                if (word.charAt(i) != chars.get(index)) break;
                int count = i - previous;
                if (count > counts.get(index)) break;
                if (count != counts.get(index) && counts.get(index) < 3) break;
                if (i == word.length() - 1 && index == chars.size() - 1) valid = true;
                index++;
                previous = i;
            }
        }       
        if (valid) {
            cnt++;
        }
        valid = false;
        previous = -1;
        index = 0;
    }
    return cnt;
}
```

**Solution2:**

```java
class Solution {
    public int expressiveWords(String S, String[] words) {
        RLE R = new RLE(S);
        int ans = 0;

        search: for (String word: words) {
            RLE R2 = new RLE(word);
            if (!R.key.equals(R2.key)) continue;
            for (int i = 0; i < R.counts.size(); ++i) {
                int c1 = R.counts.get(i);
                int c2 = R2.counts.get(i);
                if (c1 < 3 && c1 != c2 || c1 < c2)
                    continue search;
            }
            ans++;
        }
        return ans;
    }
}

class RLE {
    String key;
    List<Integer> counts;

    public RLE(String S) {
        StringBuilder sb = new StringBuilder();
        counts = new ArrayList();

        char[] ca = S.toCharArray();
        int N = ca.length;
        int prev = -1;
        for (int i = 0; i < N; ++i) {
            if (i == N-1 || ca[i] != ca[i+1]) {
                sb.append(ca[i]);
                counts.add(i - prev);
                prev = i;
            }
        }

        key = sb.toString();
    }
}
```

### 788. Rotated Digits

X is a good number if after rotating each digit individually by 180 degrees, we get a valid number that is different from X.  Each digit must be rotated - we cannot choose to leave it alone.

A number is valid if each digit remains a digit after rotation. 0, 1, and 8 rotate to themselves; 2 and 5 rotate to each other; 6 and 9 rotate to each other, and the rest of the numbers do not rotate to any other number and become invalid.

Now given a positive number `N`, how many numbers X from `1` to `N` are good?

**Solution:**

validTable 的思想。

```java
public int rotatedDigits(int N) {
    int cnt = 0;
    // 0,1,8 -> 0, 2,5,6,9 ->1, 3,4,7 -> -1
    int[] validTable = {0,0,1,-1,-1,1,1,-1,0,1};
    for (int i = 1; i <= N; ++i) {
        if (validNumber(validTable, i)) ++cnt;
    }
    return cnt;
}
private boolean validNumber(int[] validTable, int n) {
    boolean isDifferent = false;
    while (n > 0) {
        int mode = n % 10;
        if (validTable[mode] == -1) return false;
        if (validTable[mode] == 1) isDifferent = true;
        n /= 10;
    }
    return isDifferent;
}
```

### 802. Find Eventual Safe States

In a directed graph, we start at some node and every turn, walk along a directed edge of the graph.  If we reach a node that is terminal (that is, it has no outgoing directed edges), we stop.

Now, say our starting node is *eventually safe* if and only if we must eventually walk to a terminal node.  More specifically, there exists a natural number `K` so that for any choice of where to walk, we must have stopped at a terminal node in less than `K` steps.

Which nodes are eventually safe?  Return them as an array in sorted order.

The directed graph has `N` nodes with labels `0, 1, ..., N-1`, where `N` is the length of `graph`.  The graph is given in the following form: `graph[i]` is a list of labels `j` such that `(i, j)` is a directed edge of the graph.

```
Example:
Input: graph = [[1,2],[2,3],[5],[0],[5],[],[]]
Output: [2,4,5,6]
Here is a diagram of the above graph.
```

![Illustration of graph](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/03/17/picture1.png)

**Solution:**

**Reverse Edges + BFS**

**Intuition**The crux of the problem is whether you can reach a cycle from the node you start in. If you can, then there is a way to avoid stopping indefinitely; and if you can't, then after some finite number of steps you'll stop.

Thinking about this property more, a node is eventually safe if all it's outgoing edges are to nodes that are eventually safe.

This gives us the following idea: we start with nodes that have no outgoing edges - those are eventually safe. Now, we can update any nodes which only point to eventually safe nodes - those are also eventually safe. Then, we can update again, and so on.

**Algorithm**We'll keep track of `graph`, a way to know for some node `i`, what the outgoing edges `(i, j)` are. We'll also keep track of `rgraph`, a way to know for some node `j`, what the incoming edges `(i, j)` are.

Now for every node `j` which was declared eventually safe, we'll process them in a queue. We'll look at all parents `i = rgraph[j]` and remove the edge `(i, j)` from the graph (from `graph`). If this causes the `graph` to have no outgoing edges `graph[i]`, then we'll declare it eventually safe and add it to our queue.

Also, we'll keep track of everything we ever added to the queue, so we can read off the answer in sorted order later.

没有任何outgoing edges的node是safe的，那么只有一种走法走到这些已经safe的node的nodes是safe的，所以我们倒着来，先把直接没有outgoing edges的放到queue中，然后一层一层去找只有一条路通向这些nodes的node。

```java
public List<Integer> eventualSafeNodes(int[][] graph) {
    int n = graph.length;
    boolean[] safe = new boolean[n];

    List<Set<Integer>> outgraph = new ArrayList<>();
    // rgraph[i] 的set中存的是所有可以到第i个node的nodes
    List<Set<Integer>> ingraph = new ArrayList<>();
    for (int i = 0; i < n; ++i) {
        outgraph.add(new HashSet<>());
        ingraph.add(new HashSet<>());
    }

    Queue<Integer> queue = new LinkedList();

    for (int i = 0; i < n; ++i) {
        if (graph[i].length == 0) { 
            safe[i] = true;
            queue.offer(i);
        }
        for (int j : graph[i]) {
            outgraph.get(i).add(j);
            ingraph.get(j).add(i);
        }
    }

    while (!queue.isEmpty()) {
        int j = queue.poll();
        for (int i : ingraph.get(j)) {
            outgraph.get(i).remove(j);
            if (outgraph.get(i).isEmpty()) {
                queue.offer(i);
                safe[i] = true;
            }
        }
    }

    List<Integer> ans = new ArrayList<>();
    for (int i = 0; i < n; ++i) {
        if (safe[i]) ans.add(i);
    }
    return ans;
}
```

**Solution2:**

**Intuition**

As in *Approach #1*, the crux of the problem is whether you reach a cycle or not.

Let us perform a "brute force": a cycle-finding DFS algorithm on each node individually. This is a classic "white-gray-black" DFS algorithm that would be part of any textbook on DFS. We mark a node gray on entry, and black on exit. If we see a gray node during our DFS, it must be part of a cycle. In a naive view, we'll clear the colors between each search.

**Algorithm**

We can improve this approach, by noticing that we don't need to clear the colors between each search.

When we visit a node, the only possibilities are that we've marked the entire subtree black (which must be eventually safe), or it has a cycle and we have only marked the members of that cycle gray. So indeed, the invariant that gray nodes are always part of a cycle, and black nodes are always eventually safe is maintained.

In order to exit our search quickly when we find a cycle (and not paint other nodes erroneously), we'll say the result of visiting a node is `true` if it is eventually safe, otherwise `false`. This allows information that we've reached a cycle to propagate up the call stack so that we can terminate our search early.

```java
public List<Integer> eventualSafeNodes(int[][] graph) {
    int N = graph.length;
    int[] color = new int[N];
    List<Integer> ans = new ArrayList();

    for (int i = 0; i < N; ++i)
        if (dfs(i, color, graph))
            ans.add(i);
    return ans;
}

// colors: WHITE 0, GRAY 1, BLACK 2;
public boolean dfs(int i, int[] color, int[][] graph) {
    if (color[i] > 0)
        return color[i] == 2;

    color[i] = 1;
    for (int nei: graph[i]) {
        if (color[i] == 2)
            continue;
        if (color[nei] == 1 || !dfs(nei, color, graph))
            return false;
    }

    color[i] = 2;
    return true;
}
```

### 833. Find And Replace in String

To some string `S`, we will perform some replacement operations that replace groups of letters with new ones (not necessarily the same size).

Each replacement operation has `3` parameters: a starting index `i`, a source word `x` and a target word `y`.  The rule is that if `x` starts at position `i` in the **original** **string** **S**, then we will replace that occurrence of `x` with `y`.  If not, we do nothing.

For example, if we have `S = "abcd"` and we have some replacement operation `i = 2, x = "cd", y = "ffff"`, then because `"cd"` starts at position `2` in the original string `S`, we will replace it with `"ffff"`.

Using another example on `S = "abcd"`, if we have both the replacement operation `i = 0, x = "ab", y = "eee"`, as well as another replacement operation `i = 2, x = "ec", y = "ffff"`, this second operation does nothing because in the original string `S[2] = 'c'`, which doesn't match `x[0] = 'e'`.

All these operations occur simultaneously.  It's guaranteed that there won't be any overlap in replacement: for example, `S = "abc", indexes = [0, 1], sources = ["ab","bc"]` is not a valid test case.

**Example 1:**

```
Input: S = "abcd", indexes = [0,2], sources = ["a","cd"], targets = ["eee","ffff"]
Output: "eeebffff"
Explanation: "a" starts at index 0 in S, so it's replaced by "eee".
"cd" starts at index 2 in S, so it's replaced by "ffff".
```

**Solution:**

这道题难点在于要同时把sources在相应的位置改变，那么长度就不一定了。所以要先记录下哪些位置可以改变，同时，也要知道这个需要在String S上改变的位置对应的是indexes,sources,targets中的哪一个，可以用一个map来记录对应关系。这时候map经常会用一个array来代替。array的index作为key，代表的是要在String S的哪个位置开始改变，array[index]记录的是这是indexes中的哪一位。

```java
public String findReplaceString(String S, int[] indexes, String[] sources, String[] targets) {
    // List<Integer> valid = new ArrayList<>();
    int[] match = new int[S.length()];
    Arrays.fill(match, -1);
    // Map<Integer, Integer> map = new HashMap<>();
    for (int i = 0; i < indexes.length; ++i) {
        int index = indexes[i];
        if (index + sources[i].length()-1 < S.length()) {
            if (S.substring(index, index + sources[i].length()).equals(sources[i])) {
                match[index] = i;
                // valid.add(indexes[i]);
                // map.put(indexes[i], i);
            }
        }
    }
    // Collections.sort(valid);
    int index = 0;
    char[] arr = S.toCharArray();
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < arr.length;) {
        // if (index < valid.size() && i == valid.get(index)) {
        //     sb.append(targets[map.get(valid.get(index))]);
        //     i += sources[map.get(valid.get(index))].length();
        //     index++;
        // } else {
        //     sb.append(S.charAt(i++));
        // }
        if (match[i] >= 0) {
            sb.append(targets[match[i]]);
            i += sources[match[i]].length();
        } else {
            sb.append(S.charAt(i++));
        }
    }
    return sb.toString();
}
```

### 844. Backspace String Compare

Given two strings `S` and `T`, return if they are equal when both are typed into empty text editors. `#` means a backspace character.

**Example 1:**

```
Input: S = "ab#c", T = "ad#c"
Output: true
Explanation: Both S and T become "ac".
```

**Solution:**

这种走走还要倒退的问题可以考虑stack。

```java
public boolean backspaceCompare(String S, String T) {
    String finalS = afterBackspace(S);
    String finalT = afterBackspace(T);
    return finalS.equals(finalT);
}
private String afterBackspace(String s) {
    Stack<Character> stack = new Stack<>();
    for (int i = 0; i < s.length(); ++i) {
        char c = s.charAt(i);
        if (c == '#') {
            if(!stack.isEmpty()) {
                stack.pop();
            }
        } else {
            stack.push(c);
        }
    }
    StringBuilder sb = new StringBuilder();
    while (!stack.isEmpty()) {
        sb.append(stack.pop());
    }
    return sb.toString();
}
```

### 846. Hand of Straights

Alice has a `hand` of cards, given as an array of integers.

Now she wants to rearrange the cards into groups so that each group is size `W`, and consists of `W` consecutive cards.

Return `true` if and only if she can.

**Example 1:**

```
Input: hand = [1,2,3,6,2,3,4,7,8], W = 3
Output: true
Explanation: Alice's hand can be rearranged as [1,2,3],[2,3,4],[6,7,8].
```

**Solution:**

TreeMap brute force(straight forward)

既要order又要search -> TreeMap

```java
public boolean isNStraightHand(int[] hand, int W) {
    TreeMap<Integer, Integer> count = new TreeMap();
    for (int card : hand) {
        if (!count.containsKey(card)) count.put(card,1);
             // treemap replace
        else count.replace(card, count.get(card) + 1);
    }

    while (count.size() > 0) {
        int first = count.firstKey();
        for (int card = first; card < first + W; ++card) {
            if (!count.containsKey(card)) return false;
            int c = count.get(card);
            if (c == 1) count.remove(card);
            else count.replace(card, c - 1);
        }
    }

    return true;
}
```

### 855. Exam Room

In an exam room, there are `N` seats in a single row, numbered `0, 1, 2, ..., N-1`.

When a student enters the room, they must sit in the seat that maximizes the distance to the closest person.  If there are multiple such seats, they sit in the seat with the lowest number.  (Also, if no one is in the room, then the student sits at seat number 0.)

Return a class `ExamRoom(int N)` that exposes two functions: `ExamRoom.seat()` returning an `int` representing what seat the student sat in, and `ExamRoom.leave(int p)` representing that the student in seat number `p` now leaves the room.  It is guaranteed that any calls to `ExamRoom.leave(p)` have a student sitting in seat `p`.

**Solution:**

We need a data structure to record which seats are not available. 并且在seat的时候要找到距离这些被坐的座位最大的距离的座位。在remove时候直接将这个座位remove出去就好。那么就是怎么找最大的距离，那就是找相邻两个座位并且其之间距离最大。这样我们就需要一个sorted data structure，为了方便online，我们不用sort，直接用treeset去记录当前坐了座位的index。

Note：第一眼看到题straitforward 的想法有一个长度为N的array，array为true或者某个数表示有人，这样也行，但其实，没有人的座位不需要遍历，所以只记录坐人的座位的index就好了。

找最大的距离，就是相邻两个距离，大则更新。注意第零个到第一个有人的位置和最后一个有人的位置到N-1.

```java
class ExamRoom {
    int N;
    TreeSet<Integer> students;
    public ExamRoom(int N) {
        this.N = N;
        students = new TreeSet();
    }
    
    public int seat() {
        if (students.size() == 0) {
            students.add(0);
            return 0;
        }
        int student = 0;
        int dist = students.first();
        Integer prev = null;
        for (Integer s : students) {
            if (prev != null) {
                int d = (s - prev) / 2;
                if (d > dist) {
                    dist = d;
                    student = prev + d;
                }
            }
            prev = s;
        }
        if (N - 1 - students.last() > dist) student = N - 1;
        
        students.add(student);
        return student;
    }
    
    public void leave(int p) {
        students.remove(p);
    }
}
```

### 889. Construct Binary Tree from Preorder and Postorder Traversal

Return any binary tree that matches the given preorder and postorder traversals.

Values in the traversals `pre` and `post` are distinct positive integers.

**Example 1:**

```
Input: pre = [1,2,4,5,3,6,7], post = [4,5,2,6,7,3,1]
Output: [1,2,3,4,5,6,7]
```

 **Solution:**

A preorder traversal is:

- `(root node) (preorder of left branch) (preorder of right branch)`

While a postorder traversal is:

- `(postorder of left branch) (postorder of right branch) (root node)`

Inorder:

`(postorder of left branch)(root node)(postorder of right branch)`

这三个order都不同于tree的表示方法，tree是从root开始，一层一层从左到右遍历，null即为null。

树的问题一般会用recursion写，对某一个子树成立对任意一个树都成立。找到相应的规律即可。

For example, if the final binary tree is `[1, 2, 3, 4, 5, 6, 7]` (serialized), then the preorder traversal is `[1] + [2, 4, 5] + [3, 6, 7]`, while the postorder traversal is `[4, 5, 2] + [6, 7, 3] + [1]`.

If we knew how many nodes the left branch had, we could partition these arrays as such, and use recursion to generate each branch of the tree.

**Algorithm**

Let's say the left branch has L*L* nodes. We know the head node of that left branch is `pre[1]`, but it also occurs last in the postorder representation of the left branch. So `pre[1] = post[L-1]` (because of uniqueness of the node values.) Hence, `L = post.indexOf(pre[1]) + 1`.

Now in our recursion step, the left branch is represnted by `pre[1 : L+1]` and `post[0 : L]`, while the right branch is represented by `pre[L+1 : N]` and `post[L : N-1]`.

**用Recursion,创建根节点，不断的去找左右子树的根节点，根据pre[1] = post[L]，去找左子树的长度，N-L为右子树的长度**

```java
class Solution {
    public TreeNode constructFromPrePost(int[] pre, int[] post) {
        return make(0, 0, pre.length, pre, post);
    }
    
    // recursion定义: 把pre[iPre]当作根结点 分别把左右子树链接到本根节点上。
    //  pre[iPre:iPre+N] 和 post[iPost:iPost+N] 对应的是左子树混着右子树
    // recursion出口: 1. leaf的左右子树，长度为0，判断条件N==0
    // recursion出口: 2. leaf，长度为1，判断条件N==1
    // recursion递推关系，本树的长度为N，找到本树(root)的左右子root
    // 一直都是先走到N==0，即leaf的左右子树，然后N==1
    private TreeNode make (int iPre, int iPost, int N, int[] pre, int[] post) {
        if (N == 0) return null;
        TreeNode root = new TreeNode(pre[iPre]);
        // if N==1, pre[iPre] = post[iPost] 为本子树的根节点，左右无子树。
        if (N == 1) return root;
        
        //L = 以root为根节点的左子树的长度，N-L即为右子树的长度
        // pre[0]为根节点，pre[1]为左子树的根节点
        int L = 1;
        for (;L < N; ++L) {
            if (pre[iPre + 1] == post[iPost+L-1]) break;
        }
        root.left = make(iPre+1, iPost, L, pre, post);
        root.right = make(iPre+L+1, iPost+L, N-1-L, pre, post);
        return root;
    }
}
```

### 934.Shortest Bridge

In a given 2D binary array `A`, there are two islands.  (An island is a 4-directionally connected group of `1`s not connected to any other 1s.)

Now, we may change `0`s to `1`s so as to connect the two islands together to form 1 island.

Return the smallest number of `0`s that must be flipped.  (It is guaranteed that the answer is at least 1.)

**Solution:**

用dfs去找到一个岛，并把这个岛标为2，同时把岛的边界（外面一圈为0的pair）加到queue中。

以第一个岛外围的0作为第一层开始bfs直到找到第二个岛为止。

```java
class Pair {
    int x;
    int y;
    public Pair(int x, int y) {
        this.x = x;
        this.y = y;
    }
}
class Solution {
    public int[][] directions = new int[][]{{-1,0},{1,0},{0,1},{0,-1}};
    public int shortestBridge(int[][] A) {
        if (A == null || A.length == 0) return 0;
        int m = A.length, n = A[0].length;
        Queue<Pair> queue = new LinkedList<>();
        boolean found = false;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (A[i][j] == 1) {
                    dfs(A, i, j, m, n, queue);
                    found = true;
                    break;
                }
            }
            if (found) break;
        }
        int path = 0;
        while (!queue.isEmpty()) {
            int size = queue.size();
            path++;
            for (int i = 0; i < size; ++i) {
                Pair p = queue.poll();
                for (int[] d : directions) {
                    int newx = p.x + d[0];
                    int newy = p.y + d[1];
                    if (newx < 0 || newx >= m || newy < 0 || newy >= n || A[newx][newy] == 2) continue;
                    if (A[newx][newy] == 1) return path;
                    A[newx][newy] = 2;
                    queue.add(new Pair(newx, newy));
                }
            }
        }
        return path;
    }
    
    //find an island
    private void dfs(int[][] A, int i, int j, int m, int n, Queue<Pair> queue) {
        if (A[i][j] == 0) { 
            queue.add(new Pair(i, j)); 
            return; 
        }
        if (A[i][j] == 2) return;
        A[i][j] = 2;
        for (int[] d : directions) {
            if (i + d[0] >= 0 && i + d[0] < m && j + d[1] >= 0 && j + d[1] < n) {
                dfs(A, i + d[0], j + d[1], m, n, queue);
            }
        }
    }
}
```

### 939. Minimum Area Rectangle

Given a set of points in the xy-plane, determine the minimum area of a rectangle formed from these points, with sides parallel to the x and y axes.If there isn't any rectangle, return 0.

**Solution:**

把所有点按照x group到一起，HashMap,key->x, value->list of y。把list of y全部sort一遍。遍历两个column，去找相同的y值，有两对的话就可以组成一个长方形。

Note：

1. Integer的相等要用equals
2. 用一个x list去存x，当作遍历HashMap时的index。防止重复两次遍历一对x。有了index就可以用j+1，防止重复之前遍历过的。
3. 一些加速的小技巧

```java
public int minAreaRect(int[][] points) {
    Map<Integer, List<Integer>> map = new HashMap<>();
    List<Integer> x = new ArrayList<>();
    for (int[] p : points) {
        if (map.containsKey(p[0])) {
            map.get(p[0]).add(p[1]);
        } else {
            x.add(p[0]);
            map.put(p[0], new ArrayList<>());
            map.get(p[0]).add(p[1]);
        }
    }
    for (Map.Entry<Integer, List<Integer>> entry : map.entrySet()) {
        Collections.sort(entry.getValue());
    }
    int lasty = 0;
    int minArea = Integer.MAX_VALUE;
    boolean last = false;
    for (int i = 0; i < x.size(); ++i) {
        if (map.get(x.get(i)).size() <= 1) continue;
        for (int j = i + 1; j < x.size(); ++j) {
            if (map.get(x.get(j)).size() <= 1) continue;
            int m = 0, n = 0;
            List<Integer> col1 = map.get(x.get(i));
            List<Integer> col2 = map.get(x.get(j));
            while (m < col1.size() && n < col2.size()) {
                if (col1.get(m) < col2.get(n)) {
                    ++m;
                } else if (col1.get(m) > col2.get(n)) {
                    ++n;
                } else {
                     if (!last) {
                        last = true;
                    } else {
                        minArea = Math.min(minArea, Math.abs((x.get(i)-x.get(j)) * (col1.get(m) - lasty)));
                    }
                    lasty = col1.get(m);
                    ++m;
                    ++n;
                }
            }
            last = false;
        }
    }
    return minArea < Integer.MAX_VALUE ? minArea : 0;
}
```

### 947. Most Stones Removed with Same Row or Column

On a 2D plane, we place stones at some integer coordinate points.  Each coordinate point may have at most one stone.

Now, a *move* consists of removing a stone that shares a column or row with another stone on the grid.

What is the largest possible number of moves we can make?

**Solution:**

Number of islands 变种。求component个数，用结点个数减去conponent个数。dfs把与当前的点相连的所有点都标为visited。

```java
public int removeStones(int[][] stones) {
    Map<Integer, Set<Integer>> map = new HashMap<>();
    int n = stones.length;
    for (int i = 0; i < n; ++i) {
        Set<Integer> set = new HashSet<>();
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            if (stones[i][0] == stones[j][0] || stones[i][1] == stones[j][1]) {
                set.add(j);
            }
        }
        map.put(i, set);
    }
    boolean[] visited = new boolean[n];
    int res = 0;
    for (int i = 0; i < stones.length; ++i) {
        if (!visited[i]) {
            dfs(i, map, visited);
            res++;
        }
    }
    return n - res;
}

private void dfs(int index, Map<Integer, Set<Integer>> map, boolean[] visited) { 
    if (visited[index]) return;
    visited[index] = true;
    for (int i : map.get(index)) {
        dfs(i, map, visited);
    }
}
```

### 951. Flip Equivalent Binary Trees

For a binary tree T, we can define a flip operation as follows: choose any node, and swap the left and right child subtrees.

A binary tree X is *flip equivalent* to a binary tree Y if and only if we can make X equal to Y after some number of flip operations.

Write a function that determines whether two binary trees are *flip equivalent*.  The trees are given by root nodes `root1` and `root2`.

**Example 1:**

```
Input: root1 = [1,2,3,4,5,6,null,null,null,7,8], root2 = [1,3,2,null,6,4,5,null,null,null,null,8,7]
Output: true
Explanation: We flipped at nodes with values 1, 3, and 5.
```

 **Solution:**

递推：recursion。（也可以理解为是个dfs，dfs(recursion写法）本质上就是在用recursion去搜索）。

主要要把recursion单独当作一种方法，不要和dfs混淆，只是dfs经常用recursion来写，因为快且方便。

**Recursion 的要素**

- 递归的定义（递归函数求的是什么，完成了什么功能，类似dp[i]表示什么）

- 递归的拆解 （这次递归和之前的递归有什么关系，在本次递归调用递归传参，return等等，类似dp fucntion）

- 递归的出口 （什么时候可以return）

**写recursion的时候，assume对于当前的node是正确的，那么对于所有的node一定正确。**

**tree等需要recursion结构 判断identical类问题四步：**

1. 两个root皆为null return true （出口）

2. 两个root其中一个为null return false （出口）

3. 两个root的值不相等 return false （出口）

4. 根据题意，判断左右子树调用问题传参（递归的拆解）

   本题的recursion就是判断当前节点正确，且左右子树正确

```java
public boolean flipEquiv(TreeNode root1, TreeNode root2) {
    if (root1 == null && root2 == null) return true;
    if (root1 == null || root2 == null) return false;
    if (root1.val != root2.val) return false;
    return (flipEquiv(root1.left, root2.left) && flipEquiv(root1.right, root2.right)) || (flipEquiv(root1.left, root2.right) && flipEquiv(root1.right, root2.left));
}
```

### 963. Minimum Area Rectangle II

Given a set of points in the xy-plane, determine the minimum area of **any** rectangle formed from these points, with sides **not necessarily parallel** to the x and y axes.

If there isn't any rectangle, return 0.

**Solution:**

O($n^3$)遍历。先把所有的点加入set，取三个点，判断满足条件的第四个点在不在set种。

判断四个点p1, p2, p3, p4是rectangle(假设p1,p2是对角)：

1. p1.x + p2.x = p3.x + p4.x && p1.y + p2.y = p3.y + p4.y—> 保证了平行（对角线中点是同一个点）
2. (p1.x-p3.x) * (p2.x-p3.x) + (p1.y-p3.y) * (p2.y-p3.y) == 0 保证了垂直

```java
import java.awt.Point;

class Solution {
    
    public double minAreaFreeRect(int[][] points) {
        if (points == null  || points.length <= 3) return 0;
        int n = points.length;
        Point[] A = new Point[n];
        Set<Point> set = new HashSet<>();
        for (int i = 0; i < n; ++i) {
            A[i] = new Point(points[i][0], points[i][1]);
            set.add(A[i]);
        }
        
        double res = Double.MAX_VALUE;
        for (int i = 0; i < n; ++i) {
            for (int j = i+1; j < n; ++j) {
                for (int m = j + 1; m < n; ++m) {
                    Point candidate1 = new Point(A[i].x + A[j].x - A[m].x, A[i].y + A[j].y - A[m].y);   
                    double area;
                    if (checkRec(A[i], A[j], A[m], candidate1, set)) {
                        area = A[i].distance(A[m]) * A[j].distance(A[m]);
                        res = Math.min(res, area);
                    }  
                    Point candidate2 = new Point(A[i].x + A[m].x - A[j].x, A[i].y + A[m].y - A[j].y);
                    if (checkRec(A[i], A[m], A[j], candidate2, set)) {
                        area = A[i].distance(A[j]) * A[m].distance(A[j]);
                        res = Math.min(res, area);
                    }
                    Point candidate3 = new Point(A[j].x + A[m].x - A[i].x, A[j].y + A[m].y - A[i].y);
                    if (checkRec(A[j], A[m], A[i], candidate3, set)) {
                        area = A[j].distance(A[i]) * A[m].distance(A[i]);
                        res = Math.min(res, area);
                    }
                }
            }
        }
        return res < Double.MAX_VALUE ? res : 0;
    }
    
    private boolean checkRec(Point p1, Point p2, Point p3, Point p4, Set<Point> set) {
        if (!set.contains(p4)) return false;
        if ((p1.x - p3.x) * (p2.x - p3.x) + (p1.y - p3.y) * (p2.y - p3.y) == 0) return true;
        return false;
    } 
}
```

### 981. Time Based Key-Value Store

Create a timebased key-value store class `TimeMap`, that supports two operations.

\1. `set(string key, string value, int timestamp)`

- Stores the `key` and `value`, along with the given `timestamp`.

\2. `get(string key, int timestamp)`

- Returns a value such that `set(key, value, timestamp_prev)` was called previously, with `timestamp_prev <= timestamp`.
- If there are multiple such values, it returns the one with the largest `timestamp_prev`.
- If there are no values, it returns the empty string (`""`).

 **Solution:**

数据结构可以用一个HashMap<key, List<Pair<timestamp,String>>> 这样set是O(1), get正常情况是O(n)（暴力搜一遍）。或者set,get都是O(lgn)。用TreeMap，treemap的get和put都是O(lgn)，所以这里的set,get都是O(lgn)。

```java
class TimeMap {
    /** Initialize your data structure here. */
    private Map<String, TreeMap<Integer, String>> map;
    public TimeMap() {
        map = new HashMap();
    }
    
    public void set(String key, String value, int timestamp) {
        if (!map.containsKey(key)) {
            map.put(key, new TreeMap<>());
        } 
        map.get(key).put(timestamp, value);
        
    }
    
    public String get(String key, int timestamp) {
        if (map.containsKey(key)) {
            Integer t = map.get(key).floorKey(timestamp);
            return t == null ? "" : map.get(key).get(t);
        } else {
            return "";
        }
    }
}
```

### 987. Vertical Order Traversal of a Binary Tree

Given a binary tree, return the *vertical order* traversal of its nodes values.

For each node at position `(X, Y)`, its left and right children respectively will be at positions `(X-1, Y-1)` and `(X+1, Y-1)`.

Running a vertical line from `X = -infinity` to `X = +infinity`, whenever the vertical line touches some nodes, we report the values of the nodes in order from top to bottom (decreasing `Y` coordinates).

If two nodes have the same position, then the value of the node that is reported first is the value that is smaller.

Return an list of non-empty reports in order of `X` coordinate.  Every report will have a list of values of nodes.

**Example 1:**

![img](https://assets.leetcode.com/uploads/2019/01/31/1236_example_1.PNG)

```
Input: [3,9,20,null,null,15,7]
Output: [[9],[3,15],[20],[7]]
Explanation: 
Without loss of generality, we can assume the root node is at position (0, 0):
Then, the node with value 9 occurs at position (-1, -1);
The nodes with values 3 and 15 occur at positions (0, 0) and (0, -2);
The node with value 20 occurs at position (1, -1);
The node with value 7 occurs at position (2, -2).
```

**Soluiton:**

用某种搜索或者traversal去遍历每个node，把其位置和val放进去。

BFS + sort

```java
public List<List<Integer>> verticalTraversal(TreeNode root) {
    if (root == null) return null;
    Queue<TreeNode> queue = new LinkedList<>();
    List<List<Integer>> result = new ArrayList<>();
    //int[0]:x int[1]:y int[2]:val
    List<int[]> infos = new ArrayList<>();
    queue.offer(root);
    infos.add(new int[]{0,0,root.val});
    int index = -1;
    while (!queue.isEmpty()) {
        TreeNode node = queue.poll();
        index++;
        int cur_x = infos.get(index)[0];
        int cur_y = infos.get(index)[1];
        if (node.left != null) {
            queue.offer(node.left);
            infos.add(new int[]{cur_x - 1, cur_y - 1, node.left.val});
        }
        if (node.right != null) {
            queue.offer(node.right);
            infos.add(new int[]{cur_x + 1, cur_y - 1, node.right.val});
        }
    }
    Collections.sort(infos, new Comparator<int[]>() {
        public int compare(int[] a, int[] b) {
            int d = a[0]-b[0];
            if (d == 0) {
                d = b[1] - a[1];
                if (d == 0) {
                    d = a[2] - b[2];
                }
            }
            return d;
        }
    });
    int x = Integer.MAX_VALUE;
    List<Integer> col = null;
    for (int[] each : infos) {
        if (each[0] != x) {
            if (col != null) {
                result.add(col);
            }
            col = new ArrayList<>();
        }
        col.add(each[2]);
        x = each[0];
    }
    result.add(col);
    return result;
}
```

### 1011. Capacity To Ship Packages Within D Days

A conveyor belt has packages that must be shipped from one port to another within `D` days.

The `i`-th package on the conveyor belt has a weight of `weights[i]`.  Each day, we load the ship with packages on the conveyor belt (in the order given by `weights`). We may not load more weight than the maximum weight capacity of the ship.

Return the least weight capacity of the ship that will result in all the packages on the conveyor belt being shipped within `D` days.

**Example:**

```
Input: weights = [1,2,3,4,5,6,7,8,9,10], D = 5
Output: 15
Explanation: 
A ship capacity of 15 is the minimum to ship all the packages in 5 days like this:
1st day: 1, 2, 3, 4, 5
2nd day: 6, 7
3rd day: 8
4th day: 9
5th day: 10

Note that the cargo must be shipped in the order given, so using a ship of capacity 14 and splitting the packages into parts like (2, 3, 4, 5), (1, 6, 7), (8), (9), (10) is not allowed. 
```

**Solution:**

这种题一上来要想到二分，首先得有个思想就是知道会有一个helper function去算给定一个capacity和weights，找到相应的天数或者判断在D天内能不能完成该任务。二分的话就是去找个lower bound, upper bound, 然后利用helper function去判断属于哪一边。

关于lower bound可以是1，也可以是max(average, maxWeight),实际上用1差不多，upperbound 是 totalwieght(最快就是一天完成嘛，不可能0天)。

然后在[lowerbound,upperBound]二分去找满足shipValid的最小的数，也就是第一个满足shipValid的数。

```java
public int shipWithinDays(int[] weights, int D) {
    int maxWeight = Integer.MIN_VALUE;
    int totalWeight = 0;
    for (int w : weights) {
        maxWeight = Math.max(w, maxWeight);
        totalWeight += w;
    }
    int average = totalWeight % D == 0 ? totalWeight/D : totalWeight/D + 1;
    int left = Math.max(average, maxWeight), right = totalWeight;
    while (left + 1 < right) {
        int mid = left + (right-left)/2;
        if (shipValid(weights, D, mid)) {
            right = mid;
        } else {
            left = mid;
        }
    }
    if (shipValid(weights, D, left)) return left;
    return right; 
}
private boolean shipValid(int[] weights, int D, int capacity) {
    int cnt = 1, tmp = capacity;
    for (int i = 0; i < weights.length;) {
        tmp -= weights[i];
        if (tmp < 0) {
            tmp = capacity;
            cnt += 1;
        } else {
            ++i;
        }
    }
    return cnt <= D;
}
```

###1047. Remove All Adjacent Duplicates In String

Given a string `S` of lowercase letters, a *duplicate removal* consists of choosing two adjacent and equal letters, and removing them.

We repeatedly make duplicate removals on S until we no longer can.

Return the final string after all such duplicate removals have been made.  It is guaranteed the answer is unique.

**Example 1:**

```
Input: "abbaca"
Output: "ca"
Explanation: 
For example, in "abbaca" we could remove "bb" since the letters are adjacent and equal, and this is the only possible move.  The result of this move is that the string is "aaca", of which only "aa" is possible, so the final string is "ca".
```

**Solution:**

Stack, 别忘了最后要reverse。

```java
public String removeDuplicates(String S) {
    if (S == null || S.length() == 0) return S;
    Stack<Character> stack = new Stack<Character>();
    for (int i = 0; i < S.length(); ++i) {
        char c = S.charAt(i);
        if (!stack.isEmpty()) {
            if (stack.peek() == c) {
                stack.pop();
            } else {
                stack.push(c);
            }
        } else {
            stack.push(c);
        }
    }
    StringBuilder sb = new StringBuilder();
    while (!stack.isEmpty()) {
        sb.append(stack.pop());
    }
    return sb.reverse().toString();
}
```

### 1055. Shortest Way to Form String

From any string, we can form a *subsequence* of that string by deleting some number of characters (possibly no deletions).

Given two strings `source` and `target`, return the minimum number of subsequences of `source` such that their concatenation equals `target`. If the task is impossible, return `-1`.

**Example 1:**

```
Input: source = "abc", target = "abcbc"
Output: 2
Explanation: The target "abcbc" can be formed by "abc" and "bc", which are subsequences of source "abc".
```

**Solution:**

先用一个map去判断是不是target中出现的character在source中出现过，如果没有返回-1，如果全都出现过说明一定可以form 成功。

Greedy，遍历target，再遍历source去cancate target中的subarray。把target从头开始覆盖，遍历source去尽可能的从头顺次覆盖，这样得到的是最短的way。

```java
public int shortestWay(String source, String target) {
    int m = source.length(), n = target.length();
    int[] chars = new int[256];
    for (char c : source.toCharArray()) {
        chars[c] = 1;
    }
    for (char c : target.toCharArray()) {
        if (chars[c] != 1) return -1;
    }
    int i = 0, j = 0;
    int res = 0;
    while (i < n) {
        while (j < m && i < n) {
            if (target.charAt(i) == source.charAt(j)) i++;
            j++;
        }
        res++;
        j = 0;
    }
    return res;
}
```

### 1057. Campus Bikes

On a campus represented as a 2D grid, there are `N` workers and `M` bikes, with `N <= M`. Each worker and bike is a 2D coordinate on this grid.

Our goal is to assign a bike to each worker. Among the available bikes and workers, we choose the (worker, bike) pair with the shortest Manhattan distance between each other, and assign the bike to that worker. (If there are multiple (worker, bike) pairs with the same shortest Manhattan distance, we choose the pair with the smallest worker index; if there are multiple ways to do that, we choose the pair with the smallest bike index). We repeat this process until there are no available workers.

The Manhattan distance between two points `p1` and `p2` is `Manhattan(p1, p2) = |p1.x - p2.x| + |p1.y - p2.y|`.

Return a vector `ans` of length `N`, where `ans[i]` is the index (0-indexed) of the bike that the `i`-th worker is assigned to.

**Example:**

![img](https://assets.leetcode.com/uploads/2019/03/06/1261_example_1_v2.png)

```
Input: workers = [[0,0],[2,1]], bikes = [[1,2],[3,3]]
Output: [1,0]
Explanation: 
Worker 1 grabs Bike 0 as they are closest (without ties), and Worker 0 is assigned Bike 1. So the output is [1, 0].
```

**Solution:**

用一个tuple去存distance，workerId，bikeId。根据dis，workId,bikeId  sort。然后依次把bikeid存袋相应的workid为index的答案中。注意 index该用什么。

两个加速的trick:

1. 用array代替set，不需要worker的set，把ans视为set，-1代表还没分配bike。
2. **用一个counter去记录已经分配了几个woker，到达worker的数量，提前break，常见的一种提前跳出循环的方法，而且加速作用很大**

```java
public class Tuple {
    public int dis;
    public int workerId;
    public int bikeId;
    public Tuple(int dis, int workerId, int bikeId) {
        this.dis = dis;
        this.workerId = workerId;
        this.bikeId = bikeId;
    }
}
public int[] assignBikes(int[][] workers, int[][] bikes) {
    if (workers == null || workers.length == 0) return null;
    int m = workers.length, n = bikes.length;
    Tuple[] tuples = new Tuple[m*n];
    int index = 0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            tuples[index++] = new Tuple(distance(workers[i],bikes[j]), i, j);;
        }
    }
    Arrays.sort(tuples, new Comparator<Tuple>(){
        public int compare(Tuple a, Tuple b) {
            int res = a.dis - b.dis;
            if (res == 0) {
                res = a.workerId - b.workerId;
                if (res == 0) {
                    res = a.bikeId - b.bikeId;
                }
            }
            return res;
        }
    });
    int cnt = 0;
    int[] ans = new int[m];
    Arrays.fill(ans, -1);
    int[] bikeValid = new int[n];
    for (int i = 0; i < tuples.length; ++i) {
        if (ans[tuples[i].workerId] == -1 && bikeValid[tuples[i].bikeId] == 0) {
            ans[tuples[i].workerId] = tuples[i].bikeId;
            bikeValid[tuples[i].bikeId] = 1;
            cnt++;
        } 
        if (cnt == m) break;
    }
    return ans;
}
private int distance(int[] a, int[] b) {
    return Math.abs(a[0]-b[0]) + Math.abs(a[1] - b[1]);
}
```

### 1087. Brace Expansion

A string `S` represents a list of words.

Each letter in the word has 1 or more options.  If there is one option, the letter is represented as is.  If there is more than one option, then curly braces delimit the options.  For example, `"{a,b,c}"` represents options `["a", "b", "c"]`.

For example, `"{a,b,c}d{e,f}"` represents the list `["ade", "adf", "bde", "bdf", "cde", "cdf"]`.

Return all words that can be formed in this manner, in lexicographical order.

**Example 1:**

```
Input: "{a,b}c{d,e}f"
Output: ["acdf","acef","bcdf","bcef"]
```

**Example 2:**

```
Input: "abcd"
Output: ["abcd"]
```

**Solution:**

看作是个图，去搜索。是个排列组合问题，用DFS/backtracking。

**dfs要素**

dfs recursion 定义是，把当前这一层（也可能是与入口所有相连的结点，或者可能相连的结点，取决于进入dfs返回，还是判断后再进入dfs）所有结点进行处理(这里需要用一个for循环来遍历当前层的结点)。每遍历一个点，继续dfs遍历它所连接的点。

本题的dfs定义是把当前这一层的character加到cur的string中。+操作每次返回一个新的String，不改变原来的Stirng，原来的String还是保留的。相当于开辟了很多个String的空间。而backtracking则是对一个referecnce进行append, remove操作。没有开辟新的空间。

```java
public String[] expand(String S) {
    List<String> result = new ArrayList<>();
    dfs(S, 0, result, "");
    String[] ans = new String[result.size()];
    Collections.sort(result);
    int i = 0;
    for (String s : result) {
        ans[i++] = s;
    }
    return ans;
}

// dfs: 把index开始的当前的一层加到cur
private void dfs(String S, int index, List<String> result, String cur) {
    if (index == S.length()) {
        result.add(cur.toString());
        return;
    }
    List<Character> chars = new ArrayList<>();
    if (S.charAt(index) != '{') {
        chars.add(S.charAt(index++));
    } else {
        index++;
        while (S.charAt(index) != '}') {
            if (S.charAt(index) != ',') {
                chars.add(S.charAt(index));
            }
            ++index;
        }
        ++index;
    }
    for (char c : chars) {
        dfs(S, index, result, cur+c);
    }
}
```

**Solution2:**

backtracking, 先append，传进去，结束后，delete最后一位。

```java
public String[] expand(String S) {
    List<String> result = new ArrayList<>();
    dfs(S, 0, result, new StringBuilder());
    String[] ans = new String[result.size()];
    Collections.sort(result);
    int i = 0;
    for (String s : result) {
        ans[i++] = s;
    }
    return ans;
}

// dfs: 把index开始的当前的一层加到cur
private void dfs(String S, int index, List<String> result, StringBuilder cur) {
    if (index == S.length()) {
        result.add(cur.toString());
        return;
    }
    List<Character> chars = new ArrayList<>();
    if (S.charAt(index) != '{') {
        chars.add(S.charAt(index++));
    } else {
        index++;
        while (S.charAt(index) != '}') {
            if (S.charAt(index) != ',') {
                chars.add(S.charAt(index));
            }
            ++index;
        }
        ++index;
    }
    for (char c : chars) {
        cur.append(c);
        dfs(S, index, result, cur);
        cur.delete(cur.length()-1, cur.length());
    }
}
```

### 1091. Shortest Path in Binary Matrix

In an N by N square grid, each cell is either empty (0) or blocked (1).

A *clear path from top-left to bottom-right* has length `k` if and only if it is composed of cells `C_1, C_2, ..., C_k` such that:

- Adjacent cells `C_i` and `C_{i+1}` are connected 8-directionally (ie., they are different and share an edge or corner)
- `C_1` is at location `(0, 0)` (ie. has value `grid[0][0]`)
- `C_k` is at location `(N-1, N-1)` (ie. has value `grid[N-1][N-1]`)
- If `C_i` is located at `(r, c)`, then `grid[r][c]` is empty (ie. `grid[r][c] == 0`).

Return the length of the shortest such clear path from top-left to bottom-right.  If such a path does not exist, return -1.

**Example 1:**

```
Input: [[0,1],[1,0]]
Output: 2
```

**Solution:**

8-way BFS

1. 注意判断x,y exceeds the bound
2. check if the positon is valid(grid[x] [y] == 0)
3. 设置visited，不用再设置回去，因为一层一层保证进行到当前为止是最短距离。我们在加到queue之前判断visited，加到queue之后设置为visited。同一层的node具有相同的优先级。

```java
class Pair {
    int x;
    int y;
    public Pair(int x, int y) {
        this.x = x;
        this.y = y;
    }
}
class Solution {
    int[][] directions = new int[][]{{-1,-1},{-1,0},{-1,1},{0,1},{0,-1},{1,1},{1,0},{1,-1}};
    public int shortestPathBinaryMatrix(int[][] grid) {
        if (grid == null || grid[0][0] == 1) return -1;
        int n = grid.length;
        if (grid[0][0] == 0 && n == 1) return 1;
        boolean[][] visited = new boolean[n][n];
        Queue<Pair> queue = new LinkedList<>();
        queue.offer(new Pair(0,0));
        visited[0][0] = true;
        int level = 0;
        while(!queue.isEmpty()) {
            int size = queue.size();
            ++level;
            for (int i = 0; i < size; ++i) {
                Pair cur = queue.poll();
                for (int[] d : directions) {
                    int newX = cur.x + d[0];
                    int newY = cur.y + d[1];
                    if (newX < 0 || newX >= n || newY < 0 || newY >= n || grid[newX][newY] == 1) continue;
                    if (newX == n-1 && newY == n-1) return level+1;
                    if (!visited[newX][newY]) {
                        queue.offer(new Pair(newX, newY));
                        visited[newX][newY] = true;
                    }
                }
            }
        }
        return -1;
    }
}
```

### 1110. Delete Nodes And Return Forest

Given the `root` of a binary tree, each node in the tree has a distinct value.

After deleting all nodes with a value in `to_delete`, we are left with a forest (a disjoint union of trees).

Return the roots of the trees in the remaining forest.  You may return the result in any order.

**Example 1:**

**![img](https://assets.leetcode.com/uploads/2019/07/01/screen-shot-2019-07-01-at-53836-pm.png)**

```
Input: root = [1,2,3,4,5,6,7], to_delete = [3,5]
Output: [[1,2,null,4],[6],[7]]
```

 **Solution:**

Recursion, (dfs)， recursion完成的功能是，处理root左右结点，相当于左右结点之下的都已经处理好了，该删的删掉了，该加的加到forest中了，然后处理本结点。（有点像post order tarversal）是一种思路处理好左右结点再处理自己。

如果先处理自己再左右结点的话，会出现，如果2，4都被删除，那么处理2的时候就先把4加到结果中，后边把4置为null，只是把当时root不再reference到4结点，但是4这个object还在,并且list这个容器的相应位置还是reference到了4这个object。

```java
public List<TreeNode> delNodes(TreeNode root, int[] to_delete) {        
    Set<Integer> set = new HashSet<>();
    List<TreeNode> list = new ArrayList<>();
    for (int i : to_delete) {
        set.add(i);
    }
    root = helper(root, set, list);
    if (root != null) list.add(root);
    return list;
}

private TreeNode helper(TreeNode root, Set set, List list) {
    if (root == null) return null;
    root.left = helper(root.left, set, list);
    root.right = helper(root.right, set, list);
    if (set.contains(root.val)) {
        if (root.left != null) list.add(root.left);
        if (root.right != null) list.add(root.right);
        root = null;
    }
    return root;
}
```

### 1146. Snapshot Array

Implement a SnapshotArray that supports the following interface:

- `SnapshotArray(int length)` initializes an array-like data structure with the given length.  **Initially, each element equals 0**.
- `void set(index, val)` sets the element at the given `index` to be equal to `val`.
- `int snap()` takes a snapshot of the array and returns the `snap_id`: the total number of times we called `snap()` minus `1`.
- `int get(index, snap_id)` returns the value at the given `index`, at the time we took the snapshot with the given `snap_id`

**Example 1:**

```
Input: ["SnapshotArray","set","snap","set","get"]
[[3],[0,5],[],[0,6],[0,0]]
Output: [null,null,0,null,5]
Explanation: 
SnapshotArray snapshotArr = new SnapshotArray(3); // set the length to be 3
snapshotArr.set(0,5);  // Set array[0] = 5
snapshotArr.snap();  // Take a snapshot, return snap_id = 0
snapshotArr.set(0,6);
snapshotArr.get(0,0);  // Get the value of array[0] with snap_id = 0, return 5
```

 **Solution:**

为了节省空间，不开长度为length的array，而是用一个arraylist，每次set的时候把index之前的设为0，get时候如果index比当前add到的index大，就直接返回0.

用一个hashmap去存snap_id 和当时的array(copy)。

```java
class SnapshotArray {
    ArrayList<Integer> arr = null;
    int snap_id = 0;
    HashMap<Integer,ArrayList<Integer>> map = new HashMap<>();
    public SnapshotArray(int length) {
        arr = new  ArrayList<Integer>();
    }
    
    public void set(int index, int val) {
        if(arr.size() <= index){
            while(arr.size() != index){
                arr.add(0);
            }
            arr.add(index,val);    
        }else{
            arr.set(index,val);
        }
        
    }
    
    public int snap() {
        ArrayList<Integer> temp = new ArrayList<Integer>();
    		temp.addAll(arr);
        map.put(snap_id,temp);
        snap_id++;
        return snap_id-1;
    }
    
    public int get(int index, int snap_id) {
       ArrayList<Integer> temp =  map.get(snap_id);
        if(temp.size() > index)
            return temp.get(index);
        else 
            return 0;
    }
}
```

###1170. Compare Strings by Frequency of the Smallest Character

Let's define a function `f(s)` over a non-empty string `s`, which calculates the frequency of the smallest character in `s`. For example, if `s = "dcce"` then `f(s) = 2` because the smallest character is `"c"` and its frequency is 2.

Now, given string arrays `queries` and `words`, return an integer array `answer`, where each `answer[i]` is the number of words such that `f(queries[i])` < `f(W)`, where `W` is a word in `words`.

**Example:**

```
Input: queries = ["bbb","cc"], words = ["a","aa","aaa","aaaa"]
Output: [1,2]
Explanation: On the first query only f("bbb") < f("aaaa"). On the second query both f("aaa") and f("aaaa") are both > f("cc").
```

**Solution:**

本道题的二分是去找nums中比target大的个数，也就是求nums中第一个大于target的元素，也就是分为，小于等于target和大于target，所以二分的条件是 (小于等于) 和 (大于), 逼近之后逐一判断left，right（不想细想最后倒是逼近到哪了）。

极端情况是全部都比target大，最后left，right是在length-2，length-1，都不满足>的话，返回right+1。

二分的话具体情况具体分析。举例思考。

```java
public int[] numSmallerByFrequency(String[] queries, String[] words) {
    int m = queries.length, n = words.length;
    int[] fqueries = new int[m];
    int[] fwords = new int[n];
    int[] ans = new int[m];
    for (int i = 0; i < m; ++i) {
        fqueries[i] = f(queries[i]);
    }
    for (int i = 0; i < n; ++i) {
        fwords[i] = f(words[i]);
    }
    Arrays.sort(fwords);
    for (int i = 0; i < m; ++i) {
        ans[i] = n - binarySearch(fqueries[i], fwords);
    }
    return ans;
}
private int f(String s) {
    Map<Character, Integer> map = new HashMap<Character, Integer>();
    char min = 'z';
    for (int i = 0; i < s.length(); ++i) {
        char c = s.charAt(i);
        if (map.containsKey(c)) {
            map.put(c, map.get(c)+1);
        } else {
            map.put(c, 1);
        }
        if (c < min) min = c;
    }
    return map.get(min);
}
private int binarySearch(int target, int[] nums) {
    int left = 0, right = nums.length - 1;
    while (left + 1 < right) {
        int mid = left + (right - left)/2;
        if (nums[mid] > target) {
            right = mid;
        } else if (nums[mid] <= target) {
            left = mid;
        } 
    }
    if (nums[left] > target) return left;
    if (nums[right] > target) return right;
    return right+1;
}
```

###1197. Minimum Knight Moves

In an **infinite** chess board with coordinates from `-infinity` to `+infinity`, you have a **knight** at square `[0, 0]`.

A knight has 8 possible moves it can make, as illustrated below. Each move is two squares in a cardinal direction, then one square in an orthogonal direction.

![img](https://assets.leetcode.com/uploads/2018/10/12/knight.png)

Return the minimum number of steps needed to move the knight to the square `[x, y]`.  It is guaranteed the answer exists.

**Example 1:**

```
Input: x = 2, y = 1
Output: 1
Explanation: [0, 0] → [2, 1]
```

**Solution:**

1. 记住directions 的写法是花括号
2. 记住queue怎么define，用的是Queue 和 linkedlist
3. 因为本题一定有答案，而且chess没有边界，不会越界，所以不用在for loop中判断
4. 本题层数按照$8^n$增长，遍历过的position不要再遍历了，节省时间
5. 考虑到用set去存一个pair，但是pair 是 reference，不能contains去看，要重写equals 和hashcode函数，但是String等包装类，已经重写好了equals函数，treemap等重写好了compare所以可以直接用contians。一个tricky的办法：可以建一个x-y的string存到map里
6. 本题因为pair跑不过，所以加了很多减支，正负其实是一样的，另外用一个Integer去存了两个x,y在里面。

```java
class Solution {
    final int[][] directions = {{2,1},{1,2},{2,-1},{1,-2},{-1,-2},{-2,-1},{-2,1},{-1,2}};
    public int minKnightMoves(int x, int y) {
        x = Math.abs(x);
        y = Math.abs(y);
        Queue<Integer> queue = new LinkedList<>();
        Set<Integer> set = new HashSet<>();
        queue.add(0);
        set.add(0);
        int step = 0;
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                Integer cur = queue.poll();
                int cr = cur >> 10, cc = cur - (cr << 10);
                if (cr == x && cc == y) return step;
                for (int[] d : directions) {
                    int m = cr + d[0];
                    int n = cc + d[1];
                    if (m < -2 || n < -2) continue;
                    int next = (m << 10) + n;
                    if (set.contains(next)) continue;
                    set.add(next);
                    queue.add(next);
                }
            }
            step++;
        }
        return -1;
    }
}
```

