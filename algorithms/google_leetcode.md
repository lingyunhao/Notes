

## Google tag

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
4. 排序的时间复杂度是O(nlog(n)), 此solution复杂度也是O (nlg(n)).

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

