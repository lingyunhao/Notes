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

