## Google tag

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

