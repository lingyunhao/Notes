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

### 206. Reverse Linked List

```
Input: 1->2->3->4->5->NULL
Output: 5->4->3->2->1->NULL
```

```java
public ListNode reverseList(ListNode head) {
    ListNode prev = null;
    while(head != null) {
        ListNode tmp = head.next;
        head.next = prev;
        prev = head;
        head = tmp;
    }
    return prev;
}
```

### 207. Course Schedule

There are a total of *n* courses you have to take, labeled from `0` to `n-1`.

Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: `[0,1]`

Given the total number of courses and a list of prerequisite **pairs**, is it possible for you to finish all courses?

**Example 1:**

```
Input: 2, [[1,0]] 
Output: true
Explanation: There are a total of 2 courses to take. 
             To take course 1 you should have finished course 0. So it is possible.
```

**Example 2:**

```
Input: 2, [[1,0],[0,1]]
Output: false
Explanation: There are a total of 2 courses to take. 
             To take course 1 you should have finished course 0, and to take course 0 you should
             also have finished course 1. So it is impossible.
```

**Solution:**

topological sort

```java
public boolean canFinish(int numCourses, int[][] prerequisites) {
    // Topological Sort
    // need a hashmap to save the indegree to each node(each course)
    Map<Integer, Integer> node_to_indegree = new HashMap<>();
    // 先给每一个node都在map里 initialize the indegree as 0
    for ( int i = 0; i < numCourses; i++){
        node_to_indegree.put(i, 0);
    }

    int length = prerequisites.length;
    for ( int i = 0; i < length; i++){
        node_to_indegree.put(prerequisites[i][0], node_to_indegree.getOrDefault(prerequisites[i][0], 0) + 1);
    }

    Deque<Integer> q = new LinkedList<>();
    // offer the coursed whose indegree is 0 into queue
    for(Integer key : node_to_indegree.keySet()){
        if(node_to_indegree.get(key) == 0){
            q.offer(key);
        }
    }

    List<Integer> result = new ArrayList<>();
    while(!q.isEmpty()){
        Integer curCourse = q.poll();
        result.add(curCourse);
        // check all the next course whose prerequisites is curCourse and deduct their indegree by 1, offer them into the queue when the indegree == 0
        for (int i = 0; i < length; i++){
            if(prerequisites[i][1] == curCourse){
                // node_to_indegree.get(prerequisites[i][0]--);
                node_to_indegree.put(prerequisites[i][0], node_to_indegree.getOrDefault(prerequisites[i][0], 0) - 1);
                if(node_to_indegree.get(prerequisites[i][0]) == 0){
                    q.offer(prerequisites[i][0]);
                }
            }
        }	
    }

    if (result.size() == numCourses){
        return true;
    }else{
        return false;
    }
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

### 210. Course Schedule II

There are a total of *n* courses you have to take, labeled from `0` to `n-1`.

Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: `[0,1]`

Given the total number of courses and a list of prerequisite **pairs**, return the ordering of courses you should take to finish all courses.

There may be multiple correct orders, you just need to return one of them. If it is impossible to finish all courses, return an empty array.

**Example 1:**

```
Input: 4, [[1,0],[2,0],[3,1],[3,2]]
Output: [0,1,2,3] or [0,2,1,3]
Explanation: There are a total of 4 courses to take. To take course 3 you should have finished both     
             courses 1 and 2. Both courses 1 and 2 should be taken after you finished course 0. 
             So one correct course order is [0,1,2,3]. Another correct ordering is [0,2,1,3] .
```

**Solution:**

```java
public int[] findOrder(int numCourses, int[][] prerequisites) {

boolean isPossible = true;
Map<Integer, List<Integer>> adjList = new HashMap<Integer, List<Integer>>();
int[] indegree = new int[numCourses];
int[] topologicalOrder = new int[numCourses];

// Create the adjacency list representation of the graph
for (int i = 0; i < prerequisites.length; i++) {
  int dest = prerequisites[i][0];
  int src = prerequisites[i][1];
  List<Integer> lst = adjList.getOrDefault(src, new ArrayList<Integer>());
  lst.add(dest);
  adjList.put(src, lst);

  // Record in-degree of each vertex
  indegree[dest] += 1;
}

// Add all vertices with 0 in-degree to the queue
Queue<Integer> q = new LinkedList<Integer>();
for (int i = 0; i < numCourses; i++) {
  if (indegree[i] == 0) {
    q.add(i);
  }
}

int i = 0;
// Process until the Q becomes empty
while (!q.isEmpty()) {
  int node = q.remove();
  topologicalOrder[i++] = node;

  // Reduce the in-degree of each neighbor by 1
  if (adjList.containsKey(node)) {
    for (Integer neighbor : adjList.get(node)) {
      indegree[neighbor]--;

      // If in-degree of a neighbor becomes 0, add it to the Q
      if (indegree[neighbor] == 0) {
        q.add(neighbor);
      }
    }
  }
}

// Check to see if topological sort is possible or not.
if (i == numCourses) {
  return topologicalOrder;
}

return new int[0];
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

### 222. Count Complete Tree Nodes

Given a **complete** binary tree, count the number of nodes.

**Note:**

**Definition of a complete binary tree from Wikipedia:**
In a complete binary tree every level, except possibly the last, is completely filled, and all nodes in the last level are as far left as possible. It can have between 1 and 2h nodes inclusive at the last level h.

**Example:**

```
Input: 
    1
   / \
  2   3
 / \  /
4  5 6
Output: 6
```

**Solution1**

cnt表示最后一层的个数，别忘了循环只进行了h-1次，最后一层一定到了最后一层的根节点上，但是没有进行判断把这个节点加进去，所以最后要判断最后节点，如果此节点不为null要+1.

Math.pow(2, h-1)是前h-1层的结点个数之和。

求高度是O(lgn)的复杂度，在一个h-1的for循环中，又求高度，所以是O(lgn * lgn)的复杂度， 比O(n)小很多。 可以用搜索这样如solution2每个节点遍历一遍，时间复杂度为O(n)。

a to the power of b

```java
public int countNodes(TreeNode root) {
    if (root == null) return 0;
    int h = countHeight(root);
    int cnt = 0;
    for (int i = h - 1; i > 0; --i) {
        if (countHeight(root.right) == i) {
            cnt += (int)Math.pow(2, i-1); // two to the power of i-1
            root = root.right;
        } else {
            root = root.left;
        }
    }
    if (root != null) cnt++;
    return (int)Math.pow(2, h-1) - 1 + cnt;
}
private int countHeight(TreeNode root) {
    return (root == null) ? 0 : countHeight(root.left) + 1;
}
```

**Solution2:**

O(n) 慢，失去了complete的意义

```java
class Solution {
  public int countNodes(TreeNode root) {
    return root != null ? 1 + countNodes(root.right) + countNodes(root.left) : 0;
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

### 234. Palindrome Linked List

Could you do it in O(n) time and O(1) space?

**Solution:**

快慢指针 + reverse。本题用的判断条件是 fast != null && fast.next != null, 所以slow会停在偏右的地方，因为没有断开两条链，所以还是连着的，尔reverse函数返回的是新的prev。我们把slow的next置为null，head还是连到了原来的slow那里，对于偶数个数的链表，只比较slow后边的个数

[1,2,3,4]  —> head : 1 -> 2 -> 3

​                       slow : 4 -> 3

[1,2,2,1]  —> head : 1-> 2 -> 2

​                       slow :  1 -> 2           比较到前两个就停下了

[1,2,3,2,1] —> head: 1 -> 2 -> 3

​                         slow: 1 -> 2 -> 3  这里的val为3的node实际上同一个，就是快慢指针之后的slow，然后在reverse的时候把slow.next 置为null了

```java
public boolean isPalindrome(ListNode head) {
    if (head == null) return true;
    ListNode slow = head;
    ListNode fast = head;
    while (fast != null && fast.next != null) {
        slow = slow.next;
        fast = fast.next.next;
    }
    slow = reverse(slow);

    ListNode test = head;
     while (test != null) {
         System.out.println(test.val);
         test = test.next;
     }

    while (slow != null) {
        if (head.val != slow.val) {
            return false;
        }
        head = head.next;
        slow = slow.next;
    }
    return true;
}

private ListNode reverse(ListNode head) {
    ListNode prev = null, tmp = null;
    while (head != null) {
        tmp = head.next;
        head.next = prev;
        prev = head;
        head = tmp;
    }
    return prev;
}
```



### 237. Delete Node in a Linked List

Write a function to delete a node (except the tail) in a singly linked list, given only access to that node.

Given linked list -- head = [4,5,1,9], which looks like following:

![img](https://assets.leetcode.com/uploads/2018/12/28/237_example.png)

 

**Example 1:**

```
Input: head = [4,5,1,9], node = 5
Output: [4,1,9]
Explanation: You are given the second node with value 5, the linked list should become 4 -> 1 -> 9 after calling your function.
```

**Solution:**

We don;'t have access to the head, 存一个当前node(prev), 把当前的node的值改为next node的值，知道node到了倒数第二个也就是node.next == null， 然后把最后一个砍掉。

```java
public void deleteNode(ListNode node) {
    ListNode pre = node;
    while (node.next != null) {
        pre = node;
        node = node.next;
        pre.val = node.val;
    }
    pre.next = null;
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

### **257. Binary Tree Paths**

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

### 278. First Bad Version

**Solution:**

Binary Search 之境界二，OOOOOXXXXX find the first bad version. While 循环去逼近first bad version, 正常情况下应该left和right应该定位到中间的OX上，一般情况下应该返回right，但是若第一版就是bad，就需要返回left，因为left，right达到了左边的极限left=1，right=2。

```java
public int firstBadVersion(int n) {
    int left = 1, right = n;
    while (left + 1 < right) {
        int mid = left + (right - left) / 2;
        if (isBadVersion(mid)) {
            right = mid;
        } else {
            left = mid;
        }
    }
    if(isBadVersion(left)) return left;
    return right;
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

### 283. Move Zeroes

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



### 297. Serialize and Deserialize Binary Tree:

**Example:** 

```
You may serialize the following tree:

    1
   / \
  2   3
     / \
    4   5

as "[1,2,3,null,null,4,5]"
```

**Solution:**

serialize:

1. 开一个arraylist，把root丢进去，BFS用一层循环把每层节点都丢进去，判断条件是i<queue.size(), queue的size每层是变化的。依次把所有节点的左右节点丢进去，遇到null跳出，直到不再丢进去，同时也运行到最后一个节点。
2. 使用while循环把最后一层尾部的null全部去掉
3. 把TreeNode的queue(arraylist) 中的节点的val code成一串字符串

Deserialize：

把String按照","split成String array，建立 一个Arraylist去存所有TreeNode，index为当前进行到哪个node，用isLeftNode去判断左右子节点。

```java
// Encodes a tree to a single string.
public String serialize(TreeNode root) {
    if (root == null) return "[]";

    List<TreeNode> queue = new ArrayList<TreeNode>();
    queue.add(root);

    for (int i = 0; i < queue.size(); i++) {
        TreeNode node = queue.get(i);
        if (node == null) continue;
        queue.add(node.left);
        queue.add(node.right);
    }

    while (queue.get(queue.size() - 1) == null) {
        queue.remove(queue.size() - 1);
    }

    StringBuilder sb = new StringBuilder();
    sb.append("[");
    sb.append(queue.get(0).val);
    for (int i = 1; i < queue.size(); i++) {
        if (queue.get(i) == null) {
            sb.append(",null");
        } else {
            sb.append(",");
            sb.append(queue.get(i).val);
        }
    }
    sb.append("]");

    return sb.toString();
}

// Decodes your encoded data to tree.
public TreeNode deserialize(String data) {
    if (data.equals("[]")) return null;

    String[] vals = data.substring(1, data.length()-1).split(",");

    List<TreeNode> queue = new ArrayList<TreeNode>();
    TreeNode root = new TreeNode(Integer.parseInt(vals[0]));
    queue.add(root);

    int index = 0;
    boolean isLeftNode = true;
    for (int i = 1; i < vals.length; i++) {
        if (!vals[i].equals("null")) {
            TreeNode node = new TreeNode(Integer.parseInt(vals[i]));
            if (isLeftNode) {
                queue.get(index).left = node;
            } else {
                queue.get(index).right = node;
            }
            queue.add(node);
        }

        if (!isLeftNode) {
            index++;
        }

        isLeftNode = !isLeftNode;
    }
    return root;
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

### 300. Longest Increasing Subsequence

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

