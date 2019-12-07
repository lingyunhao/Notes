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

### 796. Rotate String

```java
public boolean rotateString(String A, String B) {
    return A.length() == B.length() && (A+A).contains(B);
}
```

