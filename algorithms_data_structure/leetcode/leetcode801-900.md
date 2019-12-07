## LeetCode Problems 801-900

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

### 843. Guess the Word

This problem is an **interactive problem** new to the LeetCode platform.

We are given a word list of unique words, each word is 6 letters long, and one word in this list is chosen as **secret**.

You may call `master.guess(word)` to guess a word.  The guessed word should have type `string` and must be from the original list with 6 lowercase letters.

This function returns an `integer` type, representing the number of exact matches (value and position) of your guess to the **secret word**.  Also, if your guess is not in the given wordlist, it will return `-1` instead.

For each test case, you have 10 guesses to guess the word. At the end of any number of calls, if you have made 10 or less calls to `master.guess` and at least one of these guesses was the **secret**, you pass the testcase.

Besides the example test case below, there will be 5 additional test cases, each with 100 words in the word list.  The letters of each word in those testcases were chosen independently at random from `'a'` to `'z'`, such that every word in the given word lists is unique.

```
Example 1:
Input: secret = "acckzz", wordlist = ["acckzz","ccbazz","eiowzz","abcczz"]

Explanation:

master.guess("aaaaaa") returns -1, because "aaaaaa" is not in wordlist.
master.guess("acckzz") returns 6, because "acckzz" is secret and has all 6 matches.
master.guess("ccbazz") returns 3, because "ccbazz" has 3 matches.
master.guess("eiowzz") returns 2, because "eiowzz" has 2 matches.
master.guess("abcczz") returns 4, because "abcczz" has 4 matches.

We made 5 calls to master.guess and one of them was the secret, so we pass the test case.
```

**Solution:**

```java
int[][] H;
public void findSecretWord(String[] wordlist, Master master) {
    int N = wordlist.length;
    H = new int[N][N];
    for (int i = 0; i < N; ++i)
        for (int j = i; j < N; ++j) {
            int match = 0;
            for (int k = 0; k < 6; ++k)
                if (wordlist[i].charAt(k) == wordlist[j].charAt(k))
                    match++;
            H[i][j] = H[j][i] = match;
        }

    List<Integer> possible = new ArrayList();
    List<Integer> path = new ArrayList();
    for (int i = 0; i < N; ++i) possible.add(i);

    while (!possible.isEmpty()) {
        int guess = solve(possible, path);
        int matches = master.guess(wordlist[guess]);
        if (matches == wordlist[0].length()) return;
        List<Integer> possible2 = new ArrayList();
        for (Integer j: possible) if (H[guess][j] == matches) possible2.add(j);
        possible = possible2;
        path.add(guess);
    }

}

public int solve(List<Integer> possible, List<Integer> path) {
    if (possible.size() <= 2) return possible.get(0);
    List<Integer> ansgrp = possible;
    int ansguess = -1;

    for (int guess = 0; guess < H.length; ++guess) {
        if (!path.contains(guess)) {
            ArrayList<Integer>[] groups = new ArrayList[7];
            for (int i = 0; i < 7; ++i) groups[i] = new ArrayList<Integer>();
            for (Integer j: possible) if (j != guess) {
                groups[H[guess][j]].add(j);
            }

            ArrayList<Integer> maxgroup = groups[0];
            for (int i = 0; i < 7; ++i)
                if (groups[i].size() > maxgroup.size())
                    maxgroup = groups[i];

            if (maxgroup.size() < ansgrp.size()) {
                ansgrp = maxgroup;
                ansguess = guess;
            }
        }
    }

    return ansguess;
}
```

```java
public int countDiff (String a, String b) {
    int count = 0;
    for (int i = 0; i < 6; i ++) {
        if (a.charAt(i) == b.charAt(i)) count ++;
    }
    return count;
}
public void findSecretWord(String[] wordlist, Master master) {       
    LinkedList<Integer> left = new LinkedList<>();
    for (int i = 0; i < wordlist.length; i ++) left.offer(i);
    while(true) {
        if (left.isEmpty()) break;
        Collections.shuffle(left);
        int current = left.poll();
        int diff = master.guess(wordlist[current]);
        int size = left.size();
        for (int j = 0; j < size; j ++) {
            int next = left.poll();
            if (countDiff(wordlist[current], wordlist[next]) == diff) left.offer(next);
        }
    }        
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

### 852. Peak Index in a Mountain Array

Binary Search 之境界二，find the last element which is bigger the previous one. 考虑两个边界条件,[0210],最后停在[2,1]，返回left正确。[3,4,5,1] 最后停在[5,1] 返回left正确。

```java
public int peakIndexInMountainArray(int[] A) {
    int left = 1, right = A.length - 1, mid;
    while(left + 1 < right) {
        mid = left + (right - left) / 2;
        if (A[mid] > A[mid-1]) {
            left = mid;
        } else {
            right = mid;
        }
    }
    return left;
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

### 