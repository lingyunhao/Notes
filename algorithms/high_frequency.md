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

### 399. Evaluate Division

Equations are given in the format `A / B = k`, where `A` and `B` are variables represented as strings, and `k` is a real number (floating point number). Given some queries, return the answers. If the answer does not exist, return `-1.0`.

**Example:**
Given `a / b = 2.0, b / c = 3.0.`
queries are: `a / c = ?, b / a = ?, a / e = ?, a / a = ?, x / x = ? .`
return `[6.0, 0.5, -1.0, 1.0, -1.0 ].`

The input is: `vector<pair<string, string>> equations, vector<double>& values, vector<pair<string, string>> queries `, where `equations.size() == values.size()`, and the values are positive. This represents the equations. Return `vector<double>`.

According to the example above:

```
equations = [ ["a", "b"], ["b", "c"] ],
values = [2.0, 3.0],
queries = [ ["a", "c"], ["b", "a"], ["a", "e"], ["a", "a"], ["x", "x"] ]. 
```

**Solution:**

题目描述有图的关系，是一种搜索题。dfs + graph. 。 

```java
public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
    HashMap<String, HashMap<String, Double>> map = new HashMap<>();
    double[] results = new double[queries.size()];
    for(int i = 0; i < equations.size(); i++)
    {
        map.computeIfAbsent(equations.get(i).get(0), value -> new HashMap<>()).put(equations.get(i).get(1), values[i]);         
        map.computeIfAbsent(equations.get(i).get(1), value -> new HashMap<>()).put(equations.get(i).get(0), 1.0/values[i]);             
    }

    int i = 0;
    for(List<String> query : queries)
    {
        String param1 = query.get(0);
        String param2 = query.get(1);
        if (!(map.containsKey(param1) && map.containsKey(param2)))
            results[i++] = -1.0;
        else
        {
            results[i] = dfs(map, param1, param2, 1.0, new HashSet<>());
            if(results[i] == Double.MAX_VALUE)
                results[i] = -1.0;
            else
            {
                map.get(param1).put(param2, results[i]);
                map.get(param2).put(param1, 1.0/results[i]);
            }
            i++;
        }
    }

    return results;
}

// dfs求的是 src/dest的结果
private double dfs(HashMap<String, HashMap<String, Double>> map, String src, String dest, double currProd, Set<String> visited)
{
    if(src.equals(dest))
        return currProd;
    visited.add(src);
    if(map.get(src).containsKey(dest))
        return currProd * map.get(src).get(dest);
    Map<String, Double> entry = map.get(src);
    double res = Double.MAX_VALUE;
    for(Map.Entry<String, Double> e : entry.entrySet())
    {
        if(!visited.contains(e.getKey()))
            res = Math.min(dfs(map, e.getKey(), dest, currProd * e.getValue(), visited), res);
    }
    visited.remove(src);
    return res;
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

### 658. Find K Closest Elements

Given a sorted array, two integers `k` and `x`, find the `k` closest elements to `x` in the array. The result should also be sorted in ascending order. If there is a tie, the smaller elements are always preferred.

**Example 1:**

```
Input: [1,2,3,4,5], k=4, x=3
Output: [1,2,3,4]
```

**Solution:**

Binary search + two pointers  O(logn + k)

```java
public List<Integer> findClosestElements(int[] arr, int k, int x) {
    List<Integer> res = new ArrayList<>();
    if (arr == null || arr.length == 0) return res;
    int left = findLastEqualOrSmaller(arr, x);
    int right = left + 1;
    for (int i = 0; i < k; i++) {
        // 相同的差要取小的那个 所以后边是小于等于
        if (right >= arr.length || (left >= 0 && (Math.abs(arr[left] - x) <= Math.abs(arr[right] - x)))) {
            left--;
        } else {
            right++;
        }
    }
    for (int i = left + 1; i < right; i++) {
        res.add(arr[i]);
    }
    return res;
}
public int findLastEqualOrSmaller(int[] array, int x) {
    int start = 0;
    int end = array.length - 1;
    int mid;
    while (start + 1 < end) {
        mid = start + (end - start) / 2;
        if (array[mid] == x) {
            start = mid;
        } else if (array[mid] < x) {
            start = mid;
        } else {
            end = mid;
        }
    }
    if (array[end] <= x) {
        return end;
    } else {
        return start;
    }
}
```

**给定一个数组，和一个target，要求返回这个数组中能不能有subset的sum是target，followup问的是返回有多少这样的subset，以及如果不仅仅是和，还可以减，或者乘除怎么办，还有如果是要求返回subset本身怎么办
E.g Given [3,5,6,7,3,2,6,1] t=9
Return [5,3,1], [7,2] etc.**

### 78. Subsets (no duplicate)

```java
public List<List<Integer>> subsets(int[] nums) {
    List<List<Integer>> subsets = new ArrayList<>();
    List<Integer> tem = new ArrayList<>();
    for(int size=0; size<=nums.length; size++) {
        backtracking(subsets, tem, 0, size, nums);
    }
    return subsets;
}

// backtracking: add all of the subsets with size(input) into results
private void backtracking(List<List<Integer>> subsets, List<Integer> tem, int start, int size, int[] nums) {
    if(tem.size() == size) {
        subsets.add(new ArrayList<>(tem));
        return;
    }

    for(int i=start; i<nums.length; i++) {
        tem.add(nums[i]);
        backtracking(subsets, tem, i+1, size, nums);
        tem.remove(tem.size()-1);
    }
}
```

### 90. Subsets II (duplicate elements)

```java
public List<List<Integer>> subsetsWithDup(int[] nums) {
    List<List<Integer>> subsets = new ArrayList<>();
    List<Integer> temp = new ArrayList<>();
    Arrays.sort(nums);
    for (int size = 0; size <= nums.length; size++) {
        backtracking(subsets, temp, 0, size, nums);
    }
    return subsets;
}

private void backtracking(List<List<Integer>> subsets, List<Integer> temp, int start, int size, int[] nums) {
    if (temp.size() == size) {
        subsets.add(new ArrayList<>(temp));
        return;
    }
    for (int i = start; i < nums.length; i++) {
        if (i != start && nums[i] == nums[i-1]) continue;
        temp.add(nums[i]);
        backtracking(subsets, temp, i+1, size, nums);
        temp.remove(temp.size()-1);
    }
}
```

### 40. Combination Sum II

Given a collection of candidate numbers (`candidates`) and a target number (`target`), find all unique combinations in `candidates` where the candidate numbers sums to `target`.

Each number in `candidates` may only be used **once** in the combination.

**Note:**

- All numbers (including `target`) will be positive integers.
- The solution set must not contain duplicate combinations.

**Example 1:**

```
Input: candidates = [10,1,2,7,6,1,5], target = 8,
A solution set is:
[
  [1, 7],
  [1, 2, 5],
  [2, 6],
  [1, 1, 6]
]
```

**Solution:**

```java
public List<List<Integer>> combinationSum2(int[] candidates, int target) {
    List<List<Integer>> combines = new ArrayList<>();
    Arrays.sort(candidates);
    backtracking(new ArrayList<>(), combines, 0, candidates, target);
    return combines;
}
    
private void backtracking(List<Integer> combineList, List<List<Integer>> combines, int start, int[] candidates, int target) {
    if (target == 0) {
        combines.add(new ArrayList<>(combineList));
        return;
    }
    for (int i = start; i<candidates.length; i++) {
        if (i != start && candidates[i] == candidates[i-1]) continue;
        if (candidates[i] <= target) {
            combineList.add(candidates[i]);
            backtracking(combineList, combines, i + 1, candidates, target-candidates[i]);
            combineList.remove(combineList.size() - 1);
        }
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

### 54. Spiral Matrix

Given a matrix of *m* x *n* elements (*m* rows, *n* columns), return all elements of the matrix in spiral order.

**Example 1:**

```
Input:
[
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]
Output: [1,2,3,6,9,8,7,4,5]
```

**Solution:**

```java
public List<Integer> spiralOrder(int[][] matrix) {
    List ans = new ArrayList();
    if (matrix.length == 0) return ans;
    int R = matrix.length, C = matrix[0].length;
    boolean[][] seen = new boolean[R][C];
    int[] dr = {0, 1, 0, -1};
    int[] dc = {1, 0, -1, 0};
    int r = 0, c = 0, di = 0;
    // for 循环的长度 = result 的长度，num of elements m * n
    for (int i = 0; i < R * C; i++) {
        ans.add(matrix[r][c]);
        seen[r][c] = true;
        int cr = r + dr[di];
        int cc = c + dc[di];
        if (0 <= cr && cr < R && 0 <= cc && cc < C && !seen[cr][cc]){
            r = cr;
            c = cc;
        } else {
            di = (di + 1) % 4;
            r += dr[di];
            c += dc[di];
        }
    }
    return ans;
}
```

### 417. Pacific Atlantic Water Flow

Given an `m x n` matrix of non-negative integers representing the height of each unit cell in a continent, the "Pacific ocean" touches the left and top edges of the matrix and the "Atlantic ocean" touches the right and bottom edges.

Water can only flow in four directions (up, down, left, or right) from a cell to another one with height equal or lower.

Find the list of grid coordinates where water can flow to both the Pacific and Atlantic ocean.

**Solution:**

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

### 329. Longest Increasing Path in a Matrix

Given an integer matrix, find the length of the longest increasing path.

From each cell, you can either move to four directions: left, right, up or down. You may NOT move diagonally or move outside of the boundary (i.e. wrap-around is not allowed).

**Example 1:**

```
Input: nums = 
[
  [9,9,4],
  [6,6,8],
  [2,1,1]
] 
Output: 4 
Explanation: The longest increasing path is [1, 2, 6, 9].
```

**Solution:**

Topological sort

```java
// Topological Sort Based Solution
// An Alternative Solution
public class Solution {
    private static final int[][] dir = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
    private int m, n;
    public int longestIncreasingPath(int[][] grid) {
        int m = grid.length;
        if (m == 0) return 0;
        int n = grid[0].length;
        // padding the matrix with zero as boundaries
        // assuming all positive integer, otherwise use INT_MIN as boundaries
        int[][] matrix = new int[m + 2][n + 2];
        for (int i = 0; i < m; ++i)
            System.arraycopy(grid[i], 0, matrix[i + 1], 1, n);

        // calculate outdegrees
        int[][] outdegree = new int[m + 2][n + 2];
        for (int i = 1; i <= m; ++i)
            for (int j = 1; j <= n; ++j)
                for (int[] d: dir)
                    if (matrix[i][j] < matrix[i + d[0]][j + d[1]])
                        outdegree[i][j]++;

        // find leaves who have zero out degree as the initial level
        n += 2;
        m += 2;
        List<int[]> leaves = new ArrayList<>();
        for (int i = 1; i < m - 1; ++i)
            for (int j = 1; j < n - 1; ++j)
                if (outdegree[i][j] == 0) leaves.add(new int[]{i, j});

        // remove leaves level by level in topological order
        int height = 0;
        while (!leaves.isEmpty()) {
            height++;
            List<int[]> newLeaves = new ArrayList<>();
            for (int[] node : leaves) {
                for (int[] d:dir) {
                    int x = node[0] + d[0], y = node[1] + d[1];
                    if (matrix[node[0]][node[1]] > matrix[x][y])
                        if (--outdegree[x][y] == 0)
                            newLeaves.add(new int[]{x, y});
                }
            }
            leaves = newLeaves;
        }
        return height;
    }
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

### 684. Redundant Connection

In this problem, a tree is an **undirected** graph that is connected and has no cycles.

The given input is a graph that started as a tree with N nodes (with distinct values 1, 2, ..., N), with one additional edge added. The added edge has two different vertices chosen from 1 to N, and was not an edge that already existed.

The resulting graph is given as a 2D-array of `edges`. Each element of `edges` is a pair `[u, v]` with `u < v`, that represents an **undirected** edge connecting nodes `u` and `v`.

Return an edge that can be removed so that the resulting graph is a tree of N nodes. If there are multiple answers, return the answer that occurs last in the given 2D-array. The answer edge `[u, v]` should be in the same format, with `u < v`.

**Example 1:**

```
Input: [[1,2], [1,3], [2,3]]
Output: [2,3]
Explanation: The given undirected graph will be like this:
  1
 / \
2 - 3
```

**Solution:**

```java
Set<Integer> seen = new HashSet();
int MAX_EDGE_VAL = 1000;

public int[] findRedundantConnection(int[][] edges) {
    ArrayList<Integer>[] graph = new ArrayList[MAX_EDGE_VAL + 1];
    for (int i = 0; i <= MAX_EDGE_VAL; i++) {
        graph[i] = new ArrayList();
    }

    for (int[] edge: edges) {
        seen.clear();
        if (!graph[edge[0]].isEmpty() && !graph[edge[1]].isEmpty() &&
                dfs(graph, edge[0], edge[1])) {
            return edge;
        }
        graph[edge[0]].add(edge[1]);
        graph[edge[1]].add(edge[0]);
    }
    throw new AssertionError();
}
public boolean dfs(ArrayList<Integer>[] graph, int source, int target) {
    if (!seen.contains(source)) {
        seen.add(source);
        if (source == target) return true;
        for (int nei: graph[source]) {
            if (dfs(graph, nei, target)) return true;
        }
    }
    return false;
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

### 146. LRU Cache

Design and implement a data structure for [Least Recently Used (LRU) cache](https://en.wikipedia.org/wiki/Cache_replacement_policies#LRU). It should support the following operations: `get` and `put`.

`get(key)` - Get the value (will always be positive) of the key if the key exists in the cache, otherwise return -1.
`put(key, value)` - Set or insert the value if the key is not already present. When the cache reached its capacity, it should invalidate the least recently used item before inserting a new item.

The cache is initialized with a **positive** capacity.

**Follow up:**
Could you do both operations in **O(1)** time complexity?

**Example:**

```
LRUCache cache = new LRUCache( 2 /* capacity */ );

cache.put(1, 1);
cache.put(2, 2);
cache.get(1);       // returns 1
cache.put(3, 3);    // evicts key 2
cache.get(2);       // returns -1 (not found)
cache.put(4, 4);    // evicts key 1
cache.get(1);       // returns -1 (not found)
cache.get(3);       // returns 3
cache.get(4);       // returns 4
```

#### Approach 1: Ordered dictionary

**Intuition**

We're asked to implement [the structure](https://en.wikipedia.org/wiki/Cache_replacement_policies#LRU) which provides the following operations in \mathcal{O}(1)O(1) time :

- Get the key / Check if the key exists
- Put the key
- Delete the first added key

The first two operations in \mathcal{O}(1)O(1) time are provided by the standard hashmap, and the last one - by linked list.

> There is a structure called *ordered dictionary*, it combines behind both hashmap and linked list. In Python this structure is called [*OrderedDict*](https://docs.python.org/3/library/collections.html#collections.OrderedDict) and in Java [*LinkedHashMap*](https://docs.oracle.com/javase/8/docs/api/java/util/LinkedHashMap.html).

```java
class LRUCache extends LinkedHashMap<Integer, Integer>{
    private int capacity;
    
    public LRUCache(int capacity) {
        super(capacity, 0.75F, true);
        this.capacity = capacity;
    }

    public int get(int key) {
        return super.getOrDefault(key, -1);
    }

    public void put(int key, int value) {
        super.put(key, value);
    }

    @Override
    protected boolean removeEldestEntry(Map.Entry<Integer, Integer> eldest) {
        return size() > capacity; 
    }
}
```

#### Approach 2: Hashmap + DoubleLinkedList

**Intuition**

This Java solution is an extended version of the [the article published on the Discuss forum](https://leetcode.com/problems/lru-cache/discuss/45911/Java-Hashtable-%2B-Double-linked-list-(with-a-touch-of-pseudo-nodes)).

The problem can be solved with a hashmap that keeps track of the keys and its values in the double linked list. That results in \mathcal{O}(1)O(1) time for `put` and `get` operations and allows to remove the first added node in \mathcal{O}(1)O(1) time as well.

![compute](https://leetcode.com/problems/lru-cache/Figures/146/structure.png)

One advantage of *double* linked list is that the node can remove itself without other reference. In addition, it takes constant time to add and remove nodes from the head or tail.

One particularity about the double linked list implemented here is that there are *pseudo head* and *pseudo tail* to mark the boundary, so that we don't need to check the `null` node during the update.

![compute](https://leetcode.com/problems/lru-cache/Figures/146/new_node.png)

**Implementation**

```java
public class LRUCache {

  class DLinkedNode {
    int key;
    int value;
    DLinkedNode prev;
    DLinkedNode next;
  }

  private void addNode(DLinkedNode node) {
    /**
     * Always add the new node right after head.
     */
    node.prev = head;
    node.next = head.next;

    head.next.prev = node;
    head.next = node;
  }

  private void removeNode(DLinkedNode node){
    /**
     * Remove an existing node from the linked list.
     */
    DLinkedNode prev = node.prev;
    DLinkedNode next = node.next;

    prev.next = next;
    next.prev = prev;
  }

  private void moveToHead(DLinkedNode node){
    /**
     * Move certain node in between to the head.
     */
    removeNode(node);
    addNode(node);
  }

  private DLinkedNode popTail() {
    /**
     * Pop the current tail.
     */
    DLinkedNode res = tail.prev;
    removeNode(res);
    return res;
  }

  private Map<Integer, DLinkedNode> cache = new HashMap<>();
  private int size;
  private int capacity;
  private DLinkedNode head, tail;

  public LRUCache(int capacity) {
    this.size = 0;
    this.capacity = capacity;

    head = new DLinkedNode();
    // head.prev = null;

    tail = new DLinkedNode();
    // tail.next = null;

    head.next = tail;
    tail.prev = head;
  }

  public int get(int key) {
    DLinkedNode node = cache.get(key);
    if (node == null) return -1;

    // move the accessed node to the head;
    moveToHead(node);

    return node.value;
  }

  public void put(int key, int value) {
    DLinkedNode node = cache.get(key);

    if(node == null) {
      DLinkedNode newNode = new DLinkedNode();
      newNode.key = key;
      newNode.value = value;

      cache.put(key, newNode);
      addNode(newNode);

      ++size;

      if(size > capacity) {
        // pop the tail
        DLinkedNode tail = popTail();
        cache.remove(tail.key);
        --size;
      }
    } else {
      // update the value.
      node.value = value;
      moveToHead(node);
    }
  }
}
```

### 460. LFU Cache

Design and implement a data structure for [Least Frequently Used (LFU)](https://en.wikipedia.org/wiki/Least_frequently_used) cache. It should support the following operations: `get` and `put`.

`get(key)` - Get the value (will always be positive) of the key if the key exists in the cache, otherwise return -1.
`put(key, value)` - Set or insert the value if the key is not already present. When the cache reaches its capacity, it should invalidate the least frequently used item before inserting a new item. For the purpose of this problem, when there is a tie (i.e., two or more keys that have the same frequency), the least **recently** used key would be evicted.

Note that the number of times an item is used is the number of calls to the `get` and `put` functions for that item since it was inserted. This number is set to zero when the item is removed.

**Solution:**

```java
class LFUCache {

    HashMap<Integer, DoubleLinkedList> freqMap = new HashMap<>(); // HashMap<freq, ddl>
    HashMap<Integer, Node> map = new HashMap<>(); // HashMap<Key, Node>
    int minFreq;
    int count, capacity;
    public LFUCache(int capacity) {
        this.capacity = capacity;
        count = 0;
        minFreq = 1;
        freqMap.put(1, new DoubleLinkedList());
    }
    
    public int get(int key) {
        if(!map.containsKey(key))
            return -1;
        //1. Get the node
        Node node = map.get(key);
        //2. remove from original frequency
        DoubleLinkedList curList = freqMap.get(node.freq);
        curList.removeNode(node);
        //3. increase frequency + add to new frequency
        node.freq ++;
        if(!freqMap.containsKey(node.freq))
            freqMap.put(node.freq, new DoubleLinkedList());
        DoubleLinkedList newList = freqMap.get(node.freq);
        newList.addToHead(node);
        
        if(freqMap.get(minFreq).isEmpty())             // Update minFreq !!!
            minFreq ++;
        
        return node.val;
    }
    
    public void put(int key, int value) {
        if(capacity == 0) return;
        if(map.containsKey(key)) {
            //1. Get the node
            Node node = map.get(key);
            //2. Change its value
            node.val = value;
            //3. remove from original frequency
            DoubleLinkedList curList = freqMap.get(node.freq);
            curList.removeNode(node);
            //4. increase frequency + add to new frequency
            node.freq ++;
            if(!freqMap.containsKey(node.freq))
                freqMap.put(node.freq, new DoubleLinkedList());
            DoubleLinkedList newList = freqMap.get(node.freq);
            newList.addToHead(node);
            
            if(freqMap.get(minFreq).isEmpty())        // Update minFreq !!!
                minFreq ++;
            
            return;
        }
        
        
        //1. new node with frequency 1
        Node node = new Node(key, value, 1);
        map.put(key, node);
        //2. if count > capacity, evict the last --> update the last
        count ++;
        if(count > capacity) {
            DoubleLinkedList remove = freqMap.get(minFreq);   // Remove minFreq !!!
            Node last = remove.popTail();
            map.remove(last.key);
            count --;
        }
        //3. add the node to the list
        if(!freqMap.containsKey(node.freq))
            freqMap.put(node.freq, new DoubleLinkedList());
        DoubleLinkedList newList = freqMap.get(node.freq);
        newList.addToHead(node);
        
        minFreq = 1;                                 // Update minFreq !!!
    }
    
    class Node {
        int key;
        int val;
        int freq;
        Node prev;
        Node next;
        public Node(int k, int v, int f) {
            key = k;
            val = v;
            freq = f;
        }
    }
    class DoubleLinkedList {
        Node head, tail;
        public DoubleLinkedList() {
            head = new Node(-1,-1,-1);
            tail = new Node(-1,-1,-1);
            head.next = tail;
            tail.prev = head;
        }
        public void addToHead(Node node) {
            Node headNext = head.next;
            head.next = node;
            node.prev = head;
            node.next = headNext;
            headNext.prev = node;
        }
        public void removeNode(Node node) {
            Node prevNode = node.prev;
            Node nextNode = node.next;
            prevNode.next = nextNode;
            nextNode.prev = prevNode;
        }
        public Node popTail() {
            Node node = tail.prev;
            if(node == head) return null;
            Node prevNode = node.prev;
            prevNode.next = tail;
            tail.prev = prevNode;
            node.next = null;
            node.prev = null;
            return node;
        }
        public boolean isEmpty() {
            return head.next == tail;
        }
    }
    
}
```

### 1096. Brace Expansion II

Under a grammar given below, strings can represent a set of lowercase words.  Let's use `R(expr)` to denote the **set** of words the expression represents.

Grammar can best be understood through simple examples:

- Single letters represent a singleton set containing that word.
  - `R("a") = {"a"}`
  - `R("w") = {"w"}`
- When we take a comma delimited list of 2 or more expressions, we take the union of possibilities.
  - `R("{a,b,c}") = {"a","b","c"}`
  - `R("{{a,b},{b,c}}") = {"a","b","c"}` (notice the final set only contains each word at most once)
- When we concatenate two expressions, we take the set of possible concatenations between two words where the first word comes from the first expression and the second word comes from the second expression.
  - `R("{a,b}{c,d}") = {"ac","ad","bc","bd"}`
  - `R("a{b,c}{d,e}f{g,h}") = {"abdfg", "abdfh", "abefg", "abefh", "acdfg", "acdfh", "acefg", "acefh"}`

Formally, the 3 rules for our grammar:

- For every lowercase letter `x`, we have `R(x) = {x}`
- For expressions `e_1, e_2, ... , e_k` with `k >= 2`, we have `R({e_1,e_2,...}) = R(e_1) ∪ R(e_2) ∪ ...`
- For expressions `e_1` and `e_2`, we have `R(e_1 + e_2) = {a + b for (a, b) in R(e_1) × R(e_2)}`, where + denotes concatenation, and × denotes the cartesian product.

Given an `expression` representing a set of words under the given grammar, return the sorted list of words that the expression represents.

**Example 1:**

```
Input: "{a,b}{c,{d,e}}"
Output: ["ac","ad","ae","bc","bd","be"]
```

**Example 2:**

```
Input: "{{a,z},a{b,c},{ab,z}}"
Output: ["a","ab","ac","z"]
Explanation: Each distinct word is written only once in the final answer.
```

**Solution:**

```java
class Solution {
    public List<String> braceExpansionII(String expression) {
        Set<String> set = solve(expression);
        List<String> result = new ArrayList<>(set);
        Collections.sort(result);
        return result;
    }
    
    private Set<String> solve(String str) {
        int level = 0;
        int start = 0;
        List<Set<String>> groups = new ArrayList<>();
        groups.add(new HashSet<>());
        for (int i = 0; i < str.length(); i++) {
            if (str.charAt(i) == '{') {
                if (level == 0) start = i + 1;
                level++;
            } else if (str.charAt(i) == '}') {
                level--;
                if (level == 0) {
                    Set<String> sub = solve(str.substring(start, i));
                    groups.set(groups.size() - 1, merge(groups.get(groups.size() - 1), sub));
                }
            } else if (str.charAt(i) == ',' && level == 0) {
                groups.add(new HashSet<>());
            } else if (level == 0) {
                Set<String> tmp = new HashSet<>();
                StringBuilder builder = new StringBuilder();
                while (i < str.length() && Character.isLetter(str.charAt(i))) {
                    builder.append(str.charAt(i++));
                }
                i--;
                tmp.add(builder.toString());
                groups.set(groups.size() - 1, merge(groups.get(groups.size() - 1), tmp));
            }
        }
        
        Set<String> result = new HashSet<>();
        for (Set<String> group : groups) {
            result.addAll(group);
        }
        return result;
    }
    
    private Set<String> merge(Set<String> set1, Set<String> set2) {
        Set<String> result = new HashSet<>();
        if (set1.size() == 0) return set2;
        if (set2.size() == 0) return set1;
        for (String str1 : set1) {
            for (String str2 : set2) {
                result.add(str1 + str2);
            }
        }
        return result;
    }
}
```

### 1066. Campus Bikes II


On a campus represented as a 2D grid, there are `N` workers and `M` bikes, with `N <= M`. Each worker and bike is a 2D coordinate on this grid.

We assign one unique bike to each worker so that the sum of the Manhattan distances between each worker and their assigned bike is minimized.

The Manhattan distance between two points `p1` and `p2` is `Manhattan(p1, p2) = |p1.x - p2.x| + |p1.y - p2.y|`.

Return the minimum possible sum of Manhattan distances between each worker and their assigned bike.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2019/03/06/1261_example_1_v2.png)

```
Input: workers = [[0,0],[2,1]], bikes = [[1,2],[3,3]]
Output: 6
Explanation: 
We assign bike 0 to worker 0, bike 1 to worker 1. The Manhattan distance of both assignments is 3, so t
```

**Example 2:**

![img](https://assets.leetcode.com/uploads/2019/03/06/1261_example_2_v2.png)

```
Input: workers = [[0,0],[1,1],[2,0]], bikes = [[1,0],[2,2],[2,1]]
Output: 4
Explanation: 
We first assign bike 0 to worker 0, then assign bike 1 to worker 1 or worker 2, bike 2 to worker 2 or worker 1. Both assignments lead to sum of the Manhattan distances as 4.
```

 **Solution:**

```java
class Solution {
    
    HashMap<String, Integer> dp = new HashMap<String,Integer>();
    
    public int manHattanDistance(int[][] A, int[][] B, int i, int j){
        int dis = Math.abs(A[i][0] - B[j][0]) + Math.abs(A[i][1] - B[j][1]);
        return dis;
    }
    public int helper(int[][] workers, int[][] bikes, int count, boolean[] used, int m, int n){
        if(count >= m)
            return 0;
        
        String curr = Arrays.toString(used);
        
        if(dp.containsKey(curr)){
            return dp.get(curr);
        }
        
        int val = Integer.MAX_VALUE;
        for(int j=0;j<n;j++){
            if(!used[j]){
                used[j] = true;
                val = Math.min(val,helper(workers,bikes,count+1,used,m,n) + manHattanDistance(workers,bikes,count,j));
                used[j] = false;
            }
        }
        
        dp.put(curr,val);
        return val;
    }
    public int assignBikes(int[][] workers, int[][] bikes) {
        int m = workers.length;
        int n = bikes.length;
        boolean[] used = new boolean[n];
        String curr = Arrays.toString(used);
        return helper(workers,bikes,0,used,m,n);
        
    }

  
```