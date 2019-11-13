###Depth First Search

BFS是一层一层遍历，每一层得到所有的新节点，用队列存当前层的节点，处理本层节点的同时加入下一层新节点。而DFS是得到新节点立马对新节点进行遍历，直到没有新节点了再返回上一层，然后继续对该层进行遍历，如此重复，直到所有节点遍历完毕。

从一个节点出发，使用DFS进行遍历时，能够到达的节点都是初始节点可达的，DFS常用来求这种可达性问题。

DFS在inplement时，要注意以下问题：

- stack : 用stack来保存当前节点信息，当遍历新节点结束时返回能够继续遍历当前节点，有non-recursion版本喝recursion版本。
- 标记：和bfs一样需要对已经遍历过的节点进行标记。

找出所有方案的题，一般是DFS，DFS经常是排列、组合的题。一般DFS可以用recursion实现，（如果面试官不要求用non-recursion的办法写DFS的话）

DFS广泛用于树和图中，或者可以转化为树和图的问题。

**Recursion三要素**

- 递归的定义（递归函数求的是什么，完成了什么功能，类似dp[i]表示什么）
- 递归的拆解 （这次递归和之前的递归有什么关系，在本次递归调用递归传参，return等等，类似dp fucntion）
- 递归的出口 （什么时候可以return）

**Combination(组合搜索）**

问题模型：求出所有满足条件的组合

判断条件：组合中的元素是顺序无关的（有关的是排列）

时间复杂度：与O($2^n$)有关

**Permutation(排列搜索）**

问题模型：求出所有满足条件的排列

判断条件：组合中的元素是顺序相关的

时间复杂度： 与n!相关

**BackTracking**

BackTracking 属于DFS ：

- 普通 DFS 主要用在 **可达性问题** ，这种问题只需要执行到特点的位置然后返回即可。
- 而 Backtracking 主要用于求解 **排列组合** 问题，例如有 { 'a','b','c' } 三个字符，求解所有由这三个字符排列得到的字符串，这种问题在执行到特定的位置返回之后还会继续执行求解过程。

因为 Backtracking 不是立即就返回，而要继续求解，因此在程序实现时，需要注意对元素的标记问题：

- 在访问一个新元素进入新的递归调用时，需要将新元素标记为已经访问，这样才能在继续递归调用时不用重复访问该元素；
- 但是在递归返回时，需要将元素标记为未访问，因为只需要保证在一个递归链中不同时访问一个元素，可以访问已经访问过但是不在当前递归链中的元素。

Backtracking 修改一般有两种情况，一种是**修改最后一位输出**，比如排列组合；一种是**修改访问标记**，比如矩阵里搜字符串。

**普通DFS examples(不需要backtracking):**

DFS可以用来求最大面积，求方案总数等等（同dp）。

**判断有向图是否存在环**

1. dfs 记录每个遍历过的结点的父结点，若一个结点被再次遍历且父结点不同于上次遍历时的父结点，说明存在环。

2. topological sort。 If finally there exist node whose indegree is not 0 means there's a cycle

   若最终存在入度不为0的点，说明此拓扑排序排不出来，说明没有一种拓扑顺序，那么即图中存在环。

**判断无向图是否存在环**

union and find

###Example problems in leetcode

**695. Max Area of Island**

Given a non-empty 2D array `grid` of 0's and 1's, an **island** is a group of `1`'s (representing land) connected 4-directionally (horizontal or vertical.) You may assume all four edges of the grid are surrounded by water.求最大的联通面积。

**Solution:**

dfs recursion函数求的是，与当前元素向四个方向所联通的面积和，每个area在四个方向都加了一次。实际上是先调用dfs，然后判断满不满足条件需要return(函数的出口，当前函数return，把调用当前函数的函数pop出栈), 写的时候，是在dfs函数第一行进行判断。注意每次调用dfs把当前元素置为1，下一个方向上的元素又回头把自己加上，每个元素最终只进行了一次dfs。不需要改回来，改回来的话，相当于一个联通面积中每个元素都进行了一次dfs，求出了一个相等的area返回到主函数中进行打擂台。dfs是一种search，原则上每个元素遍历一次，有些题目必须要改回来，这种叫backtracking, 例如排列组合的题目。

```java
private int[][] direction = {{0,1}, {0,-1}, {1,0}, {-1,0}}; 
public int maxAreaOfIsland(int[][] grid) {
    if (grid == null || grid.length == 0) {
        return 0;
    }
    int m = grid.length;
    int n = grid[0].length;
    int maxArea = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            maxArea = Math.max(maxArea, dfs(grid, i, j, m, n));
        }
    }
    return maxArea;   
}
// 求的是当前位置四个方向的面积（不包括从source来的那个位置，因为那个位置已经被置为0，最终每个1被遍历一次，全部改为0）
private int dfs(int[][] grid, int r, int c, int m, int n) {
    // 递归的出口，矩阵问题一般就是这个
    if (r < 0 || r >= m || c < 0 || c >= n || grid[r][c] == 0) {
        return 0;
    }
    grid[r][c] = 0;
    // 注意 int 不是call by reference 所以要在函数中定义然后返回
    int area = 1;
    for (int[] d : direction) {
        area += dfs(grid, r+d[0], c+d[1], m, n);
    }
    return area;
}
```

**200. Number Of Islands**

**Example 1:**

```
Input:
11110
11010
11000
00000
Output: 1
```

**Example 2:**

```
Input:
11000
11000
00100
00011
Output: 3
```

**Solution:**

本题同max area of island类似，矩阵可以看为一个有向图，在主函数中调用dfs时先要判断是否为0，在dfs function中去搜索四个方向，把当前元素以及四个方向的land改为water，以后不会再遍历到这个元素。所以两层for循环，对每个land，及其联通的land标记为0，整个记为一个island。

```java
private int m,n;
private int[][] directions = {{0,1},{0,-1},{1,0},{-1,0}};
public int numIslands(char[][] grid) {
    if(grid == null || grid.length == 0) {
        return 0;
    }
    m = grid.length;
    n = grid[0].length;
    int numberOfIsland = 0;
    for(int i=0; i<m; i++) {
        for(int j=0; j<n; j++) {
            if(grid[i][j] != '0') {
                dfs(grid, i, j);
                numberOfIsland++;
            }
        }
    }
    return numberOfIsland;
}

private void dfs(char[][] grid, int i, int j) {
    if(i<0 || i>=m || j<0 || j>=n || grid[i][j] == '0') {
        return;
    }
    grid[i][j] = '0';
    for(int[] d : directions) {
        dfs(grid, i+d[0], j+d[1]);
    }
}
```



**547. Friend Circles**

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
- 本质和number of islands 是一样的

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



**130. Surrounded Regions**

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
8    for (int i = 0; i < n; ++i) {
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

###  

**417. Pacific Atlantic Water Flow**

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

###BackTracking

回溯法（Backtracking）属于 DFS。

- 普通 DFS 主要用在 **可达性问题** ，这种问题只需要执行到特点的位置然后返回即可。
- 而 Backtracking 主要用于求解 **排列组合** 问题，例如有 { 'a','b','c' } 三个字符，求解所有由这三个字符排列得到的字符串，这种问题在执行到特定的位置返回之后还会继续执行求解过程。

因为 Backtracking 不是立即就返回，而要继续求解，因此在程序实现时，需要注意对元素的标记问题：

- 在访问一个新元素进入新的递归调用时，需要将新元素标记为已经访问，这样才能在继续递归调用时不用重复访问该元素；
- 但是在递归返回时，需要将元素标记为未访问，因为只需要保证在一个递归链中不同时访问一个元素，可以访问已经访问过但是不在当前递归链中的元素。

Backtracking 修改一般有两种情况，一种是**修改最后一位输出**，比如排列组合；一种是**修改访问标记**，比如矩阵里搜字符串。

**78. Subsets (no duplicate)**

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

**90. Subsets II (duplicate elements)**

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

**40. Combination Sum II**

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

**79. Word Search**

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
            if (backtracking(0, i, j, visited, board, word)) {
                return true;
            }
        }
    }
    return false;
}

// 判断word中对应到curlen的character能否四向找到。
private boolean backtracking(int curLen, int r, int c, boolean[][] visited, final char[][] board, String word) {
    // recursion 出口
    if (curLen == word.length()) {
        return true;
    }
    // 还没找到但是已经invalid
    if (r<0 || r>=m || c<0 || c>=n 
        || board[r][c] != word.charAt(curLen) || visited[r][c]) {
        return false;
    }
    visited[r][c] = true;
   // dfs 遇到对的就一路找下去，如果不对就返回上一步看下一个方向，
    for(int[] d: directions) {
        if(backtracking(curLen+1, r+d[0], c+d[1], visited, board, word)){
            return true;
        }
    }
    visited[r][c] = false;
    return false;
}
```

**257. Binary Tree Paths**

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

**46. Permutations**

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

**47. Permutations II**

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

**77. Combinations**

Given two integers *n* and *k*, return all possible combinations of *k* numbers out of 1 ... *n*.

**Example:**

```
Input: n = 4, k = 2
Output:
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]
```

**Solution：**

背。

```java
public List<List<Integer>> combine(int n, int k) {
    List<List<Integer>> combines = new ArrayList<>();
    List<Integer> combineList = new ArrayList<>();
    backtracking(combines,combineList,1,k,n);
    return combines;
}

private void backtracking(List<List<Integer>> combines, List<Integer> combineList, int start, int k, int n) {
    if(k == 0) {
        combines.add(new ArrayList<>(combineList));
        return;
    }
    for(int i = start; i<=n-k+1; i++) {
        combineList.add(i);
        backtracking(combines, combineList, i+1, k-1, n);
        combineList.remove(combineList.size()-1);
    }
}
```

