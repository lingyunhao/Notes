##LeetCode Problems 201-400

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



### 371. Sum of Two Integers

Calculate the sum of two integers *a* and *b*, but you are **not allowed** to use the operator `+` and `-`. 

**Solution:**

对数字做运算，除了四则运算之外只能用位运算。

1. 不考虑进位，对每一位相加，相当于异或。
2. 考虑进位，只有1与1时会产生进位，相当于与。
3. 把上面两个结果想加，直到再也没有进位位置。本题也可以用while loop 判断条件是 b!=0

```java
public int getSum(int a, int b) {
    if (b == 0) return a;

    int sum = a ^ b;
    int carry = (a & b) << 1;

    return getSum(sum, carry);
}
```



### 376. Wiggle Subsequence

**Example 1:**

```
Input: [1,7,4,9,2,5]
Output: 6
Explanation: The entire sequence is a wiggle sequence.
```

**Example 2:**

```
Input: [1,17,5,10,13,15,10,5,16,8]
Output: 7
Explanation: There are several subsequences that achieve this length. One is [1,17,10,13,10,16,8].
```

**Solution:**

利用两个变量去记录上一个down,up 的位置。

```java
public int wiggleMaxLength(int[] nums) {
    if (nums == null || nums.length == 0) {
        return 0;
    }

    int up = 1, down = 1;
    for (int i = 1; i < nums.length; ++i) {
        if (nums[i] > nums[i-1]) {
            up = down + 1;
        } else if (nums[i] < nums[i-1]) {
            down = up + 1;
        }
    }
    return Math.max(down, up);
}
```