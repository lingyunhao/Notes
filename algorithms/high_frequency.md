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
        next= cur.next;
        cur.next= prev;
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
        prev=prev.next;
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

题目描述有图的关系，是一种搜索题。dfs + graph.

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

求高度是O(lgn)的复杂度，在一个h-1的for循环中，又求高度，所以是O(lgn * lgn)的复杂度， 比O(n)小很多。 可以用搜索这样每个节点遍历一遍，时间复杂度为O(n)。

```java
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

