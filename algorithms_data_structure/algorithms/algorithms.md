# Algorithms

### Time Complexity

O(1) 极少

O(logn) 几乎都是binary search

O($\sqrt{n}$) 几乎都是分解质因数，少见

O(n) 高频（遍历一个规模为n的数据，对每个数据进行O(1)的操作）

O(nlogn) 一般都可能要排序 （遍历一个规模为n的数据，对每个数据进行O(logn)的操作）

O($n^2$) 枚举、数组、动态规划 （遍历一个规模为n的数据，对每个数据进行O(n)的操作）

O($n^3$) 枚举、数组、动态规划

O($2^n$) 与组合有关的搜索

O(n!) 与排列有关的搜索

### Greedy 

保证每次操作都是局部最优的，并且结果也是最优的。

**455. Assign Cookies**

给一些孩子分配饼干，饼干有size，孩子有满足度，只有size>=满足度孩子才content，给一些饼干和孩子，求最多能content的孩子。每个孩子最多一个饼干。

**Solution:**

贪心策略，从满足度最小的孩子开始分起，从最小size的饼干开始分起，这样就可以留更多的饼干给后边的孩子。

```java
public int findContentChildren(int[] g, int[] s) {
    Arrays.sort(g);
    Arrays.sort(s);
    int gi=0,si=0;
    while(gi < g.length && si < s.length) { 
        if(g[gi] <= s[si]){
            gi++;
        }
        si++;
    }
    return gi;
}
```

**135. Candy**

There are *N* children standing in a line. Each child is assigned a rating value.

You are giving candies to these children subjected to the following requirements:

- Each child must have at least one candy.
- Children with a higher rating get more candies than their neighbors.

What is the minimum candies you must give?

**Solution:**

Greedy, 遍历ratings，只要rating比前一个大，那么久更新candies到前一个的糖果个数+1。因为这道题需要满足两边的关系，所以从左往右扫一遍还要从右往左扫一遍。

从左向右，保证右边的比左边的大的分配更多，从右往左，保证左边的比右边大的分配更多。注意第二遍本来就多的话就不用在分配了。

因为每个人至少一个candy，先给数组全分配1.

```java
public int candy(int[] ratings) {
    int  n = ratings.length;
    int[] candies = new int[n];
    Arrays.fill(candies,1);
    for (int i = 1; i < n; ++i) {
        if (ratings[i] > ratings[i-1]) {
            candies[i] = candies[i-1] + 1;
        }
    }
    int res = candies[n-1];
    for (int i = n-2; i >=0; --i) {
        // 本来就满足关系的话就不用再分配了。
        if (ratings[i] > ratings[i+1] && candies[i] <= candies[i+1]) {
            candies[i] = candies[i+1] + 1;
        }
        res += candies[i];
    }
   return res;
}
```

**406. Queue Reconstruction by Height**

Suppose you have a random list of people standing in a queue. Each person is described by a pair of integers `(h, k)`, where `h` is the height of the person and `k` is the number of people in front of this person who have a height greater than or equal to `h`. Write an algorithm to reconstruct the queue.

**Note:**
The number of people is less than 1,100.

**Example**

```
Input:
[[7,0], [4,4], [7,1], [5,0], [6,1], [5,2]]

Output:
[[5,0], [7,0], [5,2], [6,1], [4,4], [7,1]]
```

**Solution:**

Greedy，总是优先排搞身高高的，因为矮的对于高的来说是 invisible的，所以顺序是对的。

 The smaller persons are "invisible" for the taller ones, and hence one could first arrange the tallest guys as if there was no one else.

- Sort people:
  - In the descending order by height.
  - Among the guys of the same height, in the ascending order by k-values.
- Take guys one by one, and place them in the output array at the indexes equal to their k-values.
- Return output array.

```java
public int[][] reconstructQueue(int[][] people) {
    Arrays.sort(people, new Comparator<int[]>() {
        public int compare(int[] a, int[] b) {
            return a[0] == b[0] ? a[1] - b[1] : b[0] - a[0];
        }
    });
    List<int[]> res = new LinkedList<>();
    for (int[] p : people) {
        res.add(p[1],p);
    }
    int n = people.length;
    return res.toArray(new int[n][2]);
}
```

**435. Non-overlapping Intervals**

Given a collection of intervals, find the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.

**Solution:**

计算能组成的最多的不重叠区间，用总长度减去能组成的最多的不重叠区间的个数。

想要组成最多的不重叠区间，那么结尾很重要，每次贪心的选择结尾小的，就能给后边留更多的空间。

**按照end从小到大排序。每次贪心的选择结尾小的，就能给后边留更多的空间**

```java
public int eraseOverlapIntervals(int[][] intervals) {
    if(intervals.length == 0) return 0;
    Arrays.sort(intervals, new Comparator<int[]>() {
        public int compare(int[] i1, int[] i2) {
            return i1[1] - i2[1];
        }
    });
    // 第一个肯定被选
    int cnt=1;
    int end = intervals[0][1];
    for(int i=1; i<intervals.length; i++) {
        if(intervals[i][0] < end){
            continue;
        }
        end = intervals[i][1];
        cnt++;
    }
    return intervals.length - cnt;
}
```

**452. Minimum Number of Arrows to Burst Balloons**

题目描述：气球在一个水平数轴上摆放，可以重叠，飞镖垂直投向坐标轴，使得路径上的气球都会刺破。求解最小的投飞镖次数使所有气球都被刺破。

同435计算不重叠区间的个数。注意相等算不算重叠的情况。

```java
public int findMinArrowShots(int[][] points) {
    if(points.length == 0) return 0;
    Arrays.sort(points, Comparator.comparingInt(o -> o[1]));
    int cnt = 1;
    int end = points[0][1];
    for(int i=1; i<points.length; i++) {
        if(points[i][0] <= end) continue;
        cnt++;
        end = points[i][1];
    }
    return cnt;
}
```

**665. Non-decreasing Array**

题目描述：判断一个数组能不能只修改一个数就成为非递减数组。

这种两边矛盾的问题，让一边固定下来，另一遍去操作。

在出现 nums[i] < nums[i - 1] 时，需要考虑的是应该修改数组的哪个数，使得本次修改能使 i 之前的数组成为非递减数组，并且不影响后续的操作 。**优先考虑令 nums[i - 1] = nums[i]，因为如果修改 nums[i] = nums[i - 1] 的话，那么 nums[i] 这个数会变大，就有可能比 nums[i + 1] 大**，从而影响了后续操作。还有一个比较特别的情况就是 nums[i] < nums[i - 2]，只修改 nums[i - 1] = nums[i] 不能使数组成为非递减数组，只能修改 nums[i] = nums[i - 1]。

```java
public boolean checkPossibility(int[] nums) {
    int cnt = 0;
    for(int i=1; i<nums.length && cnt<2; i++) {
        if(nums[i] >= nums[i-1]) continue;
        cnt++;
        if(i-2>=0 && nums[i] < nums[i-2]) {
            nums[i] = nums[i-1];
        } else {
            nums[i-1] = nums[i];
        }
    }
    return cnt <= 1;
}
```

### Two Pointers

双指针主要用于遍历数组，两个指针指向不同的元素，从而协同完成任务。有通向双指针、相向双指针。若两个指针指向同一数组、遍历方向相同且不会相交，则也称为**滑动窗口**。

linked list问题经常会用双指针。尤其是快慢指针。

#### 快慢指针

slow和fast都从head开始，根据判断条件的不同，最后slow的位置不同，如果linked list长度为奇数，来年各种功能写法是一样的最后都停在中间的位置上，如果为偶数则不同。（记不住时，举个栗子）

```java
// version 1
// [1,2,3,4] slow 停在 3
// [1,2,3,4,5] slow 停在 3
ListNode slow = head;
ListNode fast = head;
while (fast != null && fast.next != null) {
  slow = slow.next;
  fast = fast.next.next;
}

// version 2
// [1,2,3,4] slow 停在 2
// [1,2,3,4,5] slow 停在 3
ListNode slow = head;
ListNode fast = head;
while (fast.next != null && fast.next.next != null) {
  slow = slow.next;
  fast = fast.next.next;
}
```

**633. Sum of Square Numbers**

Given a non-negative integer `c`, your task is to decide whether there're two integers `a` and `b` such that a2 + b2 = c.

**Solution:**

找到左右界限。注意区分 two pointer 和 binary search， 他们都是有隐形递增，两个端点移动，但bs是通过某种条件排除一般的candidate，需要求mid，tp只是左右端点++- -的。

```java
public boolean judgeSquareSum(int c) {
    int i = 0, j = (int)Math.sqrt(c);
    while(i<=j) {
        int powerSum = i*i+j*j;
        if (powerSum == c) return true;
        else if (powerSum < c) i++;
        else j--;
    }
    return false;
}
```

**345. Reverse Vowels of a String**

Write a function that takes a string as input and reverse only the vowels of a string.

**Solution:**

使用双指针，指向待反转的两个元音字符，一个指针从头向尾遍历，一个指针从尾到头遍历。

```java
private static final HashSet<Character> vowels = new HashSet<>(Arrays.asList('a', 'e', 'i', 'o', 'u','A', 'E', 'I', 'O', 'U'));
public String reverseVowels(String s) {
    int i = 0, j = s.length()-1;
    char[] ret = new char[s.length()];
    while(i<=j) {
        char ci = s.charAt(i);
        char cj = s.charAt(j);
        if(!vowels.contains(ci)){
            ret[i++] = ci;
        } else if(!vowels.contains(cj)) {
            ret[j--] = cj;
        } else {
            ret[i++] = cj;
            ret[j--] = ci;
        }
    }
    return new String(ret);
}
```

**680. Valid Palindrome II**

Given a non-empty string `s`, you may delete **at most** one character. Judge whether you can make it a palindrome.

**Solution:**

遇到不相同的位置，去判断删除i还是j。

```java
public boolean validPalindrome(String s) {
    int i = -1, j = s.length();
    while(i++ < j--) {
        if(s.charAt(i) != s.charAt(j)) {
            return isPalindrome(s, i, j-1) || isPalindrome(s, i+1, j);
        }
    }
    return true;
}
private boolean isPalindrome(String s, int i, int j) {
    while(i < j) {
        if(s.charAt(i++) != s.charAt(j--)) {
            return false;
        }
    }
    return true;
}
```

**88. Merge Sorted Array**

```
Input:
nums1 = [1,2,3,0,0,0], m = 3
nums2 = [2,5,6],       n = 3
Output: [1,2,2,3,5,6]
```

**Solution:**

倒着归并到nums1，最后归并剩下的那个数组。

```java
public void merge(int[] nums1, int m, int[] nums2, int n) {
    int index1 = m-1, index2 = n-1, indexMerge = m+n-1;
    while (index1>=0 || index2>=0) {
        if(index1 < 0) {
            nums1[indexMerge--] = nums2[index2--];
        } else if (index2 < 0) {
            nums1[indexMerge--] = nums1[index1--];
        } else if (nums1[index1] > nums2[index2]) {
            nums1[indexMerge--] = nums1[index1--];
        } else {
            nums1[indexMerge--] = nums2[index2--];
        }
    }
}
```

**142. Linked List Cycle II**

Given a linked list, return the node where the cycle begins. If there is no cycle, return `null`.

To represent a cycle in the given linked list, we use an integer `pos` which represents the position (0-indexed) in the linked list where tail connects to. If `pos` is `-1`, then there is no cycle in the linked list.

**Solution：**

**Example 1:**

```
Input: head = [3,2,0,-4], pos = 1
Output: tail connects to node index 1
Explanation: There is a cycle in the linked list, where tail connects to the second node.
```

![img](https://assets.leetcode.com/uploads/2018/12/07/circularlinkedlist.png)

**Example 2:**

```
Input: head = [1,2], pos = 0
Output: tail connects to node index 0
Explanation: There is a cycle in the linked list, where tail connects to the first node.
```

![img](https://assets.leetcode.com/uploads/2018/12/07/circularlinkedlist_test2.png)

**Solution:**

使用双指针，一个指针每次移动一个节点，一个指针每次移动两个节点，如果存在环，那么这两个指针一定会相遇。相遇后把其中一个指针放到链表头，再同速前进，最后相遇的位置即为环路节点。

注意corner case： example 2，判断是否存在环路等

```java
public ListNode detectCycle(ListNode head) {
    if (head == null || head.next == null) return null;
    ListNode slow = head, fast = head;
    do {
        // 判断是否存在环路
        if (fast == null || fast.next == null) return null;
        slow = slow.next;
        fast = fast.next.next;
    } while (slow != fast);
    slow = head;
    while (slow != fast) {
        slow = slow.next;
        fast = fast.next;
    }
    return slow;
}
```

**524. Longest Word in Dictionary through Deleting**

Given a string and a string dictionary, find the longest string in the dictionary that can be formed by deleting some characters of the given string. If there are more than one possible results, return the longest word with the smallest lexicographical order. If there is no possible result, return the empty string.

**Example 1:**

```
Input:
s = "abpcplea", d = ["ale","apple","monkey","plea"]
Output: 
"apple"
```

**Solution：**

用双指针判断是否是subsequence.

```java
public String findLongestWord(String s, List<String> d) {
    String ret = "";
    for(String str : d) {
        int l1 = ret.length(), l2 = str.length();
        // 字典序 比较字符串 compareTo
        if(l1 > l2 || (l1 == l2 && ret.compareTo(str) < 0)){
            continue;
        }
        if(isValid(s, str)) {
            ret = str;
        }
    }
    return ret;
}

private boolean isValid(String s, String d) {
    int i = 0,j = 0;
    while(i < s.length() && j < d.length()) {
        if(s.charAt(i) == d.charAt(j)) {
            j++;
        }
        i++;
    }
    return j == d.length();
}
```

**340. Longest Substring with At Most K Distinct Characters**

Given a string, find the length of the longest substring T that contains at most *k* distinct characters.

**Example 1:**

```
Input: s = "eceba", k = 2
Output: 3
Explanation: T is "ece" which its length is 3.
```

**Example 2:**

```
Input: s = "aa", k = 1
Output: 2
Explanation: T is "aa" which its length is 2.
```

**Solution:**

sliding window + map, map 存每个字母rightmost position, 用一个sliding window(left,right)去维持一个distince character少于k的substring，并且相对应一个map。

当size超过k，说明新的字母来了。所以要删掉map中字符中rightmost最靠左的character。并且把left移到右边。

注意这里不能简单的left+1，画图可知。

这里的map可以用 unorderedmap也就是hashmap，或者ordered map方便找最小的value，linkedhashmap，或者treemap。

**Sliding window + hash map**

```java
public int lengthOfLongestSubstringKDistinct(String s, int k) {
    int n = s.length();
    if (n * k == 0) return 0;
    // sliding window left and right pointers
    int left = 0, right = 0;
    // hashmap character -> its rightmost position
    Map<Character, Integer> map = new HashMap<>();
    int max = 1;
    while (right < n) {
        map.put(s.charAt(right), right++);
        if (map.size() == k + 1) {
            int del =  Collections.min(map.values());
            map.remove(s.charAt(del));
            left = del + 1;
        }
        max = Math.max(max, right - left);
    }
    return max;
}
```

**Sliding window + linkedhashmap**

```java
class Solution {
  public int lengthOfLongestSubstringKDistinct(String s, int k) {
    int n = s.length();
    if (n*k == 0) return 0;

    // sliding window left and right pointers
    int left = 0;
    int right = 0;
    // hashmap character -> its rightmost position 
    // in the sliding window
    LinkedHashMap<Character, Integer> hashmap = new LinkedHashMap<Character, Integer>(k + 1);

    int max_len = 1;

    while (right < n) {
      Character character = s.charAt(right);
      // if character is already in the hashmap -
      // delete it, so that after insert it becomes
      // the rightmost element in the hashmap
      if (hashmap.containsKey(character))
        hashmap.remove(character);
      hashmap.put(character, right++);

      // slidewindow contains k + 1 characters
      if (hashmap.size() == k + 1) {
        // delete the leftmost character
        Map.Entry<Character, Integer> leftmost = hashmap.entrySet().iterator().next();
        hashmap.remove(leftmost.getKey());
        // move left pointer of the slidewindow
        left = leftmost.getValue() + 1;
      }

      max_len = Math.max(max_len, right - left);
    }
    return max_len;
  }
}
```

### Binary Search

**Time Complexity : O(logn)**  

Binary Search 是通过O(1)的时间，将规模为n的问题变为规模为n/2的问题。例如通过一个if判断去掉一半的不可能的答案。 这类算法的时间复杂度是O(logn)。

T(n) = T(n/2) + O(1) = (T(n/4) + O(1)) + O(1) = T(8/n) + 3*O(1) = … = T(n/n) + logn * O(1) = T(1) + O(logn) = O(logn) (省略了以2为底，底数都可以提到log外面作为系数，所以都一样)

若面试中用了O(n)的解法，仍然需要优化就很有可能是二分。比O(n)更好的解法就是O(logn)。根据时间复杂度倒推算法。

**Binary Search 的三种境界**

1. Given a sorted integer array - nums, and an integer - target, find any/first/last postion of the target
2. 给一个数组满足前半段和后半段在某一点不同，找第一个或者最后一个满足某种条件的位置 OOOOOXXXXXXXX， 这种情况判断结果只有两种，是O或者不是O，最后出了循环再去判断一下left，right指针，return相应值去避免corner case
3. 可能无法找到某个条件使得前半段和后半段不同，但二分的本质是每次去掉无解的一半保留有解的一半

**二分模版：**

left + 1 < right  : 最终结束循环时，left 和 right 相邻且在 [0,nums.length-1]区间内，这种情况下，left，right =mid 和 left = mid +1，right = mid - 1是一样的。求first，last positoin时做相应的改变即可。 比如求last position 时，当 nums[mid] == target时，left 不能 mid+1，否则后面一旦没有nums[mid] == target 就可能错过了这个值，所以left = mid, 因为 mid = left + (right - left) / 2, 是左倾的，可能算出来的mid 仍然等于left，就造成了死循环。用left+1<right就会避免死循环。 

while循环去逼近target，在循环外面判断left，right是否等于target。

```java
// normal 
public int search(int[] nums, int target) {
    if (nums == null || nums.length == 0) return -1;
    int left = 0, right = nums.length - 1; // 左闭右闭
    while (left + 1 < right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) {
            right = mid;
        } else if (nums[mid] > target) {
            right = mid;
        } else {
            left = mid;
        }
    }
    if (nums[left] == target) return left;
    if (nums[right] == target) return right;
    return -1;
}
```

```java
// find the first position of target, if doesn't exist return -1
public int first_postion(int[] nums, int target) {
    if (nums == null || nums.length == 0) return -1;
    int left = 0, right = nums.length - 1; 
    while (left + 1 < right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) {
            right = mid;
        } else if (nums[mid] > target) {
            right = mid;
        } else {
            left = mid;
        }
    }
    if (nums[left] == target) return left;
    if (nums[right] == target) return right;
    return -1;
}
```

```java
// find the last position of target, if doesn't exist return -1
public int last_position(int[] nums, int target) {
    if (nums == null || nums.length == 0) return -1;
    int left = 0, right = nums.length - 1;
    while (left + 1 < right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) {
            left = mid;
        } else if (nums[mid] > target) {
            right = mid;
        } else {
            left = mid;
        }
    }
    if (nums[right] == target) return right;
    if (nums[left] == target) return left;
    return -1;
}
```



**Examples:**

**704. Binary Search**

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



**34. Find First and Last Position of Element in Sorted Array**

Given an array of integers `nums` sorted in ascending order, find the starting and ending position of a given `target` value.

Your algorithm's runtime complexity must be in the order of *O*(log *n*).

If the target is not found in the array, return `[-1, -1]`.

**Example 1:**

```
Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]
```

**Example 2:**

```
Input: nums = [5,7,7,8,8,10], target = 6
Output: [-1,-1]
```

**Solution:**

分别找first，last position，找不到返回-1。

```java
public int[] searchRange(int[] nums, int target) {
    if (nums == null || nums.length == 0) return new int[]{-1, -1};
    int first = first_position(nums, target);
    int last = last_position(nums, target);
    return new int[]{first, last};
}

private int first_position(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    while (left + 1 < right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] >= target) {
            right = mid;
        } else {
            left = mid;
        }
    }
    if (nums[left]  == target) return left;
    if (nums[right] == target) return right;
    return -1;
}

private int last_position(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    while (left + 1 < right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] > target) {
            right = mid;
        } else {
            left = mid;
        }
    }
    if (nums[right] == target) return right;
    if (nums[left]  == target) return left;
    return -1;
}
```

**278. First Bad Version**

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



**153. Find Minimum in Rotated Sorted Array**

Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.Find the minimum element. You may assume no duplicate exists in the array.

**Example :**

```
Input: [3,4,5,1,2] 
Output: 1
```

**Solution:**

Binary Search 之境界二，OOOOXXXXXX。

Find the first element which is less than or equal to the last number. 不会出现nums[mid] = nums[nums.length -1] 的情况，因为left,right最大的可能是length-2，length-1。 正常情况下，应该返回的是right,但是会出现 nums = [1,2,3,4], left 和right会停在1,2(index 0,1) 的位置，此时应该返回left, 而[5,4,3,2,1]的话，left,right会停在length-2,length-1的地方，此时返回right是正确的。

```java
public int findMin(int[] nums) {
    if (nums == null || nums.length == 0) return -1;
    int left = 0, right = nums.length-1;
    while (left + 1 < right) {
        int mid = left + (right - left) / 2;
        if(nums[mid] <= nums[nums.length - 1]) {
            right = mid;
        } else {
            left = mid;
        }
    }
    return Math.min(nums[left],nums[right]);
    // or
    // if (left == 0 && nums[left] < nums[right]) return nums[left];
    // return nums[right];
}
```



**852. Peak Index in a Mountain Array**

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



**74. Search in a 2D Array**

Write an efficient algorithm that searches for a value in an *m* x *n*matrix. This matrix has the following properties:

- Integers in each row are sorted from left to right.
- The first integer of each row is greater than the last integer of the previous row.

**Example:**

```
Input:
matrix = [
  [1,   3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 50]
]
target = 3
Output: true
```

**Solution:**

整个数组按行展开是sorted,在有序数组中search->binary search，left = 0， wight = 元素个数 -1， 将mid进行division和mod操作后转化成二维数组的坐标。本题没有重复也不求first，last，所以就使用了 left <= right 为判断条件（注意等于号不可丢保证每个元素都进行过判断，并且出循环后，left = right + 1），直接在while循环中遇到等于就返回。（其他非简单情况用left + 1 < right) 的模版。

```java
public boolean searchMatrix(int[][] matrix, int target) {
    if (matrix ==  null || matrix.length == 0) return false;
    int m = matrix.length, n = matrix[0].length;
    int left = 0, right = m * n - 1;
    int mid,r,c;
    while (left <= right) {
        mid = left + (right - left) / 2;
        r = mid / n;
        c = mid % n;
        if (matrix[r][c] == target) {
            return true;
        } else if (matrix[r][c] > target) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return false;
}
```



**35. Search Insert Position**

**Example 1:**

```
Input: [1,3,5,6], 5
Output: 2
```

**Example 2:**

```
Input: [1,3,5,6], 2
Output: 1
```

**Solution:**

本题是简单binary search(no duplicate)，用简单模版(left <= right, left = mid + 1, right  = mid - 1, while中间return), 最后效果是，如果找到直接return mid，找不到例如 example 2，right = 0， left = 1, 找到的两个位置正好是target的前一个和后一个，并且right = left - 1。

```java
public int searchInsert(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    while (left <= right) {
        int mid = left + (right - left)/2;
        if (nums[mid] == target) {
            return mid;
        } else if (nums[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return left;
}
```



**Smallest Rectangle Enclosing Black Pixels**

**33. Search In Rotated Sorted Array**

Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

(i.e., `[0,1,2,4,5,6,7]` might become `[4,5,6,7,0,1,2]`).

You are given a target value to search. If found in the array return its index, otherwise return `-1`.

You may assume no duplicate exists in the array.

Your algorithm's runtime complexity must be in the order of *O*(log *n*).

**Example 1:**

```
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
```

**Solution:**

境界三，画图区分，主要是边界条件的判断。双重判断，还是丢掉一半，保留一半。

```java
public int search(int[] nums, int target) {
    if(nums == null || nums.length == 0) return -1;
    int start = 0;
    int end = nums.length-1;

    while(start+1 < end) {
        int mid = start + (end - start)/2;
        if(nums[mid] >= nums[start]) {
            if(target >= nums[start] && target <= nums[mid]) {
                end = mid;
            } else {
                start = mid;
            }
        } else {
            if(target >= nums[mid] && target <= nums[end]) {
                start = mid;
            } else {
                end = mid;
            }
        }
    }
    if(nums[start] == target) return start;
    if(nums[end] == target) return end;
    return -1;
}
```



**Find Peak Element**



### Breadth First Search

BFS in Binary Tree

BFS in Graph -> Topological sorting

BFS in board

**使用BFS的cases**

Traverse in graph(Tree is one kind of graph)

* Level order traversal(层级遍历)
* Connected component
* Topological sorting

Shortest path in simple graph(仅限每条边长度都为1，且没有方向)

最短路径：BFS, dp

所有方案：DFS或BFS, dp

BFS: Queue (stack 也可以，但顺序是反的，没人用的)

DFS: Stack

non-recursion的DFS不好写，recursion会造成stack overflow。

**模板：**

BFS写法几乎都一样(参考102)：

1. 创建一个队列，把起始节点都放到里面去
2. while队列不空，处理队列中的节点并扩展出新的节点

如果不需要分层，则只需要一个循环

**Binary Tree Serialization:**

将内存中结构化的数据变成String的过程。

Seriazation: Object -> String

Deserialization: String -> Object 

**BFS in Graph VS BFS in Tree:**

图中存在环，意味着有可能有节点要重复进入队列，解决办法是用hashset或者hashmap记录是否在队列中。

**图的表示方法：**

1. Map<Integer, Set<Integer>>
2. Node class中加neighbours

**拓扑排序：**

Given an directed graph, a topological order of the graph nodes is defined as follow:

- For each directed edge `A -> B` in graph, A must before B in the order list.
- The first node in the order can be any node in the graph with no nodes direct to it.

Find any topological order for the given graph.

You can assume that there is at least one topological order in the graph.

**Example：**

For graph as follow:

![图片](https://media-cdn.jiuzhang.com/markdown/images/8/6/91cf07d2-b7ea-11e9-bb77-0242ac110002.jpg)

The topological order can be:

```
[0, 1, 2, 3, 4, 5]
[0, 2, 3, 1, 5, 4]
...
```



**Examples:**

**102. Binary Tree Level Order Traversal**

**Example:**

Given binary tree `[3,9,20,null,null,15,7]`

```
    3
   / \
  9  20
    /  \
   15   7
```

return its level order traversal as:

```
[
  [3],
  [9,20],
  [15,7]
]
```

**Solution:**

```java
public List<List<Integer>> levelOrder(TreeNode root) {
    List<List<Integer>> results = new ArrayList<>();
    if(root == null) return results;

    Queue<TreeNode> queue = new LinkedList<>();
    queue.offer(root);

    while(!queue.isEmpty()) {
        int size = queue.size();
        List<Integer> level = new ArrayList<>();
        for(int i=0; i<size; i++) {
            TreeNode node = queue.poll();
            level.add(node.val);
            if(node.left != null) queue.offer(node.left);
            if(node.right != null) queue.offer(node.right);
        }
        results.add(level);
    }
    return results;
}
```



**297. Serialize and Deserialize Binary Tree**

(queue写法不太好，暂时不看)

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

1. 开一个queue，把root丢进去，BFS用一层循环把每层节点都丢进去，判断条件是i<queue.size(), queue的size每层是变化的。依次把所有节点的左右节点丢进去，遇到null跳出，直到不再丢进去，同时也运行到最后一个节点。

2. 使用while循环把最后一层尾部的null全部去掉

3. 把TreeNode的queue(arraylist) 中的节点的val code成一串字符串

Deserialize：

把String按照","split成String array，建立 一个Arraylist去存所有TreeNode，index为当前进行到哪个node，用isLeftNode去判断左右子节点。

**本题是用List->arrylist去写的一个queue，arraylist其实更像一个stack，之所以可以用是本题巧妙在其实第一个for loop，每次i<queue.size(), 相当于一直在往queue里加东西，并没有remove，所以还是先进来的点，现add他的子节点。其实本质就是个arraylist，命名不好**

queue是个abstract class，应当用实现类。Queue<> queue = new LinkedList<>();

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



**261. Graph Valid Tree**

Given `n` nodes labeled from `0` to `n - 1` and a list of `undirected` edges (each edge is a pair of nodes), write a function to check whether these edges make up a valid tree.

You can assume that no duplicate edges will appear in edges. Since all edges are `undirected`, `[0, 1]` is the same as `[1, 0]` and thus will not appear together in edges.

**Example 1:**

```
Input: n = 5 edges = [[0, 1], [0, 2], [0, 3], [1, 4]]
Output: true.
```

**Example 2:**

```
Input: n = 5 edges = [[0, 1], [1, 2], [2, 3], [1, 3], [1, 4]]
Output: false.
```

**Solution：**

Graph is a tree if and only if 1. There are n-1 edges. 2. n nodes are connected

用基本数据结构表示图的方法：Map<Integer, Set<Integer>>, key为node index，value为与该node相连的node的index构成的set。

```java
    public boolean validTree(int n, int[][] edges) {
        if (n == 0) {
            return false;
        }
        
        if (edges.length != n - 1) {
            return false;
        }
        
        Map<Integer, Set<Integer>> graph = initializeGraph(n, edges);
        
        // bfs
        Queue<Integer> queue = new LinkedList<>();
        Set<Integer> hash = new HashSet<>();
        
        queue.offer(0);
        hash.add(0);
        while (!queue.isEmpty()) {
            int node = queue.poll();
            for (Integer neighbor : graph.get(node)) {
                if (hash.contains(neighbor)) {
                    continue;
                }
                hash.add(neighbor);
                queue.offer(neighbor);
            }
        }
        
        return (hash.size() == n);
    }
    
    private Map<Integer, Set<Integer>> initializeGraph(int n, int[][] edges) {
        Map<Integer, Set<Integer>> graph = new HashMap<>();
        for (int i = 0; i < n; i++) {
            graph.put(i, new HashSet<Integer>());
        }
        
        for (int i = 0; i < edges.length; i++) {
            int u = edges[i][0];
            int v = edges[i][1];
            graph.get(u).add(v);
            graph.get(v).add(u);
        }
        
        return graph;
    }
```

### Topological Sort

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
```

### Dynamic Programming

动态规划和递归(divide conquer)都是将原问题拆成多个字问题然后进行求解，他们之间最本质的区别是dp保留了子问题的解避免了重复计算。实际上，dp就相当于dfs + memorization。

**注意：为了方便处理初始情况，一个常见的操作是建立一个n+1长度的dp数组，把初始值设置在dp[0]处。**

**注意：DFS+Memoization基本等价DP，需要个数的时候用DP，需要输出的时候用DFS/backtracking**

**Examples:**

**120. Triangle**

Given a triangle, find the minimum path sum from top to bottom. Each step you may move to adjacent numbers on the row below.

**Example:**

```
[
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]
```

The minimum path sum from top to bottom is `11` (i.e., **2** + **3** + **5**+ **1** = 11).

**Solution 1:**

DP bottom up with O($n^2$) time complexity and exrta O($n^2$) space.

自底向上的DP, 开一个NN的二维数组, 存的是从(i,j)出发走到最底层的最小路径，先初始化最后一层为其本身，两层for循环遍历前n-1层，每个节点的值depend on (i+1,j) 和 (i+1,j+1) 两个节点，取其最小值再加上本身即为从(i,j)出发到bottom的最短路径，直到求到(0,0)为止， return(0,0) (从(0,0)出发到bottom 的最小值)。

```java
public int minimumTotal(List<List<Integer>> triangle) {
    int n = triangle.size();

    // Record the minimum path from (i,j) to the bottom
    int dp[][] = new int[n][n];

    // Initialize the bottom
    for(int i = 0; i < n; ++i) {
        dp[n-1][i] = triangle.get(n-1).get(i); 
    }

    // DP function
    for(int i = n - 2; i >= 0; --i) {
        for(int j = 0; j <= i; ++j) {
            dp[i][j] = Math.min(dp[i+1][j], dp[i+1][j+1]) + triangle.get(i).get(j);
        }
    }

    return dp[0][0];
}
```

初始化也可以放在两层赋值for循环中，加个if语句即可。

```java
public int minimumTotal(List<List<Integer>> triangle) {
    int n = triangle.size();

    // Record the minimum path from (i,j) to the bottom
    int dp[][] = new int[n][n];

    // DP function
    for(int i = n - 1; i >= 0; --i) {
        for(int j = 0; j <= i; ++j) {
            // Initialize
            if(i == n - 1) {
                dp[i][j] = triangle.get(i).get(j);
                continue;
            }
            dp[i][j] = Math.min(dp[i+1][j], dp[i+1][j+1]) + triangle.get(i).get(j);
        }
    }

    return dp[0][0];
}
```

**Solution 2:**

DP bottom up with O($n^2$) time complexity and exrta O($n^2$) space.

与Solution 1不同的是，dp(i,j) represents the minimum path from (0,0) to (i,j). 赋值时需要取两个前继节点的最小值取其本身，三角形左边右边只有一个前继节点，需要对三角形左边(i,0)右边(i,i)初始化为其唯一的前继节点+其本身。先初始化顶点，然后左边后边，初始化可以在外边也可以在两层for循环内加if语句完成。最后在最底层打擂台得到状态矩阵最底层的最小值。

```java
public int minimumTotal(List<List<Integer>> triangle) {
    int n = triangle.size();

    // Record the minimum path from (i,j) to the bottom
    int dp[][] = new int[n][n];

    // Initialize
    dp[0][0] = triangle.get(0).get(0);
    for(int i = 1; i < n; ++i) {
        dp[i][0] = dp[i-1][0] + triangle.get(i).get(0);
        dp[i][i] = dp[i-1][i-1] + triangle.get(i).get(i);
    }

    // DP function
    for(int i = 1; i < n; ++i) {
        for(int j = 1; j < i; ++j) {
            dp[i][j] = Math.min(dp[i-1][j-1], dp[i-1][j]) + triangle.get(i).get(j);
        }
    }

    int result = Integer.MAX_VALUE;
    for(int i = 0; i < n; ++i) {
        result = Math.min(dp[n-1][i], result);
    }

    return result;
}
```

**Solution 3:**

将上面的算法优化到O(n)extra space. 实际上可以只开一个2 * n的矩阵，只保留计算当前这一层需要的前一层和此层，和n * n算法上并没有什么需别，而开1 * n的矩阵就需要考虑到谁先更新的问题。

**Solution 4：**

DP with no extra space with bottom up.

不开新的矩阵，直接在输入上进行操作，bottom up，不需要初始化最底层

**64. Minimum Path Sum**

**Example:**

```
Input:
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
Output: 7
Explanation: Because the path 1→3→1→1→1 minimizes the sum.
```

**Solution 1:**

O($m*n$) extra space. 本题求最短路径。

state: dp[i] [j] 表示从起点到当前位置的最短路径

function: dp[i] [j] = min(dp[i-1] [j], dp[i] [j-1]) + grid[i] [j]

initialize: 第0行，第0列为前一个数加上其本身，实际上就是从0到给该位置的和

answer: dp[m-1] [n-1]

能用dp做的题一定不存在循环依赖，即怎么走都走不出一个环，例如本题只能向下或者向右走，如果四个方向都能走的话，用BFS。规定了只能向右向下走的话，BFS只能解决耗费相同，而dp可以解决耗费不同。

初始化一个二维数组的话先初始化它的第零行第零列。本题的初始化包含在for循环中用if判断了，也可以写在外面先初始化然后再两层for循环。

```java
public int minPathSum(int[][] grid) {
    int m = grid.length;
    int n = grid[0].length;
    int[][] dp = new int [m][n];

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if(i == 0 && j == 0) {
                dp[i][j] = grid[0][0];
            } else if (i == 0) {
                dp[i][j] = dp[i][j-1] + grid[i][j];
            } else if (j == 0) {
                dp[i][j] = dp[i-1][j] + grid[i][j];
            } else {
                dp[i][j] = Math.min(dp[i-1][j], dp[i][j-1]) + grid[i][j];
            }
        }
    }
    return dp[m-1][n-1];
}
```

**Solution 2:**

O(n) extra space. dp[2] [n] 滚动数组，设置pre, cur 两个index变量，处理cur，把pre当作上一层，下次循环时把cur赋给pre。

```java
public int minPathSum(int[][] grid) {
    int m = grid.length;
    int n = grid[0].length;
    int[][] dp = new int [2][n];
    int pre = 0, cur = 0;
    for (int i = 0; i < m; i++) {
        pre = cur;
        cur = 1 - cur;
        for (int j = 0; j < n; j++) {
            if(i == 0 && j == 0) {
                dp[cur][j] = grid[0][0];
            } else if (i == 0) {
                dp[cur][j] = dp[cur][j-1] + grid[i][j];
            } else if (j == 0) {
                dp[cur][j] = dp[pre][j] + grid[i][j];
            } else {
                dp[cur][j] = Math.min(dp[pre][j], dp[cur][j-1]) + grid[i][j];
            }
        }
    }
    return dp[cur][n-1];
}
```

**Solution 3:**

O(n) extra space. dp[1] [n] 因为本题向下向右所以当前状态依赖左边和上一层，那么可以不断更新当前层即可。 

```java
public int minPathSum(int[][] grid) {
    int m = grid.length;
    int n = grid[0].length;
    int[][] dp = new int [1][n];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if(i == 0 && j == 0) {
                dp[0][j] = grid[0][0];
            } else if (i == 0) {
                dp[0][j] = dp[0][j-1] + grid[i][j];
            } else if (j == 0) {
                dp[0][j] = dp[0][j] + grid[i][j];
            } else {
                dp[0][j] = Math.min(dp[0][j], dp[0][j-1]) + grid[i][j];
            }
        }
    }
    return dp[0][n-1];
}
```



**62. Unique Paths**

A robot is located at the top-left corner of a *m* x *n* grid (marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

How many possible unique paths are there?

![img](https://assets.leetcode.com/uploads/2018/10/22/robot_maze.png)
Above is a 7 x 3 grid. How many possible unique paths are there?

**Note:** *m* and *n* will be at most 100.

**Solution:**

求方案总数。初始化第一行第一列全为1，dp[i] [j] = dp[i] [j-1] + dp[i-1] [j], return dp[m-1] [n-1]

求方案总数的问题要把上一步可能在的位置相加，求最短路径是取上一步可能在的位置的最小值。

```java
public int uniquePaths(int m, int n) {
    if (m == 0 || n == 0) return 1;
    int[][] dp = new int[1][n];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (i == 0 || j == 0) {
                dp[0][j] = 1;
                continue;
            }
            dp[0][j] = dp[0][j-1] + dp[0][j];
        }
    }
    return dp[0][n-1];
}
```



**70. Climbing Stairs**

You are climbing a stair case. It takes *n* steps to reach to the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

**Note:** Given *n* will be a positive integer.

**Example:**

```
Input: 3
Output: 3
Explanation: There are three ways to climb to the top.
1. 1 step + 1 step + 1 step
2. 1 step + 2 steps
3. 2 steps + 1 step
```

**Solution:**

dp[i] presents the number of ways to climb to the ith floor. dp[i] = dp[i-1] + dp[i-2]. dp[i] is only related to dp[i-1] and dp[i-2] which is finite, we could use two variables to store and update dp[i-1] and dp[i-2]. 需要每次先用一个cur存两个变量的和, 再更新两个变量。

```java
public int climbStairs(int n) {
    if(n<=2) return n;
    int pre1 = 1, pre2 = 2;
    for(int i = 3; i <= n; ++i) {
        int cur = pre1 + pre2;
        pre1 = pre2;
        pre2 = cur;
    }
    return pre2;
}
```



**300. Longest Increasing Subsequence**

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

**413. Arithmetic Slices**

A sequence of number is called arithmetic if it consists of at least three elements and if the difference between any two consecutive elements is the same.

**Example:**

```
A = [1, 2, 3, 4]

return: 3, for 3 arithmetic slices in A: [1, 2, 3], [2, 3, 4] and [1, 2, 3, 4] itself.
```

**Solution:**

求方案总数，dp[i] 表示的是以A[i]结尾的number of arithmetic slices，如果dp[i] - dp[i-1] = dp[i-1] - dp[i-2], 那么以A[i-1]结尾的所有等差递增序列在A[i]也成立，不过是长度+1，除此之外，会多一个子序列dp[i-2],dp[i-1],dp[i]。而如果上面等式不成立的话，将没有以A[i]结尾的arithmetic slices，所以dp[i]为0。在初始化dp时已经定义为0。所以dp[i] = dp[i-1] + 1。

Note: 本题求的是所有arithmatic slices，也就是说以任何一个A[i]结尾的arithmetic slices的个数之和，所以最后要遍历dp求和,因此本题也不能压缩dp数组到几个变量。

```java
public int numberOfArithmeticSlices(int[] A) {
    if (A == null || A.length <= 2) return 0;
    int n = A.length;
    int[] dp = new int[n];

    for(int i = 2; i < n; i++) {
        if (A[i] - A[i-1] == A[i-1] - A[i-2]) {
            dp[i] = dp[i-1] + 1;
        }
    }
    int total = 0;
    for(int each : dp) {
        total += each;
    }
    return total;
}
```

**221. Maximal Square**

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

为了维持正方形的关系，取三向的min

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

**343. Integer Break**

Given a positive integer *n*, break it into the sum of **at least** two positive integers and maximize the product of those integers. Return the maximum product you can get.

**Solution：**

```java
public int integerBreak(int n) {
    int[] dp = new int[n+1];
    dp[1] = 1;
    for (int i = 2; i <= n; ++i) {
        for (int j = 1; j <= i - 1; ++j) {
            dp[i] = Math.max(dp[i], Math.max(j * dp[i-j], j * (i - j)));
        }
    }
    return dp[n];
}
```

**1143. Longest Common Subsequence**

Given two strings `text1` and `text2`, return the length of their longest common subsequence.

A *subsequence* of a string is a new string generated from the original string with some characters(can be none) deleted without changing the relative order of the remaining characters. (eg, "ace" is a subsequence of "abcde" while "aec" is not). A *common subsequence* of two strings is a subsequence that is common to both strings.

If there is no common subsequence, return 0.

**Solution:**

定义一个二维数组 dp 用来存储最长公共子序列的长度，其中 dp[i][j] 表示 S1 的前 i 个字符与 S2 的前 j 个字符最长公共子序列的长度。考虑 S1i 与 S2j 值是否相等，分为两种情况：

- 当 S1i==S2j 时，那么就能在 S1 的前 i-1 个字符与 S2 的前 j-1 个字符最长公共子序列的基础上再加上 S1i 这个值，最长公共子序列长度加 1 ，即 dp[i][j] = dp[i-1][j-1] + 1。
- 当 S1i != S2j 时，此时最长公共子序列为 S1 的前 i-1 个字符和 S2 的前 j 个字符最长公共子序列，与 S1 的前 i 个字符和 S2 的前 j-1 个字符最长公共子序列，它们的最大者，即 dp[i][j] = max{ dp[i-1][j], dp[i][j-1] }。

对于长度为 N 的序列 S1 和 长度为 M 的序列 S2，dp[N][M] 就是序列 S1 和序列 S2 的最长公共子序列长度。

与最长递增子序列相比，最长公共子序列有以下不同点：

- 针对的是两个序列，求它们的最长公共子序列。
- 在最长递增子序列中，dp[i] 表示以 Si 为结尾的最长递增子序列长度，子序列必须包含 Si ；在最长公共子序列中，dp[i][j] 表示 S1 中前 i 个字符与 S2 中前 j 个字符的最长公共子序列长度，不一定包含 S1i 和 S2j 。
- 在求最终解时，最长公共子序列中 dp[N][M] 就是最终解，而最长递增子序列中 dp[N] 不是最终解，因为以 SN 为结尾的最长递增子序列不一定是整个序列最长递增子序列，需要遍历一遍 dp 数组找到最大者。

```java
public int longestCommonSubsequence(String text1, String text2) {
    int m = text1.length(), n = text2.length();
    int[][] dp = new int[m+1][n+1];
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            dp[i][j] = text1.charAt(i-1) == text2.charAt(j-1)? 1 + dp[i-1][j-1]: Math.max(dp[i-1][j], dp[i][j-1]);
        }
    }
    return dp[m][n];
}
```

**139. Word Break**

Given a **non-empty** string *s* and a dictionary *wordDict* containing a list of **non-empty** words, determine if *s* can be segmented into a space-separated sequence of one or more dictionary words.

**Solution: dp**

```java
public boolean wordBreak(String s, List<String> wordDict) {
    int n = s.length();
    boolean[] dp = new boolean[n+1];
    dp[0] = true;
    for (int i = 1; i <= n; ++i) {
        for (String word : wordDict) {
            int len = word.length();
            if (i < len) continue;
            if (s.substring(i - len, i).equals(word)) dp[i] = dp[i] || dp[i-len];
        }
    }
    return dp[n];
}
```

**583. Delete Operation for Two Strings**

Given two words *word1* and *word2*, find the minimum number of steps required to make *word1* and *word2* the same, where in each step you can delete one character in either string.

**Example 1:**

```
Input: "sea", "eat"
Output: 2
Explanation: You need one step to make "sea" to "ea" and another step to make "eat" to "ea".
```

**Solution:**

等价于最长公共子序列问题。

```java
public int minDistance(String word1, String word2) {
    int m = word1.length(), n = word2.length();
    int[][] dp = new int[m+1][n+1];
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            dp[i][j] = word1.charAt(i-1) == word2.charAt(j-1) ? 1 + dp[i-1][j-1] : Math.max(dp[i-1][j], dp[i][j-1]);
        }
    }
    return m + n - 2 * dp[m][n];
}
```

**72. Edit Distance**

Given two words *word1* and *word2*, find the minimum number of operations required to convert *word1* to *word2*.

You have the following 3 operations permitted on a word:

1. Insert a character
2. Delete a character
3. Replace a character

**Example 1:**

```
Input: word1 = "horse", word2 = "ros"
Output: 3
Explanation: 
horse -> rorse (replace 'h' with 'r')
rorse -> rose (remove 'r')
rose -> ros (remove 'e')
```

**Solution:**

```java
public int minDistance(String word1, String word2) {
    int m = word1.length(), n = word2.length();
    int[][] distance = new int[m+1][n+1];
    for (int i = 0; i <= m; ++i) {
        for (int j = 0; j <=n; ++j) {
            if (i == 0) distance[i][j] = j;
            else if (j == 0) distance[i][j] = i;
            else distance[i][j] = Math.min(distance[i-1][j-1] + ((word1.charAt(i-1) == word2.charAt(j-1)) ? 0 : 1), Math.min(distance[i-1][j] + 1, distance[i][j-1] + 1));
        }
    }
    return distance[m][n];
}
```

**DP之背包问题**

有N个物品和容量为W的背包，要用这个背包装下物品的价值最大，这些物品有两个属性：体积 w 和价值 v。

定义一个二维数组 dp 存储最大价值，其中 dp[i][j] 表示前 i 件物品体积不超过 j 的情况下能达到的最大价值。设第 i 件物品体积为 w，价值为 v，根据第 i 件物品是否添加到背包中，可以分两种情况讨论：

- 第 i 件物品没添加到背包，总体积不超过 j 的前 i 件物品的最大价值就是总体积不超过 j 的前 i-1 件物品的最大价值，dp[i][j] = dp[i-1][j]。
- 第 i 件物品添加到背包中，dp[i][j] = dp[i-1][j-w] + v。

第 i 件物品可添加也可以不添加，取决于哪种情况下最大价值更大。

综上，0-1 背包的状态转移方程为：dp[i][j] = max(dp[i - 1][j], dp[i-1][j-w] + v)。

```
int knapsack(vector<int> weights, vector<int> values, int N, int W) {
    vector<vector<int>> dp (N + 1, vector<int>(W + 1, 0));
    for (int i = 1; i <= N; ++i) {
        int w = weights[i-1], v = values[i-1];
        for (int j = 1; j <= W; ++j) {
            if (j >= w) {
                dp[i][j] = max(dp[i-1][j], dp[i-1][j-w] + v);
            } else {
                dp[i][j] = dp[i-1][j];
            }
        }
    }
    return dp[N][W];
}
```

**空间优化**

在程序实现时可以对 0-1 背包做优化。观察状态转移方程可以知道，前 i 件物品的状态仅由前 i-1 件物品的状态有关，因此可以将 dp 定义为一维数组，其中 dp[j] 既可以表示 dp[i-1][j] 也可以表示 dp[i][j]。此时，dp[i] = max(dp[j], dp[j-w] + v)。

因为 dp[j-w] 表示 dp[i-1][j-w]，因此不能先求 dp[i][j-w]，以防止将 dp[i-1][j-w] 覆盖。也就是说要先计算 dp[i][j] 再计算 dp[i][j-w]，在程序实现时需要按倒序来循环求解。

```c++
int knapsack(vector<int> weights, vector<int> values, int N, int W) {
    vector<int> dp (W + 1, 0);
    for (int i = 1; i <= N; ++i) {
        int w = weights[i-1], v = values[i-1];
        for (int j = W; j >= w; --j) dp[j] = max(dp[j], dp[j-w] + v);
    }
    return dp[W];
}
```

**变种**

- 完全背包：物品数量为无限个
- 多重背包：物品数量有限制
- 多维费用背包：物品不仅有重量，还有体积，同时考虑这两种限制
- 其它：物品之间相互约束或者依赖

**完全背包**

```java
int knapsack(vector<int> weights, vector<int> values, int N, int W) {
    vector<vector<int>> dp (N + 1, vector<int>(W + 1, 0));
    for (int i = 1; i <= N; ++i) {
        int w = weights[i-1], v = values[i-1];
        for (int j = 1; j <= W; ++j) {
            if (j >= w) {
                dp[i][j] = max(dp[i-1][j], dp[i][j-w] + v);  // i - 1 changed to i
            } else {
                dp[i][j] = dp[i-1][j];
            }
        }
    }
    return dp[N][W];
}
```

**完全背包 vs 0-1背包的空间优化**

对于压缩内存的写法，**0-1背包对物品的迭代放在外层，里层的重量或价值从后往前遍历；完全背包对物品的迭代放在里层，外层则正常从前往后遍历重量或价值**。（若完全背包的依赖方向在矩阵上是左和上，而这个依赖关系在调转行列后仍然成立，那么在这种情况下里层外层可以互换；为了保险，完全背包都把物品放在里层即可）



### Trie

Trie又称字典树或者前缀树，用来判断字符串是否有某种前缀，或者字符串是否存在。背吧就。

**208. Implement Trie (Prefix Tree)**

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

### 一些常用的辅助function

#### Reverse Linked List

注意调用此函数会把传入的head.next 置为null，如果原来head有prev结点的话，就会使原来的linked list在head这里断掉，head变成了tail。然后返回的节点是原来的tail结点。

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

#### 快慢指针

slow和fast都从head开始，根据判断条件的不同，最后slow的位置不同，如果linked list长度为奇数，来年各种功能写法是一样的最后都停在中间的位置上，如果为偶数则不同。（记不住时，举个栗子）

```java
// version 1
// [1,2,3,4] slow 停在 3
// [1,2,3,4,5] slow 停在 3
ListNode slow = head;
ListNode fast = head;
while (fast != null && fast.next != null) {
  slow = slow.next;
  fast = fast.next.next;
}

// version 2
// [1,2,3,4] slow 停在 2
// [1,2,3,4,5] slow 停在 3
ListNode slow = head;
ListNode fast = head;
while (fast.next != null && fast.next.next != null) {
  slow = slow.next;
  fast = fast.next.next;
}
```

###Dijkstra 无负边单源最短路

求start到所有点（或给定点）的最短距离，时间复杂度是O((N+E)logE)。原理和写法十分类似Prim's Algorithm，不同点是，不用visited去重，而是再次比较距离，保留更短的那个。

```c++
void dijkstra(int start, int N, vector<vector<int>>& connections) {
    vector<vector<pair<int, int>>> graph(N, vector<pair<int, int>>());
    // pair<最短距离，节点编号>
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq; 
    for (const auto& conn : connections) {
        graph[conn[0]].push_back(make_pair(conn[2], conn[1]));
        // graph[conn[1]].push_back(make_pair(conn[2], conn[0]));  // 如果是无向图，加上反向边
    }
    vector<int> d(N, INT_MAX);  // 最短距离
    d[start] = 0;
    pq.push(make_pair(0, start));
    while (!pq.empty()) {
        auto [old_cost, from] = pq.top(); pq.pop();
        if (d[from] < old_cost) continue;
        for (const auto & v: graph[from]) {
            auto [cost, to] = v;
            if (d[to] > d[from] + cost) {
                d[to] = d[from] + cost;
                pq.push(make_pair(d[to], to));
            }
        }
    }
}
```

### 数学

1. i式筛法  求1-n之间的所有素数 i从2开始依次删掉2的<n所有倍数（右界限 sqrt(N))
2. 两个数最大公因数,最小公倍数： 

```java
//最大公因数
int gcd(int a, int b) {
    return b == 0 ? a : gcd(b, a% b);
}

// 最小公倍数
int lcm(int a, int b) {
    return a * b / gcd(a, b);
}
```

3. 在O(n)的时间内找到中位数： quick selection

### bit operation

x ^ 0s = x       x & 0s = 0       x | 0s = x

x ^ 1s = ~x     x & 1s = x       x | 1s = 1s

x ^ x = 0        x & x = x          x | x = x

### 排序

1. merge sort O(nlgn)
2. quick sort O(nlgn) / O($n^2$) - > quick selection -> 第k大的元素 average O(n)
3. 桶排序 O(n) 知道最大的和最小的->hashmap去group.

非online的order需要排序

### Divide Conquer

master theory 

给表达式加括号 241

### 待学习知识点

排序(quick sort, quick selection, bucket sort,merge sort)

bfs+耗费不同， dijkastra