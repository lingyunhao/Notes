### 2. Add Two Numbers

You are given two **non-empty** linked lists representing two non-negative integers. The digits are stored in **reverse order** and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

**Example:**

```
Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
Explanation: 342 + 465 = 807.
```

**Solution:**

The reverseness of linked list avoid the problem of alignment. 并不需要把数字reverse回来加起来后再reverse回去。倒着加就好了，相当于直接从各位开始加起。记录carry，最后需要处理一下carry。

用一个dummy node，和cur node，最后返回dummy.next。

```java
public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
    int carry = 0;
    ListNode dummy_head = new ListNode(0);
    ListNode cur = dummy_head;
    while (l1 != null || l2 != null) {
        if (l1 != null) {
            carry += l1.val;
            l1 = l1.next;
        }
        if (l2 != null) {
            carry += l2.val;
            l2 = l2.next;
        }
        cur.next = new ListNode(carry % 10);
        carry /= 10;
        cur = cur.next;
    }
    if (carry != 0) {
        cur.next = new ListNode(carry);
        cur = cur.next;
    }
    cur.next = null;
    return dummy_head.next;
}
```

### 3. Longest Substring Without Repeating Characters

Given a string, find the length of the **longest substring** without repeating characters.

**Example 1:**

```
Input: "abcabcbb"
Output: 3 
Explanation: The answer is "abc", with the length of 3. 
```

**Solution:**

Sliding window, 用双指针i,j去指向不重复的字符串，window的两端。

```java
public int lengthOfLongestSubstring(String s) {
    int n = s.length(), ans = 0;
    int[] index = new int[128];
    Arrays.fill(index, -1);
    int i = 0;
    //sliding window [i, j]
    for (int j = 0; j < n; ++j) {
        // 如果上次出现的s.charAt(j)在当前的sliding window中，则表明重复
        if (index[s.charAt(j)] < i) {
            ans = Math.max(ans, j - i + 1);
        } else {
            i = index[s.charAt(j)] + 1;
        }
        index[s.charAt(j)] = j;
    }
    return ans;
}
```

### 5. Longest Palindromic Substring

Given a string **s**, find the longest palindromic substring in **s**. You may assume that the maximum length of **s** is 1000.

**solution:**

注意初始化（没看懂）、还有两个forloop怎么写。

To improve over the brute force solution, we first observe how we can avoid unnecessary re-computation while validating palindromes. Consider the case "ababa". If we already knew that "bab" is a palindrome, it is obvious that "ababa" must be a palindrome since the two left and right end letters are the same.

We define P(i,j)*P*(*i*,*j*) as following:

P(i,j) = \begin{cases} \text{true,} &\quad\text{if the substring } S_i \dots S_j \text{ is a palindrome}\\ \text{false,} &\quad\text{otherwise.} \end{cases}*P*(*i*,*j*)={true,false,if the substring *S**i*…*S**j* is a palindromeotherwise.

Therefore,

P(i, j) = ( P(i+1, j-1) \text{ and } S_i == S_j )*P*(*i*,*j*)=(*P*(*i*+1,*j*−1) and *S**i*==*S**j*)

The base cases are:

P(i, i) = true*P*(*i*,*i*)=**true**

P(i, i+1) = ( S_i == S_{i+1} )*P*(*i*,*i*+1)=(*S**i*==*S**i*+1)

This yields a straight forward DP solution, which we first initialize the one and two letters palindromes, and work our way up finding all three letters palindromes, and so on...

```java
public String longestPalindrome(String s) {
    if (s == null || s.length() == 0) return s;
    int n = s.length();
    char[] chars = s.toCharArray();
    boolean[][] dp = new boolean[n][n];
    int row = 0, col = 0;
    for (int i = n - 1; i >= 0; --i) {
        for (int j = i; j < n; ++j) {
            // 初始化 innilize one and two letters palindrome
            if (j == i || (chars[i] == chars[j] && j-i <= 1)) {
                dp[i][j] = true;
            } else {
                //通项公式
                dp[i][j] = dp[i+1][j-1] && chars[i] == chars[j];
            }
            if (dp[i][j] && (j - i > col - row)) {
                col = j;
                row = i;
            }
        }
    }
    return s.substring(row, col+1);
}
```

### 13. Roman to Integer

Roman numerals are represented by seven different symbols: `I`, `V`, `X`, `L`, `C`, `D` and `M`.

```
Symbol       Value
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
```

For example, two is written as `II` in Roman numeral, just two one's added together. Twelve is written as, `XII`, which is simply `X` + `II`. The number twenty seven is written as `XXVII`, which is `XX` + `V` + `II`.

Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not `IIII`. Instead, the number four is written as `IV`. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as `IX`. There are six instances where subtraction is used:

**Example:**

```
Input: "III" Output: 3
Input: "IX" Output: 9
Input: "LVIII" Output: 58 Explanation: L = 50, V= 5, III = 3.
Input: "MCMXCIV" Output: 1994 Explanation: M = 1000, CM = 900, XC = 90 and IV = 4.
```

**Solution:**

Roman数字的规则：

1. 把Stirng的所有Character代表的数值进行加或者减的操作

2. 从左向右是从大到小的顺序

3. 每一位要看和后一位的关系决定自己是加还是减

4. 不会出现 “IIX”， 意味着，减的情况不会重复超过两位去判断，只需要判断其后一位就好了

   从后往前遍历，如果比前一位小则减，如果大于等于则加。

```java
public int romanToInt(String s) {
    Map<Character, Integer> map = new HashMap<Character, Integer>() {
        {
            put('I', 1); put('V', 5); put('X', 10); put('L', 50); put('C', 100); put('D', 500); put('M', 1000);
        }
    };
    if (s == null || s.length() == 0) return 0;
    int n = s.length();
    int cur = map.get(s.charAt(n-1)), post = cur, res = cur;
    for (int i = n - 2; i >= 0; i--) {
        cur = map.get(s.charAt(i));
        if (cur < post) res -= cur; 
        else res += cur;
        post = cur;
    }
    return res;
}
```

### 14. Longest Common Prefix

Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string `""`.

**Example 1:**

```
Input: ["flower","flow","flight"] Output: "fl"
没有则返回 ""
```

**Solution:**

horizontal scanning: 拿第一个字符串作为标准，在剩下的字符串中进行遍历，遇到不满足条件则停止，每个字符最多被遍历一次。所以时间复杂度 O(s), s为所有character的个数

```java
public String longestCommonPrefix(String[] strs) {
    if (strs == null || strs.length == 0) return "";
    for (int i = 0; i < strs[0].length() ; i++){
        char c = strs[0].charAt(i);
        for (int j = 1; j < strs.length; j++) {
            if (i == strs[j].length() || strs[j].charAt(i) != c)
                return strs[0].substring(0, i);             
        }
    }
    return strs[0];
}
```

### 26. Remove Duplicates from Sorted Array

**Example:**

```
Given nums = [1,1,2],

Your function should return length = 2, with the first two elements of nums being 1 and 2 respectively.

It doesn't matter what you leave beyond the returned length.
```

**Solution:**

双指针，

### 34. Find First and Last Position of Element in Sorted Array

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

### 33. Search In Rotated Sorted Array

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

先判断是在ascding part还是desceding part。

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

### 39. Combination Sum

Given a **set** of candidate numbers (`candidates`) **(without duplicates)** and a target number (`target`), find all unique combinations in `candidates` where the candidate numbers sums to `target`.

The **same** repeated number may be chosen from `candidates` unlimited number of times.

**Note:**

- All numbers (including `target`) will be positive integers.
- The solution set must not contain duplicate combinations.

**Example 1:**

```
Input: candidates = [2,3,6,7], target = 7,
A solution set is:
[
  [7],
  [2,2,3]
]
```

**Solution:**

可以重复使用。不用sort, 不用在dfs中判断 nums[i] == nums[i-1], dfs 时，i in stead of i+1

```java
public List<List<Integer>> combinationSum(int[] candidates, int target) {
    List<List<Integer>> combines = new ArrayList<>();
    backtracking(new ArrayList<>(), combines, 0, candidates, target);
    return combines;
}

private void backtracking(List<Integer> combineList, List<List<Integer>> combines, int start, int[] candidates, int target) {
    if(target == 0) {
        combines.add(new ArrayList<>(combineList));
        return;
    }
    for(int i=start; i<candidates.length; i++) {
        if(candidates[i] <= target) {
            combineList.add(candidates[i]);
            backtracking(combineList, combines, i,candidates, target-candidates[i]);
            combineList.remove(combineList.size()-1);
        }
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
    backtracking(new ArrayList<>(), combines, 0, candidates, target, visited);
    return combines;
}

private void backtracking(List<Integer> combineList, List<List<Integer>> combines, int start, int[] candidates, int target) {

    if(target == 0) {
        combines.add(new ArrayList<>(combineList));
        return;
    }
    for(int i=start; i<candidates.length; i++) {
        if(i!=start && candidates[i] == candidates[i-1]) continue;
        if(candidates[i] <= target) {
   
            combineList.add(candidates[i]);
            backtracking(combineList, combines, i+1, candidates, target-candidates[i]);
            combineList.remove(combineList.size()-1);
           
        }
    }
}
```



### 46. Permutations

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
    backtracking(permutes, 0, nums);
    return permutes;
}

private void backtracking(List<List<Integer>> permutes, int start, int[] nums) {
    if (start == nums.length - 1) {
        permutes.add(new ArrayList<>(nums));
        return;
    }
    for(int i = start; i < nums.length; i++) {
        swap(nums[i], nums[start]);
        backtracking(permutes, start+1, nums);
        swap(nums[start], nums[i]);
    }
}
```



### 47. Permutations II

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
public List<List<Integer>> permute(int[] nums) {
    List<List<Integer>> permutes = new ArrayList<>();
    backtracking(permutes, 0, nums);
    return permutes;
}

private void backtracking(List<List<Integer>> permutes, int start, int[] nums) {
    if (start == nums.length - 1) {
        permutes.add(new ArrayList<>(nums));
        return;
    }
    Arrays.sort(nums, start, nums.length - 1);
    for(int i = start; i < nums.length; i++) {
        if (i != start && nums[i] == nums[i-1]) continue;
        swap(nums[i], nums[start]);
        backtracking(permutes, start+1, nums);
        swap(nums[start], nums[i]);
    }
}
```

### 