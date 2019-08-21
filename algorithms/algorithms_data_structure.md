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

### Binary Search

**Time Complexity : O(logn)**  

Binary Search 是通过O(1)的时间，将规模为n的问题变为规模为n/2的问题。例如通过一个if判断去掉一半的不可能的答案。 这类算法的时间复杂度是O(logn)。

T(n) = T(n/2) + O(1) = (T(n/4) + O(1)) + O(1) = T(8/n) + 3*O(1) = … = T(n/n) + logn * O(1) = T(1) + O(logn) = O(logn) (省略了以2为底，底数都可以提到log外面作为系数，所以都一样)

若面试中用了O(n)的解法，仍然需要优化就很有可能是二分。比O(n)更好的解法就是O(logn)。根据时间复杂度倒推算法。

**Binary Search 的三种境界**

1. Given a sorted integer array - noms, and an integer - target, find any/first/last postion of the target
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











