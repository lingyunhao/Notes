## LeetCode Problems

#### 283. Move Zeroes

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



