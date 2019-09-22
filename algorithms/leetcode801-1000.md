## LeetCode Problems 801-1000

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

