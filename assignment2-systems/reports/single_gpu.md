# Problem (gradient_checkpointing): Memory-Optimal Gradient Checkpointing 

## (a)
把 N 个 block 一分为二，每一半都 checkpoint；进到每一半里面，再一分为二，每半再 checkpoint……一直递归到单个 block 为止。深度 log_2(N) 层。
```
def run(blocks, x):
    if len(blocks) == 1:
        return blocks[0](x)
    mid = len(blocks) // 2
    left  = lambda x: run(blocks[:mid], x)
    right = lambda x: run(blocks[mid:], x)
    x = checkpoint(left,  x, use_reentrant=False)
    x = checkpoint(right, x, use_reentrant=False)
    return x

y = run(all_blocks, x)

```
空间复杂度是O(log_2(N))，时间复杂度是O(Nlog_2(N))

## (b)
理论上是N/k + k，k是多少层弄成一个block，均值不等式的话，就是k=\sqrt(N)
下面是实际跑的数据结果：
segment_size    num_segments    peak_GiB        snapshot        status
0       -       77.97   /mnt/cephfs/user_crzaxchen/336/assignment2-systems/reports/mem/mem_xl_full_ctx512_fp32_baseline.pickle  OK
4       8       52.96   /mnt/cephfs/user_crzaxchen/336/assignment2-systems/reports/mem/mem_xl_full_ctx512_fp32_seg4.pickle      OK
8       4       54.75   /mnt/cephfs/user_crzaxchen/336/assignment2-systems/reports/mem/mem_xl_full_ctx512_fp32_seg8.pickle      OK
16      2       58.32   /mnt/cephfs/user_crzaxchen/336/assignment2-systems/reports/mem/mem_xl_full_ctx512_fp32_seg16.pickle     OK