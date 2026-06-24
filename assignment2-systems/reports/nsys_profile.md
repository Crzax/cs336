# Nsys Profile
## （a）
一样的，符合的

## (b)
Time	Total Time	Instances	Avg	Med	Min	Max	StdDev	Name
30.4%	5.211 ms	4	1.303 ms	1.304 ms	1.271 ms	1.333 ms	31.766 μs	sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize256x64x8_stage3_warpsize2x2x1_ffma_aligna4_alignc4_execute_kernel__5x_cublas

Time	Total Time	Instances	Avg	Med	Min	Max	StdDev	Name
18.6%	50.978 ms	39	1.307 ms	1.322 ms	1.270 ms	1.337 ms	26.496 μs	sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize256x64x8_stage3_warpsize2x2x1_ffma_aligna4_alignc4_execute_kernel__5x_cublas

## (c)
Time	Total Time	Instances	Avg	Med	Min	Max	StdDev	Name
11.0%	2.001 ms	2	1.001 ms	1.001 ms	999.167 μs	1.002 ms	2.035 μs	void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::<unnamed>::where_kernel_impl(at::TensorIterator &)::[lambda() (instance 1)]::operator ()() const::[lambda() (instance 11)]::operator ()() const::[lambda(bool, float, float) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)

## (d)
# forward：matmul 累计时间占比
nsys stats --report cuda_gpu_kern_sum --filter-nvtx=forward --format csv "$F" \
| awk -F, 'NR>1 && $2+0>0 {tot+=$2; if(tolower($0)~/gemm|cutlass/) mm+=$2}
           END{ if(tot>0) printf "forward  matmul=%.1f%% (mm=%d / tot=%d ns)\n",100*mm/tot,mm,tot;
                else print "no rows parsed" }'

# 整个 step：过滤父区间 step_0
nsys stats --report cuda_gpu_kern_sum --filter-nvtx=step_0 --format csv "$F" \
| awk -F, 'NR>1 && $2+0>0 {tot+=$2; if(tolower($0)~/gemm|cutlass/) mm+=$2}
           END{ if(tot>0) printf "fullstep matmul=%.1f%% (mm=%d / tot=%d ns)\n",100*mm/tot,mm,tot;
                else print "no rows parsed" }'
forward  matmul=74.4% (mm=12800587 / tot=17206979 ns)
fullstep matmul=70.2% (mm=66291189 / tot=94376255 ns)

## (e)
基于数据（matmul = QK 55.16 + AV 34.10 = **89.3 ms/fwd**，softmax = **103.0 ms/fwd**）：

在 self-attention 的 forward 中，softmax 约 103 ms/forward，比两个矩阵乘（QK^T + 权重·V）合计的 ~89 ms 还要慢（≈1.2×）；然而矩阵乘的 FLOPs 约为 softmax 的 50 倍（~1.65 TFLOP vs ~0.03 TFLOP/forward）。这说明 runtime 与 FLOPs 严重不成比例——GEMM 是 compute-bound 且高度优化，而 softmax 是 memory-bound（需对整个 $L\times L$ 分数矩阵做 max/exp/sum/除多趟读写），因此耗时占比远超其计算量占比。