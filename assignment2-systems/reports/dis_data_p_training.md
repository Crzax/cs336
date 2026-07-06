# (distributed_communication_single_node): Distributed Communication (Single Node)
size(MB)\GPUs |          2 |          4 |          6
----------------------------------------------------
           1 |      0.029 |      0.028 |      0.027
          10 |      0.056 |      0.077 |      0.118
         100 |      0.401 |      0.498 |      0.493
        1000 |      3.291 |      4.505 |      4.128


# Problem (naive_ddp_benchmarking): Naïve DDP Benchmarking 
step_max=1116.989 ms, comm_max=65.072 ms, comm_ratio=0.058