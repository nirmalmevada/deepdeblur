mkdir tmp
mpirun -np 8 \
 --allow-run-as-root \
 -bind-to none -map-by slot \
 -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
 -mca pml ob1 -mca btl ^openib \
 python horovod_test.py
