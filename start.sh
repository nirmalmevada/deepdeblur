mkdir tmp
horovodrun -np 8 -H localhost:8 python horovod_test.py