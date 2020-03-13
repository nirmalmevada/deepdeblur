mkdir tmp
horovodrun -np 8 -H localhost:8 python densenet_gan.py -e $1