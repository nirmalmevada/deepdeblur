mkdir tmp
horovodrun -np 8 -H localhost:8 python unet_to_gan.py -e $1 -l1 $2 -l2 $3 -l3 $4