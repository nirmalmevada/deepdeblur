mkdir tmp
horovodrun -np 1 -H localhost:1 python unet_to_gan.py -e $1