mkdir tmp
horovodrun -np 8 -H localhost:8 python unet_to_gan.py -e $1