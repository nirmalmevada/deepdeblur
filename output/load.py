import numpy as np
import matplotlib
import matplotlib.pyplot as plt

x = np.load('log_array.npy')
print(x)
split = np.hsplit(x,10)
x = split[0]
x = x.ravel()
l1 = split[1]
l1 = l1.ravel()
l2 = split[2]
l2 = l2.ravel()
genloss = split[3]
genloss = genloss.ravel()
dis_f = split[4]
dis_f = dis_f.ravel()
dis = split[5]
dis = dis.ravel()
psnr_exp = split[6]
psnr_exp = psnr_exp.ravel()
psnr_res = split[7]
psnr_res = psnr_res.ravel()
ssim_exp = split[8]
ssim_exp = ssim_exp.ravel()
ssim_res = split[9]
ssim_res = ssim_res.ravel()

plt.title('L1')
plt.plot(l1)
plt.show()
plt.title('L2')
plt.plot(l2)
plt.show()
plt.title('Gen loss total')
plt.plot(genloss)
plt.show()
plt.title('Discriminator fake output loss')
plt.plot(dis_f)
plt.show()
plt.title('Discriminator total loss')
plt.plot(dis)
plt.show()

plt.title('psnr_exp')
plt.plot(psnr_exp)
plt.show()
plt.title('psnr_res')
plt.plot(psnr_res)
plt.show()
plt.title('ssim_exp')
plt.plot(ssim_exp)
plt.show()
plt.title('ssim_res')
plt.plot(ssim_res)
plt.show()

psnr_diff = abs(psnr_exp - psnr_res)
ssim_diff = abs(ssim_exp - ssim_res)
plt.title('ssim_diff')
plt.plot(ssim_diff, label = 'ssim')
plt.show()

plt.title('psnr_diff')
plt.plot(psnr_diff, label = 'psnr')
plt.show()

