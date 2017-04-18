#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from feature_extraction import *
import glob

img = read_and_scale_image('test_images/test0.jpg', (0., 1.))

img_cvt = convert_color(img, 'HLS')
# print(check_scale(img_cvt))
plt.imshow(img_cvt)
plt.savefig('output_images/test.png')
print(np.max(img_cvt[:,:,0]), np.max(img_cvt[:,:,1]), np.max(img_cvt[:,:,2]))

