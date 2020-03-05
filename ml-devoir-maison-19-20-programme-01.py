#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 10:49:17 2019

@author: rubenthaler_sylvain
"""

# resize image and force a new shape
from PIL import Image
# load the image
image = Image.open('./index.jpeg')
# report the size of the image
print(image.size)
# resize image and ignore original aspect ratio
img_resized = image.resize((100,100))
img_resized_gs=img_resized.convert('L')
# report the size of the thumbnail
print(img_resized_gs.size)
img_resized_gs.save('./thumb-c_c.jpeg')
img_resized_gs.show()
