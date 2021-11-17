from PIL import Image
import numpy as np
with Image.open("/home/tan/tjtanaa/ml-gsn/data/carla_sequence_3_64x64/train/00/00233178_rgb.png") as im:
    print(im.size)

    img_np = np.array(im)
    print("min: ", np.min(img_np), " max: ", np.max(img_np))
    print(img_np)