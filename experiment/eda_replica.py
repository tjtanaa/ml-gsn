from PIL import Image
with Image.open("/home/tan/tjtanaa/ml-gsn/data/replica_all/train/00/000_depth.tiff") as im:
    print(im.size)