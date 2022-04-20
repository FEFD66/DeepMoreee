import matplotlib.pyplot as plt
import imageio, os

images = []
filenames = sorted((fn for fn in os.listdir(r'E:\more\pic\arpl') if fn.endswith('.png')))
for filename in filenames:
    images.append(imageio.imread(r'E:\more\pic\arpl\\' + filename))
imageio.mimwrite(r'E:\more\pic\arpl\gif.gif', images, duration=0.05, loop=1)
