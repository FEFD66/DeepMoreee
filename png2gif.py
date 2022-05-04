import matplotlib.pyplot as plt
import imageio, os

name='osr123567'
images = []
filenames = sorted((fn for fn in os.listdir(r'E:\more\pic\\'+name) if fn.endswith('.png')))
for filename in filenames:
    images.append(imageio.imread(r'E:\more\pic\\'+name+'\\' + filename))
imageio.mimwrite(r'E:\more\pic\\'+name+'\\gif.gif', images, duration=0.05, loop=1)
