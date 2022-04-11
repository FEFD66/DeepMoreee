import matplotlib.pyplot as plt
import imageio,os
images = []
filenames=sorted((fn for fn in os.listdir(r'E:\pic\more') if fn.endswith('.png')))
for filename in filenames:
    images.append(imageio.imread(r'E:\pic\more\\'+filename))
imageio.mimsave(r'E:\pic\more\gif.gif', images,duration=0.1)
