import os
import imageio

directory = 'figures/'
img = []
for file_name in sorted(os.listdir(directory)):
    if 'image' in file_name:
        path = os.path.join(directory, file_name)
        img.append(imageio.imread(path))
imageio.mimsave('gif/train_result.gif', img, duration=0.8, fps=60)