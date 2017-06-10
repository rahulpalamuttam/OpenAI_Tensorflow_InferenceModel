import pickle
import numpy as np
from numpy import vstack
from numpy import zeros
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf
import cv2

import collections

from random import shuffle
def TileImage(imgs, picturesPerRow=16):
    """ Convert to a true list of 16x16 images
    """
    
    # Calculate how many columns
    picturesPerColumn = imgs.shape[0]/picturesPerRow + 1*((imgs.shape[0]%picturesPerRow)!=0)
    
    # Padding
    rowPadding = picturesPerRow - imgs.shape[0]%picturesPerRow
    imgs = vstack([imgs,zeros([rowPadding,imgs.shape[1]])])
    
    # Reshaping all images
    imgs = imgs.reshape(imgs.shape[0],16,16)

    # Tiling Loop (The conditionals are not necessary anymore)
    tiled = []
    for i in range(0,picturesPerColumn*picturesPerRow,picturesPerRow):
        tiled.append(hstack(imgs[i:i+picturesPerRow,:,:]))
        
        
    return vstack(tiled)
                                                            
def generate_embedding_plot(embedding):
    import matplotlib.pyplot as plt; plt.rcdefaults()
    import numpy as np
    import matplotlib.pyplot as plt
    
    objects = ('Up', 'Left', 'Down', 'Right', 'X - boost')
    y_pos = np.arange(len(objects))
    performance = np.sum(embedding, axis=0)
    
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.xlabel('Move')
    plt.ylabel('Number of Frames')
    plt.title('Human Agent Move Distribution')
    
    plt.show()    

def _process_frame_flash(frame):
    frame = frame[90:600, 15:815, :]
    frame = cv2.resize(frame, (200, 128))
    frame = frame.mean(2).astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [128, 200, 1])
    return frame

data1 = pickle.load(open('test_data.pkl', 'rb'))
#im = plt.imshow(np.concatenate(t, axis=1))
#plt.hist(images[num].ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
#plt.show()

separated = [list(t) for t in zip(*data1)]

print(len(data1[0]))
images= [_process_frame_flash(t) for t in separated[0]]
tur = np.asarray(images)

img = np.concatenate([separated[0][t] for t in range(0, 100)], axis=1)
print(img.shape)
im = plt.imshow(img)

plt.show()
sys.exit()
embedding = np.asarray([np.asarray(t) for t in separated[1]])

embedding_image_dict = collections.defaultdict(list)
for i in range(0, len(images)):
    
    embedding_image_dict[str(separated[1][i])].append(images[i])

final_set = []
for k, v in embedding_image_dict.items():
    print(str(k) + " " + str(len(v)))
    if(k == "[0, 0, 0, 1, 0]"):
        right = np.asarray([0, 0, 0, 1, 0])
        duplicates = [(val, right) for val in v for i in range(0,6)]
        print(str(len(duplicates)))
        final_set.extend(duplicates)
    if(k == "[0, 0, 0, 0, 1]"):
        up = np.asarray([0, 0, 0, 0, 1])
        duplicates = [(val, up) for val in v for i in range(0,700)]
        print(str(len(duplicates)))
        final_set.extend(duplicates)
    if(k == "[0, 1, 0, 0, 0]"):
        up = np.asarray([0, 1, 0, 0, 0])
        duplicates = [(val, up) for val in v for i in range(0,9)]
        print(str(len(duplicates)))
        final_set.extend(duplicates)
    if(k == "[0, 0, 1, 0, 0]"):
        # whoops never pressed down
        up = np.asarray([0, 0, 1, 0, 0])
        duplicates = [(val, up) for val in v for i in range(0,900)]
        print(str(len(duplicates)))
        final_set.extend(duplicates)

shuffle(final_set)


separated = [list(t) for t in zip(*final_set)]
images, labels = separated[0], separated[1]
separated = None
data1 = None

output = open('test_dataset.pkl', 'wb')
pickle.dump(final_set, output)
output.close()
#plt.hist(images[num].ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')



