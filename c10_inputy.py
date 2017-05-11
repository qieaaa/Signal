import tensorflow as tf
import numpy as np
import os

#%%
train_dir = '/home/hzy/My-TensorFlow/train2/'

#%%
def get_files(file_dir):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''

    fsk = []
    label_fsk = []
    ask = []
    label_ask = []
    qpsk = []
    label_qpsk = []
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0]=='fsk':
            fsk.append(file_dir + file)
            label_fsk.append(2)
        elif name[0]=='ask':
            ask.append(file_dir + file)
            label_ask.append(1)
        else:
            qpsk.append(file_dir + file)
            label_qpsk.append(3)
    print('There are %d fsk\nThere are %d ask\nThere are %d qpsk' 
          %(len(fsk),len(ask),len(qpsk)))
    
    image_list = np.hstack((ask,fsk,qpsk))
    label_list = np.hstack((label_ask,label_fsk,label_qpsk))
    
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    
    
    return image_list, label_list




#%%
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''
    
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])
    
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    
    ######################################
    # data argumentation 
    ######################################
    
    image = tf.image.resize_images(image, [image_W, image_H])
    
    # if you want to test the generated batches of images, you might want to comment the following line.
    image = tf.image.per_image_standardization(image)
    
#    image_batch, label_batch = tf.train.batch([image, label],
#                                                batch_size= batch_size,
#                                                num_threads= 64, 
#                                                capacity = capacity)
#    
    #you can also use shuffle_batch 
    image_batch, label_batch = tf.train.shuffle_batch([image,label],
                                                      batch_size=batch_size,
                                                      num_threads=64,
                                                      capacity=capacity,
                                                      min_after_dequeue=capacity-1)
    
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    
#    return image_batch, label_batch

## ONE-HOT      
    n_classes = 3
    label_batch = tf.one_hot(label_batch, depth= n_classes)
        
        
    return image_batch, tf.reshape(label_batch, [batch_size, n_classes])
 
#%% TEST




#import matplotlib.pyplot as plt
##from skimage import data, io
##from PIL import Image
#
#BATCH_SIZE = 2
#CAPACITY = 256
#IMG_W = 400
#IMG_H = 400
#
##train_dir = '/home/hzy/My-TensorFlow/train2/'
#train_dir = '/home/hzy/My-TensorFlow/imagesc/'
#
#
#
#image_list, label_list = get_files(train_dir)
#image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#
#with tf.Session() as sess:
#    i = 0
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#    
#    try:
#        while not coord.should_stop() and i<1:
#            
#            img, label = sess.run([image_batch, label_batch])
#            
#            # just test one batch
#            for j in np.arange(BATCH_SIZE):
##                print('label: %d' %label[j])
#                plt.imshow(img[j,:,:,:])
#                plt.show()
##                a = img[j,:,:,:]
##                plt.imshow(a)
##                a = img[j,:,:,:]
##                a.show()
#            i+=1
#            
#    except tf.errors.OutOfRangeError:
#        print('done!')
#    finally:
#        coord.request_stop()
#    coord.join(threads)


#%%





    
