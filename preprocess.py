import numpy as np
from random import shuffle
import os
import scipy.io as io
from sklearn.preprocessing import OneHotEncoder
import argparse
from helper import *
import threading
import time
import itertools
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='Indian_pines')
parser.add_argument('--patch_size', type=int, default=3)
parser.add_argument('--mode', type=str, default='Default')

opt = parser.parse_args()

#Somehow this is necessary, even I cannot tell why -_-
if opt.data in ('KSC', 'Botswana'):
    filename = opt.data
else:
    filename = opt.data.lower()

print("Dataset: " + filename )

#Try loading data from the folder... Otherwise download from online
try:
    print("Using images from Data folder...")
    input_mat = io.loadmat('./data/' + opt.data + '.mat')[filename]
    target_mat = io.loadmat('./data/' + opt.data + '_gt.mat')[filename + '_gt']

except:
    print("Data not found, downloading input images and labelled images!\n\n")
    if opt.data == "Indian_pines":
        opt.url1 = "http://www.ehu.eus/ccwintco/uploads/2/22/Indian_pines.mat"
        opt.url2 = "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat"
    
    elif opt.data == "Salinas":
        opt.url1 = "http://www.ehu.eus/ccwintco/uploads/f/f1/Salinas.mat"
        opt.url2 = "http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat"

    elif opt.data == "KSC":
        opt.url1 = "http://www.ehu.eus/ccwintco/uploads/2/26/KSC.mat"
        opt.url2 = "http://www.ehu.eus/ccwintco/uploads/a/a6/KSC_gt.mat"
    
    elif opt.data == "Botswana":
        opt.url1 = "http://www.ehu.eus/ccwintco/uploads/7/72/Botswana.mat"
        opt.url2 = "http://www.ehu.eus/ccwintco/uploads/5/58/Botswana_gt.mat" 
    
    else:
        raise Exception("Available datasets are:: Indian_pines, Salinas, KSC, Botswana")

    os.system('wget -P' + ' ' + './data/' + ' ' + opt.url1)
    os.system('wget -P' + ' ' + './data/' + ' ' + opt.url2)

    input_mat = io.loadmat('./data/' + opt.data + '.mat')[filename]
    target_mat = io.loadmat('./data/' + opt.data + '_gt.mat')[filename + '_gt']

PATCH_SIZE = opt.patch_size
HEIGHT = input_mat.shape[0]
WIDTH = input_mat.shape[1]
BAND = input_mat.shape[2]
CLASSES = []
COUNT = 200 #Number of patches of each class
OUTPUT_CLASSES = np.max(target_mat)
mode = opt.mode
print("MODE : " + mode )

''' For debug use, uncomment to see image information
print("+-------------------------------------+")
print("MODE : " + mode )
print("Patch size", PATCH_SIZE)
print("Number of output classes: " +str(OUTPUT_CLASSES))
print("Lower number of classes: " +str(np.min(target_mat)) )
print("Input Height: " + str(HEIGHT))
print("Input Width: " + str(WIDTH))
print("Frequency dimension (Band) " +str(BAND))
print('Target_mat shape = (' + str(target_mat.shape[0]) + ',' + str(target_mat.shape[1])+')')
print("+-------------------------------------+\n")
'''

# Normalise image data
input_mat = input_mat.astype(float)
statlie_image = input_mat
input_mat -= np.min(input_mat)
input_mat /= np.max(input_mat)
statlie_image_2 = input_mat

# List label = list of class
# train_idx = list of numbers, each number represent number of samples in each class.
if mode == 'small':
    if opt.data == 'Indian_pines':
        list_labels = [2,3,5,6,8,10,11,12,14] 
        train_idx = [178, 178, 178, 178, 178, 178, 178, 178, 178] #Sum = 1600

    elif opt.data == 'Salinas':
        list_labels = range(1,OUTPUT_CLASSES+1)
        train_idx = [170]*OUTPUT_CLASSES #was 175
    
    else:
        raise Exception("KSC or Botswana does not offer 'small' mode, try removing '--size small' from command line")

else:
    if opt.data == "Indian_pines":
    # There's some classes with lack of samples so we only using the 9 that has sufficient samples
        list_labels = [2,3,5,6,8,10,11,12,14] 
        train_idx = [800, 600, 275, 350, 275, 450, 850, 430, 750] #Average

    elif opt.data == "Salinas":
        list_labels = range(1,OUTPUT_CLASSES+1)
        train_idx = [750]*OUTPUT_CLASSES #was 175
        # train_idx = [1500,2500, 1300,800, 1800, 2800, 2200, 6000, 3000, 2300, 500,1300, 500]

    elif opt.data == 'KSC':
        print('Please be patent..........')
        list_labels = [1,2,3,4,6,8,9,10,11,12,13]
        train_idx = [300,200,200,200,190,350,400,340,340,420,600]

    elif opt.data == 'Botswana':
        print('Please be patent..........')
        list_labels = [1,3,4,5,6,7,8,9,10,11,13]
        train_idx = [200,200,165,190,190,190,150,225,190,220,110,200]
    else:
        print("Impossible!")


def Patch(height_index,width_index):
    """
    Returns a mean-normalized patch, the top left corner of which
    is at (height_index, width_index)

    Inputs:
    height_index - row index of the top left corner of the image patch
    width_index - column index of the top left corner of the image patch

    Outputs:
    mean_normalized_patch - mean normalized patch of size (PATCH_SIZE, PATCH_SIZE)
    whose top left corner is at (height_index, width_index)
    """
    # transpose_array = np.transpose(input_mat,(2,0,1))
    transpose_array = input_mat
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)

    patch = transpose_array[:, height_slice, width_slice]
    mean_normalized_patch = []
    for i in range(patch.shape[0]):
        mean_normalized_patch.append(patch[i] - MEAN_ARRAY[i])
    return np.array(mean_normalized_patch)

# For showing a animation only
end_loading = False
def animate():
    global end_loading
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if end_loading:
            break
        sys.stdout.write('\rExtracting '+ opt.data + ' dataset features...' + c)
        sys.stdout.flush()
        time.sleep(0.1)
        sys.stdout.write('\rFinished!\t')


print("+-------------------------------------+")
print('Input_mat shape: ' +  str(input_mat.shape) )

MEAN_ARRAY = np.ndarray(shape=(BAND,),dtype=float)
new_input_mat = []

input_mat = np.transpose(input_mat,(2,0,1))
statlie_image_3 = input_mat
print('Input mat after transpose shape: ' +  str(input_mat.shape) )

calib_value_for_padding = int( (PATCH_SIZE-1)/2)
for i in range(BAND):
    MEAN_ARRAY[i] = np.mean(input_mat[i,:,:])
    new_input_mat.append(np.pad(input_mat[i,:,:],calib_value_for_padding,'constant',constant_values = 0))

print('Input_mat shape after padding: ' + str( np.array(new_input_mat).shape) )
print("+-------------------------------------+")
input_mat = np.array(new_input_mat)

class_label_counter = [0] * OUTPUT_CLASSES  #Class that
for i in range(OUTPUT_CLASSES):
    CLASSES.append([])


t = threading.Thread(target=animate).start()
start = time.time()
calib_value = int((PATCH_SIZE-1)/2)
count = 0
image = []
image_label = []
problem_data_set = []
for i in range(HEIGHT-1):
    for j in range(WIDTH-1):
        curr_inp = Patch(i,j)
        curr_tar = target_mat[i , j]

        if(curr_tar!=0): # Ignore patches with unknown landcover type for the central pixel
            CLASSES[curr_tar-1].append(curr_inp)
            class_label_counter[curr_tar-1] += 1
            count += 1

end_loading = True
end = time.time()
print("Total excution time..." + str(end-start)+'seconds')
print('Total number of K (things that can be identified): ' + str(count))
showClassTable(class_label_counter)

TRAIN_PATCH,TRAIN_LABELS,TEST_PATCH,TEST_LABELS,VAL_PATCH, VAL_LABELS = [],[],[],[],[],[]
# FULL_TRAIN_PATCH = []
# FULL_TRAIN_LABELS = []
count = 0

# Ringo's version
counter = 0 #Represent train_index position
for i, data in enumerate(CLASSES):
    if i+1 in list_labels:
        shuffle(data)
        print('Class '+ str(i+1)+ ' is accepted')
        #Size of validation set = 15% of training set
        val_size = int(train_idx[counter]*0.15)
        #index position between validation and test 
        test_cutoff = train_idx[counter] + val_size

        TRAIN_PATCH += data[:train_idx[counter]]
        TRAIN_LABELS += [counter] * train_idx[counter]
        #print('Check equal', len(TRAIN_PATCH), len(TRAIN_LABELS) )

        VAL_PATCH += data[train_idx[counter]:test_cutoff]
        VAL_LABELS += [counter] * val_size
       
        TEST_PATCH += data[test_cutoff:]
        tail_length = len(data)-test_cutoff
        TEST_LABELS += [counter] * tail_length

        counter += 1

    else:
        print('-Class '+ str(i+1)+ ' is rejected')

#FULL_TRAIN_LABELS = TRAIN_LABELS + VAL_LABELS
#FULL_TRAIN_PATCH = TRAIN_PATCH + VAL_PATCH

TRAIN_LABELS = np.array(TRAIN_LABELS)
TRAIN_PATCH = np.array(TRAIN_PATCH)
TEST_PATCH = np.array(TEST_PATCH)
TEST_LABELS = np.array(TEST_LABELS)
VAL_PATCH = np.array(VAL_PATCH)
VAL_LABELS = np.array(VAL_LABELS)
#FULL_TRAIN_LABELS = np.array(FULL_TRAIN_LABELS)
#FULL_TRAIN_PATCH = np.array(FULL_TRAIN_PATCH)

#print('Train, Test, Validation, Full_train', (len(TRAIN_PATCH)), (len(TEST_PATCH)), (len(VAL_PATCH)),  (len(FULL_TRAIN_PATCH))); #TODO: print size
print("+-------------------------------------+")
print("Size of Training data: " + str(len(TRAIN_PATCH)) )
print("Size of Validation data: " + str(len(VAL_PATCH))  )
print("Size of Testing data: " + str(len(TEST_PATCH)) )
print("+-------------------------------------+")

train_idx = list(range(len(TRAIN_PATCH)))
shuffle(train_idx)
TRAIN_PATCH = TRAIN_PATCH[train_idx]
TRAIN_LABELS = TRAIN_LABELS[train_idx]
test_idx = list(range(len(TEST_PATCH)))
shuffle(test_idx)
TEST_PATCH = TEST_PATCH[test_idx]
TEST_LABELS = TEST_LABELS[test_idx]
val_idx = list(range(len(VAL_PATCH)))
shuffle(val_idx)
VAL_PATCH = VAL_PATCH[val_idx]
VAL_LABELS = VAL_LABELS[val_idx]

'''
full_train_idx = shuffle(range(len(FULL_TRAIN_PATCH)))
FULL_TRAIN_PATCH = FULL_TRAIN_PATCH[full_train_idx]
FULL_TRAIN_LABELS = FULL_TRAIN_LABELS[full_train_idx]
'''

onehot_encoder = OneHotEncoder(sparse=False)

TRAIN_LABELS = np.reshape(TRAIN_LABELS, (len(TRAIN_LABELS),1) )
TRAIN_LABELS = onehot_encoder.fit_transform(TRAIN_LABELS).astype(np.uint8)
TRAIN_PATCH = np.transpose(TRAIN_PATCH,(0,2,3,1)).astype(np.float32)
train = {}
train["train_patch"] = TRAIN_PATCH
train["train_labels"] = TRAIN_LABELS
io.savemat("./data/" + opt.data + "_Train_patch_" + str(PATCH_SIZE) + ".mat", train)

TEST_LABELS = np.reshape(TEST_LABELS, (len(TEST_LABELS),1) )
TEST_LABELS = onehot_encoder.fit_transform(TEST_LABELS).astype(np.uint8)
TEST_PATCH = np.transpose(TEST_PATCH,(0,2,3,1)).astype(np.float32)
test = {}
test["test_patch"] = TEST_PATCH
test["test_labels"] = TEST_LABELS
io.savemat("./data/" + opt.data + "_Test_patch_" + str(PATCH_SIZE) + ".mat", test)

VAL_LABELS = np.reshape(VAL_LABELS, (len(VAL_LABELS),1) )
VAL_LABELS = onehot_encoder.fit_transform(VAL_LABELS).astype(np.uint8)
VAL_PATCH = np.transpose(VAL_PATCH,(0,2,3,1)).astype(np.float32)
val = {}
val["val_patch"] = VAL_PATCH
val["val_labels"] = VAL_LABELS
io.savemat("./data/" + opt.data + "_Val_patch_" + str(PATCH_SIZE) + ".mat", val)

'''
FULL_TRAIN_LABELS = FULL_TRAIN_LABELS.T
FULL_TRAIN_LABELS = np.reshape(FULL_TRAIN_LABELS, (len(FULL_TRAIN_LABELS),1) )
FULL_TRAIN_LABELS = onehot_encoder.fit_transform(FULL_TRAIN_LABELS).astype(np.uint8)
full_train = {}
full_train["train_patch"] = FULL_TRAIN_PATCH
full_train["train_labels"] = FULL_TRAIN_LABELS
'''

print("+-------------------------------------+")
print("Summary")
print('Train_patch.shape: '+ str(TRAIN_PATCH.shape) )
print('Train_label.shape: '+ str(TRAIN_LABELS.shape) )
print('Test_patch.shape: ' + str(TEST_PATCH.shape))
print('Test_label.shape: ' + str(TEST_LABELS.shape))
print("Validation batch Shape: " + str(VAL_PATCH.shape) )
print("Validation label Shape: " + str(VAL_LABELS.shape) )
print("+-------------------------------------+")
print("\nFinished processing.......\n Looking at some sample images")

'''
Below Code written by Ringo to Visualise lamdba distribution

'''
plot_random_spec_img(TRAIN_PATCH, TRAIN_LABELS)
plot_random_spec_img(TEST_PATCH, TEST_LABELS)
plot_random_spec_img(VAL_PATCH, VAL_LABELS)

# Show origin statlie image
plotStatlieImage(statlie_image)
# Show normalised statlie image
plotStatlieImage(statlie_image_2)
# Show transposed statlie image (reflection along x=y asix)
plotStatlieImage(statlie_image_3, bird=True)