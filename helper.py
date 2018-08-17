'''
Arguments: whole test data, test label,
return randomized test data, test label of 'size'
'''
def getValidDataset(test, test_label, size=250):
    from numpy import array
    from random import shuffle

    assert test.shape[0] == test_label.shape[0]

    idx = list(range(test.shape[0]))
    shuffle(idx)
    idx = idx[:size]
    accuracy_x, accuracy_y = [], []
    for i in idx:
        accuracy_x.append(test[i])
        accuracy_y.append(test_label[i])

    return array(accuracy_x), array(accuracy_y)

"""
Take first hyperspectral image from dataset and plot spectral data distribution
Arguements pic = list of images in size (?, height, width, bands), where ? represents any number > 0
            true_labels = lists of ground truth corrospond to pic
"""
def plot_random_spec_img(pic, true_label):
    pic = pic[0]  #Take first data only
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from numpy import mean, argmax

    print("Image Shape: " + str(pic.shape) )
    print("Label of this image is -> " + str(true_label[0] ) )

    title = argmax(true_label[0], axis=0)
    # Calculate mean of all elements in the 3d element
    mean_value = mean(pic)
    # Replace element with less than mean by zero
    pic[pic < mean_value] = 0
    
    x = []
    y = []
    z = []
    # Coordinate position extractions
    for z1 in range(pic.shape[0]): 
        for x1 in range(pic.shape[1]):
            for y1 in range(pic.shape[2]):
                if pic[z1,x1,y1] != 0:
                    z.append(z1)
                    x.append(x1)
                    y.append(y1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('True class = '+ str(title) + ' ,' + str(indianPineLUT(true_label[0])) )
    ax.scatter(x, y, z, color='#0606aa', marker='o', s=0.5)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Spectral Label')
    ax.set_zlabel('Y Label')
    plt.show()




# Arguement: data = 3D image in size (h,w,bands)
def plotStatlieImage(data, bird=False):
    from matplotlib.pyplot import imshow, show, subplots, axis, figure
    print('\nPlotting a band image')
    fig, ax = subplots(nrows=3, ncols=3)
    i = 1
    for row in ax:
        for col in row:
            i += 11
            if bird:
                col.imshow(data[i,:,:])
            else:
                col.imshow(data[:,:,i])
            axis('off')
    show()


def showClassTable(number_of_list):
    import pandas as pd 
    print("\n+------------Show Table---------------+")
    lenth = len(number_of_list)
    column1 = range(1, lenth+1)
    table = {'Class#': column1, 'Number of samples': number_of_list}
    table_df = pd.DataFrame(table).to_string(index=False)
    print(table_df)   
    print("+-----------Close Table---------------+")



"""
Indian pine look up table
Original dataset 
[2 ,3 ,5 ,6 ,8 ,10,11,12,14] 
Now becomes 
[0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ]

"""
def indianPineLUT(a):
    a = a.tolist()
    a = a.index(1)
    if a == 0:
        return 'Corn-notill'
    elif a == 1:
        return 'Corn-mintill'
    elif a == 2:
        return 'Grass-pasture'
    elif a == 3:
        return 'Grass-trees'
    elif a == 4:
        return 'Hay-windrowed'
    elif a == 5:
        return 'Soybean-notill'
    elif a == 6:
        return 'Soybean-mintill'
    elif a == 7:
        return 'Soybean-clean'
    elif a == 8:
        return 'Woods'
    else:
        return 'None!'

