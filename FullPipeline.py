import nibabel as nib
import numpy as np
import numpy
import itk
import math
from scipy.ndimage import label
from scipy.ndimage import sobel
from scipy.ndimage import uniform_filter
import cv2
from medpy.filter.smoothing import anisotropic_diffusion

exp_threshold = .05
import time

start_time = time.time()


def load_nii(file_path):
    # Loads the NIfTI
    nii_img = nib.load(file_path)  # Load .nii.gz file
    img_data = nii_img.get_fdata()  # Extract the data
    return img_data


def vec3_length(vec):
    return math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)


def vec3_dot(vec1, vec2):
    return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]


def threshold_image(img_data, threshold=-700):
    binary_img = img_data < threshold  # Convert to binary (1 for label, 0 for background)
    return binary_img


def calculate_connected_components(binary_img):
    # Calculates the connected components
    labeled_img, num_features = label(binary_img)  # Label connected components
    return labeled_img, num_features


def segment():
    list_components = []
    final_counts = numpy.array([])
    # Load the NIfTI

    # Threshold image
    binary_img = threshold_image(img_data)

    # Calculate connected components
    label_img, num_connected_components = calculate_connected_components(binary_img)

    # Remove small components
    filtered_labels, counts = np.unique(label_img, return_counts=True)
    for i in range(counts.size):
        if counts[i] < 100000:
            label_img[label_img == i] = 0
        else:
            new_component = label_img == i
            new_component = new_component.astype(numpy.int32)
            final_counts = numpy.append(final_counts, [counts[i]])
            list_components.append(new_component)
    # Remove Body layer
    body_layer = list_components[0]
    del list_components[0]
    final_counts = numpy.delete(final_counts, 0)


    # Remove background layer
    remove_id = -1
    for id in range(len(list_components)):
        if list_components[id][0][0][0] == 1:
            remove_id = id
            break
    del list_components[remove_id]
    final_counts = numpy.delete(final_counts, remove_id)

    shape = label_img.shape
    # remove lung layer and bed
    remove_ids = [0] * len(list_components)
    for posX in range(shape[0]):
        for posZ in range(shape[1]):
            for id in range(len(list_components)):
                if list_components[id][posX][posZ][0] == 1:
                    remove_ids[id] = 1
    for r_Id in reversed(range(len(list_components))):
        if remove_ids[r_Id] == 1:
            del list_components[r_Id]
            final_counts = numpy.delete(final_counts, r_Id)



    #Future optimization
    # crop images to bounding box
    # Determine the bounding box
    '''non_zero = np.array(np.nonzero(body_layer))
    min_coords = non_zero.min(axis=1)
    max_coords = non_zero.max(axis=1)'''

    # Crop nii image
    '''img_data = img_data[
                   min_coords[0]:max_coords[0] + 1,  # Crop along x
                   min_coords[1]:max_coords[1] + 1,  # Crop along y
                   min_coords[2]:max_coords[2] + 1  # Crop along z
                   ]
    shape = img_data.shape

    outputParsed = nib.Nifti1Image(img_data, affine=np.eye(4))
    nib.save(outputParsed, "CroppedTest.nii")'''


    # Output results
    return list_components, final_counts


#Bridge the gap between boundary layer and liquid pocket
def upwardExp(voxel,working_label):
    global final_label
    if  final_label[tuple([voxel[0],voxel[1]+2,voxel[2]])] >0:
        final_label[tuple([voxel[0], voxel[1] + 1, voxel[2]])]= 20
        working_label[tuple([voxel[0], voxel[1] + 1, voxel[2]])] = 20
    elif  final_label[tuple([voxel[0],voxel[1]+3,voxel[2]])] >0:
        final_label[tuple([voxel[0], voxel[1] + 1, voxel[2]])]= 20
        final_label[tuple([voxel[0], voxel[1] + 2, voxel[2]])] = 60
        working_label[tuple([voxel[0], voxel[1] + 1, voxel[2]])] = 20
        working_label[tuple([voxel[0], voxel[1] + 2, voxel[2]])] = 60
    elif  final_label[tuple([voxel[0],voxel[1]+4,voxel[2]])] >0:
        final_label[tuple([voxel[0], voxel[1] + 1, voxel[2]])]= 20
        final_label[tuple([voxel[0], voxel[1] + 2, voxel[2]])] = 60
        final_label[tuple([voxel[0], voxel[1] + 3, voxel[2]])] = 100
        working_label[tuple([voxel[0], voxel[1] + 1, voxel[2]])] = 20
        working_label[tuple([voxel[0], voxel[1] + 2, voxel[2]])] = 60
        working_label[tuple([voxel[0], voxel[1] + 3, voxel[2]])] = 100

#expand label based on sobel operator while limited on Y to min max
def expandVoxel(working_label,voxel,min,max):
    global sobelY
    global final_label
    shape = sobelY.shape
    to_expand = []
    up =  [0, 0, 1]
    left =  [0, 1, 0]
    front =  [1, 0, 0]
    down =  [0, 0, -1]
    right =  [0, -1, 0]
    back =  [-1, 0, 0]
    shape = img_data.shape
    for aux in range(3):
        up[aux] += voxel[aux]
        left[aux] += voxel[aux]
        front[aux] += voxel[aux]
        down[aux] += voxel[aux]
        right[aux] += voxel[aux]
        back[aux] += voxel[aux]
    if up[2] < shape[2]  :
        if working_label[tuple(up)] == 0 and sobelY[tuple(up)]<0:
            to_expand.append(up)
            working_label[tuple(up)] = 1
            final_label[tuple(up)] = 1

    if left[1] < shape[1] and left[1] < max and left[1] > min:
        if working_label[tuple(left)] == 0 and sobelY[tuple(left)]<0:
            to_expand.append(left)
            working_label[tuple(left)] = 1
            final_label[tuple(left)] = 1
    if front[0] < shape[0]:
        if working_label[tuple(front)] == 0 and sobelY[tuple(front)]<0:
            to_expand.append(front)
            working_label[tuple(front)] = 1
            final_label[tuple(front)] = 1
    if down[2] > 0 :
        if working_label[tuple(down)] == 0 and sobelY[tuple(down)]<0:
            to_expand.append(down)
            working_label[tuple(down)] = 1
            final_label[tuple(down)] = 1

    if right[1] > 0 and right[1] < max and right[1] > min:
        if working_label[tuple(right)] == 0 and sobelY[tuple(right)]<0:
            to_expand.append(right)
            working_label[tuple(right)] = 1
            final_label[tuple(right)] =1
    if back[0] > 0:
        if working_label[tuple(back)] == 0 and sobelY[tuple(back)]<0:
            to_expand.append(back)
            working_label[tuple(back)] = 1
            final_label[tuple(back)] = 1

    return to_expand



def expandLabel(label):
    shape = label.shape
    exp_list = []
    posY_min = 9999
    posY_max = 0
    #create list of voxels to expand
    for posX in range(shape[0]):
        for posY in range(shape[1]):
            for posZ in range(shape[2]):
                if label[posX, posY, posZ] >0:
                    if posY > posY_max:
                        posY_max = posY
                    if posY < posY_min:
                        posY_min = posY
                    exp_list.append([posX, posY, posZ])
    posY_min -=4
    posY_max += 8
    while exp_list:
        voxel = exp_list.pop()
        #expand current voxel and add new voxels to processing queue
        exp_list= exp_list + expandVoxel(label,voxel,posY_min,posY_max)
        #for each voxel in the boundary layer if possible "reach" the liquid pocket bridging the gap
        upwardExp(voxel,label)

def find_flat():
    global sobelY
    surface_mask = []
    total_labels = numpy.zeros(list_components[0].shape).astype(np.int16)
    expansion_labels = numpy.zeros(list_components[0].shape).astype(np.int16)
    # NO LONGER REQUIRED,find border voxels for each component
    '''for id in range(counts.size):
        surface_mask.append( (np.logical_xor(list_components[id], np.roll(list_components[id], shift=1, axis=0)) | \
                       np.logical_xor(list_components[id], np.roll(list_components[id], shift=1, axis=1)) | np.logical_xor(list_components[id], np.roll(list_components[id], shift=-1, axis=0)) |np.logical_xor(list_components[id], np.roll(list_components[id], shift=-1, axis=1)) |np.logical_xor(list_components[id], np.roll(list_components[id], shift=-1, axis=2)) | \
                       np.logical_xor(list_components[id], np.roll(list_components[id], shift=1, axis=2))*1 ).astype(numpy.int32))
        total_labels += surface_mask[id]'''
    for id in range(counts.size):
        total_labels += list_components[id]

    grad_x = sobel(total_labels, axis=0)
    sobelY = grad_y = sobel(total_labels, axis=1)
    grad_z = sobel(total_labels, axis=2)

    #DEBUG PRINTS
    output = nib.Nifti1Image(grad_x, nii.affine)
    nib.save(output, "sobelX.nii")

    output = nib.Nifti1Image(grad_y, nii.affine)
    nib.save(output, "sobelY.nii")

    output = nib.Nifti1Image(grad_z, nii.affine)
    nib.save(output, "sobelZ.nii")


    shape = total_labels.shape
    #Per axis sobel operator stacked to give a gradient matrix
    surface_normals = np.stack((grad_x, grad_y, grad_z), axis=-1)
    for x in range(20,shape[0]-20):
        for y in range(20,shape[1]-20):
            for z in range(20,shape[2]-20):

                if vec3_length(surface_normals[x][y][z]) > 0:
                    #normalize surface normals
                    surface_normals[x][y][z] = surface_normals[x][y][z] / vec3_length(surface_normals[x][y][z])
                    #only select voxels with normal similar to "pointing down"
                    if vec3_dot(surface_normals[x][y][z], [0, -1, 0]) >= 0.6 and sobelY[x][y][z] < 0:
                        
                        #all vertically close voxels are examined to improve layer thickness
                        # no border

                        if img_data[x][y - 1][z] < -600 and img_data[x][y][z] > 90:
                            expansion_labels[x][y + 1][z] = 1
                            expansion_labels[x][y][z] = 1
                            expansion_labels[x][y - 1][z] = 1
                        # one voxel thick layer
                        if img_data[x][y - 1][z] < -600 and img_data[x][y + 1][z] > 90:
                            expansion_labels[x][y + 1][z] = 1
                            expansion_labels[x][y][z] = 1
                            expansion_labels[x][y - 1][z] = 1
                        # two voxels thick layer
                        elif img_data[x][y + 2][z] < -600 and img_data[x][y + 1][z] > 90:
                            expansion_labels[x][y - 2][z] = 1
                            expansion_labels[x][y + 1][z] = 1
                            expansion_labels[x][y][z] = 1
                            expansion_labels[x][y - 1][z] = 1
                        elif img_data[x][y - 1][z] < -600 and img_data[x][y + 2][z] > 90:
                            expansion_labels[x][y + 1][z] = 1
                            expansion_labels[x][y][z] = 1
                            expansion_labels[x][y - 1][z] = 1
                        # three voxels thick layer
                        elif img_data[x][y + 3][z] < -600 and img_data[x][y + 1][z] > 90:
                            expansion_labels[x][y + 1][z] = 1
                            expansion_labels[x][y][z] = 1
                            expansion_labels[x][y - 1][z] = 1
                        elif img_data[x][y - 1][z] < -600 and img_data[x][y + 3][z] > 90:
                            expansion_labels[x][y + 1][z] = 1
                            expansion_labels[x][y][z] = 1
                            expansion_labels[x][y - 1][z] = 1
                        elif img_data[x][y - 2][z] < -600 and img_data[x][y + 2][z] > 90:
                            expansion_labels[x][y + 1][z] = 1
                            expansion_labels[x][y][z] = 1
                            expansion_labels[x][y - 1][z] = 1

                        # expansion_labels[x][y + 2][z] = 1
                        # expansion_labels[x][y + 1][z] = 1
                        # expansion_labels[x][y][z] = 1
                        # expansion_labels[x][y - 1][z] = 1
                        # expansion_labels[x][y - 2][z] = 1

    flat_components, num_components = calculate_connected_components(expansion_labels)
    filtered_labels, voxel_count = np.unique(flat_components, return_counts=True)
    count_pass = voxel_count.size -1
    list_fl = []
    count = 0


    #remove small flat labels and create a list of labels
    for i in range(voxel_count.size):
        if voxel_count[i] < 200:
            count_pass -= 1
        elif i != 0:
            new_cmp = flat_components == count
            new_cmp = new_cmp.astype(numpy.int16)
            list_fl.append(new_cmp)
            count += 1

    flat_components = numpy.zeros(list_components[0].shape).astype(np.int16)
    for i in range(count_pass):
        flat_components += list_fl[i]*i

    return total_labels, count_pass, flat_components , list_fl

def filter_stack():
    clamp()

    histogram_filter()

    anisotropic()

def  histogram_filter():
    global threshold_data
    # Normalize the intensity range to 8-bit (0-255)
    data_normalized = cv2.normalize(threshold_data, None, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

    # Apply histogram equalization slice-by-slice
    equalized_data = np.zeros_like(data_normalized)
    for i in range(data_normalized.shape[-1]):  # Iterate through slices along the z-axis
        equalized_data[..., i] = cv2.equalizeHist(data_normalized[..., i])

    # Rescale the equalized data back to the original intensity range
    threshold_data  =  cv2.normalize(equalized_data, None, threshold_data.min(), threshold_data.max(),cv2.NORM_MINMAX)


def clamp ():
    global threshold_data
    # clamp image
    WINDOW_LEVEL = 40
    WINDOW_WIDTH = 400

    lower_bound = WINDOW_LEVEL - WINDOW_WIDTH / 2
    upper_bound = WINDOW_LEVEL + WINDOW_WIDTH / 2
    image_clamped = np.clip(threshold_data, lower_bound, upper_bound)
    threshold_data = ((image_clamped - np.min(image_clamped)) / (np.max(image_clamped) - np.min(image_clamped)) * 255.0)

def anisotropic ():
    global threshold_data
    threshold_data = anisotropic_diffusion(threshold_data, niter=15, kappa=55, gamma=0.12, voxelspacing=None)

def create_pockets():

    # image with new high value pockets
    high_labels = numpy.zeros(flat_labels.shape).astype(np.int16)
    global final_label

    # shape of og image
    shape = img_data.shape

    list_components = []

    point_list = []
    #initial treshold of liquids
    for posX in range(shape[0]):
        for posY in range(shape[1]):
            for posZ in range(shape[2]):
                pos = posX, posY, posZ
                if threshold_data[pos] > 210:
                    high_labels[pos] = 1
                    point_list.append(pos)
                if flat_labels[pos] > 0:
                    img_data[pos] = -1200
                # if og_data[pos] < -900:
                #    final_label[pos] = 1

    #conected components to separate each liquid pocket
    labeled_img, num_features = label(high_labels)
    filtered_labels, counts = np.unique(labeled_img, return_counts=True)
    counter = 0
    for i in range(counts.size):
        if counts[i] < 5000 or counts[i] >  0.02 * img_data.shape[0]* img_data.shape[1]*  img_data.shape[2]:
            labeled_img[labeled_img == i] = 0
        else:

            new_component = labeled_img == i
            labeled_img[labeled_img == i] = counter - 1
            new_component = new_component.astype(numpy.int32)
            list_components.append(new_component)
            counter += 1

    # remove background layer
    del list_components[0]

    #DEBUG print
    outputParsed = nib.Nifti1Image(flat_labels, nii.affine)
    nib.save(outputParsed, "flatLabels.nii")

    valid_components = numpy.zeros(counter - 1).astype(np.int16)
    valid_flats = numpy.zeros(flat_counts).astype(np.int16)

    #if pocket intersects with label than its valid
    for pos in point_list:
        if flat_labels[pos] != 0 and labeled_img[pos] >= 0:
            valid_components[labeled_img[pos]] = 1
            valid_flats[flat_labels[pos]] = 1

    # update the final label with only the new boundary layer/ liquid pocket paired
    for i in range(counter - 1):
        if valid_components[i] != 0:
            final_label += list_components[i]
    for i in range(flat_counts):
        if valid_flats[i] != 0:
            expandLabel(label_list[i])
            outputParsed = nib.Nifti1Image(label_list[i], nii.affine)
            nib.save(outputParsed, "flatLabels.nii")
            final_label += label_list[i]

    for pos in point_list:
        if final_label[pos] != 0:
            img_data[pos] = -1200





sobelY = np.array
img_data = load_nii("Test.nii")
nii = nib.load("Test.nii")
final_label = numpy.zeros(img_data.shape).astype(np.int16)
threshold_data = img_data


filter_stack()


output = nib.Nifti1Image(threshold_data, nii.affine)
nib.save(output,"testLabel.nii")

#segment air and remove extra components
list_components, counts =segment()
expansion_reg, flat_counts, flat_labels , label_list = find_flat()




outputParsed = nib.Nifti1Image(expansion_reg, nii.affine)
nib.save(outputParsed, "outputParsed.nii")


#create contrast enhanced imagee


create_pockets()

final_label = final_label + expansion_reg

output = nib.Nifti1Image(final_label, nii.affine)
nib.save(output,"finalSegmentation.nii")

output = nib.Nifti1Image(img_data, nii.affine)
nib.save(output,"BaseContrast.nii")



print("time elapsed: {:.2f}s".format(time.time() - start_time))
# Convert to itk
itk_components = []
# for i in range(counts.size):
#    itk_image = itk.image_from_array(list_components[i].data)
#    itk_image.SetSpacing(np.sqrt((list_components[i].affine ** 2).sum(axis=0))[:3])
#   itk_components.append(itk_image)

# for i in range(counts):
#    outputParsed = nib.Nifti1Image(expansion_reg[i], affine=np.eye(4))
#   nib.save(outputParsed,"outputParsed" + str(i) + ".nii")



