import cv2
import numpy as np

# Connect smalls pixels in larger blobs
def morph(image, kernel = (5, 5), show=False):
    element_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
    img = cv2.morphologyEx(image, cv2.MORPH_CLOSE, element_closing)
    if show:
        cv2.imshow(f'Kernel: {kernel}', img)
    return img

# Remove elements smaller than minimum size
# https://stackoverflow.com/questions/42798659/how-to-remove-small-connected-objects-using-opencv
def component(image, connectivity = 8, min_size = 100, show=False):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity= connectivity)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    img = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img[output == i + 1] = 255
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    if show:
        cv2.imshow(f'Min size: {min_size}', img)
    return img

# Approximation of the contour
def approx_cont(contours, look_back):
    contour = []
    for n in range(len(contours)):
        cont = []
        cont_back = np.concatenate([contours[n][-look_back:], contours[n]])
        for i in range(len(cont_back[:-look_back])):
            inner_list = []
            slice = cont_back[i: i + (2*look_back)].mean(axis=0)
            inner_list.append(int(slice[0][0]))
            inner_list.append(int(slice[0][1]))
            cont.append(inner_list)
        contour.append(np.array(cont).reshape((-1, 1,2)).astype(np.int32))
    return contour

# Applaying Sobel Edge detections
def sobel(image):
    sobelX = cv2.Sobel(image ,cv2.CV_64F,1,0, ksize=-1)
    sobelY = cv2.Sobel(image,cv2.CV_64F,0,1, ksize=-1)
    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))
    image_sobel = cv2.bitwise_or(sobelX, sobelY)
    return image_sobel

def merger(imageGRAY):
    image_morph = morph(imageGRAY, kernel=(3, 3), show=False)
    # Remove smalls bloobs
    image_conn = component(image_morph, min_size=250, show=False)
    # Merge pixels using opencv closing morphology functions
    image_morph = morph(image_conn, kernel=(11, 11), show=False)
    # Remove smalls bloobs
    image_conn = component(image_morph, min_size=750, show=False)
    return image_conn
