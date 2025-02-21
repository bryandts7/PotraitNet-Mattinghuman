import numpy as np
import cv2

def image_t(im, scale=1.0, rot=45, trans=(50,-50)):
    # Get image dimensions
    rows, cols = im.shape[:2]

    # Convert angle to radians
    angle_rad = np.radians(rot)

    # Compute the center of the image to perform rotation about the center
    center = (cols / 2, rows / 2)

    # Construct scaling matrix
    scale_matrix = np.array([
        [scale, 0, 0],
        [0, scale, 0]
    ], dtype=np.float32)

    # Get the rotation matrix (considering only rotation)
    rotation_matrix = cv2.getRotationMatrix2D(center, rot, scale)

    # Add translation to the transformation matrix
    rotation_matrix[0, 2] += trans[0]
    rotation_matrix[1, 2] += trans[1]

    # Apply the affine transformation using the computed matrix
    result = cv2.warpAffine(im, rotation_matrix, (cols, rows))

    return result


if __name__ == '__main__':
    # Read the image
    im = cv2.imread('./misc/pearl.jpeg')
    
    scale  = 0.5
    rot    = 45
    trans  = (50, -50)
    
    # Apply the transformation
    result = image_t(im, scale, rot, trans)
    
    # Save the result
    cv2.imwrite('./results/affine_result.png', result)
