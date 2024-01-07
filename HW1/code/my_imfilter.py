import numpy as np

def my_imfilter(image, imfilter):
    """function which imitates the default behavior of the build in scipy.misc.imfilter function.

    Input:
        image: A 3d array represent the input image.
        imfilter: The gaussian filter.
    Output:
        output: The filtered image.
    """
    # =================================================================================
    # TODO:                                                                           
    # This function is intended to behave like the scipy.ndimage.filters.correlate    
    # (2-D correlation is related to 2-D convolution by a 180 degree rotation         
    # of the filter matrix.)                                                          
    # Your function should work for color images. Simply filter each color            
    # channel independently.                                                          
    # Your function should work for filters of any width and height                   
    # combination, as long as the width and height are odd (e.g. 1, 7, 9). This       
    # restriction makes it unambigious which pixel in the filter is the center        
    # pixel.                                                                          
    # Boundary handling can be tricky. The filter can't be centered on pixels         
    # at the image boundary without parts of the filter being out of bounds. You      
    # should simply recreate the default behavior of scipy.signal.convolve2d --       
    # pad the input image with zeros, and return a filtered image which matches the   
    # input resolution. A better approach is to mirror the image content over the     
    # boundaries for padding.                                                         
    # Uncomment if you want to simply call scipy.ndimage.filters.correlate so you can 
    # see the desired behavior.                                                       
    # When you write your actual solution, you can't use the convolution functions    
    # from numpy scipy ... etc. (e.g. numpy.convolve, scipy.signal)                   
    # Simply loop over all the pixels and do the actual computation.                  
    # It might be slow.                        
    
    # NOTE:                                                                           
    # Some useful functions:                                                        
    #     numpy.pad (https://numpy.org/doc/stable/reference/generated/numpy.pad.html)      
    #     numpy.sum (https://numpy.org/doc/stable/reference/generated/numpy.sum.html)                                     
    # =================================================================================

    # ============================== Start OF YOUR CODE ===============================
    output = np.zeros_like(image)

    # print("image shape:", image.shape)
    # print("imfilter shape:", imfilter.shape)
    # print(image)

    kernel_height, kernel_width = imfilter.shape
    image_height, image_width, depth = image.shape

    # 不擴充陣列下卷積後之輸出size
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1
    # print(output_height, output_width)
    # print(pad_adding_height, pad_adding_width)

    # 擴充pad大小
    pad_adding_height = int((kernel_height - 1) / 2)
    pad_adding_width = int((kernel_width - 1) / 2)

    # padding image
    pad_image = np.pad(image, 
                       ((pad_adding_height, pad_adding_width), (pad_adding_height, pad_adding_width), (0, 0)),
                       "constant", 
                       constant_values=(0, 0))
    # print(pad_image.shape)

    for i in range(image_height):
        for j in range(image_width):
            for d in range(depth):
                input_slice = pad_image[i:i+kernel_height, j:j+kernel_width, d]
                output[i, j, d] = np.sum(input_slice * imfilter)

    
    # =============================== END OF YOUR CODE ================================

    # Uncomment if you want to simply call scipy.ndimage.filters.correlate so you can
    # see the desired behavior.
    # import scipy.ndimage as ndimage
    # output = np.zeros_like(image)
    # for ch in range(image.shape[2]):
    #    output[:,:,ch] = ndimage.filters.correlate(image[:,:,ch], imfilter, mode='constant')

    return output