import numpy

from chainer.backends import cuda

from train_utils.datatypes import Size

if cuda.available:
    create_mask_kernel = cuda.cupy.ElementwiseKernel(
        'T originalMask, raw T corners, int32 inputHeight, int32 inputWidth, T fillValue',
        'T mask',
        '''
            // determine our current position in the array
            int w = i % inputWidth;
            int h = (i / inputWidth) % inputHeight;
            int batchIndex = i / inputWidth / inputHeight;
            int cornerIndex = batchIndex * 4 * 2;
    
            // calculate vectors for each side of the box
            int2 topVec = {
                corners[cornerIndex + 1 * 2] - corners[cornerIndex],
                corners[cornerIndex + 1 * 2 + 1] - corners[cornerIndex + 1]
            };
    
            int2 bottomVec = {
                corners[cornerIndex + 3 * 2] - corners[cornerIndex + 2 * 2],
                corners[cornerIndex + 3 * 2 + 1] - corners[cornerIndex + 2 * 2 + 1]
            };
    
            int2 leftVec = {
                corners[cornerIndex] - corners[cornerIndex + 3 * 2],
                corners[cornerIndex + 1] - corners[cornerIndex + 3 * 2 + 1]
            }; 
    
            int2 rightVec = {
                corners[cornerIndex + 2 * 2] - corners[cornerIndex + 1 * 2],
                corners[cornerIndex + 2 * 2 + 1] - corners[cornerIndex + 1 * 2 + 1]
            };
    
            // calculate cross product for each side of array
            int crossTop = topVec.x * (h - corners[cornerIndex + 1]) - topVec.y * (w - corners[cornerIndex]);
            int crossRight = rightVec.x * (h - corners[cornerIndex + 3]) - rightVec.y * (w - corners[cornerIndex + 2]);
            int crossBottom = bottomVec.x * (h - corners[cornerIndex + 5]) - bottomVec.y * (w - corners[cornerIndex + 4]);
            int crossLeft = leftVec.x * (h - corners[cornerIndex + 7]) - leftVec.y * (w - corners[cornerIndex + 6]);
    
            // our point is inside as long as every cross product is greater or equal to 0
            bool inside = crossTop >= 0 && crossRight >= 0 && crossBottom >= 0 && crossLeft >= 0;
    
            mask = inside ? fillValue : originalMask;
        ''',
        name='bbox_to_mask',
)


class ImageMasker:
    """
        This class implements a CPU/GPU method for adding a mask on an array. Think of it as drawing a polygon that is defined
        by a bounding box, given by four points.
    """

    def __init__(self, fill_value, mask_dtype=numpy.int32):
        self.fill_value = fill_value
        self.mask_dtype = mask_dtype

    @staticmethod
    def extract_corners(bboxes, xp):
        top_left = xp.clip(bboxes[:, :, 0, 0], -1, 1)
        top_right = xp.clip(bboxes[:, :, 0, -1], -1, 1)
        bottom_right = xp.clip(bboxes[:, :, -1, -1], -1, 1)
        bottom_left = xp.clip(bboxes[:, :, -1, 0], -1, 1)
        return xp.stack((top_left, top_right, bottom_right, bottom_left), axis=1)

    @staticmethod
    def scale_bboxes(bboxes, image_size):
        bboxes = (bboxes + 1) / 2
        bboxes[:, :, 0] *= image_size.width
        bboxes[:, :, 1] *= image_size.height
        return bboxes

    def create_mask_cpu(self, base_mask, corners):
        top_width_vectors = corners[:, 1, :] - corners[:, 0, :]
        bottom_width_vectors = corners[:, 3, :] - corners[:, 2, :]
        left_height_vectors = corners[:, 0, :] - corners[:, 3, :]
        right_height_vectors = corners[:, 2, :] - corners[:, 1, :]

        for idx in numpy.ndindex(base_mask.shape):
            batch_index, h, w = idx
            top_width_vector = top_width_vectors[batch_index]
            bottom_width_vector = bottom_width_vectors[batch_index]
            left_height_vector = left_height_vectors[batch_index]
            right_height_vector = right_height_vectors[batch_index]

            # determine cross product of each vector with our current point
            cross_products = []
            for corner, vector in zip(corners[batch_index], [top_width_vector, right_height_vector, bottom_width_vector, left_height_vector]):
                cross_product = vector[0] * (h - corner[1]) - vector[1] * (w - corner[0])
                cross_products.append(cross_product >= 0)

            # if the point is on the right of all lines (i.e positive) the point is inside the box and we can mark it
            if all(cross_products):
                base_mask[idx] = self.fill_value

        return base_mask

    def create_mask_gpu(self, base_mask, corners):
        batch_size, height, width = base_mask.shape
        mask = create_mask_kernel(base_mask, corners, height, width, self.fill_value)
        return mask

    def create_mask(self, mask_shape, corners, xp):
        base_mask = xp.zeros(mask_shape, dtype=self.mask_dtype)
        return self.mask_array(base_mask, corners, xp)

    def mask_array(self, array, corners, xp):
        """
        Create a mask on a given array, using the OOB specified by corners

        :param array: An array with three dimensions. The first dimension is the batch dimension (if you want to mask a
         4 dimensional array, you'll first need to collapse the batch and channel dimension, but make sure to also pad
         the given corners!, that means increase the number of corners to match batch size * number_of_channels).
         The second dimension is the height of the feature map or image. The third dimension is the
         width of the feature map or image.

        :param corners: An array with three dimensions. The first dimension is the batch axis (similar to the array
        parameter). The shape of the second dimension must be 4, as we are using bounding boxes with four corner points.
        The shape of the third dimension (the actual points of each corner) should be 2. The first element in this
        dimension must be the x coordinate and the second element must be the y coordinate.

        :param xp: either numpy or cupy, depending on whether the code should run on CPU or GPU

        :return: an array where all elements inside the bounding box get the value that is saved in `self.fill_value`.
        """
        if xp == cuda.cupy:
            return self.create_mask_gpu(array, corners)
        else:
            return self.create_mask_cpu(array, corners)


class InhibitionOfReturnImageMasker(ImageMasker):

    def __init__(self, fill_value, mask_dtype=numpy.int32, mask_full_box=False, mask_fraction=0.25):
        super().__init__(fill_value, mask_dtype=mask_dtype)
        self.mask_full_box = mask_full_box
        self.mask_fraction = mask_fraction

    def add_inhibition_of_return_mask(self, images, bboxes):
        xp = cuda.get_array_module(bboxes.data)
        corners = self.extract_corners(bboxes.data, xp)
        corners = self.scale_bboxes(corners, Size._make(images.shape[-2:]))

        batch_size, num_channels, height, width = images.shape
        masked_images = xp.reshape(images, (batch_size * num_channels, height, width))
        if not self.mask_full_box:
            corners = self.create_inhibition_of_return_box(corners, xp)
        corners = self.tile_box(corners, num_channels, xp)
        masked_images = self.mask_array(masked_images, corners, xp)
        return xp.reshape(masked_images, (batch_size, num_channels, height, width))

    @staticmethod
    def tile_box(box, tile_factor, xp):
        batch_size, num_points, num_coordinates = box.shape
        box = xp.tile(box[:, xp.newaxis, ...], (1, tile_factor, 1, 1))
        return xp.reshape(box, (batch_size * tile_factor, num_points, num_coordinates))

    def create_inhibition_of_return_box(self, corners, xp):
        height_vector = corners[:, 3, :] - corners[:, 0, :]
        vector_length = xp.sqrt(xp.square(height_vector[:, 0]) + xp.square(height_vector[:, 1]))

        # pad vector legth in order to do elementwise division
        padded_vector_length = xp.tile(vector_length[:, xp.newaxis], (1, 2))
        normalized_height_vector = height_vector / padded_vector_length
        new_length = vector_length * abs(self.mask_fraction)
        new_length = xp.tile(new_length[:, xp.newaxis], (1, 2))
        height_add_vector = normalized_height_vector * new_length

        top_left = corners[:, 0, :] + height_add_vector
        bottom_left = corners[:, 3, :] - height_add_vector
        top_right = corners[:, 1, :] + height_add_vector
        bottom_right = corners[:, 2, :] - height_add_vector

        box = xp.stack([top_left, top_right, bottom_right, bottom_left], axis=1)
        return box
