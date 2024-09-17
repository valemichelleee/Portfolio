import numpy as np

class Pooling:
    def __init__(self, stride_shape, pooling_shape):
        self.trainable = False
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.p_h, self.p_w = pooling_shape
        self.s_h, self.s_w = stride_shape
        self.input_tensor = None
        self.indices = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        n, c, h, w = input_tensor.shape
        output_height = ((h - self.p_h) // self.s_h) + 1
        output_width = ((w - self.p_w) // self.s_w) + 1

        output_tensor = np.zeros((n,c, output_height, output_width))
        self.indices = np.zeros((n, c, output_height, output_width), dtype=object)

        for batch in range(n):
            for channel in range(c):
                for height in range(output_height):
                    start_h = height * self.s_h
                    end_h = start_h + self.p_h
                    for width in range(output_width):
                        start_w = width * self.s_w
                        end_w = start_w + self.p_w

                        pool = input_tensor[batch, channel, start_h:end_h, start_w:end_w]
                        max_val = np.max(pool)
                        output_tensor[batch, channel, height, width] = max_val

                        idx_h, idx_w = np.unravel_index(np.argmax(pool), pool.shape)
                        self.indices[batch, channel, height, width] = (start_h + idx_h, start_w + idx_w)

        return output_tensor

    def backward(self, error_tensor):
        n_error, c_error, h_error, w_error = error_tensor.shape
        n, c, h, w = self.input_tensor.shape
        next_error_tensor = np.zeros(self.input_tensor.shape)

        for batch in range(n):
            for channel in range(c):
                for height in range(h_error):
                    for width in range(w_error):
                        h_index, w_index = self.indices[batch, channel, height, width]
                        next_error_tensor[batch, channel, h_index, w_index] += error_tensor[batch, channel, height, width]

        return next_error_tensor
