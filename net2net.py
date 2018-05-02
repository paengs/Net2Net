"""
Implementation of Net2Net (http://arxiv.org/abs/1511.05641)
Numpy modules for Net2Net
- Net2Wider
- Net2Deeper

Written by Kyunghyun Paeng

"""
import numpy as np

class Net2Net(object):
    def __init__(self, error=1e-4):
        self._error_th = error
        print('Net2Net module initialize...')

    def deeper(self, weight, verification=True):
        """ Net2Deeper operation
          
        All weights & biases should be 'numpy' array.
        If it is 'conv' type, weight.ndim = 4 (kH, kW, InChannel, OutChannel)
        If it is 'fc' type, weight.ndim = 2 (In, Out)

        Args:
            weight: weight matrix where the layer to be deepened

        Returns:
            Identity matrix & bias fitted to input weight
        """
        assert weight.ndim == 4 or weight.ndim == 2, 'Check weight.ndim'
        if weight.ndim == 2:
            deeper_w = np.eye(weight.shape[1])
            deeper_b = np.zeros(weight.shape[1])
            if verification:
                err = np.abs(np.sum(np.dot(weight, deeper_w)-weight))
                assert err < 1e-5, 'Verification failed: [ERROR] {}'.format(err)
        else:
            deeper_w = np.zeros((weight.shape[0], weight.shape[1], weight.shape[3], weight.shape[3]))
            assert weight.shape[0] % 2 == 1 and weight.shape[1] % 2 == 1, 'Kernel size should be odd'
            center_h = (weight.shape[0]-1)//2
            center_w = (weight.shape[1]-1)//2
            for i in range(weight.shape[3]):
                tmp = np.zeros((weight.shape[0], weight.shape[1], weight.shape[3]))
                tmp[center_h, center_w, i] = 1
                deeper_w[:, :, :, i] = tmp
            deeper_b = np.zeros(weight.shape[3])
            if verification:
                import scipy.signal
                inputs = np.random.rand(weight.shape[0]*4, weight.shape[1]*4, weight.shape[2])
                ori = np.zeros((weight.shape[0]*4, weight.shape[1]*4, weight.shape[3]))
                new = np.zeros((weight.shape[0]*4, weight.shape[1]*4, weight.shape[3]))
                for i in range(weight.shape[3]):
                    for j in range(inputs.shape[2]):
                        if j==0: tmp = scipy.signal.convolve2d(inputs[:,:,j], weight[:,:,j,i], mode='same')
                        else: tmp += scipy.signal.convolve2d(inputs[:,:,j], weight[:,:,j,i], mode='same')
                    ori[:,:,i] = tmp
                for i in range(deeper_w.shape[3]):
                    for j in range(ori.shape[2]):
                        if j==0: tmp = scipy.signal.convolve2d(ori[:,:,j], deeper_w[:,:,j,i], mode='same')
                        else: tmp += scipy.signal.convolve2d(ori[:,:,j], deeper_w[:,:,j,i], mode='same')
                    new[:,:,i] = tmp
                err = np.abs(np.sum(ori-new))
                assert err < self._error_th, 'Verification failed: [ERROR] {}'.format(err)
        return deeper_w, deeper_b

    def wider(self, weight1, bias1, weight2, new_width, verification=True):
        """ Net2Wider operation
        
        All weights & biases should be 'numpy' array.
        If it is 'conv' type, weight.ndim = 4 (kH, kW, InChannel, OutChannel)
        If it is 'fc' type, weight.ndim = 2 (In, Out)
        
        Args:    
            weight1: weight matrix of a target layer
            bias1: biases of a target layer, bias1.ndim = 1
            weight2: weight matrix of a next layer
            new_width: It should be larger than old width.
                     (i.e., 'conv': weight1.OutChannel < new_width,
                            'fc'  : weight1.Out < new_width )
        Returns:
            Transformed weights & biases (w1, b1, w2)
        """
        # Check dimensions
        assert bias1.squeeze().ndim==1, 'Check bias.ndim'
        assert weight1.ndim == 4 or weight1.ndim == 2, 'Check weight1.ndim'
        assert weight2.ndim == 4 or weight2.ndim == 2, 'Check weight2.ndim'
        bias1 = bias1.squeeze()
        if weight1.ndim == 2:
            assert weight1.shape[1] == weight2.shape[0], 'Check shape of weight'
            assert weight1.shape[1] == len(bias1), 'Check shape of bias'
            assert weight1.shape[1] < new_width, 'new_width should be larger than old width'
            return self._wider_fc(weight1, bias1, weight2, new_width, verification)
        else:
            assert weight1.shape[3] == weight2.shape[2], 'Check shape of weight'
            assert weight1.shape[3] == len(bias1), 'Check shape of bias'
            assert weight1.shape[3] < new_width, 'new_width should be larger than old width'
            return self._wider_conv(weight1, bias1, weight2, new_width, verification)
    
    def wider_rand(self, weight1, bias1, weight2, new_width):
        """ Net2Wider operation with random pad (baseline)
        
        All weights & biases should be 'numpy' array.
        If it is 'conv' type, weight.ndim = 4 (kH, kW, InChannel, OutChannel)
        If it is 'fc' type, weight.ndim = 2 (In, Out)
        
        Args:    
            weight1: weight matrix of a target layer
            bias1: biases of a target layer, bias1.ndim = 1
            weight2: weight matrix of a next layer
            new_width: It should be larger than old width.
                     (i.e., 'conv': weight1.OutChannel < new_width,
                            'fc'  : weight1.Out < new_width )
        Returns:
            Transformed weights & biases (w1, b1, w2)
        """
        # Check dimensions
        assert bias1.squeeze().ndim==1, 'Check bias.ndim'
        assert weight1.ndim == 4 or weight1.ndim == 2, 'Check weight1.ndim'
        assert weight2.ndim == 4 or weight2.ndim == 2, 'Check weight2.ndim'
        bias1 = bias1.squeeze()
        if weight1.ndim == 2:
            assert weight1.shape[1] == weight2.shape[0], 'Check shape of weight'
            assert weight1.shape[1] == len(bias1), 'Check shape of bias'
            assert weight1.shape[1] < new_width, 'new_width should be larger than old width'
            return self._wider_fc_rand(weight1, bias1, weight2, new_width)
        else:
            assert weight1.shape[3] == weight2.shape[2], 'Check shape of weight'
            assert weight1.shape[3] == len(bias1), 'Check shape of bias'
            assert weight1.shape[3] < new_width, 'new_width should be larger than old width'
            return self._wider_conv_rand(weight1, bias1, weight2, new_width)
           
    def _wider_conv(self, teacher_w1, teacher_b1, teacher_w2, new_width, verification):
        rand = np.random.randint(teacher_w1.shape[3], size=(new_width-teacher_w1.shape[3]))
        replication_factor = np.bincount(rand)
        student_w1 = teacher_w1.copy()
        student_w2 = teacher_w2.copy()
        student_b1 = teacher_b1.copy()
        # target layer update (i)
        for i in range(len(rand)):
            teacher_index = rand[i]
            new_weight = teacher_w1[:, :, :, teacher_index]
            new_weight = new_weight[:, :, :, np.newaxis]
            student_w1 = np.concatenate((student_w1, new_weight), axis=3)
            student_b1 = np.append(student_b1, teacher_b1[teacher_index])
        # next layer update (i+1)
        for i in range(len(rand)):
            teacher_index = rand[i]
            factor = replication_factor[teacher_index] + 1
            assert factor > 1, 'Error in Net2Wider'
            new_weight = teacher_w2[:, :, teacher_index, :]*(1./factor)
            new_weight_re = new_weight[:, :, np.newaxis, :]
            student_w2 = np.concatenate((student_w2, new_weight_re), axis=2)
            student_w2[:, :, teacher_index, :] = new_weight
        if verification:
            import scipy.signal
            inputs = np.random.rand(teacher_w1.shape[0]*4, teacher_w1.shape[1]*4, teacher_w1.shape[2])
            ori1 = np.zeros((teacher_w1.shape[0]*4, teacher_w1.shape[1]*4, teacher_w1.shape[3]))
            ori2 = np.zeros((teacher_w1.shape[0]*4, teacher_w1.shape[1]*4, teacher_w2.shape[3]))
            new1 = np.zeros((teacher_w1.shape[0]*4, teacher_w1.shape[1]*4, student_w1.shape[3]))
            new2 = np.zeros((teacher_w1.shape[0]*4, teacher_w1.shape[1]*4, student_w2.shape[3]))
            for i in range(teacher_w1.shape[3]):
                for j in range(inputs.shape[2]):
                    if j==0: tmp = scipy.signal.convolve2d(inputs[:,:,j], teacher_w1[:,:,j,i], mode='same')
                    else: tmp += scipy.signal.convolve2d(inputs[:,:,j], teacher_w1[:,:,j,i], mode='same')
                ori1[:,:,i] = tmp + teacher_b1[i]
            for i in range(teacher_w2.shape[3]):
                for j in range(ori1.shape[2]):
                    if j==0: tmp = scipy.signal.convolve2d(ori1[:,:,j], teacher_w2[:,:,j,i], mode='same')
                    else: tmp += scipy.signal.convolve2d(ori1[:,:,j], teacher_w2[:,:,j,i], mode='same')
                ori2[:,:,i] = tmp
            for i in range(student_w1.shape[3]):
                for j in range(inputs.shape[2]):
                    if j==0: tmp = scipy.signal.convolve2d(inputs[:,:,j], student_w1[:,:,j,i], mode='same')
                    else: tmp += scipy.signal.convolve2d(inputs[:,:,j], student_w1[:,:,j,i], mode='same')
                new1[:,:,i] = tmp + student_b1[i]
            for i in range(student_w2.shape[3]):
                for j in range(new1.shape[2]):
                    if j==0: tmp = scipy.signal.convolve2d(new1[:,:,j], student_w2[:,:,j,i], mode='same')
                    else: tmp += scipy.signal.convolve2d(new1[:,:,j], student_w2[:,:,j,i], mode='same')
                new2[:,:,i] = tmp
            err = np.abs(np.sum(ori2-new2))
            assert err < self._error_th, 'Verification failed: [ERROR] {}'.format(err)
        return student_w1, student_b1, student_w2

    def _wider_conv_rand(self, teacher_w1, teacher_b1, teacher_w2, new_width):
        size = new_width-teacher_w1.shape[3]
        student_w1 = teacher_w1.copy()
        student_w2 = teacher_w2.copy()
        student_b1 = teacher_b1.copy()
        # target layer update (i)
        for i in range(size):
            shape = teacher_w1[:,:,:,0].shape
            new_weight = np.random.normal(0, 0.1, size=shape)
            new_weight = new_weight[:, :, :, np.newaxis]
            student_w1 = np.concatenate((student_w1, new_weight), axis=3)
            student_b1 = np.append(student_b1, 0.1)
        # next layer update (i+1)
        for i in range(size):
            shape = teacher_w2[:,:,0,:].shape
            new_weight = np.random.normal(0, 0.1, size=shape)
            new_weight_re = new_weight[:, :, np.newaxis, :]
            student_w2 = np.concatenate((student_w2, new_weight_re), axis=2)
        return student_w1, student_b1, student_w2
        
    def _wider_fc(self, teacher_w1, teacher_b1, teacher_w2, new_width, verification):
        rand = np.random.randint(teacher_w1.shape[1], size=(new_width-teacher_w1.shape[1]))
        replication_factor = np.bincount(rand)
        student_w1 = teacher_w1.copy()
        student_w2 = teacher_w2.copy()
        student_b1 = teacher_b1.copy()
        # target layer update (i)
        for i in range(len(rand)):
            teacher_index = rand[i]
            new_weight = teacher_w1[:, teacher_index]
            new_weight = new_weight[:, np.newaxis]
            student_w1 = np.concatenate((student_w1, new_weight), axis=1)
            student_b1 = np.append(student_b1, teacher_b1[teacher_index])
        # next layer update (i+1)
        for i in range(len(rand)):
            teacher_index = rand[i]
            factor = replication_factor[teacher_index] + 1
            assert factor > 1, 'Error in Net2Wider'
            new_weight = teacher_w2[teacher_index,:]*(1./factor)
            new_weight = new_weight[np.newaxis, :]
            student_w2 = np.concatenate((student_w2, new_weight), axis=0)
            student_w2[teacher_index,:] = new_weight
        if verification:
            inputs = np.random.rand(1, teacher_w1.shape[0])
            ori1 = np.dot(inputs, teacher_w1) + teacher_b1
            ori2 = np.dot(ori1, teacher_w2)
            new1 = np.dot(inputs, student_w1) + student_b1
            new2 = np.dot(new1, student_w2)
            err = np.abs(np.sum(ori2-new2))
            assert err < self._error_th, 'Verification failed: [ERROR] {}'.format(err)
        return student_w1, student_b1, student_w2
    
    def _wider_fc_rand(self, teacher_w1, teacher_b1, teacher_w2, new_width):
        size = new_width-teacher_w1.shape[1]
        student_w1 = teacher_w1.copy()
        student_w2 = teacher_w2.copy()
        student_b1 = teacher_b1.copy()
        # target layer update (i)
        for i in range(size):
            shape = teacher_w1[:,0].shape
            new_weight = np.random.normal(0, 0.1, size=shape)
            new_weight = new_weight[:, np.newaxis]
            student_w1 = np.concatenate((student_w1, new_weight), axis=1)
            student_b1 = np.append(student_b1, 0.1)
        # next layer update (i+1)
        for i in range(size):
            shape = teacher_w2[0,:].shape
            new_weight = np.random.normal(0, 0.1, size=shape)
            new_weight = new_weight[np.newaxis, :]
            student_w2 = np.concatenate((student_w2, new_weight), axis=0)
        return student_w1, student_b1, student_w2

if __name__ == '__main__':
    """ Net2Net Class Test """
    obj = Net2Net()

    w1 = np.random.rand(100, 50)
    obj.deeper(w1)
    print('Succeed: Net2Deeper (fc)')
    
    w1 = np.random.rand(3,3,16,32)
    obj.deeper(w1)
    print('Succeed: Net2Deeper (conv)')
    
    w1 = np.random.rand(100, 50)
    b1 = np.random.rand(50,1)
    w2 = np.random.rand(50, 10)
    obj.wider(w1, b1, w2, 70)
    print('Succeed: Net2Wider (fc)')

    w1 = np.random.rand(3,3,16,32)
    b1 = np.random.rand(32)
    w2 = np.random.rand(3,3,32,64)
    obj.wider(w1, b1, w2, 48)
    print('Succeed: Net2Wider (conv)')
