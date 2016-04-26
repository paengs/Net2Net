# Net2Net: Accelerating Learning via Knowledge Transfer

## Numpy-based Net2Net module
- Net2Wider
- Net2Deeper

## Net2Net using Tensorflow
- Test in MNIST dataset

# Dependencies

## Net2Net core module
- Numpy
- Scipy (for verification)

## Tensorflow examples
- Tensorflow
- Slim

# Results

@ Test in MNIST dataset

- Baseline architecture

5x5x32(conv1)-pool1-5x5x64(conv2)-pool2-1024(fc1)-10(fc2)

1. Train a teacher network

2. Resume training in same architecture

3. Train a student network (Net2Wider)
  - # of filters in 'conv1' layer [32->128]

4. Train a student network (Net2Deeper)
  - Insert a new layer after 'conv1' layer

@ Results

NOTE: All parameters are fixed.

1. validation accuracy: 96.39%

2. validation accuracy: 97.39% 

3. validation accuracy: 97.85%

4. validation accuracy: 97.75%
