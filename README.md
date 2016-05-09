# Net2Net : Accelerating Learning via Knowledge Transfer

- Numpy-based Net2Net module
  - Net2Wider
  - Net2Deeper
  
- Net2Net using Tensorflow
  - Test in MNIST dataset

## Dependencies

- Net2Net core module
  - Numpy
  - Scipy

- Tensorflow examples
  - Tensorflow
  - Slim

## Results

- Baseline architecture
  
  ```
  5x5x32(conv1)-pool1-5x5x64(conv2)-pool2-1024(fc1)-10(fc2)
  ```

- [EXP 1] Train a teacher network
  
  ```
  [Iter: 100] Validation Accuracy : 0.8732
  [Iter: 200] Validation Accuracy : 0.9025
  [Iter: 300] Validation Accuracy : 0.9313
  [Iter: 400] Validation Accuracy : 0.9408
  [Iter: 500] Validation Accuracy : 0.9363
  [Iter: 600] Validation Accuracy : 0.9466
  [Iter: 700] Validation Accuracy : 0.9379
  [Iter: 800] Validation Accuracy : 0.9582
  [Iter: 900] Validation Accuracy : 0.9583
  ```

- [EXP 2] Train a student network (Net2Wider)
  - # of filters in 'conv1' layer [32->128]
  
  ```
  [Iter: 100] Validation Accuracy : 0.9136
  [Iter: 200] Validation Accuracy : 0.9689
  [Iter: 300] Validation Accuracy : 0.9645
  [Iter: 400] Validation Accuracy : 0.9757
  [Iter: 500] Validation Accuracy : 0.9762
  [Iter: 600] Validation Accuracy : 0.9757
  [Iter: 700] Validation Accuracy : 0.9752
  [Iter: 800] Validation Accuracy : 0.9765
  [Iter: 900] Validation Accuracy : 0.9777
  ```

- [EXP 3] Net2Wider baseline (Random pad)
  
  ```
  [Iter: 100] Validation Accuracy : 0.9255
  [Iter: 200] Validation Accuracy : 0.9361
  [Iter: 300] Validation Accuracy : 0.9418
  [Iter: 400] Validation Accuracy : 0.9551
  [Iter: 500] Validation Accuracy : 0.9608
  [Iter: 600] Validation Accuracy : 0.9653
  [Iter: 700] Validation Accuracy : 0.9677
  [Iter: 800] Validation Accuracy : 0.9659
  [Iter: 900] Validation Accuracy : 0.9690
  ```

- [EXP 4] Train a student network (Net2Deeper)
  - Insert a new layer after 'conv1' layer
  
  ```
  [Iter: 100] Validation Accuracy : 0.9673
  [Iter: 200] Validation Accuracy : 0.9646
  [Iter: 300] Validation Accuracy : 0.9718
  [Iter: 400] Validation Accuracy : 0.9731
  [Iter: 500] Validation Accuracy : 0.9765
  [Iter: 600] Validation Accuracy : 0.9612
  [Iter: 700] Validation Accuracy : 0.9783
  [Iter: 800] Validation Accuracy : 0.9812
  [Iter: 900] Validation Accuracy : 0.9785
  ```

- [EXP 5] Net2Deeper baseline (Random initialization)
  
  ```
  [Iter: 100] Validation Accuracy : 0.9057
  [Iter: 200] Validation Accuracy : 0.9059
  [Iter: 300] Validation Accuracy : 0.9446
  [Iter: 400] Validation Accuracy : 0.9489
  [Iter: 500] Validation Accuracy : 0.9541
  [Iter: 600] Validation Accuracy : 0.9581
  [Iter: 700] Validation Accuracy : 0.9607
  [Iter: 800] Validation Accuracy : 0.9499
  [Iter: 900] Validation Accuracy : 0.9663
  ```
  
## Notes
- All parameters are fixed except new weights from Net2Net.
- The Net2Net core module (net2net.py) can be used in various deep learning libraries (theano, caffe etc.) because it has only numpy dependency. 

