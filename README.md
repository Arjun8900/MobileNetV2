# MobileNetV2
MobileNetV2 implementation using Keras.

MobileNetV2 is very similar to the original MobileNet,except that it uses inverted residual blocks with bottlenecking features. It has a drastically lower parameter count than the original MobileNet. MobileNets support any input size greater than 32 x 32, with larger image sizes offering better performance.

Channel axis is assumed to be -1 i.e, channel last. Change it to 1 if it uses channel first.
