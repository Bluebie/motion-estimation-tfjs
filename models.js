const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node')

let modelBuilders = {
  // first attempt at a convolutional 
  motionConv1: (shape)=> {
    const model = tf.sequential();
    model.add(tf.layers.conv2d({
      inputShape: shape,
      filters: 8,
      kernelSize: 8,
      strides: 2,
      activation: 'relu',
    }));
    // max pooling's probably the wrong answer here!
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({units: 42, activation: 'relu'}));
    model.add(tf.layers.dense({units: 2, activation: 'relu'}));
    return model
  },

  motionDense1: (shape)=> {
    const model = tf.sequential();
    model.add(tf.layers.flatten({inputShape: shape}));
    model.add(tf.layers.dense({units: shape[0] * shape[1] * shape[2], activation: 'relu'}));
    model.add(tf.layers.dense({units: 2, activation: 'relu'}));
    return model;
  }
}

module.exports = modelBuilders;