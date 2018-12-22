const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node')

let modelBuilders = {
  // what if i use conv operations to pretty much build a densely connected network, but then exploit that to let
  // tensorflow handle the strides across the images and avoid having to do crops and stuff
  trueMotionConv1: (shape, mul)=> {
    const model = tf.sequential()
    model.add(tf.layers.conv2d({
      //inputShape: shape,
      inputShape: [null, null, 6],
      filters: Math.round(shape[0] * shape[1] * mul),
      kernelSize: [shape[0], shape[1]],
      pad: 'valid',
      strides: [shape[0], shape[1]],
      activation: 'tanh'
    }))
    model.add(tf.layers.conv2d({filters: Math.round(shape[0] * shape[1] * mul), kernelSize: 1, activation: 'tanh'}))
    model.add(tf.layers.conv2d({filters: Math.round(shape[0] * shape[1] * mul), kernelSize: 1, activation: 'relu'}))
    model.add(tf.layers.conv2d({filters: 2, kernelSize: 1, activation: 'linear'}))
    return model
  },

  // can we drop a layer and do as well?
  trueMotionConv2: (shape, mul)=> {
    const model = tf.sequential()
    model.add(tf.layers.conv2d({
      //inputShape: shape,
      inputShape: [null, null, 6],
      filters: Math.round(shape[0] * shape[1] * mul),
      kernelSize: [shape[0], shape[1]],
      pad: 'valid',
      strides: [shape[0], shape[1]],
      activation: 'tanh'
    }))
    model.add(tf.layers.conv2d({filters: Math.round(shape[0] * shape[1] * mul), kernelSize: 1, activation: 'tanh'}))
    //model.add(tf.layers.conv2d({filters: Math.round(shape[0] * shape[1] * mul), kernelSize: 1, activation: 'relu'}))
    model.add(tf.layers.conv2d({filters: 2, kernelSize: 1, activation: 'linear'}))
    return model
  },

  // what if it has more of a funnel shape?
  trueMotionConv3: (shape, mul)=> {
    const model = tf.sequential()
    model.add(tf.layers.conv2d({
      inputShape: [null, null, 6],
      filters: Math.round(shape[0] * shape[1] * mul),
      kernelSize: [shape[0], shape[1]],
      pad: 'valid',
      strides: [shape[0], shape[1]],
      activation: 'tanh'
    }))
    model.add(tf.layers.conv2d({filters: Math.round(shape[0] * shape[1] * mul * 0.50), kernelSize: 1, activation: 'tanh'}))
    model.add(tf.layers.conv2d({filters: Math.round(shape[0] * shape[1] * mul * 0.25), kernelSize: 1, activation: 'relu'}))
    model.add(tf.layers.conv2d({filters: 2, kernelSize: 1, activation: 'linear'}))
    return model
  },

  // what if there's only (x + y) * 2 filters instead of x*y filters?
  trueMotionConv4: (shape, mul)=> {
    const model = tf.sequential()
    model.add(tf.layers.conv2d({
      inputShape: [null, null, 6],
      filters: Math.round((shape[0] + shape[1]) * 2 * mul),
      kernelSize: [shape[0], shape[1]],
      pad: 'valid',
      strides: [shape[0], shape[1]],
      activation: 'tanh'
    }))
    model.add(tf.layers.conv2d({filters: Math.round((shape[0] + shape[1]) * 2 * mul), kernelSize: 1, activation: 'tanh'}))
    model.add(tf.layers.conv2d({filters: Math.round((shape[0] + shape[1]) * 2 * mul), kernelSize: 1, activation: 'relu'}))
    model.add(tf.layers.conv2d({filters: 2, kernelSize: 1, activation: 'linear'}))
    return model
  }
}

module.exports = modelBuilders;