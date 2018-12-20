const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node')

let modelBuilders = {
  // high score: bs 250 ~85%
  motionDense1: (shape, multiplier)=> {
    const model = tf.sequential()
    model.add(tf.layers.flatten({inputShape: shape}))
    model.add(tf.layers.dense({units: Math.round(shape[0] * shape[1] * shape[2] * multiplier), activation: 'relu'}))
    model.add(tf.layers.dense({units: Math.round(shape[0] * shape[1] * shape[2] * multiplier), activation: 'relu'}))
    model.add(tf.layers.dense({units: 2, activation: 'linear'}))
    return model
  },

  // what if we can make this really small tho
  // high score: 89.1% after 71000 cycles
  motionDense2: (shape, multiplier)=> {
    const model = tf.sequential()
    model.add(tf.layers.flatten({inputShape: shape}))
    model.add(tf.layers.dense({units: Math.round(shape[0] * shape[1] * multiplier), activation: 'tanh'}))
    model.add(tf.layers.dense({units: Math.round(shape[0] * shape[1] * multiplier), activation: 'tanh'}))
    model.add(tf.layers.dense({units: Math.round(shape[0] * shape[1] * multiplier), activation: 'relu'}))
    model.add(tf.layers.dense({units: 2, activation: 'linear'}))
    return model
  },

  // high score: bs 250 ~86%; ~92-94% by 15,000 cycles
  motionConv1: (shape, multiplier)=> {
    const model = tf.sequential()
    model.add(tf.layers.conv2d({
      inputShape: shape,
      filters: 3,
      kernelSize: 4,
      strides: 1,
      activation: 'relu'
    }))
    model.add(tf.layers.flatten())
    model.add(tf.layers.dense({units: Math.round(shape[0] * shape[1] * shape[2] * multiplier), activation: 'relu'}))
    model.add(tf.layers.dense({units: Math.round(shape[0] * shape[1] * shape[2] * multiplier), activation: 'relu'}))
    model.add(tf.layers.dense({units: 2, activation: 'linear'}))
    return model
  },

  // Model Accuracy on testset: 70.1% - batch 5051
  motionConv2: (shape, multiplier)=> {
    const model = tf.sequential()
    model.add(tf.layers.conv2d({
      inputShape: shape,
      filters: 3,
      kernelSize: 4,
      activation: 'relu'
    }))
    model.add(tf.layers.conv2d({
      filters: 3,
      kernelSize: 4,
      activation: 'relu'
    }))
    model.add(tf.layers.flatten())
    model.add(tf.layers.dense({units: Math.round(42 * multiplier), activation: 'relu'}))
    model.add(tf.layers.dense({units: 2, activation: 'linear'}))
    return model
  },

  // what if softmax would help? it didn't. it was bad
  motionConv3: (shape, multiplier)=> {
    const model = tf.sequential()
    model.add(tf.layers.conv2d({
      inputShape: shape,
      filters: 2,
      kernelSize: 2,
      activation: 'relu'
    }))
    model.add(tf.layers.flatten())
    // thinking this layer can create a bitmap of probabilities that it aligns with that pixel
    model.add(tf.layers.dense({units: shape[0]*shape[1], activation: 'tanh'}))
    // and this can transform that in to x/y
    model.add(tf.layers.dense({units: 2, activation: 'linear'}))
    return model
  },

  // what if the front end is just a tiny conv that can invent something kinda like hsv if it wants
  // gets to 90% after 100,000 cycles
  motionConv4: (shape, mul)=> {
    const model = tf.sequential()
    model.add(tf.layers.conv2d({inputShape: shape, filters: 6, kernelSize: 1, activation: 'tanh'}))
    model.add(tf.layers.flatten({inputShape: shape}))
    model.add(tf.layers.dense({units: Math.round(shape[0] * shape[1]) * 2, activation: 'tanh'}))
    model.add(tf.layers.dense({units: Math.round(shape[0] * shape[1]), activation: 'tanh'}))
    model.add(tf.layers.dense({units: 2, activation: 'linear'}))
    return model
  },

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
  }
}

module.exports = modelBuilders;