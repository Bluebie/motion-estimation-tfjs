const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node')

let modelBuilders = {
  // first attempt at a convolutional 
  // motionConv1: (shape, multiplier)=> {
  //   const model = tf.sequential();
  //   model.add(tf.layers.conv2d({
  //     inputShape: shape,
  //     filters: 8,
  //     kernelSize: 8,
  //     strides: 2,
  //     activation: 'relu',
  //   }));
  //   // max pooling's probably the wrong answer here!
  //   model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
  //   model.add(tf.layers.flatten());
  //   model.add(tf.layers.dense({units: Math.round(42 * multiplier), activation: 'relu'}));
  //   model.add(tf.layers.dense({units: 2, activation: 'relu'}));
  //   return model
  // },

  // high score: bs 250 ~85%
  motionDense1: (shape, multiplier)=> {
    const model = tf.sequential()
    model.add(tf.layers.flatten({inputShape: shape}))
    model.add(tf.layers.dense({units: Math.round(shape[0] * shape[1] * shape[2] * multiplier), activation: 'relu'}))
    model.add(tf.layers.dense({units: Math.round(shape[0] * shape[1] * shape[2] * multiplier), activation: 'relu'}))
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

  // high score: bs 250, ~85% accurate after like 15,000 training cycles (this thing learns slowly)
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
    model.add(tf.layers.conv2d({
      filters: 3,
      kernelSize: 4,
      activation: 'linear'
    }))
    model.add(tf.layers.globalAveragePooling2d({}))
    //model.add(tf.layers.flatten())
    //model.add(tf.layers.dense({units: Math.round(42 * multiplier), activation: 'relu'}))
    model.add(tf.layers.dense({units: 2, activation: 'linear'}))
    return model
  },
}

module.exports = modelBuilders;