const sharp = require('sharp')
const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node')
const fs = require('fs')
const path = require('path')
const {promisify} = require('util')

class TrainingData {
  constructor(datapath) {
    this.path = datapath
    this.outputShape = [2]
  }

  // load all the jpegs, decoded, in to ram
  async load() {
    let shape, floatArray
    let files = await promisify(fs.readdir)(this.path)
    let fullPaths = files.filter((x) => x.match(/\.(jpg|jpeg|png)$/i)).map((filename)=> path.join(this.path, filename))
    // to speed up debug
    //fullPaths = fullPaths.slice(0, 5)
    for (let imgIdx = 0; imgIdx < fullPaths.length; imgIdx++) {
      let imgPath = fullPaths[imgIdx]
      let { data, info } = await sharp(imgPath).raw().toBuffer({ resolveWithObject: true })
      //console.log(`loading ${imgPath}:`, info)
      // if this is the first image, note it's shape for later validation
      if (!shape) shape = [info.height, info.width, info.channels]
      // validate that the image is the same shape as the last image
      if (shape[0] != info.height || shape[1] != info.width || shape[2] != info.channels)
        throw new Error(`${imgPath} doesn't have the same shape as previous image`)
      // setup a float32array to store all the images if one isn't already setup
      if (!floatArray) floatArray = new Float32Array(fullPaths.length * info.height * info.width * info.channels)
      
      // scan out the pixel data from the image in to the big float array
      let imgByteSize = shape[0] * shape[1] * shape[2]
      data.forEach((b, idx)=> floatArray[(imgIdx * imgByteSize) + idx] = b / 255)
      // let tensor = tf.tensor3d(floatArray, [info.width, info.height, info.channels])
      // this.images.push(tensor)

    }
    this.dataset = tf.tensor4d(floatArray, [fullPaths.length, shape[0], shape[1], shape[2]])
    return this
  }

  // release memory for image tensors
  dispose() {
    this.dataset.dispose()
  }

  // internal, gets random coordinates with some constraints, for random cropping to generate samples
  getRandomConstrinedBoxPair(width, height, size, offsetX, offsetY) {
    let halfSize = size / 2
    let x = halfSize + (Math.random() * (width - size - Math.abs(offsetX)))
    let y = halfSize + (Math.random() * (height - size - Math.abs(offsetX)))
    return [
      [ (y - halfSize - (offsetY / 2)) / height,
        (x - halfSize - (offsetX / 2)) / width, 
        (y + halfSize - (offsetX / 2)) / height,
        (x + halfSize - (offsetX / 2)) / width],
      [ (y - halfSize + (offsetY / 2)) / height,
        (x - halfSize + (offsetX / 2)) / width,
        (y + halfSize + (offsetY / 2)) / height,
        (x + halfSize + (offsetX / 2)) / width]
    ]
  }

  // get an x/y pair to use for motion training with a specified offset
  async getRandomTrainingPairs(size, randomShift, trainingSize) {
    let cropJobsA = []
    let cropJobsB = []
    let boxIndexes = []
    let randomMotions = []

    // generate a random selection of box pairs from the dataset
    for (let i = 0; i < trainingSize; i++) {
      let randomX = Math.round(((Math.random() * 2) - 1) * randomShift)
      let randomY = Math.round(((Math.random() * 2) - 1) * randomShift)
      let [box_a, box_b] = this.getRandomConstrinedBoxPair(this.dataset.shape[2], this.dataset.shape[1], size, randomX, randomY)
      cropJobsA.push(box_a)
      cropJobsB.push(box_b)
      boxIndexes.push(Math.floor(Math.random() * this.dataset.shape[0])) // choose a random image to take this sample from
      randomMotions.push([-(randomX / size), -(randomY / size)])
    }

    // crop and resize the samples to the spec, and interlace the frame data so we end up with 6 channel sample images
    // now arranged RGBRGB, with the older frame first, then the newer frame
    let interlace = tf.tidy(()=> {
      let cropJobsAT = tf.tensor2d(cropJobsA, [trainingSize, 4], 'float32')
      let cropJobsBT = tf.tensor2d(cropJobsB, [trainingSize, 4], 'float32')
      let boxIdxT = tf.tensor1d(boxIndexes, 'int32')

      let cropsA = tf.image.cropAndResize(this.dataset, cropJobsAT, boxIdxT, [size, size])
      let cropsB = tf.image.cropAndResize(this.dataset, cropJobsBT, boxIdxT, [size, size])
      return tf.concat([cropsA, cropsB], 3)
    })

    return {
      x: interlace,
      y: tf.tidy(()=> tf.tensor(randomMotions, [trainingSize].concat(this.outputShape)))
    }
  }
}

// async shortcut to instantiate and load training data
// returns a promise, resolved when everything's loaded in to RAM
TrainingData.load = function(...args) {
  return (new TrainingData(...args)).load()
}

module.exports = TrainingData