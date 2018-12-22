// trying to make a little network that does motion estimation like optical flow kinda thing!
// idk maths so, machine learning, plz save me!
// network recieves input that's a square RGB tile, in RRGGBB format, alternating between earlier frame and later frame
// output should be two numbers, x and y, between 0.0 and 1.0 with 0.5 being no motion, and lower being left or up
const fs = require('fs')
const {promisify} = require('util')
const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node')
const TrainingData = require('./training-data')
const modelFactory = require('./models')

// config
const samplePicsPath = './sample-pics'

// training config
const trainingEpochs = 1
const batchSizeMul = 1
const batchSize = 2000 * batchSizeMul
const trainingCycles = 7000 / batchSizeMul * 100
const modelName = process.argv.slice(-1)[0]
const inputPatchSize = 8 // how big the square tile is that the estimator sees
const maxMotionEstimate = inputPatchSize - 2 // how big the maximum movement between the two frames is, in pixels, in the auto training set
const inputShape = [inputPatchSize, inputPatchSize, 6] // 6 channels because two rgb patches are combined
const modelMultiplier = 1

const saveModelInterval = 500 / batchSizeMul
const learningRate = 0.0005
const optimizer = tf.train.adam(learningRate)


// main program
async function main() {
  console.log("Loading training dataset")
  let training = await TrainingData.load(samplePicsPath)
  if (modelName.match(/trueMotionConv/)) training.outputShape = [1,1,2]
  console.log("Loaded pictures")

  console.log("Setting up model...")
  let model = modelFactory[modelName](inputShape, modelMultiplier)
  // compile model
  model.compile({ optimizer, loss: 'meanAbsoluteError', metrics: ['accuracy'] })

  // generate test data for regular validation
  const testData = await training.getRandomTrainingPairs(inputPatchSize, maxMotionEstimate, batchSize)

  let trainingNum = 0
  let nextTrainingBatch = training.getRandomTrainingPairs(inputPatchSize, maxMotionEstimate, batchSize)
  while (trainingNum < trainingCycles) {
    let batchData = await nextTrainingBatch
    nextTrainingBatch = training.getRandomTrainingPairs(inputPatchSize, maxMotionEstimate, batchSize)

    await model.fit(batchData.x, batchData.y, { batchSize, epochs: trainingEpochs })

    // clear that memory out now
    batchData.x.dispose()
    batchData.y.dispose()

    if (trainingNum % saveModelInterval == 0) {
      let testAccPercent = tf.tidy(()=> model.evaluate(testData.x, testData.y)[1].dataSync()[0] * 100)
      console.log(`============================= Model Accuracy on testset: ${testAccPercent.toFixed(1)}% - batch ${trainingNum + 1}`)

      // save model
      const saveResult = await model.save(`file://./trained-models/${modelName}`)
      //console.log('model save result: ', saveResult)
      fs.writeFileSync(`./trained-models/${modelName}/accuracy.txt`, `accuracy: ${testAccPercent.toFixed(1)}%\n`)
      fs.writeFileSync(`./trained-models/${modelName}/batchNum.txt`, `${trainingNum + 1}`)
    }

    trainingNum += 1
  }
  console.log("Training Complete. Evaluating...")

  let testAccPercent = tf.tidy(()=> model.evaluate(testData.x, testData.y)[1].dataSync()[0] * 100)
  console.log(`============================= Final test accuracy: ${testAccPercent.toFixed(1)}%`)
  const saveResult = await model.save(`file://./trained-models/${modelName}`)

  testData.x.dispose()
  testData.y.dispose()
}

main()