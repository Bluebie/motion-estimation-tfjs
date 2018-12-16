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
const batchSize = 250
const trainingCycles = 50000
const modelName = 'motionDense1'
const inputPatchSize = 12 // how big the square tile is that the estimator sees
const maxMotionEstimate = inputPatchSize / 2 // how big the maximum movement between the two frames is, in pixels, in the auto training set
const inputShape = [inputPatchSize, inputPatchSize, 6] // 6 channels because two rgb patches are combined
const modelMultiplier = 1

const saveModelInterval = 10
const learningRate = 0.0005
const optimizer = tf.train.adam(learningRate)


// main program
async function main() {
  console.log("Loading training dataset")
  let training = await TrainingData.load(samplePicsPath)
  console.log("Loaded pictures")

  console.log("Setting up model...")
  let model = modelFactory[modelName](inputShape, modelMultiplier)
  // compile model
  model.compile({ optimizer, loss: 'meanAbsoluteError', metrics: ['accuracy'] })

  // generate test data for regular validation
  const testData = await training.getRandomTrainingPairs(inputPatchSize, maxMotionEstimate, batchSize)

  let trainingNum = 0
  while (trainingNum < trainingCycles) {
    console.log("Training, batches remaining: " + batchSize)

    let batchData = await training.getRandomTrainingPairs(inputPatchSize, maxMotionEstimate, batchSize)

    await model.fit(batchData.x, batchData.y, { batchSize, epochs: trainingEpochs })

    // clear that memory out now
    batchData.x.dispose()
    batchData.y.dispose()

    let testResult = model.evaluate(testData.x, testData.y);
    let testAccPercent = testResult[1].dataSync()[0] * 100;
    console.log(`============================= Model Accuracy on testset: ${testAccPercent.toFixed(1)}% - batch ${trainingNum + 1}`)

    // if (trainingNum % saveModelInterval == 0) {
    //   // save model
    //   const saveResult = await model.save(`file://./${modelName}`)
    //   //console.log('model save result: ', saveResult)
    //   fs.writeFileSync(`./${modelName}/accuracy.txt`, `accuracy: ${testAccPercent.toFixed(1)}%\n`)
    //   fs.writeFileSync(`./${modelName}/batchNum.txt`, `${trainingNum + 1}`)
    // }

    trainingNum += 1
  }
  console.log("Training Complete. Evaluating...")

  const testResult = model.evaluate(testData.x, testData.y);
  const testAccPercent = testResult[1].dataSync()[0] * 100;
  console.log(`============================= Final test accuracy: ${testAccPercent.toFixed(1)}%`)
  //const saveResult = await model.save(`file://./${modelName}`)

  testData.x.dispose()
  testData.y.dispose()
}

main()