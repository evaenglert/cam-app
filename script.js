const status = document.getElementById('status');
if (status) {
  status.innerText = 'Loaded TensorFlow.js - version: ' + tf.version.tfjs;
}

const video = document.getElementById('webcam');
const liveView = document.getElementById('liveView');
const demosSection = document.getElementById('demos');
const enableWebcamButton = document.getElementById('webcamButton');

// Check if webcam access is supported.
function getUserMediaSupported() {
  return !!(navigator.mediaDevices &&
    navigator.mediaDevices.getUserMedia);
}

// If webcam supported, add event listener to button for when user
// wants to activate it to call enableCam function which we will
// define in the next step.
if (getUserMediaSupported()) {
  enableWebcamButton.addEventListener('click', enableCam);
} else {
  console.warn('getUserMedia() is not supported by your browser');
}

// Enable the live webcam view and start classification.
function enableCam(event) {
  // Only continue if the COCO-SSD has finished loading.
  if (!model) {
    return;
  }

  // Hide the button once clicked.
  event.target.classList.add('removed');

  // getUsermedia parameters to force video but not audio.
  const constraints = {
    video: true
  };

  // Activate the webcam stream.
  navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
    video.srcObject = stream;
    video.addEventListener('loadeddata', predictWebcam);
  });
}

var children = [];
var i = 0;

function predictWebcam() {
  // Now let's start classifying a frame in the stream.
  // console.log(model);
  // console.log(video);
  // console.log(video.srcObject);


  i = i + 1;

  detector.estimateHands(video).then(function (hands) {
    for (let i = 0; i < children.length; i++) {
      liveView.removeChild(children[i]);
    }
    children.splice(0);

    if (hands.length !== 0) {

      const min_x = hands[0].keypoints.reduce((min, current) => {
        return current.x < min.x ? current : min
      }, hands[0].keypoints[0]).x;

      const max_x = hands[0].keypoints.reduce((min, current) => {
        return current.x > min.x ? current : min
      }, hands[0].keypoints[0]).x;

      const min_y = hands[0].keypoints.reduce((min, current) => {
        return current.y < min.y ? current : min
      }, hands[0].keypoints[0]).y;

      const max_y = hands[0].keypoints.reduce((min, current) => {
        return current.y > min.y ? current : min
      }, hands[0].keypoints[0]).y;

      const centre_x = (min_x + max_x) / 2;
      const centre_y = (min_y + max_y) / 2;

      const p = document.createElement('p');

      p.style = 'margin-left: ' + min_x + 'px; margin-top: '
        + (max_y - 10) + 'px; width: '
        + ((max_x - min_x) - 10) + 'px; top: 0; left: 0;';

      const highlighter = document.createElement('div');
      highlighter.setAttribute('class', 'highlighter');

      highlighter.style = 'left: ' + min_x + 'px; top: '
          + min_y + 'px; width: '
          + (max_x - min_x) + 'px; height: '
        + (max_y - min_y) + 'px;';

      liveView.appendChild(highlighter);
      liveView.appendChild(p);
      children.push(highlighter);
      children.push(p);



    }
  });

  // for

  // shape [480, 640, 3]

  // model.detect(video).then(function (predictions) {
  //   // Remove any highlighting we did previous frame.
  //   for (let i = 0; i < children.length; i++) {
  //     liveView.removeChild(children[i]);
  //   }
  //   children.splice(0);

  //   // Now lets loop through predictions and draw them to the live view if
  //   // they have a high confidence score.
  //   for (let n = 0; n < predictions.length; n++) {
  //     // If we are over 66% sure we are sure we classified it right, draw it!
  //     if (predictions[n].score > 0.66) {
  //       const p = document.createElement('p');
  //       p.innerText = predictions[n].class + ' - with '
  //         + Math.round(parseFloat(predictions[n].score) * 100)
  //         + '% confidence.';
  //       p.style = 'margin-left: ' + predictions[n].bbox[0] + 'px; margin-top: '
  //         + (predictions[n].bbox[1] - 10) + 'px; width: '
  //         + (predictions[n].bbox[2] - 10) + 'px; top: 0; left: 0;';

  //       const highlighter = document.createElement('div');
  //       highlighter.setAttribute('class', 'highlighter');
  //       highlighter.style = 'left: ' + predictions[n].bbox[0] + 'px; top: '
  //         + predictions[n].bbox[1] + 'px; width: '
  //         + predictions[n].bbox[2] + 'px; height: '
  //         + predictions[n].bbox[3] + 'px;';

  //       liveView.appendChild(highlighter);
  //       liveView.appendChild(p);
  //       children.push(highlighter);
  //       children.push(p);
  //     }
    // }

  //   // Call this function again to keep predicting when the browser is ready.
    window.requestAnimationFrame(predictWebcam);
  // });
}

// Store the resulting model in the global scope of our app.
var model = undefined;
const handDetectModel = handPoseDetection.SupportedModels.MediaPipeHands;
const detectConfig = {
  runtime: 'tfjs',
  modelType: 'lite'
}
var detector = undefined;


handPoseDetection.createDetector(handDetectModel, detectConfig).then(function (det) {
  detector = det;
})

// Before we can use COCO-SSD class we must wait for it to finish
// loading. Machine Learning models can be large and take a moment
// to get everything needed to run.
// Note: cocoSsd is an external object loaded from our index.html
// script tag import so ignore any warning in Glitch.

// cocoSsd.load().then(function (loadedModel) {
//   model = loadedModel;
//   // Show demo section now model is ready to use.
//   demosSection.classList.remove('invisible');
// });
async function runModel() {
  model = await tf.loadLayersModel('saved_model/model.json');
  demosSection.classList.remove('invisible');
}

runModel()
