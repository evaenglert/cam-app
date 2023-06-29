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
  if (!model | !labels) {
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

function predictWebcam() {
  // Now let's start classifying a frame in the stream.

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

      const canvas = document.createElement('canvas');
      var max_dist = Math.max(max_x - min_x, max_y - min_y) * 1.1;
      canvas.width = 400;
      canvas.height = 400;

      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, centre_x - max_dist / 2, centre_y - max_dist / 2, max_dist, max_dist, 0, 0, canvas.width, canvas.height);
      const croppedImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

      // Create a new two-dimensional array to store the RGB values
      const pixels = new Array(1);
      pixels[0] = new Array(canvas.height);
      for (let i = 0; i < canvas.height; i++) {
        pixels[0][i] = new Array(canvas.width);
      }

      // Iterate over the pixel data and extract the RGB values
      const data = croppedImageData.data;
      for (let i = 0; i < data.length; i += 4) {
        const x = (i / 4) % canvas.width;
        const y = Math.floor((i / 4) / canvas.width);
        pixels[0][y][x] = [data[i], data[i + 1], data[i + 2]];
      }

      var input = tf.tensor4d(pixels, [1, 400, 400, 3]);

      var predictions = model.predict(input).dataSync();

      var pred_idx = predictions.indexOf(Math.max(...predictions));

      p.innerText = labels[pred_idx] + ' - with '
        + Math.round(parseFloat(predictions[pred_idx]) * 100)
          + '% confidence.';
      p.style = 'margin-left: ' + min_x + 'px; margin-top: '
        + (max_y - 10) + 'px; width: '
        + ((max_x - min_x) - 10) + 'px; top: 0; left: 0;';

    }
  });
  //   // Call this function again to keep predicting when the browser is ready.
    window.requestAnimationFrame(predictWebcam);
  // });
}

// Store the resulting model in the global scope of our app.
var model = undefined;
var labels = undefined;

const handDetectModel = handPoseDetection.SupportedModels.MediaPipeHands;
const detectConfig = {
  runtime: 'tfjs',
  modelType: 'lite'
}
var detector = undefined;


handPoseDetection.createDetector(handDetectModel, detectConfig).then(function (det) {
  detector = det;
})

tf.loadLayersModel('mobile_saved_model/model.json').then(function (loadedModel) {
  model = loadedModel;

  fetch('mobile_saved_model/mobile_labels.json')
    .then(function (response) {return response.json()})
    .then(function (response) { labels = response});


  demosSection.classList.remove('invisible');
})
