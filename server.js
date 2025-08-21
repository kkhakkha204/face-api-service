const express = require('express');
const faceapi = require('@vladmandic/face-api');
const axios = require('axios');
const tf = require('@tensorflow/tfjs-node');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

// Load models khi start
async function loadModels() {
  await faceapi.nets.ssdMobilenetv1.loadFromDisk('./models');
  await faceapi.nets.faceLandmark68Net.loadFromDisk('./models');
  await faceapi.nets.faceRecognitionNet.loadFromDisk('./models');
  console.log('Models loaded');
}

app.post('/detect', async (req, res) => {
  try {
    const { image_url } = req.body;
    
    // Download image
    const response = await axios.get(image_url, { responseType: 'arraybuffer' });
    const buffer = Buffer.from(response.data);
    
    // Decode image
    const tensor = tf.node.decodeImage(buffer);
    
    // Detect faces
    const detections = await faceapi
      .detectAllFaces(tensor)
      .withFaceLandmarks()
      .withFaceDescriptors();
    
    // Format response
    const faces = detections.map(d => ({
      embedding: Array.from(d.descriptor),
      area: {
        x: d.detection.box.x,
        y: d.detection.box.y,
        w: d.detection.box.width,
        h: d.detection.box.height
      }
    }));
    
    tensor.dispose();
    
    res.json({ faces });
  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ error: error.message });
  }
});

const PORT = process.env.PORT || 5000;
loadModels().then(() => {
  app.listen(PORT, () => {
    console.log(`Face API service running on port ${PORT}`);
  });
});