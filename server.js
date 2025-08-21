const express = require('express');
const faceapi = require('@vladmandic/face-api');
const axios = require('axios');
const tf = require('@tensorflow/tfjs-node');
const cors = require('cors');
const path = require('path');
const canvas = require('canvas');

const app = express();
app.use(cors());
app.use(express.json());

// Configure face-api to use canvas
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

// Load models
async function loadModels() {
  const modelPath = path.join(__dirname, 'models');
  
  try {
    await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);
    await faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath);
    await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath);
    console.log('Models loaded successfully');
  } catch (error) {
    console.error('Error loading models:', error);
    const MODEL_URL = 'https://raw.githubusercontent.com/vladmandic/face-api/master/model';
    await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
    await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
    await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
    console.log('Models loaded from URL');
  }
}

// Health check
app.get('/', (req, res) => {
  res.json({ status: 'Face API service is running' });
});

app.post('/detect', async (req, res) => {
  try {
    const { image_url } = req.body;
    console.log('Processing image:', image_url);
    
    // Download image
    const response = await axios.get(image_url, { responseType: 'arraybuffer' });
    const buffer = Buffer.from(response.data);
    
    // Load image using canvas
    const img = await canvas.loadImage(buffer);
    
    // Detect faces
    const detections = await faceapi
      .detectAllFaces(img)
      .withFaceLandmarks()
      .withFaceDescriptors();
    
    console.log('Detections found:', detections.length);
    
    // Format response
    const faces = detections.map(d => ({
      embedding: Array.from(d.descriptor),
      area: {
        x: Math.round(d.detection.box.x),
        y: Math.round(d.detection.box.y),
        w: Math.round(d.detection.box.width),
        h: Math.round(d.detection.box.height)
      }
    }));
    
    res.json({ faces });
  } catch (error) {
    console.error('Error detecting faces:', error);
    res.status(500).json({ error: error.message });
  }
});

const PORT = process.env.PORT || 5000;
loadModels().then(() => {
  app.listen(PORT, () => {
    console.log(`Face API service running on port ${PORT}`);
  });
}).catch(error => {
  console.error('Failed to start:', error);
});