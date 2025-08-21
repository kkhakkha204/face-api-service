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
    const { image_url, return_all_faces = false } = req.body;
    console.log('Processing image:', image_url);
    
    // Download image
    const response = await axios.get(image_url, { responseType: 'arraybuffer' });
    const buffer = Buffer.from(response.data);
    
    // Load image using canvas
    const img = await canvas.loadImage(buffer);
    
    // Detect faces với nhiều options
    const detections = await faceapi
      .detectAllFaces(img, new faceapi.SsdMobilenetv1Options({ 
        minConfidence: 0.5,
        maxResults: return_all_faces ? 100 : 1
      }))
      .withFaceLandmarks()
      .withFaceDescriptors();
    
    console.log('Detections found:', detections.length);
    
    // Format response
    const faces = detections.map((d, index) => ({
      embedding: Array.from(d.descriptor),
      area: {
        x: Math.round(d.detection.box.x),
        y: Math.round(d.detection.box.y),
        w: Math.round(d.detection.box.width),
        h: Math.round(d.detection.box.height)
      },
      confidence: d.detection.score,
      index: index
    }));
    
    res.json({ faces });
  } catch (error) {
    console.error('Error detecting faces:', error);
    res.status(500).json({ error: error.message });
  }
});

// New endpoint for comparing faces
app.post('/compare', async (req, res) => {
  try {
    const { embedding1, embedding2 } = req.body;
    
    const distance = faceapi.euclideanDistance(embedding1, embedding2);
    const similarity = Math.max(0, 1 - distance);
    
    res.json({ 
      distance,
      similarity,
      is_match: distance < 0.4
    });
  } catch (error) {
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