const express = require('express');
const faceapi = require('@vladmandic/face-api');
const axios = require('axios');
const tf = require('@tensorflow/tfjs-node');
const cors = require('cors');
const path = require('path');
const { createCanvas, loadImage } = require('canvas');

const app = express();
app.use(cors());
app.use(express.json({ limit: '50mb' }));

// Configure face-api to use canvas
const { Canvas, Image, ImageData } = require('canvas');
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

// Load models once at startup
async function loadModels() {
  const modelPath = path.join(__dirname, 'models');
  
  try {
    await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);
    await faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath);
    await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath);
    console.log('Models loaded successfully from disk');
  } catch (error) {
    console.error('Loading from disk failed, trying URL:', error.message);
    const MODEL_URL = 'https://raw.githubusercontent.com/vladmandic/face-api/master/model';
    await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
    await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
    await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
    console.log('Models loaded from URL');
  }
}

// Helper function to clean up resources
function cleanupTensors() {
  const numTensors = tf.memory().numTensors;
  if (numTensors > 100) {
    console.log(`Cleaning up ${numTensors} tensors`);
    tf.disposeVariables();
    if (global.gc) {
      global.gc();
    }
  }
}

// Health check
app.get('/', (req, res) => {
  const memory = tf.memory();
  res.json({ 
    status: 'Face API service is running',
    memory: {
      numTensors: memory.numTensors,
      numBytes: memory.numBytes
    }
  });
});

app.post('/detect', async (req, res) => {
  let tempCanvas = null;
  let img = null;
  
  try {
    const { image_url, return_all_faces = false } = req.body;
    
    console.log(`\n=== Processing image: ${image_url} ===`);
    console.log('Memory before:', tf.memory().numTensors, 'tensors');
    
    // Download image with timeout and no-cache
    const response = await axios.get(image_url, { 
      responseType: 'arraybuffer',
      timeout: 15000,
      headers: {
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
        'Expires': '0'
      },
      // Add timestamp to URL to prevent caching
      params: {
        t: Date.now()
      }
    });
    
    const buffer = Buffer.from(response.data);
    console.log('Image downloaded, size:', buffer.length, 'bytes');
    
    // Create fresh canvas for each request
    tempCanvas = createCanvas(1, 1);
    const ctx = tempCanvas.getContext('2d');
    
    // Load image
    img = await loadImage(buffer);
    console.log('Image loaded, dimensions:', img.width, 'x', img.height);
    
    // Set canvas size and draw image
    tempCanvas.width = img.width;
    tempCanvas.height = img.height;
    ctx.drawImage(img, 0, 0);
    
    // Detect faces with tf.tidy for automatic cleanup
    let detections = await tf.tidy(() => {
      return faceapi
        .detectAllFaces(tempCanvas, new faceapi.SsdMobilenetv1Options({ 
          minConfidence: 0.3,  // Lower threshold for better detection
          maxResults: return_all_faces ? 100 : 10
        }))
        .withFaceLandmarks()
        .withFaceDescriptors();
    });
    
    console.log('Initial detections found:', detections.length);
    
    // If no faces found, try with different settings
    if (detections.length === 0) {
      console.log('No faces found, trying with TinyFaceDetector...');
      
      detections = await tf.tidy(() => {
        return faceapi
          .detectAllFaces(tempCanvas, new faceapi.TinyFaceDetectorOptions({
            inputSize: 416,
            scoreThreshold: 0.3
          }))
          .withFaceLandmarks()
          .withFaceDescriptors();
      });
      
      console.log('TinyFaceDetector found:', detections.length);
    }
    
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
      index: index,
      landmarks: d.landmarks ? d.landmarks.positions.length : 0
    }));
    
    // Clear canvas
    ctx.clearRect(0, 0, tempCanvas.width, tempCanvas.height);
    
    // Cleanup tensors
    cleanupTensors();
    
    console.log('Memory after:', tf.memory().numTensors, 'tensors');
    console.log(`=== Completed: ${faces.length} faces found ===\n`);
    
    res.json({ 
      faces,
      debug: {
        imageSize: `${img.width}x${img.height}`,
        tensorsAfter: tf.memory().numTensors
      }
    });
    
  } catch (error) {
    console.error('Error detecting faces:', error);
    res.status(500).json({ 
      error: error.message,
      stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
    });
  } finally {
    // Ensure cleanup happens
    if (tempCanvas) {
      const ctx = tempCanvas.getContext('2d');
      ctx.clearRect(0, 0, tempCanvas.width, tempCanvas.height);
      tempCanvas = null;
    }
    img = null;
    
    // Force cleanup if too many tensors
    cleanupTensors();
  }
});

// Compare endpoint
app.post('/compare', async (req, res) => {
  try {
    const { embedding1, embedding2 } = req.body;
    
    if (!embedding1 || !embedding2) {
      return res.status(400).json({ error: 'Both embeddings are required' });
    }
    
    const distance = faceapi.euclideanDistance(embedding1, embedding2);
    const similarity = Math.max(0, 1 - distance);
    
    res.json({ 
      distance,
      similarity,
      is_match: distance < 0.4,
      confidence: similarity > 0.6 ? 'high' : similarity > 0.4 ? 'medium' : 'low'
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Cleanup endpoint for manual tensor cleanup
app.post('/cleanup', (req, res) => {
  const before = tf.memory();
  tf.disposeVariables();
  if (global.gc) {
    global.gc();
  }
  const after = tf.memory();
  
  res.json({
    message: 'Cleanup completed',
    before: before.numTensors,
    after: after.numTensors,
    freed: before.numTensors - after.numTensors
  });
});

const PORT = process.env.PORT || 5000;

// Start server
loadModels().then(() => {
  app.listen(PORT, () => {
    console.log(`Face API service running on port ${PORT}`);
    console.log('Initial memory:', tf.memory());
    
    // Periodic cleanup every 5 minutes
    setInterval(() => {
      console.log('Running periodic cleanup...');
      cleanupTensors();
    }, 5 * 60 * 1000);
  });
}).catch(error => {
  console.error('Failed to start:', error);
  process.exit(1);
});