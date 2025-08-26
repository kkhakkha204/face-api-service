const express = require('express');
const faceapi = require('@vladmandic/face-api');
const axios = require('axios');
const tf = require('@tensorflow/tfjs-node');
const cors = require('cors');
const path = require('path');
const { createCanvas, loadImage } = require('canvas');
const sharp = require('sharp'); // Thêm sharp để resize ảnh

const app = express();
app.use(cors());
app.use(express.json({ limit: '50mb' }));

const { Canvas, Image, ImageData } = require('canvas');
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

// Configuration
const MAX_IMAGE_SIZE = 1920; // Max width/height để resize
const BATCH_SIZE = 3; // Số lượng face detection đồng thời
const MEMORY_THRESHOLD = 50; // Cleanup khi vượt ngưỡng tensors

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

function cleanupTensors(force = false) {
  const memory = tf.memory();
  if (force || memory.numTensors > MEMORY_THRESHOLD) {
    console.log(`Cleaning up ${memory.numTensors} tensors (force=${force})`);
    tf.disposeVariables();
    if (global.gc) {
      global.gc();
    }
  }
}

// Resize ảnh để tránh out of memory
async function resizeImage(buffer, maxSize = MAX_IMAGE_SIZE) {
  try {
    const metadata = await sharp(buffer).metadata();
    console.log(`Original image: ${metadata.width}x${metadata.height}`);
    
    if (metadata.width > maxSize || metadata.height > maxSize) {
      const resized = await sharp(buffer)
        .resize(maxSize, maxSize, {
          fit: 'inside',
          withoutEnlargement: true
        })
        .jpeg({ quality: 90 })
        .toBuffer();
      
      const newMetadata = await sharp(resized).metadata();
      console.log(`Resized to: ${newMetadata.width}x${newMetadata.height}`);
      return resized;
    }
    
    return buffer;
  } catch (error) {
    console.error('Error resizing image:', error);
    return buffer;
  }
}

// Queue để xử lý tuần tự, tránh overload
const detectionQueue = [];
let isProcessing = false;

async function processQueue() {
  if (isProcessing || detectionQueue.length === 0) return;
  
  isProcessing = true;
  const batch = detectionQueue.splice(0, BATCH_SIZE);
  
  for (const task of batch) {
    try {
      const result = await detectFacesInternal(task.imageUrl, task.returnAll);
      task.resolve(result);
    } catch (error) {
      task.reject(error);
    }
    // Cleanup sau mỗi detection
    cleanupTensors();
  }
  
  isProcessing = false;
  if (detectionQueue.length > 0) {
    setTimeout(processQueue, 100);
  }
}

async function detectFacesInternal(imageUrl, returnAllFaces) {
  let tempCanvas = null;
  let img = null;
  
  try {
    console.log(`\n=== Processing: ${imageUrl} ===`);
    console.log('Memory before:', tf.memory().numTensors, 'tensors');
    
    // Download image với timeout
    const response = await axios.get(imageUrl, { 
      responseType: 'arraybuffer',
      timeout: 20000,
      maxContentLength: 50 * 1024 * 1024, // Max 50MB
      headers: {
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache'
      }
    });
    
    let buffer = Buffer.from(response.data);
    console.log('Downloaded:', buffer.length, 'bytes');
    
    // Resize nếu ảnh quá lớn
    buffer = await resizeImage(buffer);
    
    // Load và process image
    tempCanvas = createCanvas(1, 1);
    const ctx = tempCanvas.getContext('2d');
    
    img = await loadImage(buffer);
    console.log('Processing:', img.width, 'x', img.height);
    
    tempCanvas.width = img.width;
    tempCanvas.height = img.height;
    ctx.drawImage(img, 0, 0);
    
    // Detection với multiple strategies
    let detections = [];
    
    // Strategy 1: SSD MobileNet với confidence thấp hơn
    detections = await tf.tidy(() => {
      return faceapi
        .detectAllFaces(tempCanvas, new faceapi.SsdMobilenetv1Options({ 
          minConfidence: 0.2, // Giảm threshold
          maxResults: returnAllFaces ? 50 : 10
        }))
        .withFaceLandmarks()
        .withFaceDescriptors();
    });
    
    console.log('SSD detections:', detections.length);
    
    // Strategy 2: Nếu không tìm thấy, thử TinyFaceDetector
    if (detections.length === 0) {
      console.log('Trying TinyFaceDetector...');
      
      // Cleanup trước khi thử strategy khác
      cleanupTensors(true);
      
      detections = await tf.tidy(() => {
        return faceapi
          .detectAllFaces(tempCanvas, new faceapi.TinyFaceDetectorOptions({
            inputSize: 512,
            scoreThreshold: 0.2
          }))
          .withFaceLandmarks()
          .withFaceDescriptors();
      });
      
      console.log('Tiny detections:', detections.length);
    }
    
    // Strategy 3: Nếu vẫn không có, thử với ảnh enhanced
    if (detections.length === 0) {
      console.log('Trying with enhanced image...');
      
      // Enhance contrast/brightness
      ctx.filter = 'contrast(1.2) brightness(1.1)';
      ctx.drawImage(img, 0, 0);
      
      detections = await tf.tidy(() => {
        return faceapi
          .detectAllFaces(tempCanvas, new faceapi.SsdMobilenetv1Options({ 
            minConfidence: 0.15,
            maxResults: 20
          }))
          .withFaceLandmarks()
          .withFaceDescriptors();
      });
      
      console.log('Enhanced detections:', detections.length);
    }
    
    // Convert results
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
    
    // Cleanup
    ctx.clearRect(0, 0, tempCanvas.width, tempCanvas.height);
    tempCanvas = null;
    img = null;
    
    console.log(`=== Completed: ${faces.length} faces ===\n`);
    
    return { faces };
    
  } catch (error) {
    console.error('Detection error:', error.message);
    throw error;
  } finally {
    // Ensure cleanup
    if (tempCanvas) {
      const ctx = tempCanvas.getContext('2d');
      ctx.clearRect(0, 0, tempCanvas.width, tempCanvas.height);
    }
    tempCanvas = null;
    img = null;
    buffer = null;
    
    // Force cleanup after each detection
    cleanupTensors(true);
  }
}

// API Endpoints
app.get('/', (req, res) => {
  const memory = tf.memory();
  res.json({ 
    status: 'Face API service running',
    memory: {
      numTensors: memory.numTensors,
      numBytes: Math.round(memory.numBytes / 1024 / 1024) + 'MB'
    },
    queueLength: detectionQueue.length,
    config: {
      maxImageSize: MAX_IMAGE_SIZE,
      batchSize: BATCH_SIZE
    }
  });
});

app.post('/detect', async (req, res) => {
  try {
    const { image_url, return_all_faces = false } = req.body;
    
    // Add to queue
    const result = await new Promise((resolve, reject) => {
      detectionQueue.push({
        imageUrl: image_url,
        returnAll: return_all_faces,
        resolve,
        reject
      });
      processQueue();
    });
    
    res.json(result);
    
  } catch (error) {
    console.error('API error:', error);
    res.status(500).json({ 
      error: error.message,
      faces: [] // Return empty array để không break flow
    });
  }
});

app.post('/compare', async (req, res) => {
  try {
    const { embedding1, embedding2 } = req.body;
    
    if (!embedding1 || !embedding2) {
      return res.status(400).json({ error: 'Both embeddings required' });
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

app.post('/cleanup', (req, res) => {
  const before = tf.memory();
  cleanupTensors(true);
  const after = tf.memory();
  
  res.json({
    message: 'Cleanup completed',
    before: before.numTensors,
    after: after.numTensors,
    freed: before.numTensors - after.numTensors
  });
});

// Health check endpoint
app.get('/health', async (req, res) => {
  const memory = tf.memory();
  const healthy = memory.numTensors < 500 && !isProcessing;
  
  res.status(healthy ? 200 : 503).json({
    status: healthy ? 'healthy' : 'degraded',
    memory: {
      tensors: memory.numTensors,
      bytes: Math.round(memory.numBytes / 1024 / 1024) + 'MB'
    },
    queueLength: detectionQueue.length,
    isProcessing
  });
});

const PORT = process.env.PORT || 5000;

// Start server
loadModels().then(() => {
  app.listen(PORT, () => {
    console.log(`Face API service running on port ${PORT}`);
    console.log('Initial memory:', tf.memory());
    
    // Periodic cleanup
    setInterval(() => {
      const memory = tf.memory();
      if (memory.numTensors > 100) {
        console.log('Periodic cleanup...');
        cleanupTensors(true);
      }
    }, 60000); // Every minute
    
    // Force GC every 5 minutes
    setInterval(() => {
      if (global.gc) {
        console.log('Force garbage collection...');
        global.gc();
      }
    }, 5 * 60000);
  });
}).catch(error => {
  console.error('Failed to start:', error);
  process.exit(1);
});