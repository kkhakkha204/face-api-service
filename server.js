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
const MAX_IMAGE_SIZE = 1920; // Resize ảnh xuống max 1920px
const BATCH_SIZE = 1; // Xử lý từng ảnh một
const MEMORY_THRESHOLD = 50; // Giảm threshold để cleanup sớm hơn

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

// Aggressive memory cleanup
function cleanupTensors() {
  const memory = tf.memory();
  console.log(`Memory cleanup - Tensors: ${memory.numTensors}, Bytes: ${(memory.numBytes / 1048576).toFixed(2)}MB`);
  
  // Dispose all variables
  tf.disposeVariables();
  
  // Force garbage collection if available
  if (global.gc) {
    global.gc();
  }
  
  // Clear TensorFlow backend
  tf.engine().startScope();
  tf.engine().endScope();
  
  const afterMemory = tf.memory();
  console.log(`After cleanup - Tensors: ${afterMemory.numTensors}, Bytes: ${(afterMemory.numBytes / 1048576).toFixed(2)}MB`);
}

// Resize ảnh trước khi xử lý để giảm memory
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

app.get('/', (req, res) => {
  const memory = tf.memory();
  res.json({ 
    status: 'Face API service is running (optimized)',
    memory: {
      numTensors: memory.numTensors,
      numBytes: memory.numBytes,
      numBytesInMB: (memory.numBytes / 1048576).toFixed(2)
    }
  });
});

app.post('/detect', async (req, res) => {
  let tempCanvas = null;
  let img = null;
  let detections = [];
  
  try {
    const { image_url, return_all_faces = false } = req.body;
    
    console.log(`\n=== Processing image: ${image_url} ===`);
    
    // Check memory before processing
    const memoryBefore = tf.memory();
    if (memoryBefore.numTensors > MEMORY_THRESHOLD) {
      console.log('Pre-emptive memory cleanup...');
      cleanupTensors();
    }
    
    // Download image with timeout
    const response = await axios.get(image_url, { 
      responseType: 'arraybuffer',
      timeout: 15000,
      maxContentLength: 50 * 1024 * 1024, // Max 50MB
      headers: {
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache'
      }
    });
    
    let buffer = Buffer.from(response.data);
    console.log('Image downloaded, size:', (buffer.length / 1048576).toFixed(2), 'MB');
    
    // Resize image if too large
    buffer = await resizeImage(buffer);
    
    // Load and process image
    tempCanvas = createCanvas(1, 1);
    const ctx = tempCanvas.getContext('2d');
    
    img = await loadImage(buffer);
    console.log('Processing image:', img.width, 'x', img.height);
    
    tempCanvas.width = img.width;
    tempCanvas.height = img.height;
    ctx.drawImage(img, 0, 0);
    
    // Detect faces with TensorFlow scope management
    await tf.tidy(async () => {
      // Try SsdMobilenetv1 first
      detections = await faceapi
        .detectAllFaces(tempCanvas, new faceapi.SsdMobilenetv1Options({ 
          minConfidence: 0.4,
          maxResults: return_all_faces ? 50 : 10
        }))
        .withFaceLandmarks()
        .withFaceDescriptors();
      
      console.log('SsdMobilenetv1 detections:', detections.length);
      
      // If no faces found, try TinyFaceDetector
      if (detections.length === 0) {
        console.log('Trying TinyFaceDetector...');
        
        // Load TinyFaceDetector if not loaded
        if (!faceapi.nets.tinyFaceDetector.isLoaded) {
          const modelPath = path.join(__dirname, 'models');
          try {
            await faceapi.nets.tinyFaceDetector.loadFromDisk(modelPath);
          } catch {
            const MODEL_URL = 'https://raw.githubusercontent.com/vladmandic/face-api/master/model';
            await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
          }
        }
        
        detections = await faceapi
          .detectAllFaces(tempCanvas, new faceapi.TinyFaceDetectorOptions({
            inputSize: 416,
            scoreThreshold: 0.4
          }))
          .withFaceLandmarks()
          .withFaceDescriptors();
        
        console.log('TinyFaceDetector detections:', detections.length);
      }
    });
    
    // Convert detections to serializable format
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
    
    // Clean up canvas immediately
    if (tempCanvas) {
      ctx.clearRect(0, 0, tempCanvas.width, tempCanvas.height);
      tempCanvas.width = 1;
      tempCanvas.height = 1;
    }
    
    // Force cleanup
    cleanupTensors();
    
    console.log(`=== Completed: ${faces.length} faces found ===\n`);
    
    res.json({ 
      faces,
      debug: {
        imageSize: `${img.width}x${img.height}`,
        facesFound: faces.length,
        memoryUsed: (tf.memory().numBytes / 1048576).toFixed(2) + 'MB'
      }
    });
    
  } catch (error) {
    console.error('Error detecting faces:', error.message);
    res.status(500).json({ 
      error: error.message,
      type: error.name
    });
  } finally {
    // Ensure cleanup
    if (tempCanvas) {
      const ctx = tempCanvas.getContext('2d');
      ctx.clearRect(0, 0, tempCanvas.width, tempCanvas.height);
      tempCanvas = null;
    }
    img = null;
    detections = null;
    
    // Final cleanup
    cleanupTensors();
  }
});

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

app.post('/cleanup', (req, res) => {
  const before = tf.memory();
  cleanupTensors();
  const after = tf.memory();
  
  res.json({
    message: 'Cleanup completed',
    before: {
      tensors: before.numTensors,
      bytes: before.numBytes,
      mb: (before.numBytes / 1048576).toFixed(2)
    },
    after: {
      tensors: after.numTensors,
      bytes: after.numBytes,
      mb: (after.numBytes / 1048576).toFixed(2)
    },
    freed: {
      tensors: before.numTensors - after.numTensors,
      mb: ((before.numBytes - after.numBytes) / 1048576).toFixed(2)
    }
  });
});

// Health check endpoint
app.get('/health', (req, res) => {
  const memory = tf.memory();
  const memoryUsage = process.memoryUsage();
  
  res.json({
    status: 'healthy',
    tensorflow: {
      tensors: memory.numTensors,
      memoryMB: (memory.numBytes / 1048576).toFixed(2)
    },
    process: {
      heapUsedMB: (memoryUsage.heapUsed / 1048576).toFixed(2),
      heapTotalMB: (memoryUsage.heapTotal / 1048576).toFixed(2),
      rssMB: (memoryUsage.rss / 1048576).toFixed(2)
    }
  });
});

const PORT = process.env.PORT || 5000;

// Start server
loadModels().then(() => {
  app.listen(PORT, () => {
    console.log(`Face API service (optimized) running on port ${PORT}`);
    console.log('Initial memory:', tf.memory());
    
    // Periodic cleanup every 2 minutes
    setInterval(() => {
      const memory = tf.memory();
      if (memory.numTensors > MEMORY_THRESHOLD) {
        console.log('Periodic cleanup triggered...');
        cleanupTensors();
      }
    }, 2 * 60 * 1000);
    
    // Monitor memory usage
    setInterval(() => {
      const memoryUsage = process.memoryUsage();
      console.log(`Process Memory - Heap: ${(memoryUsage.heapUsed / 1048576).toFixed(2)}MB, RSS: ${(memoryUsage.rss / 1048576).toFixed(2)}MB`);
      
      // Restart if memory usage is too high (> 1GB)
      if (memoryUsage.rss > 1024 * 1024 * 1024) {
        console.error('Memory usage too high, consider restarting...');
      }
    }, 30 * 1000);
  });
}).catch(error => {
  console.error('Failed to start:', error);
  process.exit(1);
});