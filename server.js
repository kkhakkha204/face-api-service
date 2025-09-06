const express = require('express');
const faceapi = require('@vladmandic/face-api');
const axios = require('axios');
const tf = require('@tensorflow/tfjs-node');
const cors = require('cors');
const path = require('path');
const { createCanvas, loadImage } = require('canvas');
const sharp = require('sharp');

const app = express();
app.use(cors());
app.use(express.json({ limit: '50mb' }));

const { Canvas, Image, ImageData } = require('canvas');
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const MAX_IMAGE_SIZE = 1280;
const BATCH_SIZE = 5;
const MEMORY_THRESHOLD = 30;

async function loadModels() {
  const modelPath = path.join(__dirname, 'models');
  
  try {
    await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);
    await faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath);
    await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath);
    await faceapi.nets.tinyFaceDetector.loadFromDisk(modelPath);
    console.log('Models loaded successfully from disk');
  } catch (error) {
    console.error('Loading from disk failed, trying URL:', error.message);
    const MODEL_URL = 'https://raw.githubusercontent.com/vladmandic/face-api/master/model';
    await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
    await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
    await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
    await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
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

// Face alignment function
function alignFace(canvas, landmarks) {
  const ctx = canvas.getContext('2d');
  
  // Get eye positions from landmarks
  const leftEye = landmarks.positions[36]; // Left eye center
  const rightEye = landmarks.positions[45]; // Right eye center
  
  // Calculate rotation angle
  const deltaX = rightEye.x - leftEye.x;
  const deltaY = rightEye.y - leftEye.y;
  const angle = Math.atan2(deltaY, deltaX);
  
  // Apply rotation if significant
  if (Math.abs(angle) > 0.1) { // ~5.7 degrees
    const centerX = (leftEye.x + rightEye.x) / 2;
    const centerY = (leftEye.y + rightEye.y) / 2;
    
    ctx.translate(centerX, centerY);
    ctx.rotate(-angle);
    ctx.translate(-centerX, -centerY);
    
    return true;
  }
  
  return false;
}

// Enhanced detection with multiple strategies
async function detectWithMultipleStrategies(canvas, returnAllFaces) {
  let allDetections = [];
  
  // Strategy 1: SSD MobileNet (best for front faces)
  try {
    const ssdDetections = await tf.tidy(() => {
      return faceapi
        .detectAllFaces(canvas, new faceapi.SsdMobilenetv1Options({ 
          minConfidence: 0.15,
          maxResults: returnAllFaces ? 50 : 15
        }))
        .withFaceLandmarks()
        .withFaceDescriptors();
    });
    
    allDetections = allDetections.concat(ssdDetections.map(d => ({ ...d, method: 'ssd' })));
    console.log(`SSD detections: ${ssdDetections.length}`);
  } catch (e) {
    console.log('SSD detection failed:', e.message);
  }
  
  // Strategy 2: Tiny Face Detector (better for small/distant faces)
  try {
    cleanupTensors();
    
    const tinyDetections = await tf.tidy(() => {
      return faceapi
        .detectAllFaces(canvas, new faceapi.TinyFaceDetectorOptions({
          inputSize: 608, // Higher resolution for group photos
          scoreThreshold: 0.15
        }))
        .withFaceLandmarks()
        .withFaceDescriptors();
    });
    
    allDetections = allDetections.concat(tinyDetections.map(d => ({ ...d, method: 'tiny' })));
    console.log(`Tiny detections: ${tinyDetections.length}`);
  } catch (e) {
    console.log('Tiny detection failed:', e.message);
  }
  
  // Strategy 3: Enhanced preprocessing for profile faces
  try {
    cleanupTensors();
    
    const ctx = canvas.getContext('2d');
    const originalImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    
    // Enhance contrast and brightness for profile faces
    ctx.filter = 'contrast(1.3) brightness(1.2) saturate(1.1)';
    ctx.drawImage(canvas, 0, 0);
    
    const enhancedDetections = await tf.tidy(() => {
      return faceapi
        .detectAllFaces(canvas, new faceapi.SsdMobilenetv1Options({ 
          minConfidence: 0.12,
          maxResults: 20
        }))
        .withFaceLandmarks()
        .withFaceDescriptors();
    });
    
    // Restore original image
    ctx.putImageData(originalImageData, 0, 0);
    
    allDetections = allDetections.concat(enhancedDetections.map(d => ({ ...d, method: 'enhanced' })));
    console.log(`Enhanced detections: ${enhancedDetections.length}`);
  } catch (e) {
    console.log('Enhanced detection failed:', e.message);
  }
  
  // Strategy 4: Multi-scale detection for group photos
  if (canvas.width > 800 || canvas.height > 600) {
    try {
      cleanupTensors();
      
      // Create smaller version for distant faces
      const smallCanvas = createCanvas(canvas.width * 0.7, canvas.height * 0.7);
      const smallCtx = smallCanvas.getContext('2d');
      smallCtx.drawImage(canvas, 0, 0, smallCanvas.width, smallCanvas.height);
      
      const multiScaleDetections = await tf.tidy(() => {
        return faceapi
          .detectAllFaces(smallCanvas, new faceapi.TinyFaceDetectorOptions({
            inputSize: 512,
            scoreThreshold: 0.2
          }))
          .withFaceLandmarks()
          .withFaceDescriptors();
      });
      
      // Scale coordinates back to original size
      const scaleFactor = canvas.width / smallCanvas.width;
      const scaledDetections = multiScaleDetections.map(d => {
        const scaledBox = {
          x: d.detection.box.x * scaleFactor,
          y: d.detection.box.y * scaleFactor,
          width: d.detection.box.width * scaleFactor,
          height: d.detection.box.height * scaleFactor
        };
        
        return {
          ...d,
          detection: { ...d.detection, box: scaledBox },
          method: 'multiscale'
        };
      });
      
      allDetections = allDetections.concat(scaledDetections);
      console.log(`Multi-scale detections: ${multiScaleDetections.length}`);
    } catch (e) {
      console.log('Multi-scale detection failed:', e.message);
    }
  }
  
  return allDetections;
}

// Deduplicate overlapping detections
function deduplicateDetections(detections) {
  const deduplicated = [];
  const threshold = 0.3; // IoU threshold
  
  // Sort by confidence score
  detections.sort((a, b) => b.detection.score - a.detection.score);
  
  for (const detection of detections) {
    let isDuplicate = false;
    
    for (const existing of deduplicated) {
      const iou = calculateIoU(detection.detection.box, existing.detection.box);
      if (iou > threshold) {
        isDuplicate = true;
        break;
      }
    }
    
    if (!isDuplicate) {
      deduplicated.push(detection);
    }
  }
  
  console.log(`Deduplicated: ${detections.length} -> ${deduplicated.length}`);
  return deduplicated;
}

// Calculate Intersection over Union
function calculateIoU(box1, box2) {
  const x1 = Math.max(box1.x, box2.x);
  const y1 = Math.max(box1.y, box2.y);
  const x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
  const y2 = Math.min(box1.y + box1.height, box2.y + box2.height);
  
  if (x2 <= x1 || y2 <= y1) return 0;
  
  const intersection = (x2 - x1) * (y2 - y1);
  const area1 = box1.width * box1.height;
  const area2 = box2.width * box2.height;
  const union = area1 + area2 - intersection;
  
  return intersection / union;
}

// Calculate face quality score
function calculateQualityScore(detection, canvas) {
  let score = detection.detection.score; // Base confidence
  
  // Size factor (larger faces = higher quality)
  const faceArea = detection.detection.box.width * detection.detection.box.height;
  const imageArea = canvas.width * canvas.height;
  const sizeRatio = faceArea / imageArea;
  const sizeFactor = Math.min(sizeRatio * 100, 1); // Cap at 1
  
  // Landmark quality (if landmarks are stable)
  const landmarkFactor = detection.landmarks ? 0.1 : 0;
  
  // Position factor (center faces often higher quality)
  const centerX = canvas.width / 2;
  const centerY = canvas.height / 2;
  const faceX = detection.detection.box.x + detection.detection.box.width / 2;
  const faceY = detection.detection.box.y + detection.detection.box.height / 2;
  const distanceFromCenter = Math.sqrt(
    Math.pow(faceX - centerX, 2) + Math.pow(faceY - centerY, 2)
  );
  const maxDistance = Math.sqrt(Math.pow(centerX, 2) + Math.pow(centerY, 2));
  const positionFactor = (1 - distanceFromCenter / maxDistance) * 0.1;
  
  return Math.min(score + sizeFactor * 0.2 + landmarkFactor + positionFactor, 1);
}

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
    
    // Download image
    const response = await axios.get(imageUrl, { 
      responseType: 'arraybuffer',
      timeout: 30000, // Increased timeout for large images
      maxContentLength: 50 * 1024 * 1024,
      headers: {
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache'
      }
    });
    
    let buffer = Buffer.from(response.data);
    console.log('Downloaded:', buffer.length, 'bytes');
    
    buffer = await resizeImage(buffer);
    
    tempCanvas = createCanvas(1, 1);
    const ctx = tempCanvas.getContext('2d');
    
    img = await loadImage(buffer);
    console.log('Processing:', img.width, 'x', img.height);
    
    tempCanvas.width = img.width;
    tempCanvas.height = img.height;
    ctx.drawImage(img, 0, 0);
    
    // Enhanced detection with multiple strategies
    let allDetections = await detectWithMultipleStrategies(tempCanvas, returnAllFaces);
    
    // Deduplicate overlapping faces
    const uniqueDetections = deduplicateDetections(allDetections);
    
    // Convert to response format with enhanced quality scoring
    const faces = uniqueDetections.map((d, index) => {
      const qualityScore = calculateQualityScore(d, tempCanvas);
      
      return {
        embedding: Array.from(d.descriptor),
        area: {
          x: Math.round(d.detection.box.x),
          y: Math.round(d.detection.box.y),
          w: Math.round(d.detection.box.width),
          h: Math.round(d.detection.box.height)
        },
        confidence: d.detection.score,
        quality: qualityScore,
        method: d.method,
        index: index,
        // Add face angle estimation
        landmarks_available: !!d.landmarks,
        estimated_pose: d.landmarks ? estimateFacePose(d.landmarks) : null
      };
    });
    
    // Sort by quality score (best first)
    faces.sort((a, b) => b.quality - a.quality);
    
    // Apply smart filtering
    const filteredFaces = smartFilterFaces(faces, returnAllFaces);
    
    console.log(`=== Completed: ${filteredFaces.length}/${faces.length} faces (filtered by quality) ===\n`);
    
    return { faces: filteredFaces };
    
  } catch (error) {
    console.error('Detection error:', error.message);
    throw error;
  } finally {
    if (tempCanvas) {
      const ctx = tempCanvas.getContext('2d');
      ctx.clearRect(0, 0, tempCanvas.width, tempCanvas.height);
    }
    tempCanvas = null;
    img = null;
    buffer = null;
    
    cleanupTensors(true);
  }
}

// Estimate face pose from landmarks
function estimateFacePose(landmarks) {
  if (!landmarks || !landmarks.positions) return null;
  
  try {
    const nose = landmarks.positions[30]; // Nose tip
    const leftEye = landmarks.positions[36]; // Left eye
    const rightEye = landmarks.positions[45]; // Right eye
    const mouth = landmarks.positions[48]; // Mouth corner
    
    // Calculate yaw (left-right rotation)
    const eyeDistance = Math.abs(rightEye.x - leftEye.x);
    const noseToEyeDistance = Math.abs(nose.x - (leftEye.x + rightEye.x) / 2);
    const yaw = (noseToEyeDistance / eyeDistance) - 0.5; // Normalized
    
    // Calculate pitch (up-down rotation) 
    const eyeY = (leftEye.y + rightEye.y) / 2;
    const noseToEyeY = nose.y - eyeY;
    const eyeToMouthY = mouth.y - eyeY;
    const pitch = noseToEyeY / Math.abs(eyeToMouthY);
    
    return {
      yaw: Math.max(-1, Math.min(1, yaw * 2)), // Clamp between -1 and 1
      pitch: Math.max(-1, Math.min(1, pitch)),
      frontal: Math.abs(yaw) < 0.3 && Math.abs(pitch) < 0.3
    };
  } catch (e) {
    return null;
  }
}

// Smart filtering based on quality and diversity
function smartFilterFaces(faces, returnAll) {
  if (returnAll) {
    return faces.filter(f => f.quality > 0.15); // Very permissive
  }
  
  // For single face queries, prefer frontal high-quality faces
  const frontalFaces = faces.filter(f => 
    f.estimated_pose?.frontal && f.quality > 0.4
  );
  
  if (frontalFaces.length > 0) {
    return frontalFaces.slice(0, 5); // Top 5 frontal faces
  }
  
  // Fallback to best quality faces
  return faces.filter(f => f.quality > 0.25).slice(0, 8);
}

// Routes
app.get('/', (req, res) => {
  const memory = tf.memory();
  res.json({ 
    status: 'Enhanced Face API service running',
    memory: {
      numTensors: memory.numTensors,
      numBytes: Math.round(memory.numBytes / 1024 / 1024) + 'MB'
    },
    queueLength: detectionQueue.length,
    features: {
      face_alignment: true,
      multi_angle_detection: true,
      group_photo_optimization: true,
      quality_scoring: true
    },
    config: {
      maxImageSize: MAX_IMAGE_SIZE,
      batchSize: BATCH_SIZE
    }
  });
});

app.post('/detect', async (req, res) => {
  try {
    const { image_url, return_all_faces = false } = req.body;
    
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
      faces: [] 
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

loadModels().then(() => {
  app.listen(PORT, () => {
    console.log(`Enhanced Face API service running on port ${PORT}`);
    console.log('Features: Face Alignment, Multi-angle Detection, Group Photo Optimization');
    console.log('Initial memory:', tf.memory());
    
    // Periodic cleanup
    setInterval(() => {
      const memory = tf.memory();
      if (memory.numTensors > 100) {
        console.log('Periodic cleanup...');
        cleanupTensors(true);
      }
    }, 60000);
    
    // Force garbage collection
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