// DOM Elements
const webcamElement = document.getElementById('webcam');
const guideBoxElement = document.getElementById('guide-box');
const captureBtn = document.getElementById('capture-btn');
const switchCameraBtn = document.getElementById('switch-camera');
const resultPanel = document.querySelector('.result-panel');
const closeResultBtn = document.getElementById('close-result');
const tryAgainBtn = document.getElementById('try-again');
const capturedImage = document.getElementById('captured-image');
const processedImage = document.getElementById('processed-image');
const predictedLetter = document.getElementById('predicted-letter');
const confidenceBar = document.querySelector('.confidence-bar');
const confidenceText = document.getElementById('confidence-text');
const detectionStatus = document.querySelector('.detection-status');
const loadingOverlay = document.getElementById('loading-overlay');
const statusMessage = document.getElementById('status-message');

// Global variables
let stream;
let facingMode = 'environment'; // Default to back camera if available
let videoConstraints;
let configData = {};

// Fetch configuration data from the server
async function fetchConfig() {
    try {
        const response = await fetch('/get_config');
        configData = await response.json();
        
        // Note: Guide box positioning is now handled in DOMContentLoaded
        // using data attributes instead of directly here
    } catch (error) {
        console.error('Error fetching config:', error);
    }
}

// Initialize webcam
async function initializeCamera() {
    try {
        // Stop any existing streams
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
        
        // Set video constraints based on facing mode
        videoConstraints = {
            width: { ideal: configData.image_width || 640 },
            height: { ideal: configData.image_height || 480 },
            facingMode: facingMode
        };
        
        // Request camera access
        stream = await navigator.mediaDevices.getUserMedia({
            video: videoConstraints,
            audio: false
        });
        
        // Set stream to video element
        webcamElement.srcObject = stream;
        
        // Update status message
        statusMessage.textContent = 'Place your hand inside the guide box and keep it steady';
        
        // Enable capture button once camera is ready
        captureBtn.disabled = false;
    } catch (error) {
        console.error('Error accessing the camera:', error);
        statusMessage.textContent = 'Error accessing camera. Please check permissions.';
    }
}

// Switch between front and back cameras
function switchCamera() {
    facingMode = facingMode === 'environment' ? 'user' : 'environment';
    initializeCamera();
}

// Capture image from webcam
function captureImage() {
    return new Promise((resolve) => {
        const canvas = document.createElement('canvas');
        canvas.width = webcamElement.videoWidth;
        canvas.height = webcamElement.videoHeight;
        const ctx = canvas.getContext('2d');
        
        // Draw the current video frame to canvas
        ctx.drawImage(webcamElement, 0, 0, canvas.width, canvas.height);
        
        // Draw guide box on the canvas for reference
        const gbLeft = parseInt(guideBoxElement.style.left);
        const gbTop = parseInt(guideBoxElement.style.top);
        const gbWidth = parseInt(guideBoxElement.style.width);
        const gbHeight = parseInt(guideBoxElement.style.height);
        
        // Convert to data URL and resolve
        const dataUrl = canvas.toDataURL('image/jpeg');
        resolve(dataUrl);
    });
}

// Process captured image (dummy implementation for frontend demo)
async function processCapturedImage(dataUrl) {
    // Show loading overlay
    loadingOverlay.style.display = 'flex';
    
    try {
        // In a real implementation, you would send the image to the server for processing
        // For demo purposes, we'll use a dummy API call
        const response = await fetch('/dummy_predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: dataUrl
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        // For demo, we'll just return some dummy data
        return {
            processedImageUrl: dataUrl, // In real app, this would be returned from backend
            predictedClass: result.predicted_class,
            confidence: result.confidence,
            handDetected: result.hand_detected
        };
    } catch (error) {
        console.error('Error processing image:', error);
        return {
            error: 'Failed to process image',
            handDetected: false
        };
    } finally {
        // Hide loading overlay
        setTimeout(() => {
            loadingOverlay.style.display = 'none';
        }, 1000); // Simulate processing delay
    }
}

// Update the result panel with prediction results
function updateResultPanel(data) {
    if (data.error) {
        // Handle error case
        detectionStatus.className = 'detection-status error';
        detectionStatus.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${data.error}`;
        return;
    }
    
    // Display captured image
    capturedImage.src = data.processedImageUrl;
    
    // If we have a processed image from the backend, show it
    if (data.processedImageUrl) {
        processedImage.src = data.processedImageUrl;
        document.getElementById('processed-image-container').style.display = 'block';
    } else {
        document.getElementById('processed-image-container').style.display = 'none';
    }
    
    // Update detection status based on hand detection result
    if (data.handDetected) {
        detectionStatus.className = 'detection-status success';
        detectionStatus.innerHTML = '<i class="fas fa-check-circle"></i> Hand successfully detected';
        
        // Convert class number to letter or number using CLASS_MAPPING
        // For now, just use A as dummy data
        predictedLetter.textContent = 'A';
        
        // Update confidence bar
        const confidence = data.confidence * 100;
        confidenceBar.style.width = `${confidence}%`;
        confidenceText.textContent = `${confidence.toFixed(0)}% Confidence`;
    } else {
        detectionStatus.className = 'detection-status warning';
        detectionStatus.innerHTML = '<i class="fas fa-exclamation-triangle"></i> No hand detected, please try again';
        predictedLetter.textContent = '?';
        confidenceBar.style.width = '0%';
        confidenceText.textContent = 'No prediction';
    }
    
    // Show result panel
    resultPanel.style.display = 'block';
}

// Event Handlers
captureBtn.addEventListener('click', async () => {
    captureBtn.disabled = true;
    statusMessage.textContent = 'Capturing...';
    
    try {
        const imageData = await captureImage();
        const result = await processCapturedImage(imageData);
        updateResultPanel({
            ...result,
            processedImageUrl: imageData // In real app, this would come from backend
        });
    } catch (error) {
        console.error('Error during capture process:', error);
        updateResultPanel({
            error: 'Failed to process image',
            handDetected: false
        });
    } finally {
        captureBtn.disabled = false;
        statusMessage.textContent = 'Place your hand inside the guide box and keep it steady';
    }
});

switchCameraBtn.addEventListener('click', () => {
    switchCamera();
});

closeResultBtn.addEventListener('click', () => {
    resultPanel.style.display = 'none';
});

tryAgainBtn.addEventListener('click', () => {
    resultPanel.style.display = 'none';
});

// Initialize the app
document.addEventListener('DOMContentLoaded', async () => {
    // Fetch config first
    await fetchConfig();
    
    // Set guide box position from data attributes
    const guideBox = document.getElementById('guide-box');
    const x = guideBox.getAttribute('data-x');
    const y = guideBox.getAttribute('data-y');
    const width = guideBox.getAttribute('data-width');
    const height = guideBox.getAttribute('data-height');
    
    guideBox.style.left = `${x}px`;
    guideBox.style.top = `${y}px`;
    guideBox.style.width = `${width}px`;
    guideBox.style.height = `${height}px`;
    
    // Then initialize camera
    await initializeCamera();
    
    // Check if camera is available
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        statusMessage.textContent = 'Camera access not supported in your browser';
        captureBtn.disabled = true;
        return;
    }
});

// Window resize handler to adjust guide box positioning if needed
window.addEventListener('resize', () => {
    // You might need to adjust guide box position on window resize
    // This depends on your layout requirements
});

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
});