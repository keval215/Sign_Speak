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
// Process captured image with real prediction endpoint
async function processCapturedImage(dataUrl) {
    // Show loading overlay
    loadingOverlay.style.display = 'flex';
    statusMessage.textContent = 'Processing image...';
    
    try {
        // Send the image to the server for processing
        const response = await fetch('/predict', {
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
        
        // Return processed data
        return {
            processedImageUrl: result.processed_image_url,
            visualizationUrl: result.visualization_url,
            annotatedImageUrl: result.annotated_image_url,
            predictedClass: result.predicted_class,
            classLabel: result.class_label,
            confidence: result.confidence,
            handDetected: result.hand_detected,
            error: result.error
        };
    } catch (error) {
        console.error('Error processing image:', error);
        return {
            error: 'Failed to process image: ' + error.message,
            handDetected: false
        };
    } finally {
        // Hide loading overlay
        loadingOverlay.style.display = 'none';
        statusMessage.textContent = 'Place your hand inside the guide box and keep it steady';
    }
}

// Function to scroll to the result panel
function scrollToResultPanel() {
    resultPanel.scrollIntoView({ behavior: 'smooth' });
}

// Call scrollToResultPanel when the result panel is updated
function updateResultPanel(data) {
    if (data.error) {
        // Handle error case
        detectionStatus.className = 'detection-status error';
        detectionStatus.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${data.error}`;
        
        // If there's an input image despite the error, show it
        if (data.processedImageUrl) {
            capturedImage.src = data.processedImageUrl;
        }
        
        // Clear other elements
        processedImage.src = '';
        predictedLetter.textContent = '?';
        confidenceBar.style.width = '0%';
        confidenceText.textContent = 'No prediction';
        
        // Show result panel even with error
        resultPanel.style.display = 'block';
        scrollToResultPanel();  // Scroll to the result panel
        return;
    }
    
    // Display captured image (use annotated image if available)
    if (data.annotatedImageUrl) {
        capturedImage.src = data.annotatedImageUrl;
    } else {
        capturedImage.src = data.processedImageUrl || '';
    }
    
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
        
        // Use the class label from the server if available
        if (data.classLabel) {
            predictedLetter.textContent = data.classLabel;
        } else {
            // Fallback to the class index
            predictedLetter.textContent = `Class ${data.predictedClass}`;
        }
        
        // Update confidence bar
        const confidence = data.confidence * 100;
        confidenceBar.style.width = `${confidence}%`;
        confidenceText.textContent = `${confidence.toFixed(0)}% Confidence`;
        
        // If we have a visualization image, show it in a new container
        if (data.visualizationUrl) {
            let visContainer = document.getElementById('visualization-container');
            if (!visContainer) {
                visContainer = document.createElement('div');
                visContainer.id = 'visualization-container';
                visContainer.className = 'image-container';
                visContainer.innerHTML = `  
                    <h3>Hand Detection Visualization</h3>
                    <img id="visualization-image" alt="Visualization" class="result-image">
                `;
                document.querySelector('.result-images').appendChild(visContainer);
            }
            
            // Set the visualization image
            document.getElementById('visualization-image').src = data.visualizationUrl;
            visContainer.style.display = 'block';
        }
    } else {
        detectionStatus.className = 'detection-status warning';
        detectionStatus.innerHTML = '<i class="fas fa-exclamation-triangle"></i> No hand detected, please try again';
        predictedLetter.textContent = '?';
        confidenceBar.style.width = '0%';
        confidenceText.textContent = 'No prediction';
    }
    
    // Show result panel
    resultPanel.style.display = 'block';
    scrollToResultPanel();  // Scroll to the result panel
}


// Event Handlers
captureBtn.addEventListener('click', async () => {
    captureBtn.disabled = true;
    statusMessage.textContent = 'Capturing...';
    
    try {
        const imageData = await captureImage();
        const result = await processCapturedImage(imageData);
        updateResultPanel(result);
    } catch (error) {
        console.error('Error during capture process:', error);
        updateResultPanel({
            error: 'Failed to process image: ' + error.message,
            handDetected: false
        });
    } finally {
        captureBtn.disabled = false;
        statusMessage.textContent = 'Place your hand inside the guide box and keep it steady';
    }
});

// Add a function to fetch class mapping from the server
async function fetchClassMapping() {
    try {
        const response = await fetch('/get_class_mapping');
        const mapping = await response.json();
        
        // Store mapping globally for later use
        window.classMapping = mapping;
        console.log('Class mapping loaded:', mapping);
    } catch (error) {
        console.error('Error fetching class mapping:', error);
    }
}

// Initialize the app - update to include class mapping fetch
document.addEventListener('DOMContentLoaded', async () => {
    // Fetch config first
    await fetchConfig();
    
    // Fetch class mapping
    await fetchClassMapping();
    
    // Set guide box position from data attributes
    const guideBox = document.getElementById('guide-box');
    const x = guideBox.getAttribute('data-x') || configData.guide_box_x || 100;
    const y = guideBox.getAttribute('data-y') || configData.guide_box_y || 100;
    const width = guideBox.getAttribute('data-width') || configData.guide_box_w || 300;
    const height = guideBox.getAttribute('data-height') || configData.guide_box_h || 300;
    
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

}

);

// Function to generate random matrix text
function generateRandomMatrixText() {
    const characters = '01ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'; // You can add more characters
    let randomText = '';
    for (let i = 0; i < 50; i++) { // Adjust the length of each line for randomness
        randomText += characters.charAt(Math.floor(Math.random() * characters.length));
    }
    return randomText;
}

// Dynamically add the random text to the background
const matrixBackground = document.querySelector('body::before');
matrixBackground.textContent = generateRandomMatrixText();

