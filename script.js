let model = null;
let webcamElement = document.getElementById('webcam');
let canvasElement = document.getElementById('output');
let ctx = canvasElement.getContext('2d');
let toggleCamBtn = document.getElementById('toggle-cam');
let loadingOverlay = document.getElementById('loading-overlay');
let loadingText = document.getElementById('loading-text');

let isWebcamActive = false;
let localStream = null;
let customClassifier = null;

let lastFrameTime = performance.now();
let frameCount = 0;
let currentFps = 0;

// Internal Alphabet Lookup Mapping Array
const ALPHABET = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y'];

// Initialize App: Load Models sequentially
async function setupApp() {
    try {
        // Step 1: Initialize MediaPipe Handpose base tracker
        model = await handpose.load();
        updateStatusIndicator('status-model', 'green', 'Handpose Loaded');
        
        // Step 2: Extract custom weights layer array on top of basic Handpose landmarks
        loadingText.textContent = "Loading ASL Weights Model...";
        const response = await fetch('model.json');
        customClassifier = await response.json();
        
        updateStatusIndicator('status-model', 'green', 'System Online');
        
        // Release button access states
        loadingOverlay.style.opacity = '0';
        setTimeout(() => loadingOverlay.style.display = 'none', 400);
        
        toggleCamBtn.removeAttribute('disabled');
        toggleCamBtn.classList.remove('disabled');
    } catch (err) {
        console.error("Critical boot structure error: ", err);
        loadingText.innerHTML = `<span style="color: #f87171">Initialization Failed.<br>Verify JSON weights source pathways.</span>`;
        document.querySelector('.spinner').style.display = 'none';
    }
}

// Camera I/O State toggles
async function toggleWebcam() {
    if (isWebcamActive) {
        // Terminate ongoing streaming threads cleanly
        if (localStream) {
            localStream.getTracks().forEach(track => track.stop());
        }
        webcamElement.srcObject = null;
        isWebcamActive = false;
        toggleCamBtn.innerHTML = `<span class="material-icons">videocam_off</span> Start Camera`;
        ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        updateStatusIndicator('status-fps', 'gray', 'FPS: --');
        resetPredictionDisplay();
    } else {
        try {
            localStream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: "user" },
                audio: false
            });
            webcamElement.srcObject = localStream;
            isWebcamActive = true;
            toggleCamBtn.innerHTML = `<span class="material-icons">videocam</span> Stop Camera`;
            
            // Sync Canvas mapping ratios on stream meta detection triggers
            webcamElement.onloadedmetadata = () => {
                canvasElement.width = webcamElement.videoWidth;
                canvasElement.height = webcamElement.videoHeight;
                requestAnimationFrame(renderAnalysisLoop);
            };
        } catch (error) {
            console.error("Camera access blocked: ", error);
            alert("Could not access camera. Please inspect system permission prompts.");
        }
    }
}

// Performance Frame Metric calculation
function trackPerformanceMetrics() {
    frameCount++;
    const now = performance.now();
    if (now >= lastFrameTime + 1000) {
        currentFps = Math.round((frameCount * 1000) / (now - lastFrameTime));
        updateStatusIndicator('status-fps', 'green', `FPS: ${currentFps}`);
        frameCount = 0;
        lastFrameTime = now;
    }
}

// Real-time loop structure evaluating tracking configurations
async function renderAnalysisLoop() {
    if (!isWebcamActive) return;

    ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    trackPerformanceMetrics();

    try {
        const predictions = await model.estimateHands(webcamElement);
        
        if (predictions.length > 0) {
            const hand = predictions[0];
            drawHandSkeleton(hand.landmarks);
            
            // Extract feature input vector from coordinates
            const normalizedFeatures = generateFeatureVector(hand.landmarks);
            
            // Compute matching outputs against loaded JSON matrix layers
            evaluateGestureVector(normalizedFeatures);
        } else {
            resetPredictionDisplay();
        }
    } catch (e) {
        console.warn("Frame analysis processing drop error occurred: ", e);
    }

    requestAnimationFrame(renderAnalysisLoop);
}

// Draw skeleton joints and lines on canvas layer
function drawHandSkeleton(landmarks) {
    ctx.fillStyle = '#38bdf8';
    ctx.strokeStyle = 'rgba(248, 250, 252, 0.6)';
    ctx.lineWidth = 3;

    // Define connection map pathways
    const fingers = {
        thumb: [0, 1, 2, 3, 4],
        index: [0, 5, 6, 7, 8],
        middle: [0, 9, 10, 11, 12],
        ring: [0, 13, 14, 15, 16],
        pinky: [0, 17, 18, 19, 20]
    };

    // Render connection segments
    Object.values(fingers).forEach(path => {
        ctx.beginPath();
        ctx.moveTo(landmarks[path[0]][0], landmarks[path[0]][1]);
        for (let i = 1; i < path.length; i++) {
            ctx.lineTo(landmarks[path[i]][0], landmarks[path[i]][1]);
        }
        ctx.stroke();
    });

    // Draw individual joints
    landmarks.forEach(point => {
        ctx.beginPath();
        ctx.arc(point[0], point[1], 5, 0, 2 * Math.PI);
        ctx.fill();
    });
}

// Convert absolute pixel positions into scaled relative feature distances
function generateFeatureVector(landmarks) {
    const wrist = landmarks[0];
    let distances = [];
    
    // Compute distance offsets relative to wrist coordinate bounds
    for (let i = 1; i < landmarks.length; i++) {
        const dx = landmarks[i][0] - wrist[0];
        const dy = landmarks[i][1] - wrist[1];
        const dz = landmarks[i][2] - wrist[2];
        distances.push(Math.sqrt(dx*dx + dy*dy + dz*dz));
    }
    
    // Normalize absolute variations by dividing max bounding space
    const maxDist = Math.max(...distances);
    return distances.map(d => d / (maxDist || 1));
}

// Custom internal evaluation utilizing Euclidean distance configurations
function evaluateGestureVector(features) {
    if (!customClassifier || !customClassifier.gestures) return;

    let bestMatch = "—";
    let highestScore = 0;

    customClassifier.gestures.forEach(gesture => {
        let distanceSum = 0;
        // Compare target vector indexes matching base parameters
        for(let i = 0; i < features.length; i++) {
            let diff = features[i] - gesture.weights[i];
            distanceSum += diff * diff;
        }
        
        let calculatedScore = 1 / (1 + Math.sqrt(distanceSum)); // Convert error inverse scale score
        if (calculatedScore > highestScore) {
            highestScore = calculatedScore;
            bestMatch = gesture.label;
        }
    });

    // Post processing UI translation updates
    if (highestScore > 0.65) {
        document.getElementById('prediction-output').textContent = bestMatch;
        let percentString = Math.round(highestScore * 100) + '%';
        document.getElementById('confidence-text').textContent = percentString;
        document.getElementById('confidence-bar').style.width = percentString;
    } else {
        resetPredictionDisplay();
    }
}

// Visual layout helper routines
function updateStatusIndicator(elementId, colorClass, message) {
    const targetLi = document.getElementById(elementId);
    if(targetLi) {
        targetLi.innerHTML = `<span class="indicator ${colorClass}"></span>${message}`;
    }
}

function resetPredictionDisplay() {
    document.getElementById('prediction-output').textContent = "—";
    document.getElementById('confidence-text').textContent = "0%";
    document.getElementById('confidence-bar').style.width = "0%";
}

// Event hooks setup
toggleCamBtn.addEventListener('click', toggleWebcam);
window.addEventListener('load', setupApp);
