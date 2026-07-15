// Target Document Object Model Binding References
const videoElement = document.getElementById('webcam');
const smoothedOutput = document.getElementById('smoothed-output');
const consoleLog = document.getElementById('console-log');
const pipelineStatusLog = document.getElementById('pipeline-status-log');
const placeholderView = document.getElementById('placeholder-view');
const placementGuide = document.getElementById('placement-guide');
const liveIndicator = document.getElementById('live-indicator');
const liveStatusText = document.getElementById('live-status-text');
const modelStatusBadge = document.getElementById('model-status');
const inferenceTimeDisplay = document.getElementById('inference-time');

// Control Buttons DOM Access Bindings
const toggleGuideBtn = document.getElementById('toggle-guide-btn');
const clearBufferBtn = document.getElementById('clear-buffer-btn');

// System Operational Pipeline Constants
const MODEL_PATH = './model.json';
const BUFFER_SIZE = 5;
const ASL_ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'];

// Application Pipeline State Registers
let model = null;
let handsPipeline = null;
let camera = null;
let predictionBuffer = [];
let lastSpokenLetter = '';

// Logger utility wrapper matching custom console element properties
function logSystemStatus(message, isRawBoxUpdate = true) {
    const timestamp = new Date().toLocaleTimeString();
    if (isRawBoxUpdate) {
        consoleLog.innerHTML = `[${timestamp}] ${message}`;
    }
    console.log(`[${timestamp}] ${message}`);
}

// 1. Model Initialization Vector Stack Function
async function loadTensorflowModel() {
    try {
        logSystemStatus('Loading local model arrays...');
        pipelineStatusLog.innerText = 'Downloading local structural weights maps...';
        
        // Non-strict layout compilation resolves Keras 3 schema variations smoothly
        model = await tf.loadLayersModel(MODEL_PATH, { strict: false });
        
        logSystemStatus('Model loaded cleanly.');
        modelStatusBadge.innerText = 'ACTIVE';
        modelStatusBadge.style.color = '#10B981';
    } catch (error) {
        logSystemStatus(`Model crash state flag: ${error.message}`);
        pipelineStatusLog.innerText = 'Model weights structural compilation failure.';
        modelStatusBadge.innerText = 'CRASHED';
        modelStatusBadge.style.color = '#EF4444';
        throw error;
    }
}

// 2. Transformative Spatial Normalization Mathematics (Position & Scale Invariance)
function normalizeHandLandmarks(landmarks) {
    const wristAnchor = landmarks[0];
    let zeroAnchoredFeatures = [];

    // Zero-anchor relative translation offset calculations step
    for (let i = 0; i < landmarks.length; i++) {
        zeroAnchoredFeatures.push(landmarks[i].x - wristAnchor.x);
        zeroAnchoredFeatures.push(landmarks[i].y - wristAnchor.y);
        zeroAnchoredFeatures.push(landmarks[i].z - wristAnchor.z);
    }

    // Scale parameter extraction factor adjustments block
    const absoluteMaxMagnitude = Math.max(...zeroAnchoredFeatures.map(Math.abs));
    if (absoluteMaxMagnitude > 0) {
        zeroAnchoredFeatures = zeroAnchoredFeatures.map(value => value / absoluteMaxMagnitude);
    }

    return zeroAnchoredFeatures; // Flattens precisely to a 63-dimensional matrix input profile
}

// 3. Frame Smoothing Window Context Filter Function
function processBufferSmoothing(newPredictionRaw) {
    predictionBuffer.push(newPredictionRaw);
    if (predictionBuffer.length > BUFFER_SIZE) {
        predictionBuffer.shift();
    }

    const frequencyHistogram = {};
    let dominantStabilizedClass = newPredictionRaw;
    let peakOccurrenceCount = 0;

    for (const token of predictionBuffer) {
        frequencyHistogram[token] = (frequencyHistogram[token] || 0) + 1;
        if (frequencyHistogram[token] > peakOccurrenceCount) {
            peakOccurrenceCount = frequencyHistogram[token];
            dominantStabilizedClass = token;
        }
    }

    return dominantStabilizedClass;
}

// 4. Text-To-Speech Output feedback engine
function queueVoiceSynthesis(textAlphaCharacter) {
    if (textAlphaCharacter !== lastSpokenLetter && textAlphaCharacter !== '') {
        lastSpokenLetter = textAlphaCharacter;
        window.speechSynthesis.cancel(); // Clears lagging tracks immediately
        const speechTrack = new SpeechSynthesisUtterance(textAlphaCharacter);
        speechTrack.rate = 1.0;
        window.speechSynthesis.speak(speechTrack);
    }
}

// 5. Native Execution Loop Process Framework Callback
async function handleVideoFrameResults(results) {
    // Clear display if tracking boundaries are out of camera frame range
    if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
        smoothedOutput.innerHTML = `<span class="placeholder-text">Awaiting sign hand detection sequence...</span>`;
        smoothedOutput.classList.remove('is-fresh');
        return;
    }

    const executionStartTimeStamp = performance.now();
    const primaryHandCoordinates = results.multiHandLandmarks[0];
    const mathematicalFeatureMatrix = normalizeHandLandmarks(primaryHandCoordinates);

    tf.tidy(() => {
        const structuralMatrixTensor = tf.tensor2d([mathematicalFeatureMatrix], [1, 63]);
        const layerPredictionsTensor = model.predict(structuralMatrixTensor);
        const predictionProbabilityArray = layerPredictionsTensor.dataSync();
        
        const highestConfidenceIndex = predictionProbabilityArray.indexOf(Math.max(...predictionProbabilityArray));
        const rawInferredChar = ASL_ALPHABET[highestConfidenceIndex];
        const smoothedInferredChar = processBufferSmoothing(rawInferredChar);

        // Render data output frames smoothly down onto custom interface sections
        smoothedOutput.classList.add('is-fresh');
        smoothedOutput.innerText = smoothedInferredChar;
        
        // Trigger voice synthesizers asynchronously
        queueVoiceSynthesis(smoothedInferredChar);
    });

    const executionDeltaTime = Math.round(performance.now() - executionStartTimeStamp);
    inferenceTimeDisplay.innerHTML = `${executionDeltaTime}<small>ms</small>`;
}

// 6. Complete Initialization Routine Implementation
async function runtimePipelineBootstrap() {
    try {
        await loadTensorflowModel();
        
        pipelineStatusLog.innerText = 'Requesting client camera system capture flags...';
        logSystemStatus('Binding MediaPipe worker parameters...');

        handsPipeline = new Hands({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        });

        handsPipeline.setOptions({
            maxNumHands: 1,
            modelComplexity: 1,
            minDetectionConfidence: 0.7,
            minTrackingConfidence: 0.5
        });

        handsPipeline.onResults(handleVideoFrameResults);

        camera = new Camera(videoElement, {
            onFrame: async () => {
                await handsPipeline.send({ image: videoElement });
            },
            width: 640,
            height: 480
        });

        logSystemStatus('Activating live media system streams...');
        await camera.start();
        
        logSystemStatus('Pipeline operations stable and live.');
        
        // Switch Layout visibility structures smoothly
        placeholderView.classList.add('is-hidden');
        videoElement.classList.add('is-active');
        placementGuide.classList.add('is-visible');
        liveIndicator.classList.add('is-live');
        liveStatusText.innerText = 'LIVE';
    } catch (error) {
        logSystemStatus(`Pipeline Initialization Fault Break: ${error.message}`);
        pipelineStatusLog.innerText = 'Hardware camera stream activation or pipeline mismatch fault.';
    }
}

// 7. Event Handler Registration Settings
toggleGuideBtn.addEventListener('click', () => {
    const isCurrentlyVisible = placementGuide.classList.contains('is-visible');
    if (isCurrentlyVisible) {
        placementGuide.classList.remove('is-visible');
        toggleGuideBtn.innerText = 'Show Guide Frame';
    } else {
        placementGuide.classList.add('is-visible');
        toggleGuideBtn.innerText = 'Hide Guide Frame';
    }
});

clearBufferBtn.addEventListener('click', () => {
    predictionBuffer = [];
    lastSpokenLetter = '';
    logSystemStatus('Inference prediction buffer state reset complete.');
});

// Launch engine processing stack
window.addEventListener('DOMContentLoaded', runtimePipelineBootstrap);
