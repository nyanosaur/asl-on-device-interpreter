// Global DOM References
const videoElement = document.getElementById('webcam');
const letterDisplay = document.getElementById('detected-letter');
const statusText = document.getElementById('status-text');
const loadingOverlay = document.getElementById('loading-overlay');
const consoleLog = document.getElementById('console-log');

// Configuration Constants
const MODEL_PATH = 'model.json';
const BUFFER_SIZE = 5;
const ASL_ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'];

// Global State Variables
let model = null;
let handsPipeline = null;
let camera = null;
let predictionBuffer = [];
let lastSpokenLetter = '';

// Logger utility helper
function logMessage(message) {
    const timestamp = new Date().toLocaleTimeString();
    consoleLog.innerHTML += `<div>[${timestamp}] ${message}</div>`;
    consoleLog.scrollTop = consoleLog.scrollHeight;
    console.log(message);
}

// 1. Load the TensorFlow.js Model
async function loadModel() {
    try {
        logMessage('Loading TensorFlow.js Keras model...');
        model = await tf.loadLayersModel(MODEL_PATH);
        logMessage('Model architecture verified and loaded successfully.');
    } catch (error) {
        logMessage(`CRITICAL ERROR loading model: ${error.message}`);
        statusText.innerText = 'Model failed to load. Check console.';
        throw error;
    }
}

// 2. Anchor Math (Position & Distance Scale Independence normalization)
function normalizeLandmarks(landmarks) {
    // MediaPipe provides 21 points with x, y, z coordinates
    const wrist = landmarks[0];
    let zeroAnchored = [];

    // Step A: Subtract wrist coordinates from all coordinates (translation invariance)
    for (let i = 0; i < landmarks.length; i++) {
        zeroAnchored.push(landmarks[i].x - wrist.x);
        zeroAnchored.push(landmarks[i].y - wrist.y);
        zeroAnchored.push(landmarks[i].z - wrist.z);
    }

    // Step B: Scale invariance (Normalize scale by maximum distance absolute value)
    const maxVal = Math.max(...zeroAnchored.map(Math.abs));
    if (maxVal > 0) {
        zeroAnchored = zeroAnchored.map(val => val / maxVal);
    }

    return zeroAnchored; // Returns a flat array of 63 values
}

// 3. 5-Frame Smoothing Window Logic
function getSmoothedPrediction(newPrediction) {
    predictionBuffer.push(newPrediction);
    if (predictionBuffer.length > BUFFER_SIZE) {
        predictionBuffer.shift();
    }

    // Tally frequency count of classes inside window
    const frequencyMap = {};
    let majorityElement = newPrediction;
    let maxCount = 0;

    for (const letter of predictionBuffer) {
        frequencyMap[letter] = (frequencyMap[letter] || 0) + 1;
        if (frequencyMap[letter] > maxCount) {
            maxCount = frequencyMap[letter];
            majorityElement = letter;
        }
    }

    return majorityElement;
}

// 4. Smart Text-to-Speech Engine
function speakLetter(letter) {
    if (letter !== lastSpokenLetter && letter !== '-') {
        lastSpokenLetter = letter;
        window.speechSynthesis.cancel(); // Stop ongoing stuttering utterances
        const utterance = new SpeechSynthesisUtterance(letter);
        utterance.rate = 1.0;
        window.speechSynthesis.speak(utterance);
    }
}

// 5. Real-Time Processing Loop Core
async function onResults(results) {
    // Clear screen UI if no hand structure tracking is present
    if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
        letterDisplay.innerText = '-';
        return;
    }

    // Capture the first tracked hand configuration array
    const landmarks = results.multiHandLandmarks[0];
    
    // Apply position and scale transformation mathematics
    const inputFeatures = normalizeLandmarks(landmarks);

    // Run execution tensor generation block safely wrapped from leakages
    tf.tidy(() => {
        const inputTensor = tf.tensor2d([inputFeatures], [1, 63]);
        const prediction = model.predict(inputTensor);
        const probabilities = prediction.dataSync();
        
        // Find index with high activation scalar value matching confidence map
        const maxIndex = probabilities.indexOf(Math.max(...probabilities));
        const rawLetter = ASL_ALPHABET[maxIndex];

        // Process through smoothing layer matrices
        const smoothedLetter = getSmoothedPrediction(rawLetter);

        // Update UI View States
        letterDisplay.innerText = smoothedLetter;
        
        // Broadcast speech synthesizers seamlessly
        speakLetter(smoothedLetter);
    });
}

// 6. Main Pipeline Initialization Routine
async function initializePipeline() {
    try {
        await loadModel();
        
        statusText.innerText = 'Starting Camera Stream...';
        logMessage('Initializing MediaPipe Hands framework context...');

        handsPipeline = new Hands({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        });

        handsPipeline.setOptions({
            maxNumHands: 1,
            modelComplexity: 1,
            minDetectionConfidence: 0.7,
            minTrackingConfidence: 0.5
        });

        handsPipeline.onResults(onResults);

        camera = new Camera(videoElement, {
            onFrame: async () => {
                await handsPipeline.send({ image: videoElement });
            },
            width: 640,
            height: 480
        });

        logMessage('Requesting client system webcam access parameters...');
        await camera.start();
        
        logMessage('System pipeline successfully operational.');
        loadingOverlay.style.display = 'none'; // Clear setup modal overlay away smoothly
    } catch (error) {
        logMessage(`Initialization Pipeline Failure: ${error.message}`);
        statusText.innerText = 'Hardware or Initialization error.';
    }
}

// Execute setup execution stack patterns
window.addEventListener('DOMContentLoaded', initializePipeline);
