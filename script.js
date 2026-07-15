// Target Document Object Model Binding References
const videoElement = document.getElementById('webcam');
const placeholderView = document.getElementById('placeholder-view');
const pipelineStatusLog = document.getElementById('pipeline-status-log');
const placementGuide = document.getElementById('placement-guide');
const cameraStatusPill = document.getElementById('camera-status-pill');
const statusDot = document.getElementById('status-dot');
const cameraStatusText = document.getElementById('camera-status-text');
const progressTrail = document.getElementById('progress-trail');

const flashcardFront = document.getElementById('flashcard-front');
const flashcardEyebrow = document.getElementById('flashcard-eyebrow');
const flashcardImage = document.getElementById('flashcard-image');
const flashcardLetterFallback = document.getElementById('flashcard-letter-fallback');
const flashcardTargetLetter = document.getElementById('flashcard-target-letter');
const flashcardHint = document.getElementById('flashcard-hint');
const confettiContainer = document.getElementById('confetti');

// Control Buttons DOM Access Bindings
const toggleGuideBtn = document.getElementById('toggle-guide-btn');
const skipLetterBtn = document.getElementById('skip-letter-btn');

// System Operational Pipeline Constants
const MODEL_PATH = './model.json';
const BUFFER_SIZE = 5;
const GAME_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'];
const CELEBRATION_DELAY_MS = 1500;
const CONFETTI_COLORS = ['#FFC33D', '#FF7A66', '#4FB6E8', '#45B478', '#F5A300'];

// Application Pipeline State Registers
let model = null;
let handsPipeline = null;
let camera = null;
let predictionBuffer = [];

// Game State Registers
let letterQueue = [];
let currentTargetLetter = '';
let letterIndex = 0;
let learnedLetters = new Set();
let isCelebrating = false;

// Simple dev-console logger (no on-screen readout for players)
function logSystemStatus(message) {
    const timestamp = new Date().toLocaleTimeString();
    console.log(`[${timestamp}] ${message}`);
}

// ============================= Game setup helpers ============================= //

// Fisher-Yates shuffle so letters appear in a fresh random order each round,
// with a guard so the same letter never plays twice in a row.
function buildShuffledQueue(avoidFirst) {
    const shuffled = [...GAME_LETTERS];
    for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    if (avoidFirst && shuffled[0] === avoidFirst && shuffled.length > 1) {
        [shuffled[0], shuffled[1]] = [shuffled[1], shuffled[0]];
    }
    return shuffled;
}

function buildProgressTrail() {
    progressTrail.innerHTML = '';
    GAME_LETTERS.forEach((letter) => {
        const dot = document.createElement('div');
        dot.className = 'progress-dot';
        dot.id = `progress-dot-${letter}`;
        dot.innerText = letter;
        progressTrail.appendChild(dot);
    });
}

function updateProgressTrail() {
    GAME_LETTERS.forEach((letter) => {
        const dot = document.getElementById(`progress-dot-${letter}`);
        if (!dot) return;
        dot.classList.toggle('is-learned', learnedLetters.has(letter));
        dot.classList.toggle('is-current', letter === currentTargetLetter && !learnedLetters.has(letter));
    });
}

function buildConfettiPieces() {
    confettiContainer.innerHTML = '';
    for (let i = 0; i < 18; i++) {
        const piece = document.createElement('span');
        piece.className = 'confetti-piece';
        piece.style.left = `${Math.random() * 100}%`;
        piece.style.background = CONFETTI_COLORS[i % CONFETTI_COLORS.length];
        piece.style.animationDelay = `${Math.random() * 0.25}s`;
        confettiContainer.appendChild(piece);
    }
}

// Renders the active flashcard, with a graceful fallback to a big letter
// if the sign illustration for that letter isn't available yet.
function renderFlashcard(letter) {
    currentTargetLetter = letter;
    flashcardTargetLetter.innerText = letter;
    flashcardHint.innerText = `Letter ${letterIndex + 1} of ${GAME_LETTERS.length}`;

    flashcardImage.classList.remove('is-hidden');
    flashcardLetterFallback.classList.remove('is-visible');
    flashcardLetterFallback.innerText = letter;
    flashcardImage.src = `./assets/flashcards/${letter}.svg`;
    flashcardImage.alt = `Sign for the letter ${letter}`;

    updateProgressTrail();
}

flashcardImage.addEventListener('error', () => {
    flashcardImage.classList.add('is-hidden');
    flashcardLetterFallback.classList.add('is-visible');
});

function advanceToNextLetter() {
    letterIndex++;
    if (letterIndex >= letterQueue.length) {
        letterQueue = buildShuffledQueue(currentTargetLetter);
        letterIndex = 0;
    }
    renderFlashcard(letterQueue[letterIndex]);
}

function startNewGame() {
    letterQueue = buildShuffledQueue();
    letterIndex = 0;
    learnedLetters = new Set();
    buildProgressTrail();
    buildConfettiPieces();
    renderFlashcard(letterQueue[0]);
}

// ============================= Camera status helpers ============================= //

function setCameraStatus(text, state) {
    cameraStatusText.innerText = text;
    cameraStatusPill.classList.remove('is-live', 'is-error');
    if (state) cameraStatusPill.classList.add(state);
}

// ============================= Model + hand tracking pipeline ============================= //

// 1. Model Initialization
async function loadTensorflowModel() {
    try {
        logSystemStatus('Loading local model arrays...');
        pipelineStatusLog.innerText = 'Getting the magic camera ready…';

        // Non-strict layout compilation resolves Keras 3 schema variations smoothly
        model = await tf.loadLayersModel(MODEL_PATH, { strict: false });

        logSystemStatus('Model loaded cleanly.');
    } catch (error) {
        logSystemStatus(`Model crash state flag: ${error.message}`);
        pipelineStatusLog.innerText = 'Oops! We had trouble loading the game.';
        setCameraStatus('Having trouble getting ready 😕', 'is-error');
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

function clearPredictionBuffer() {
    predictionBuffer = [];
}

// 4. Celebration + voice feedback engine
function celebrateMatch() {
    if (isCelebrating) return;
    isCelebrating = true;

    learnedLetters.add(currentTargetLetter);
    flashcardFront.classList.add('is-celebrating');
    updateProgressTrail();

    window.speechSynthesis.cancel();
    const cheer = new SpeechSynthesisUtterance('Awesome! You did it!');
    cheer.rate = 1.05;
    cheer.pitch = 1.15;
    window.speechSynthesis.speak(cheer);

    setTimeout(() => {
        flashcardFront.classList.remove('is-celebrating');
        clearPredictionBuffer();
        isCelebrating = false;
        advanceToNextLetter();
    }, CELEBRATION_DELAY_MS);
}

// 5. Native Execution Loop Process Framework Callback
async function handleVideoFrameResults(results) {
    if (isCelebrating) return;

    // No hand in frame — nothing to evaluate this tick
    if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
        return;
    }

    const primaryHandCoordinates = results.multiHandLandmarks[0];
    const mathematicalFeatureMatrix = normalizeHandLandmarks(primaryHandCoordinates);

    tf.tidy(() => {
        const structuralMatrixTensor = tf.tensor2d([mathematicalFeatureMatrix], [1, 63]);
        const layerPredictionsTensor = model.predict(structuralMatrixTensor);
        const predictionProbabilityArray = layerPredictionsTensor.dataSync();

        const highestConfidenceIndex = predictionProbabilityArray.indexOf(Math.max(...predictionProbabilityArray));
        const rawInferredChar = GAME_LETTERS[highestConfidenceIndex];
        const smoothedInferredChar = processBufferSmoothing(rawInferredChar);

        if (smoothedInferredChar === currentTargetLetter) {
            celebrateMatch();
        }
    });
}

// 6. Complete Initialization Routine Implementation
async function runtimePipelineBootstrap() {
    try {
        await loadTensorflowModel();

        pipelineStatusLog.innerText = 'Asking to borrow your camera…';
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

        // Switch layout visibility structures smoothly
        placeholderView.classList.add('is-hidden');
        videoElement.classList.add('is-active');
        placementGuide.classList.add('is-visible');
        setCameraStatus('Say cheese! Ready to play 🎉', 'is-live');

        startNewGame();
    } catch (error) {
        logSystemStatus(`Pipeline Initialization Fault Break: ${error.message}`);
        pipelineStatusLog.innerText = "Hmm, we couldn't reach your camera.";
        setCameraStatus('Camera trouble — check permissions 😕', 'is-error');
    }
}

// 7. Event Handler Registration Settings
toggleGuideBtn.addEventListener('click', () => {
    const isCurrentlyVisible = placementGuide.classList.contains('is-visible');
    if (isCurrentlyVisible) {
        placementGuide.classList.remove('is-visible');
        toggleGuideBtn.innerText = 'Show Hand Guide';
    } else {
        placementGuide.classList.add('is-visible');
        toggleGuideBtn.innerText = 'Hide Hand Guide';
    }
});

skipLetterBtn.addEventListener('click', () => {
    if (isCelebrating) return;
    clearPredictionBuffer();
    advanceToNextLetter();
    logSystemStatus('Player skipped to a new letter.');
});

// Launch engine processing stack
window.addEventListener('DOMContentLoaded', runtimePipelineBootstrap);
