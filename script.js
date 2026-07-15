(() => {
  'use strict';
  const MIRROR_CAMERA = false; 
  const USE_Z_SCORE = true;
  // =======================================================================

  /* ---------------------------- DOM references ---------------------------- */
  const video            = document.getElementById('webcam');
  const webcamPlaceholder = document.getElementById('webcam-placeholder');
  const handGuide         = document.getElementById('hand-guide');
  const recIndicator      = document.getElementById('rec-indicator');
  const btnCamera      = document.getElementById('btn-camera');
  const btnCameraLabel = document.getElementById('btn-camera-label');
  const btnVoice       = document.getElementById('btn-voice');
  const btnVoiceLabel  = document.getElementById('btn-voice-label');
  const smoothedEl    = document.getElementById('smoothed-sentence');
  const smoothedStatusEl = document.getElementById('smoothed-status');
  const statInference = document.getElementById('stat-inference');
  const statFps        = document.getElementById('stat-fps');

  // Flashcard DOM
  const flashcardCard     = document.getElementById('flashcard-card');
  const targetCharEl      = document.getElementById('target-char');
  const targetSubtitleEl  = document.getElementById('flashcard-subtitle');
  const targetImgEl       = document.getElementById('target-sign-img');
  const targetPlaceholder = document.getElementById('target-img-placeholder');
  const progressEl        = document.getElementById('flashcard-progress');

  const state = {
    cameraOn: false, voiceOn: false, mediaStream: null, 
    isDetecting: false, framesSinceLastFps: 0, lastFpsTime: performance.now()
  };

  const CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 
    'T', 'U', 'V', 'W', 'X', 'Y'
  ];
  
  let predictionBuffer = [];
  let targetIndex = 0;
  let isSuccessPause = false; 

  function loadFlashcard() {
    const letter = CLASSES[targetIndex];
    targetCharEl.textContent = letter;
    targetSubtitleEl.textContent = letter;
    progressEl.textContent = `Letter ${targetIndex + 1} of ${CLASSES.length}`;
    flashcardCard.classList.remove('success');
    isSuccessPause = false;

    targetImgEl.style.display = 'block';
    targetPlaceholder.style.display = 'none';
    targetImgEl.src = `assets/signs/${letter}.png`;
    targetImgEl.onerror = () => {
        targetImgEl.style.display = 'none';
        targetPlaceholder.style.display = 'block';
        targetPlaceholder.innerHTML = `Image missing:<br><code>assets/signs/${letter}.png</code>`;
    };
  }
  loadFlashcard();

  let tfModel, hands, normStats; // <--- Ensure normStats is declared here

  async function initializeAI() {
    try {
      // --- NEW: OBLITERATE CACHE & LOAD NORM STATS ---
      // This stops GitHub Pages from holding onto the broken model files
      if ('caches' in window) {
          try {
              const cacheNames = await caches.keys();
              for (const name of cacheNames) { await caches.delete(name); }
          } catch (e) {}
      }

      const cacheBuster = `?t=${new Date().getTime()}`;
      
      // Fetch the normalization stats so the JS matches the Python math
      const statsResponse = await fetch(`./norm_stats.json${cacheBuster}`);
      normStats = await statsResponse.json();
      
      // Load the Model (ensure the path is './model.json' since it is in your root folder)
      tfModel = await tf.loadLayersModel(`./model.json${cacheBuster}`, {
          requestInit: { cache: 'no-store' },
          strict: false
      });
      
      hands = new Hands({locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
      }});
      hands.setOptions({
        maxNumHands: 1, modelComplexity: 1,
        minDetectionConfidence: 0.5, minTrackingConfidence: 0.5
      });
      hands.onResults(onMediaPipeResults);
      console.log("AI Loaded successfully with FRESH weights!");
    } catch (err) {
      console.error("Failed to load AI models:", err);
      smoothedEl.innerHTML = `<span class="placeholder-text" style="color:var(--danger);">Error loading AI weights. Check console.</span>`;
    }
  }
  initializeAI();

  async function cameraLoop() {
    if (!state.cameraOn || !video.videoWidth || !hands || !tfModel || !normStats) return;

    if (!state.isDetecting) {
      state.isDetecting = true;
      const t0 = performance.now();
      await hands.send({image: video});
      statInference.innerHTML = `${(performance.now() - t0).toFixed(0)}<small>ms</small>`;
      state.isDetecting = false;
    }

    state.framesSinceLastFps++;
    if (performance.now() - state.lastFpsTime >= 1000) {
      statFps.innerHTML = `${state.framesSinceLastFps}<small>fps</small>`;
      state.framesSinceLastFps = 0;
      state.lastFpsTime = performance.now();
    }
    requestAnimationFrame(cameraLoop);
  }

  function onMediaPipeResults(results) {
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      const landmarks = results.multiHandLandmarks[0];

      // --- MIRROR FIX ---
      let coords = landmarks.map(lm => [MIRROR_CAMERA ? 1.0 - lm.x : lm.x, lm.y, lm.z]);
      
      let wrist = coords[0];
      let relativeCoords = coords.map(pt => [pt[0] - wrist[0], pt[1] - wrist[1], pt[2] - wrist[2]]);
      
      let maxDist = 0;
      for(let pt of relativeCoords) {
        let dist = Math.sqrt(pt[0]**2 + pt[1]**2 + pt[2]**2);
        if(dist > maxDist) maxDist = dist;
      }
      if(maxDist > 1e-6) {
        relativeCoords = relativeCoords.map(pt => [pt[0]/maxDist, pt[1]/maxDist, pt[2]/maxDist]);
      }

      let features = [];
      relativeCoords.forEach(pt => features.push(pt[0], pt[1], pt[2]));

      // --- Z-SCORE NORMALIZATION (CRITICAL FOR ACCURACY) ---
      if (USE_Z_SCORE && normStats) {
          for (let i = 0; i < features.length; i++) {
             let mean = normStats.mean[i] !== undefined ? normStats.mean[i] : 0;
             let std = normStats.std[i] !== undefined ? normStats.std[i] : 1.0;
             if (std === 0 || isNaN(std)) std = 1.0;
             features[i] = (features[i] - mean) / std;
          }
      }

      const tensor = tf.tensor2d([features]);
      const prediction = tfModel.predict(tensor);
      const probs = prediction.dataSync();
      const maxProb = Math.max(...probs);
      const classIdx = probs.indexOf(maxProb);
      
      tensor.dispose(); prediction.dispose();

      let rawLetter = "?";
      // Lowered threshold to 30% so you can clearly see what the AI is attempting to guess
      if (maxProb > 0.30 && CLASSES && CLASSES.length > classIdx) {
          rawLetter = CLASSES[classIdx];
      }

      predictionBuffer.push(rawLetter);
      if (predictionBuffer.length > 5) predictionBuffer.shift();

      let counts = {};
      let maxCount = 0;
      let smoothedLetter = "?";
      for (let val of predictionBuffer) {
        counts[val] = (counts[val] || 0) + 1;
        if (counts[val] > maxCount) { maxCount = counts[val]; smoothedLetter = val; }
      }

      if (smoothedLetter !== "?") {
        setSmoothedSentence(`${smoothedLetter} (${(maxProb * 100).toFixed(0)}%)`);
        smoothedStatusEl.textContent = 'Active';

        const targetLetter = CLASSES[targetIndex];
        if (!isSuccessPause && smoothedLetter === targetLetter) {
            isSuccessPause = true;
            flashcardCard.classList.add('success');
            speak(`Great job! That's ${targetLetter}`);
            setTimeout(() => {
                targetIndex = (targetIndex + 1) % CLASSES.length;
                loadFlashcard();
            }, 2000);
        }
      } else {
        setSmoothedSentence("Adjust hand...");
      }

    } else {
      predictionBuffer = [];
      smoothedStatusEl.textContent = 'Ready';
      setSmoothedSentence("No hand detected");
    }
  }

  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
      state.mediaStream = stream; video.srcObject = stream;
      video.classList.add('is-active'); webcamPlaceholder.classList.add('is-hidden');
      handGuide.classList.add('is-visible'); recIndicator.classList.add('is-live');
      state.cameraOn = true; btnCameraLabel.textContent = 'Stop Camera'; btnCamera.classList.add('is-active');
      video.onloadeddata = () => requestAnimationFrame(cameraLoop);
    } catch (err) { console.error('Camera access failed:', err); }
  }

  function stopCamera() {
    if (state.mediaStream) state.mediaStream.getTracks().forEach((track) => track.stop());
    state.mediaStream = null; video.srcObject = null;
    video.classList.remove('is-active'); webcamPlaceholder.classList.remove('is-hidden');
    handGuide.classList.remove('is-visible'); recIndicator.classList.remove('is-live');
    state.cameraOn = false; btnCameraLabel.textContent = 'Start Camera'; btnCamera.classList.remove('is-active');
    predictionBuffer = []; statInference.innerHTML = `--<small>ms</small>`; statFps.innerHTML = `--<small>fps</small>`;
  }

  btnCamera.addEventListener('click', () => { state.cameraOn ? stopCamera() : startCamera(); });

  function speak(text) {
    if (!state.voiceOn || !('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
  }

  btnVoice.addEventListener('click', () => {
    state.voiceOn = !state.voiceOn;
    btnVoice.classList.toggle('is-active', state.voiceOn);
    btnVoiceLabel.textContent = `Voice Output: ${state.voiceOn ? 'On' : 'Off'}`;
  });

  function setSmoothedSentence(text) {
    smoothedEl.innerHTML = '';
    const span = document.createElement('span'); span.textContent = text;
    smoothedEl.appendChild(span);
  }

  window.addEventListener('beforeunload', () => {
    if (state.mediaStream) state.mediaStream.getTracks().forEach((track) => track.stop());
  });
})();
