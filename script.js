/* ==========================================================================
   SignSpeak — script.js
   Vanilla JS controller for camera, voice output, and the ASL detection
   simulation hook. Replace processVideoFrame()'s internals with a real
   on-device model (e.g. MediaPipe Hands + a gesture classifier) later —
   the UI wiring below will keep working unchanged.
   ========================================================================== */

(() => {
  'use strict';

  /* ---------------------------- DOM references ---------------------------- */

  const video            = document.getElementById('webcam');
  const webcamPlaceholder = document.getElementById('webcam-placeholder');
  const handGuide         = document.getElementById('hand-guide');
  const recIndicator      = document.getElementById('rec-indicator');

  const btnCamera      = document.getElementById('btn-camera');
  const btnCameraLabel = document.getElementById('btn-camera-label');
  const btnVoice       = document.getElementById('btn-voice');
  const btnVoiceLabel  = document.getElementById('btn-voice-label');
  const btnClear       = document.getElementById('btn-clear');

  const rawStreamEl   = document.getElementById('raw-stream');
  const rawStatusEl   = document.getElementById('raw-status');
  const smoothedEl    = document.getElementById('smoothed-sentence');
  const smoothedStatusEl = document.getElementById('smoothed-status');

  const statInference = document.getElementById('stat-inference');
  const statFps        = document.getElementById('stat-fps');

  /* ---------------------------- App state ---------------------------- */

  const state = {
    cameraOn: false,
    voiceOn: false,
    mediaStream: null,
    simulationTimer: null,
    statsTimer: null,
  };

  // Vocabulary the "model" simulation draws from. Swap for real inference
  // output once a hand-detection model is wired into processVideoFrame().
  const SIMULATED_PHRASES = [
    'Hello!',
    'Thank you',
    'How are you?',
    'Yes',
    'No',
    'Please',
    'My name is Alex',
    'Nice to meet you',
  ];

  /* ============================================================
     CAMERA CONTROL
     ============================================================ */

  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });

      state.mediaStream = stream;
      video.srcObject = stream;
      video.classList.add('is-active');
      webcamPlaceholder.classList.add('is-hidden');
      handGuide.classList.add('is-visible');
      recIndicator.classList.add('is-live');

      state.cameraOn = true;
      btnCameraLabel.textContent = 'Stop Camera';
      btnCamera.classList.add('is-active');
      btnCamera.setAttribute('aria-pressed', 'true');

      rawStatusEl.textContent = 'Listening';
      smoothedStatusEl.textContent = 'Listening';

      startDetectionSimulation();
      startStatsJitter();
    } catch (err) {
      console.error('Camera access failed:', err);
      rawStatusEl.textContent = 'Camera error';
      webcamPlaceholder.querySelector('p').textContent = 'Camera unavailable';
      webcamPlaceholder.querySelector('span').textContent =
        'Check browser permissions and try again.';
    }
  }

  function stopCamera() {
    if (state.mediaStream) {
      state.mediaStream.getTracks().forEach((track) => track.stop());
      state.mediaStream = null;
    }
    video.srcObject = null;
    video.classList.remove('is-active');
    webcamPlaceholder.classList.remove('is-hidden');
    handGuide.classList.remove('is-visible');
    recIndicator.classList.remove('is-live');

    state.cameraOn = false;
    btnCameraLabel.textContent = 'Start Camera';
    btnCamera.classList.remove('is-active');
    btnCamera.setAttribute('aria-pressed', 'false');

    rawStatusEl.textContent = 'Idle';
    smoothedStatusEl.textContent = 'Ready';

    stopDetectionSimulation();
    stopStatsJitter();
  }

  btnCamera.addEventListener('click', () => {
    state.cameraOn ? stopCamera() : startCamera();
  });

  /* ============================================================
     VOICE OUTPUT (native, fully offline browser speech synthesis)
     ============================================================ */

  function speak(text) {
    if (!state.voiceOn || !('speechSynthesis' in window)) return;
    // Cancel anything mid-utterance so phrases don't queue up and lag behind.
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1;
    utterance.pitch = 1;
    window.speechSynthesis.speak(utterance);
  }

  btnVoice.addEventListener('click', () => {
    state.voiceOn = !state.voiceOn;
    btnVoice.classList.toggle('is-active', state.voiceOn);
    btnVoice.setAttribute('aria-pressed', String(state.voiceOn));
    btnVoiceLabel.textContent = `Voice Output: ${state.voiceOn ? 'On' : 'Off'}`;

    if (!state.voiceOn && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
  });

  /* ============================================================
     CLEAR INTERFACE
     ============================================================ */

  function clearInterface() {
    rawStreamEl.innerHTML = '<span class="placeholder-text">Detected signs will stream here…</span>';
    smoothedEl.innerHTML = '<span class="placeholder-text">Your interpreted sentence will appear here.</span>';
    smoothedEl.classList.remove('is-fresh');
    rawStatusEl.textContent = state.cameraOn ? 'Listening' : 'Idle';
    smoothedStatusEl.textContent = state.cameraOn ? 'Listening' : 'Ready';
    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
  }

  btnClear.addEventListener('click', clearInterface);

  /* ============================================================
     THE DUMMY HOOK — processVideoFrame()
     ------------------------------------------------------------
     This is the integration point for a real on-device model.
     A production pipeline would call this once per captured frame
     (e.g. from a requestAnimationFrame loop feeding frames into
     MediaPipe Hands + a gesture classifier running via TF.js/ONNX
     Runtime Web), with `frameData` being the pixel buffer for that
     frame. For now it's driven by a 3-second simulation timer that
     hands it a random phrase instead of real inference output.
     ============================================================ */

  function processVideoFrame(frameData) {
    const { rawToken, finalPhrase } = frameData;

    // 1. Raw detection — push the newest token into the raw stream first,
    //    the way an unsmoothed per-frame classifier would.
    rawStatusEl.textContent = 'Detecting…';
    appendRawToken(rawToken);

    // 2. Simulate the short pause a smoothing/language model would take
    //    to turn noisy raw tokens into a clean finalized phrase.
    window.setTimeout(() => {
      setSmoothedSentence(finalPhrase);
      rawStatusEl.textContent = 'Listening';
      smoothedStatusEl.textContent = 'Finalized';
      speak(finalPhrase);

      // Return the status pill to a listening state shortly after.
      window.setTimeout(() => {
        if (state.cameraOn) smoothedStatusEl.textContent = 'Listening';
      }, 1200);
    }, 550);
  }

  function appendRawToken(token) {
    rawStreamEl.innerHTML = '';
    const span = document.createElement('span');
    span.textContent = token;
    rawStreamEl.appendChild(span);
  }

  function setSmoothedSentence(text) {
    smoothedEl.innerHTML = '';
    const span = document.createElement('span');
    span.textContent = text;
    smoothedEl.appendChild(span);

    smoothedEl.classList.remove('is-fresh');
    // Force reflow so the highlight transition can re-trigger on repeats.
    void smoothedEl.offsetWidth;
    smoothedEl.classList.add('is-fresh');
  }

  /* ---------------------------- Simulation driver ---------------------------- */

  function startDetectionSimulation() {
    stopDetectionSimulation();
    state.simulationTimer = window.setInterval(() => {
      const phrase = SIMULATED_PHRASES[Math.floor(Math.random() * SIMULATED_PHRASES.length)];
      // Mimic a raw token looking rougher than the final smoothed phrase,
      // e.g. all-caps single-word glosses before language-model cleanup.
      const rawToken = phrase.replace(/[!?]/g, '').toUpperCase();
      processVideoFrame({ rawToken, finalPhrase: phrase });
    }, 3000);
  }

  function stopDetectionSimulation() {
    if (state.simulationTimer) {
      clearInterval(state.simulationTimer);
      state.simulationTimer = null;
    }
  }

  /* ---------------------------- Mock stats dashboard ---------------------------- */

  function startStatsJitter() {
    stopStatsJitter();
    state.statsTimer = window.setInterval(() => {
      const inference = (28 + Math.random() * 9).toFixed(0);
      const fps = (28 + Math.random() * 4).toFixed(0);
      statInference.innerHTML = `${inference}<small>ms</small>`;
      statFps.innerHTML = `${fps}<small>fps</small>`;
    }, 1400);
  }

  function stopStatsJitter() {
    if (state.statsTimer) {
      clearInterval(state.statsTimer);
      state.statsTimer = null;
    }
    statInference.innerHTML = '32<small>ms</small>';
    statFps.innerHTML = '30<small>fps</small>';
  }

  /* ---------------------------- Cleanup ---------------------------- */

  window.addEventListener('beforeunload', () => {
    if (state.mediaStream) {
      state.mediaStream.getTracks().forEach((track) => track.stop());
    }
  });

})();
