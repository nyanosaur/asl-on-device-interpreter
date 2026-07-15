# 🏛️ Architectural Design Document: Real-Time ASL Interpreter

This document outlines the structural framework, mathematical modeling data paths, and computing layer topologies implemented within the on-device inference ecosystem.

---

## 🏗️ Architectural Overview

The application follows an **Edge-Compute Stream Design Pattern**. Rather than capturing raw matrix frames and transferring massive tracking properties to remote processing servers, execution handles sequential steps micro-second by micro-second directly within the browser lifecycle loop.

This guarantees a **zero server-side database dependency design**—reducing server costs to absolute zero and ensuring processing execution times remain optimized for local hardware.

---

## 🎛️ Component Layer Topology

The codebase architecture relies on a strict separation of layers to isolate responsibilities:

```text
  ┌────────────────────────────────────────────────────────┐
  │                   1. Presentation Layer                │
  │     (index.html Document Model & canvas Render Context) │
  └───────────────────────────┬────────────────────────────┘
                              │ Registers Intercept Triggers
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │                   2. Stream Control Layer              │
  │    (Webcam capture hooks, RequestAnimationFrame Loops) │
  └───────────────────────────┬────────────────────────────┘
                              │ Supplies Frame Arrays
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │                   3. Feature Extraction Layer          │
  │       (MediaPipe Handpose Coordinate Node Tracking)    │
  └───────────────────────────┬────────────────────────────┘
                              │ Emits 21 x 3D Node Telemetry
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │                4. Classification & Vector Math         │
  │      (Euclidean Geometry Alignment & model.json Maps)   │
  └────────────────────────────────────────────────────────┘
```

### 1. Presentation Layer (`index.html` / `style.css`)

Provides the dark UI layout structure. Houses the twin layered viewport frame elements: a hidden raw camera video layer and a transparent active canvas element that displays real-time tracking lines.

### 2. Stream Control Layer (`script.js`)

Handles browser hardware access configurations (`navigator.mediaDevices.getUserMedia`). Once access is granted, it runs a continuous loop using `requestAnimationFrame`, updating dynamically as fast as the user's monitor refresh rate allows.

### 3. Feature Extraction Layer (MediaPipe Integration)

Processes live images frame-by-frame using internal tracking weights loaded from CDN repositories. It isolates structural hand geometries and outputs an indexed array tracking 21 specific 3D coordinate joints (each holding detailed X, Y, Z positions).

### 4. Vector Math Engine

Calculates relative finger joint extensions and compares them directly against localized classification mappings stored inside `model.json`.

---

## 📐 Mathematical Transformation Pipeline

Because pixel positions change drastically depending on a user's distance or angle relative to the webcam lens, the tracking layer transforms absolute data coordinates using Scale-Invariant Vector Profiling.

### Phase A: Establishing the Origin Coordinate Bound

The core engine defines the absolute wrist joint position (Node 0) as the origin (0, 0, 0) of a 3D grid system.

### Phase B: Calculating Spatial Relationships

For every joint index point *i* (1 to 20), the system calculates the exact spatial distance back to the wrist base node (x₀, y₀, z₀) using standard 3D Euclidean geometry calculations:

$$D_i = \sqrt{(x_i - x_0)^2 + (y_i - y_0)^2 + (z_i - z_0)^2}$$

### Phase C: Value Scaling and Standardization

To ensure tracking profiles remain consistent whether a hand is small or large, or close to or far from the camera, the system extracts the maximum distance value from the current frame:

$$D_{\text{max}} = \max(D_1, D_2, \dots, D_{20})$$

Every distance value is then divided by this maximum ceiling index, scaling all structural vector dimensions cleanly into a normalized range between 0.0 and 1.0:

$$V_{\text{normalized}}[i] = \frac{D_i}{D_{\text{max}}}$$

---

## 📊 Evaluation & Classification Layer

To verify match signatures quickly without resource-intensive neural re-evaluations, the system measures data closeness by calculating the inverse Euclidean distance against target configurations pre-loaded from `model.json`:

$$\text{Match Score} = \frac{1}{1 + \sqrt{\sum_{i=1}^{20} (V_{\text{live}}[i] - V_{\text{weight}}[i])^2}}$$

**State Transition Logic:**

- **Confidence Score ≥ 65%:** Renders the predicted character target output to the screen and expands the color-coded UI progress bar.
- **Confidence Score < 65%:** Resets the interface output to a fallback placeholder state (—), filtering out transient noise or incomplete movements.

---

## ⚡ Performance Optimization Mechanics

To maintain smooth frame-processing loops (averaging 24 to 30 FPS) on standard consumer laptops, the architecture uses specific runtime efficiency rules:

- **Garbage Collection Minimization:** Reuses canvas context boundaries directly instead of creating new drawing containers on each iteration, preventing frame drops during intensive rendering.
- **Asynchronous Processing Loops:** Runs non-blocking analysis frames sequentially using asynchronous code workflows (`async`/`await`), keeping UI rendering and asset loading entirely fluid.
- **Zero External Roundtrips:** Keeps network tracking completely clear after the initial loading phase, ensuring classification calculations run immediately within local application memory spaces.
