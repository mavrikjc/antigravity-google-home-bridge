/**
 * Antigravity Live · Google Home Bridge Client
 * Full-Duplex WebAudio PCM + Web Speech VAD + Barge-In + Google Home Room Switcher
 */

// Progressive enhancement for <input type="checkbox" switch>
if (!('switch' in HTMLInputElement.prototype)) {
  for (const el of document.querySelectorAll('input[type="checkbox"][switch]')) {
    el.setAttribute('role', 'switch');
  }
}

const settingsDialog = document.getElementById('settingsDialog');
const openSettingsBtn = document.getElementById('openSettingsBtn');
const closeSettingsBtn = document.getElementById('closeSettingsBtn');

openSettingsBtn.addEventListener('click', () => {
  settingsDialog.showModal();
});

closeSettingsBtn.addEventListener('click', () => {
  settingsDialog.close();
});

// MANDATORY fallback for browsers without native <dialog closedby="any"> support
if (!('closedBy' in HTMLDialogElement.prototype)) {
  settingsDialog.addEventListener('click', (event) => {
    if (event.target !== settingsDialog) return;
    const rect = settingsDialog.getBoundingClientRect();
    const isDialogContent = (
      rect.top <= event.clientY &&
      event.clientY <= rect.top + rect.height &&
      rect.left <= event.clientX &&
      event.clientX <= rect.left + rect.width
    );
    if (isDialogContent) return;
    settingsDialog.close();
  });
}

settingsDialog.addEventListener('close', () => {
  settingsDialog.returnValue = '';
});

// UI Elements
const connDot = document.getElementById('connDot');
const statusSubtitle = document.getElementById('statusSubtitle');
const activeRoomLabel = document.getElementById('activeRoomLabel');
const orbWrapper = document.getElementById('orbWrapper');
const micToggleBtn = document.getElementById('micToggleBtn');
const orbStateText = document.getElementById('orbStateText');
const roomPills = document.getElementById('roomPills');
const refreshSpeakersBtn = document.getElementById('refreshSpeakersBtn');
const transcriptFeed = document.getElementById('transcriptFeed');
const promptForm = document.getElementById('promptForm');
const promptInput = document.getElementById('promptInput');
const bargeInBtn = document.getElementById('bargeInBtn');
const customRoomInput = document.getElementById('customRoomInput');
const addCustomRoomBtn = document.getElementById('addCustomRoomBtn');
const waveformCanvas = document.getElementById('waveformCanvas');
const canvasCtx = waveformCanvas.getContext('2d');
const secureContextBanner = document.getElementById('secureContextBanner');
const openHttpsLink = document.getElementById('openHttpsLink');
const httpsTargetUrl = `https://${window.location.hostname}:8443`;
if (openHttpsLink) {
  openHttpsLink.href = httpsTargetUrl;
}
if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
  if (secureContextBanner) {
    secureContextBanner.hidden = false;
  }
}

// State
let ws = null;
let isLiveMicActive = false;
let audioCtx = null;
let micStream = null;
let processorNode = null;
let analyserNode = null;
let speechRecognition = null;
let nextPlayTime = 0;
let activeSources = [];
let activeRoom = 'Whole Home';
let routingMode = 'both';
let wakeLock = null;

function appendMessage(role, text) {
  const div = document.createElement('div');
  div.className = `msg msg-${role}`;
  div.textContent = text;
  transcriptFeed.appendChild(div);
  transcriptFeed.scrollTop = transcriptFeed.scrollHeight;
}

function setOrbState(state, label) {
  orbWrapper.dataset.state = state;
  if (label) {
    orbStateText.textContent = label;
  }
}

function updateRoomPills(selectedRoom, speakers = []) {
  activeRoom = selectedRoom || activeRoom;
  activeRoomLabel.textContent = activeRoom;

  const knownNames = new Set(['Whole Home', 'Office', 'Kitchen', 'Living Room', 'Bedroom']);
  for (const sp of speakers) {
    if (sp && sp.name) knownNames.add(sp.name);
  }
  knownNames.add(activeRoom);

  roomPills.innerHTML = '';
  for (const name of knownNames) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `room-pill${name.toLowerCase() === activeRoom.toLowerCase() ? ' active' : ''}`;
    btn.dataset.room = name;
    btn.textContent = name;
    btn.addEventListener('click', () => selectRoom(name));
    roomPills.appendChild(btn);
  }
}

async function selectRoom(roomName) {
  updateRoomPills(roomName);
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'set_room', room: roomName }));
  } else {
    await fetch('/api/cast/select', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ room: roomName }),
    });
  }
  appendMessage('system', `Google Home target speaker switched to: ${roomName}`);
}

// Interrupt / Barge-In: Immediately stop local WebAudio & WebSpeech playback and notify server
function triggerBargeIn() {
  if ('speechSynthesis' in window) {
    window.speechSynthesis.cancel();
  }
  for (const src of activeSources) {
    try {
      src.stop();
    } catch (_) {}
  }
  activeSources = [];
  if (audioCtx) {
    nextPlayTime = audioCtx.currentTime;
  }
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'barge_in' }));
  }
  if (isLiveMicActive) {
    setOrbState('listening', 'Listening Live');
  }
}

// Play incoming 24kHz 16-bit PCM audio chunk from Gemini Live seamlessly
function playPcmChunk(base64Pcm, sampleRate = 24000) {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate });
  }
  const binary = atob(base64Pcm);
  const sampleCount = binary.length / 2;
  const float32 = new Float32Array(sampleCount);
  for (let i = 0; i < sampleCount; i++) {
    let int16 = (binary.charCodeAt(i * 2 + 1) << 8) | binary.charCodeAt(i * 2);
    if (int16 >= 0x8000) int16 -= 0x10000;
    float32[i] = int16 / 32768.0;
  }

  const buffer = audioCtx.createBuffer(1, sampleCount, sampleRate);
  buffer.getChannelData(0).set(float32);

  const source = audioCtx.createBufferSource();
  source.buffer = buffer;
  source.connect(audioCtx.destination);

  const now = audioCtx.currentTime;
  if (nextPlayTime < now) nextPlayTime = now;
  source.start(nextPlayTime);
  nextPlayTime += buffer.duration;
  activeSources.push(source);
  setOrbState('speaking', 'Antigravity Speaking');

  source.onended = () => {
    activeSources = activeSources.filter((s) => s !== source);
    if (activeSources.length === 0) {
      setOrbState(isLiveMicActive ? 'listening' : 'idle', isLiveMicActive ? 'Listening Live' : 'Tap to Go Live');
    }
  };
}

// Browser speech synthesis fallback when running in turn-based / direct mode
function speakDirectBrowser(text) {
  if (!('speechSynthesis' in window) || !text) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1.04;
  utterance.onstart = () => setOrbState('speaking', 'Antigravity Speaking');
  utterance.onend = () => {
    setOrbState(isLiveMicActive ? 'listening' : 'idle', isLiveMicActive ? 'Listening Live' : 'Tap to Go Live');
  };
  window.speechSynthesis.speak(utterance);
}

// WebSocket Connection to Proxmox Antigravity Live Server
function connectWebSocket() {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${window.location.host}/ws/live`);

  ws.onopen = () => {
    connDot.classList.add('online');
    statusSubtitle.textContent = `Connected · Following in ${activeRoom}`;
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === 'status') {
      if (data.active_room) {
        updateRoomPills(data.active_room, data.speakers || []);
      }
      if (data.routing_mode) {
        routingMode = data.routing_mode;
      }
      statusSubtitle.textContent = `Live (${data.state || 'ready'}) · Speaker: ${activeRoom}`;
      if (data.state === 'thinking') {
        setOrbState('thinking', 'Thinking…');
      } else if (data.state === 'listening' && isLiveMicActive) {
        setOrbState('listening', 'Listening Live');
      }
    } else if (data.type === 'audio_chunk') {
      playPcmChunk(data.pcm_b64, data.sample_rate || 24000);
    } else if (data.type === 'transcript') {
      appendMessage(data.role === 'user' ? 'user' : 'agent', data.text);
      if (data.role === 'agent' && data.speak_direct) {
        speakDirectBrowser(data.text);
      }
    } else if (data.type === 'tool_call') {
      appendMessage('tool', `⚡ ${data.name}(${JSON.stringify(data.args)}) → ${data.result}`);
      if (data.active_room) {
        updateRoomPills(data.active_room);
      }
    } else if (data.type === 'interrupted') {
      setOrbState(isLiveMicActive ? 'listening' : 'idle', isLiveMicActive ? 'Listening Live' : 'Tap to Go Live');
    }
  };

  ws.onclose = () => {
    connDot.classList.remove('online');
    statusSubtitle.textContent = 'Reconnecting to Proxmox Bridge…';
    setTimeout(connectWebSocket, 2500);
  };
}

// Start / Stop Full-Duplex Microphone Capture
async function toggleLiveMicrophone() {
  if (isLiveMicActive) {
    stopLiveMicrophone();
    return;
  }

  if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    if (secureContextBanner) secureContextBanner.hidden = false;
    appendMessage(
      'system',
      `Mobile browsers require HTTPS for microphone access. Tap "Open Secure HTTPS (:8443)" above or go to ${httpsTargetUrl}`
    );
    window.location.href = httpsTargetUrl;
    return;
  }

  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
        sampleRate: 16000,
      },
    });

    audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    const source = audioCtx.createMediaStreamSource(micStream);
    analyserNode = audioCtx.createAnalyser();
    analyserNode.fftSize = 256;
    source.connect(analyserNode);

    // Stream 16kHz 16-bit PCM to server & run local energy barge-in detector
    processorNode = audioCtx.createScriptProcessor(2048, 1, 1);
    source.connect(processorNode);
    processorNode.connect(audioCtx.destination);

    processorNode.onaudioprocess = (e) => {
      if (!isLiveMicActive || !ws || ws.readyState !== WebSocket.OPEN) return;
      const input = e.inputBuffer.getChannelData(0);

      // Compute RMS for instant barge-in while Antigravity is speaking
      let sumSq = 0;
      const int16 = new Int16Array(input.length);
      for (let i = 0; i < input.length; i++) {
        const s = Math.max(-1, Math.min(1, input[i]));
        sumSq += s * s;
        int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      const rms = Math.sqrt(sumSq / input.length);
      if (rms > 0.045 && (activeSources.length > 0 || (window.speechSynthesis && window.speechSynthesis.speaking))) {
        triggerBargeIn();
      }

      const bytes = new Uint8Array(int16.buffer);
      let binary = '';
      for (let i = 0; i < bytes.byteLength; i++) {
        binary += String.fromCharCode(bytes[i]);
      }
      ws.send(JSON.stringify({ type: 'audio_chunk', pcm_b64: btoa(binary) }));
    };

    // Also start continuous Web Speech Recognition for immediate live transcript & offline fallback
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRec) {
      speechRecognition = new SpeechRec();
      speechRecognition.continuous = true;
      speechRecognition.interimResults = false;
      speechRecognition.lang = 'en-US';
      speechRecognition.onresult = (evt) => {
        const last = evt.results[evt.results.length - 1];
        if (last && last.isFinal) {
          const phrase = last[0].transcript.trim();
          if (phrase && ws && ws.readyState === WebSocket.OPEN) {
            triggerBargeIn();
            ws.send(JSON.stringify({ type: 'text', text: phrase }));
          }
        }
      };
      speechRecognition.onend = () => {
        if (isLiveMicActive) {
          try {
            speechRecognition.start();
          } catch (_) {}
        }
      };
      try {
        speechRecognition.start();
      } catch (_) {}
    }

    // Request Screen Wake Lock so walking room-to-room with phone in pocket stays active
    const wakeLockToggle = document.getElementById('wakeLockToggle');
    if (wakeLockToggle.checked && 'wakeLock' in navigator) {
      try {
        wakeLock = await navigator.wakeLock.request('screen');
      } catch (_) {}
    }

    isLiveMicActive = true;
    micToggleBtn.setAttribute('aria-pressed', 'true');
    setOrbState('listening', 'Listening Live');
    drawVisualizer();
  } catch (err) {
    appendMessage('system', `Microphone permission needed for live voice: ${err.message}`);
  }
}

function stopLiveMicrophone() {
  isLiveMicActive = false;
  micToggleBtn.setAttribute('aria-pressed', 'false');
  setOrbState('idle', 'Tap to Go Live');
  if (speechRecognition) {
    try {
      speechRecognition.stop();
    } catch (_) {}
    speechRecognition = null;
  }
  if (processorNode) {
    processorNode.disconnect();
    processorNode = null;
  }
  if (micStream) {
    for (const track of micStream.getTracks()) track.stop();
    micStream = null;
  }
  if (wakeLock) {
    wakeLock.release().catch(() => {});
    wakeLock = null;
  }
}

function drawVisualizer() {
  if (!isLiveMicActive || !analyserNode) {
    canvasCtx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    return;
  }
  requestAnimationFrame(drawVisualizer);

  const data = new Uint8Array(analyserNode.frequencyBinCount);
  analyserNode.getByteFrequencyData(data);
  let avg = 0;
  for (let i = 0; i < data.length; i++) avg += data[i];
  avg /= data.length;

  const w = waveformCanvas.width;
  const h = waveformCanvas.height;
  const cx = w / 2;
  const cy = h / 2;
  canvasCtx.clearRect(0, 0, w, h);

  const radius = 74 + (avg / 255) * 32;
  canvasCtx.beginPath();
  canvasCtx.arc(cx, cy, radius, 0, Math.PI * 2);
  canvasCtx.strokeStyle = 'rgba(56, 189, 248, 0.45)';
  canvasCtx.lineWidth = 4;
  canvasCtx.stroke();
}

// Event Listeners
micToggleBtn.addEventListener('click', toggleLiveMicrophone);
bargeInBtn.addEventListener('click', triggerBargeIn);

promptForm.addEventListener('submit', (e) => {
  e.preventDefault();
  const text = promptInput.value.trim();
  if (!text) return;
  promptInput.value = '';
  triggerBargeIn();
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'text', text }));
  }
});

refreshSpeakersBtn.addEventListener('click', async () => {
  refreshSpeakersBtn.textContent = 'Scanning…';
  try {
    const res = await fetch('/api/speakers?refresh=true');
    const data = await res.json();
    updateRoomPills(data.active_room, data.speakers);
    appendMessage('system', `Found ${data.speakers.length} Google Home speaker(s) on LAN.`);
  } finally {
    refreshSpeakersBtn.textContent = 'Scan LAN';
  }
});

addCustomRoomBtn.addEventListener('click', () => {
  const room = customRoomInput.value.trim();
  if (!room) return;
  customRoomInput.value = '';
  selectRoom(room);
});

for (const radio of document.querySelectorAll('input[name="routingMode"]')) {
  radio.addEventListener('change', (e) => {
    routingMode = e.target.value;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'set_routing', mode: routingMode }));
    }
  });
}

for (const pill of roomPills.querySelectorAll('.room-pill')) {
  pill.addEventListener('click', () => selectRoom(pill.dataset.room));
}

connectWebSocket();
