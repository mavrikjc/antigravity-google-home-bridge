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
  startAudioSettingsPolling();
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
  stopAudioSettingsPolling();
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
const activeTvLabel = document.getElementById('activeTvLabel');
const muteVoiceBtn = document.getElementById('muteVoiceBtn');
const voiceSelect = document.getElementById('voiceSelect');
const micLevelFill = document.getElementById('micLevelFill');
const micThresholdMarker = document.getElementById('micThresholdMarker');
const micGateBadge = document.getElementById('micGateBadge');
const calibrateMicBtn = document.getElementById('calibrateMicBtn');
const calibrateMicStatus = document.getElementById('calibrateMicStatus');
const micGainSlider = document.getElementById('micGainSlider');
const micGainVal = document.getElementById('micGainVal');
const micVadSlider = document.getElementById('micVadSlider');
const micVadVal = document.getElementById('micVadVal');
const micHangoverSlider = document.getElementById('micHangoverSlider');
const micHangoverVal = document.getElementById('micHangoverVal');
const geminiSilenceSlider = document.getElementById('geminiSilenceSlider');
const geminiSilenceVal = document.getElementById('geminiSilenceVal');

let audioSettingsPollTimer = null;

function renderAudioSettingsUI(data, updateSliders = true) {
  if (!data) return;
  if (updateSliders) {
    if (micGainSlider && data.hw_mic_gain !== undefined) {
      micGainSlider.value = String(data.hw_mic_gain);
      if (micGainVal) micGainVal.textContent = `${Number(data.hw_mic_gain).toFixed(1)}×`;
    }
    if (micVadSlider && data.hw_mic_vad_threshold !== undefined) {
      micVadSlider.value = String(data.hw_mic_vad_threshold);
      if (micVadVal) micVadVal.textContent = Number(data.hw_mic_vad_threshold).toFixed(3);
    }
    if (micHangoverSlider && data.hw_mic_silence_duration_sec !== undefined) {
      micHangoverSlider.value = String(data.hw_mic_silence_duration_sec);
      if (micHangoverVal) micHangoverVal.textContent = `${Number(data.hw_mic_silence_duration_sec).toFixed(1)}s`;
    }
    if (geminiSilenceSlider && data.gemini_live_silence_ms !== undefined) {
      geminiSilenceSlider.value = String(data.gemini_live_silence_ms);
      if (geminiSilenceVal) geminiSilenceVal.textContent = `${data.gemini_live_silence_ms} ms`;
    }
  }
  const meterMax = 0.12;
  if (micLevelFill && typeof data.live_post_rms === 'number') {
    const pct = Math.min(100, Math.round((data.live_post_rms / meterMax) * 100));
    micLevelFill.style.width = `${pct}%`;
  }
  if (micThresholdMarker && typeof data.hw_mic_vad_threshold === 'number') {
    const tPct = Math.min(100, Math.round((data.hw_mic_vad_threshold / meterMax) * 100));
    micThresholdMarker.style.left = `${tPct}%`;
  }
  if (micGateBadge && typeof data.gate_open === 'boolean') {
    micGateBadge.dataset.open = data.gate_open ? 'true' : 'false';
    micGateBadge.textContent = data.gate_open ? 'GATE OPEN' : 'GATE CLOSED';
  }
}

async function fetchAudioSettings(updateSliders = true) {
  try {
    const res = await fetch('/api/audio/settings');
    if (res.ok) {
      const data = await res.json();
      renderAudioSettingsUI(data, updateSliders);
    }
  } catch (_) {}
}

function startAudioSettingsPolling() {
  fetchAudioSettings(true);
  stopAudioSettingsPolling();
  audioSettingsPollTimer = setInterval(() => fetchAudioSettings(false), 250);
}

function stopAudioSettingsPolling() {
  if (audioSettingsPollTimer) {
    clearInterval(audioSettingsPollTimer);
    audioSettingsPollTimer = null;
  }
}

async function saveAudioSettings(patch) {
  try {
    const res = await fetch('/api/audio/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    });
    if (res.ok) {
      const data = await res.json();
      renderAudioSettingsUI(data, true);
    }
  } catch (_) {}
}

if (micGainSlider) {
  micGainSlider.addEventListener('input', () => {
    if (micGainVal) micGainVal.textContent = `${Number(micGainSlider.value).toFixed(1)}×`;
  });
  micGainSlider.addEventListener('change', () => {
    saveAudioSettings({ hw_mic_gain: Number(micGainSlider.value) });
  });
}

if (micVadSlider) {
  micVadSlider.addEventListener('input', () => {
    if (micVadVal) micVadVal.textContent = Number(micVadSlider.value).toFixed(3);
  });
  micVadSlider.addEventListener('change', () => {
    saveAudioSettings({ hw_mic_vad_threshold: Number(micVadSlider.value) });
  });
}

if (micHangoverSlider) {
  micHangoverSlider.addEventListener('input', () => {
    if (micHangoverVal) micHangoverVal.textContent = `${Number(micHangoverSlider.value).toFixed(1)}s`;
  });
  micHangoverSlider.addEventListener('change', () => {
    saveAudioSettings({ hw_mic_silence_duration_sec: Number(micHangoverSlider.value) });
  });
}

if (geminiSilenceSlider) {
  geminiSilenceSlider.addEventListener('input', () => {
    if (geminiSilenceVal) geminiSilenceVal.textContent = `${geminiSilenceSlider.value} ms`;
  });
  geminiSilenceSlider.addEventListener('change', () => {
    saveAudioSettings({ gemini_live_silence_ms: Number(geminiSilenceSlider.value) });
  });
}

if (calibrateMicBtn) {
  calibrateMicBtn.addEventListener('click', async () => {
    calibrateMicBtn.disabled = true;
    if (calibrateMicStatus) {
      calibrateMicStatus.textContent = 'Listening for 5s… Stay quiet 2s, then speak a normal sentence!';
    }
    try {
      const res = await fetch('/api/audio/calibrate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ duration_sec: 5.0 }),
      });
      if (res.ok) {
        const data = await res.json();
        renderAudioSettingsUI(data, true);
        if (calibrateMicStatus) {
          if (data.sample_count > 0) {
            calibrateMicStatus.textContent =
              `Calibrated! Noise=${data.noise_floor_rms}, Speech=${data.speech_rms} → Gain ${data.hw_mic_gain}×, VAD ${data.hw_mic_vad_threshold}`;
          } else {
            calibrateMicStatus.textContent =
              'No live hardware mic samples captured — ensure the Proxmox USB mic is active.';
          }
        }
      }
    } catch (_) {
      if (calibrateMicStatus) calibrateMicStatus.textContent = 'Calibration failed.';
    } finally {
      calibrateMicBtn.disabled = false;
    }
  });
}
const muteVoiceIcon = document.getElementById('muteVoiceIcon');
const muteVoiceLabel = document.getElementById('muteVoiceLabel');
const castHudBtn = document.getElementById('castHudBtn');
const castHudFeedback = document.getElementById('castHudFeedback');
const refreshProjectsBtn = document.getElementById('refreshProjectsBtn');
const activeTargetSummary = document.getElementById('activeTargetSummary');
const projectsNodeGrid = document.getElementById('projectsNodeGrid');
const activityScopeBadge = document.getElementById('activityScopeBadge');
const nodeActivityFeed = document.getElementById('nodeActivityFeed');
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
const isHomeDomain = window.location.hostname.endsWith('.home');
const httpsTargetUrl = isHomeDomain
  ? `https://${window.location.hostname}`
  : `https://${window.location.hostname}:8443`;
if (openHttpsLink) {
  openHttpsLink.href = httpsTargetUrl;
  openHttpsLink.textContent = isHomeDomain
    ? 'Open Secure HTTPS'
    : 'Open Secure HTTPS (:8443)';
}
if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
  if (secureContextBanner) {
    secureContextBanner.hidden = false;
  }
}

// State
let ws = null;
let isLiveMicActive = false;
let isVoiceMuted = false;

function updateMuteButton(muted) {
  isVoiceMuted = Boolean(muted);
  if (!muteVoiceBtn) return;
  muteVoiceBtn.dataset.muted = isVoiceMuted ? 'true' : 'false';
  muteVoiceBtn.setAttribute('aria-pressed', isVoiceMuted ? 'true' : 'false');
  if (muteVoiceIcon) muteVoiceIcon.textContent = isVoiceMuted ? '🔇' : '🎙️';
  if (muteVoiceLabel) muteVoiceLabel.textContent = isVoiceMuted ? 'Muted' : 'Mute';
}

if (muteVoiceBtn) {
  muteVoiceBtn.addEventListener('click', async () => {
    const nextMuted = !isVoiceMuted;
    updateMuteButton(nextMuted);
    try {
      const res = await fetch('/api/voice/mute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ muted: nextMuted }),
      });
      if (res.ok) {
        const data = await res.json();
        updateMuteButton(Boolean(data.voice_muted));
      }
    } catch (_) {}
  });
}

function updateVoiceSelect(voiceName, availableVoices) {
  if (!voiceSelect) return;
  if (Array.isArray(availableVoices) && availableVoices.length) {
    const existing = Array.from(voiceSelect.options).map((o) => o.value);
    const same =
      existing.length === availableVoices.length &&
      existing.every((v, idx) => v === availableVoices[idx]);
    if (!same) {
      voiceSelect.innerHTML = '';
      for (const v of availableVoices) {
        const opt = document.createElement('option');
        opt.value = v;
        opt.textContent = v;
        voiceSelect.appendChild(opt);
      }
    }
  }
  if (voiceName) {
    voiceSelect.value = voiceName;
  }
}

if (voiceSelect) {
  voiceSelect.addEventListener('change', async () => {
    const selected = voiceSelect.value;
    try {
      const res = await fetch('/api/voice', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ voice_name: selected }),
      });
      if (res.ok) {
        const data = await res.json();
        updateVoiceSelect(data.voice_name, data.available_voices);
      }
    } catch (_) {}
  });
}
let audioCtx = null; // playback context (24 kHz Gemini audio)
let captureCtx = null; // mic capture context (device-native rate)
let serverMode = null; // 'gemini-live-native-audio' when the server streams mic audio to Gemini
let micStream = null;
let processorNode = null;
let analyserNode = null;
let speechRecognition = null;
let nextPlayTime = 0;
let activeSources = [];
let activeRoom = '';
let activePresenceRoom = '';
let activeTv = '';
let voiceStatus = 'idle';
let routingMode = 'cast';
let wakeLock = null;
let knownSpeakers = [];
let knownTvs = [];
let projectsState = {
  active_project_id: null,
  active_agent_id: '101',
  active_project_agent: null,
  explicit_none: false,
  pending_project_id: null,
  projects: [],
};
let selectRequestSeq = 0;
let fetchProjectsSeq = 0;
const activeTokenEntries = new Map();
const MAX_ACTIVITY_ITEMS = 80;

function resolveActiveProjectAndAgent() {
  const projects = Array.isArray(projectsState.projects) ? projectsState.projects : [];
  const rawActiveId = projectsState.active_project_id;
  const matchedById = rawActiveId
    ? projects.find((p) => p && p.id === rawActiveId) || null
    : null;
  const isPendingUncached = Boolean(
    rawActiveId && !matchedById && projectsState.pending_project_id === rawActiveId
  );
  const allowFallback = !projectsState.explicit_none && !isPendingUncached;
  const activeProj = matchedById || (allowFallback ? (projects.find((p) => p && p.active) || null) : null);
  if (!activeProj) {
    return {
      proj: null,
      agent: null,
      agentId: projectsState.active_agent_id || '101',
      matchedById: false,
    };
  }
  const agentId = matchedById
    ? (projectsState.active_agent_id || activeProj.active_agent_id || '101')
    : (rawActiveId ? (activeProj.active_agent_id || '101') : (projectsState.active_agent_id || activeProj.active_agent_id || '101'));
  const agents = Array.isArray(activeProj.agents) ? activeProj.agents : [];
  const agent = agents.find((a) => a && a.agent_id === agentId) || null;
  return {
    proj: activeProj,
    agent,
    agentId,
    matchedById: Boolean(matchedById),
  };
}

function normalizeActiveProjects() {
  const projects = Array.isArray(projectsState.projects) ? projectsState.projects : [];
  if (!projects.length) {
    return;
  }
  const { proj: activeProj, agentId, matchedById } = resolveActiveProjectAndAgent();
  if (projectsState.active_project_id && !matchedById) {
    if (projectsState.pending_project_id === projectsState.active_project_id) {
      for (const p of projects) {
        if (p) p.active = false;
      }
      return;
    }
    projectsState.active_project_id = null;
    projectsState.active_agent_id = activeProj ? agentId : null;
    projectsState.active_project_agent = null;
  }
  if (!activeProj) {
    for (const p of projects) {
      if (p) p.active = false;
    }
    projectsState.active_project_agent = null;
    return;
  }
  for (const p of projects) {
    if (!p) continue;
    p.active = p.id === activeProj.id;
    if (p.active) {
      p.active_agent_id = agentId;
    }
  }
}

let lastUserTranscriptEl = null;

function appendMessage(role, text) {
  openTranscript = null;
  const div = document.createElement('div');
  div.className = `msg msg-${role}`;
  const span = document.createElement('span');
  span.className = 'msg-text';
  span.textContent = text;
  div.appendChild(span);
  transcriptFeed.appendChild(div);
  if (role === 'user') {
    lastUserTranscriptEl = div;
  }
  transcriptFeed.scrollTop = transcriptFeed.scrollHeight;
  return div;
}

// Live transcription arrives as word-level fragments: grow one bubble per speaker turn.
let openTranscript = null;
function appendTranscript(role, text, partial) {
  if (!partial) {
    appendMessage(role, text);
    return;
  }
  if (!openTranscript || openTranscript.role !== role) {
    const el = appendMessage(role, '');
    openTranscript = { role, el, textEl: el.querySelector('.msg-text') };
  }
  if (openTranscript.textEl) {
    openTranscript.textEl.textContent += text;
  }
  transcriptFeed.scrollTop = transcriptFeed.scrollHeight;
}

function attachPromptAudio(audioUrl) {
  if (!audioUrl || !lastUserTranscriptEl) return;
  if (lastUserTranscriptEl.querySelector('.prompt-audio-row')) return;
  const row = document.createElement('div');
  row.className = 'prompt-audio-row';

  const audio = document.createElement('audio');
  audio.className = 'prompt-audio';
  audio.controls = true;
  audio.preload = 'none';
  audio.src = audioUrl;
  audio.setAttribute('aria-label', 'Play recorded voice prompt');

  const dlBtn = document.createElement('button');
  dlBtn.type = 'button';
  dlBtn.className = 'prompt-audio-dl';
  dlBtn.textContent = '⬇ WAV';
  dlBtn.title = 'Download recorded prompt WAV';
  dlBtn.addEventListener('click', async () => {
    try {
      const res = await fetch(audioUrl);
      if (!res.ok) return;
      const blob = await res.blob();
      const objUrl = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = objUrl;
      a.download = audioUrl.split('/').pop() || 'voice_prompt.wav';
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(objUrl), 5000);
    } catch (_) {}
  });

  row.appendChild(audio);
  row.appendChild(dlBtn);
  lastUserTranscriptEl.appendChild(row);
  transcriptFeed.scrollTop = transcriptFeed.scrollHeight;
}

function int16ToBase64(int16) {
  const bytes = new Uint8Array(int16.buffer, int16.byteOffset, int16.byteLength);
  let binary = '';
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  }
  return btoa(binary);
}

function downsampleTo16k(input, inRate) {
  if (inRate === 16000) return input;
  const ratio = inRate / 16000;
  const out = new Float32Array(Math.floor(input.length / ratio));
  for (let o = 0; o < out.length; o++) {
    const start = Math.floor(o * ratio);
    const end = Math.min(input.length, Math.floor((o + 1) * ratio));
    let sum = 0;
    for (let i = start; i < end; i++) sum += input[i];
    out[o] = sum / Math.max(1, end - start);
  }
  return out;
}

function setOrbState(state, label) {
  orbWrapper.dataset.state = state;
  if (label) {
    orbStateText.textContent = label;
  }
}

function isTvDevice(dev) {
  if (!dev) return false;
  const role = String(dev.device_role || '').toLowerCase();
  const castType = String(dev.cast_type || '').toLowerCase();
  return role === 'tv' || castType === 'cast' || castType === 'tizen';
}

function updateTvDisplay(tvName, presenceRoom = null, tvs = null) {
  if (tvName !== undefined && tvName !== null) {
    activeTv = tvName;
  }
  if (presenceRoom) {
    activePresenceRoom = presenceRoom;
  }
  if (Array.isArray(tvs)) {
    knownTvs = tvs;
  }
  if (!activeTvLabel) return;

  const matchedTv = knownTvs.find(
    (t) => t && t.name && activeTv && t.name.toLowerCase() === activeTv.toLowerCase()
  );
  const parts = [];
  if (activePresenceRoom) parts.push(activePresenceRoom.replace(/_/g, ' '));
  if (activeTv) {
    const protoSuffix = matchedTv && matchedTv.protocol ? ` (${matchedTv.protocol})` : '';
    parts.push(`${activeTv}${protoSuffix}`);
  } else {
    parts.push('No TV mapped');
  }
  activeTvLabel.textContent = parts.join(' · ');
}

function updateRoomPills(selectedRoom, speakers = null) {
  activeRoom = selectedRoom || activeRoom;
  activeRoomLabel.textContent = activeRoom || 'Whole Home';
  if (Array.isArray(speakers)) {
    const filteredSpeakers = speakers.filter((sp) => sp && sp.name && !isTvDevice(sp));
    const discoveredTvs = speakers.filter((sp) => sp && sp.name && isTvDevice(sp));
    if (filteredSpeakers.length || speakers.length === 0) {
      knownSpeakers = filteredSpeakers;
    }
    if (discoveredTvs.length && knownTvs.length === 0) {
      knownTvs = discoveredTvs;
    }
  }

  // Only show speakers the server actually discovered on the LAN (no TVs or placeholders).
  const names = [];
  for (const sp of knownSpeakers) {
    if (sp && sp.name && !isTvDevice(sp) && !names.includes(sp.name)) {
      names.push(sp.name);
    }
  }
  if (activeRoom && !names.includes(activeRoom)) {
    const activeIsKnownTv = knownTvs.some(
      (t) => t && t.name && t.name.toLowerCase() === activeRoom.toLowerCase()
    );
    if (!activeIsKnownTv) names.unshift(activeRoom);
  }

  const buttons = [];
  for (const name of names) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `room-pill${name.toLowerCase() === activeRoom.toLowerCase() ? ' active' : ''}`;
    btn.dataset.room = name;
    btn.textContent = name;
    btn.addEventListener('click', () => selectRoom(name));
    buttons.push(btn);
  }
  roomPills.replaceChildren(...buttons);
}

function findProjectAgent(projectId, agentId) {
  const proj = (projectsState.projects || []).find((p) => p && p.id === projectId);
  if (!proj) return { proj: null, agent: null };
  const targetAgentId = agentId || proj.active_agent_id || '101';
  const agent = (proj.agents || []).find((a) => a && a.agent_id === targetAgentId) || null;
  return { proj, agent };
}

function appendNodeActivity(kind, scopeLabel, bodyText, taskKey = null) {
  if (!nodeActivityFeed) return;
  const emptyEl = nodeActivityFeed.querySelector('.activity-empty');
  if (emptyEl) emptyEl.remove();

  const li = document.createElement('li');
  li.className = 'node-activity-item';
  li.dataset.kind = kind;
  if (taskKey) li.dataset.taskKey = taskKey;

  const meta = document.createElement('div');
  meta.className = 'activity-item-meta';

  const kindSpan = document.createElement('span');
  kindSpan.textContent = kind.replace(/_/g, ' ').toUpperCase();
  meta.appendChild(kindSpan);

  if (scopeLabel) {
    const scopeSpan = document.createElement('span');
    scopeSpan.textContent = scopeLabel;
    meta.appendChild(scopeSpan);
  }

  const timeSpan = document.createElement('span');
  timeSpan.textContent = new Date().toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
  meta.appendChild(timeSpan);

  const body = document.createElement('p');
  body.className = 'activity-item-body';
  body.textContent = bodyText;

  li.appendChild(meta);
  li.appendChild(body);
  nodeActivityFeed.appendChild(li);

  while (nodeActivityFeed.children.length > MAX_ACTIVITY_ITEMS) {
    const first = nodeActivityFeed.firstElementChild;
    if (!first) break;
    if (first.dataset.taskKey) {
      activeTokenEntries.delete(first.dataset.taskKey);
    }
    first.remove();
  }
  nodeActivityFeed.scrollTop = nodeActivityFeed.scrollHeight;
  return li;
}

function coalesceNodeToken(data) {
  if (!nodeActivityFeed) return;
  const key = `${data.project_id || 'default'}:${data.agent_id || '101'}:${data.task_id || 'stream'}`;
  const chunk = data.text || '';
  if (!chunk) return;

  let entry = activeTokenEntries.get(key);
  if (!entry || !entry.el.isConnected) {
    const scope = data.project_agent || data.short_name || data.project_id || 'agent';
    const li = appendNodeActivity('token', scope, chunk, key);
    if (li) {
      entry = { el: li, bodyEl: li.querySelector('.activity-item-body'), text: chunk };
      activeTokenEntries.set(key, entry);
    }
    return;
  }
  entry.text += chunk;
  if (entry.text.length > 1200) {
    entry.text = `…${entry.text.slice(-1100)}`;
  }
  entry.bodyEl.textContent = entry.text;
  nodeActivityFeed.scrollTop = nodeActivityFeed.scrollHeight;
}

function finalizeTokenPreview(data) {
  const key = `${data.project_id || 'default'}:${data.agent_id || '101'}:${data.task_id || 'stream'}`;
  activeTokenEntries.delete(key);
}

function renderProjectsNodeView() {
  if (!projectsNodeGrid) return;
  normalizeActiveProjects();
  const projects = Array.isArray(projectsState.projects) ? projectsState.projects : [];
  const { proj: activeProj, agent: activeAgent, agentId: activeAgentId } = resolveActiveProjectAndAgent();
  const isPendingUncached = Boolean(
    !activeProj &&
    projectsState.active_project_id &&
    projectsState.pending_project_id === projectsState.active_project_id
  );
  const activeLabel = activeProj
    ? ((activeAgent && activeAgent.project_agent) ||
       projectsState.active_project_agent ||
       (activeProj.short_name ? `project ${activeProj.short_name}, agent ${activeAgentId}` : null) ||
       activeProj.name ||
       activeProj.id)
    : ((projects.length === 0 || isPendingUncached)
       ? (projectsState.active_project_agent || projectsState.active_project_id || 'None selected')
       : 'None selected');

  if (activeTargetSummary) {
    activeTargetSummary.textContent = `Voice Target: ${activeLabel}`;
  }
  if (activityScopeBadge) {
    activityScopeBadge.textContent = activeLabel;
  }

  if (!projects.length) {
    const empty = document.createElement('p');
    empty.className = 'nodes-empty';
    empty.textContent = 'No workspaces discovered yet.';
    projectsNodeGrid.replaceChildren(empty);
    return;
  }

  const cards = [];
  for (const proj of projects) {
    if (!proj) continue;
    const isProjActive = Boolean(activeProj && proj.id === activeProj.id);

    const card = document.createElement('article');
    card.className = 'project-node-card';
    card.dataset.projectId = proj.id;
    card.dataset.active = isProjActive ? 'true' : 'false';

    const projBtn = document.createElement('button');
    projBtn.type = 'button';
    projBtn.className = 'node-select-btn project-select-btn';
    projBtn.setAttribute('aria-pressed', isProjActive ? 'true' : 'false');
    if (isProjActive) {
      projBtn.setAttribute('aria-current', 'true');
    }
    projBtn.addEventListener('click', () => {
      selectProjectTarget({ project: proj.id, agent: proj.active_agent_id || '101' });
    });

    const topRow = document.createElement('span');
    topRow.className = 'project-btn-top';

    const nameBlock = document.createElement('span');
    nameBlock.className = 'project-name-block';
    const shortEl = document.createElement('span');
    shortEl.className = 'project-short-name';
    shortEl.textContent = proj.short_name || proj.name || proj.id;
    nameBlock.appendChild(shortEl);

    if (proj.name && proj.short_name && proj.name !== proj.short_name) {
      const rawEl = document.createElement('span');
      rawEl.className = 'project-raw-name node-meta-mono';
      rawEl.textContent = proj.name;
      nameBlock.appendChild(rawEl);
    }
    topRow.appendChild(nameBlock);

    const badges = document.createElement('span');
    badges.className = 'node-badges-row';

    if (isProjActive) {
      const activeBadge = document.createElement('span');
      activeBadge.className = 'node-badge is-active';
      activeBadge.textContent = 'Active';
      badges.appendChild(activeBadge);
    }

    const branchBadge = document.createElement('span');
    branchBadge.className = 'node-badge git-branch-chip';
    branchBadge.textContent = proj.git_branch ? `git:${proj.git_branch}` : 'no-git';
    badges.appendChild(branchBadge);

    const dirtyCount = Number(proj.uncommitted_changes || 0);
    const dirtyBadge = document.createElement('span');
    dirtyBadge.className = `node-badge${dirtyCount > 0 ? ' dirty' : ''}`;
    dirtyBadge.textContent = dirtyCount > 0 ? `${dirtyCount} dirty` : 'clean';
    badges.appendChild(dirtyBadge);

    topRow.appendChild(badges);
    projBtn.appendChild(topRow);
    card.appendChild(projBtn);

    const agentsList = document.createElement('ul');
    agentsList.className = 'agent-node-list';
    agentsList.setAttribute('role', 'list');

    const agents = Array.isArray(proj.agents) ? proj.agents : [];
    for (const agent of agents) {
      if (!agent) continue;
      const isAgentActive = isProjActive && agent.agent_id === activeAgentId;
      const li = document.createElement('li');

      const agentBtn = document.createElement('button');
      agentBtn.type = 'button';
      agentBtn.className = 'node-select-btn agent-select-btn';
      agentBtn.setAttribute('aria-pressed', isAgentActive ? 'true' : 'false');
      if (isAgentActive) {
        agentBtn.setAttribute('aria-current', 'true');
      }
      agentBtn.addEventListener('click', () => {
        selectProjectTarget({ project: proj.id, agent: agent.agent_id });
      });

      const agTop = document.createElement('span');
      agTop.className = 'agent-btn-top';

      const agName = document.createElement('span');
      agName.className = 'agent-verbal-name';
      agName.textContent = agent.project_agent || `${proj.short_name || proj.name}, agent ${agent.agent_id}`;
      agTop.appendChild(agName);

      const statusBadge = document.createElement('span');
      const st = agent.status || 'idle';
      statusBadge.className = 'node-badge';
      statusBadge.dataset.status = st;
      statusBadge.textContent = st;
      agTop.appendChild(statusBadge);
      agentBtn.appendChild(agTop);

      const convEl = document.createElement('span');
      convEl.className = 'node-meta-mono conv-id';
      convEl.textContent = `thread: ${agent.conversation_id || 'new thread'}`;
      agentBtn.appendChild(convEl);

      const summaryText = agent.current_task || agent.last_summary;
      if (summaryText) {
        const sumEl = document.createElement('span');
        sumEl.className = 'agent-summary-line';
        sumEl.textContent = agent.current_task
          ? `Running: ${agent.current_task}`
          : agent.last_summary;
        agentBtn.appendChild(sumEl);
      }

      li.appendChild(agentBtn);

      const swarmNodes = Array.isArray(agent.swarm_nodes) ? agent.swarm_nodes : [];
      if (swarmNodes.length) {
        const swarmList = document.createElement('ul');
        swarmList.className = 'swarm-chip-list';
        swarmList.setAttribute('role', 'list');
        for (const node of swarmNodes) {
          if (!node) continue;
          const chip = document.createElement('li');
          chip.className = 'swarm-chip';
          const label = node.label || node.node_id || 'swarm';
          chip.textContent = `${label} · ${node.status || 'idle'}`;
          swarmList.appendChild(chip);
        }
        li.appendChild(swarmList);
      }

      agentsList.appendChild(li);
    }

    card.appendChild(agentsList);
    cards.push(card);
  }

  projectsNodeGrid.replaceChildren(...cards);
}

async function fetchProjects() {
  const reqSeq = ++fetchProjectsSeq;
  const selectSeqAtStart = selectRequestSeq;
  try {
    const res = await fetch('/api/projects');
    if (!res.ok) return null;
    const data = await res.json();
    if (reqSeq !== fetchProjectsSeq || selectSeqAtStart !== selectRequestSeq) {
      return null;
    }
    projectsState = {
      active_project_id: data.active_project_id || null,
      active_agent_id: data.active_agent_id || '101',
      active_project_agent: data.active_project_agent || null,
      explicit_none: data.active_project_id === null,
      pending_project_id: null,
      projects: Array.isArray(data.projects) ? data.projects : [],
    };
    normalizeActiveProjects();
    renderProjectsNodeView();
    return data;
  } catch (_) {
    return null;
  }
}

async function fetchSpeakers(refresh = false) {
  try {
    const url = refresh ? '/api/speakers?refresh=true' : '/api/speakers';
    const res = await fetch(url);
    if (!res.ok) return null;
    const data = await res.json();
    const speakersList = Array.isArray(data) ? data : (data.speakers || []);
    const tvsList = Array.isArray(data)
      ? data.filter((d) => isTvDevice(d))
      : (data.tvs || []);
    updateRoomPills(data.active_room, speakersList);
    updateTvDisplay(data.active_tv, data.active_presence_room, tvsList);
    if (data.routing_mode) {
      syncRoutingRadios(data.routing_mode);
    }
    return data;
  } catch (_) {
    return null;
  }
}

async function fetchHealth() {
  const selectSeqAtStart = selectRequestSeq;
  const fetchSeqAtStart = fetchProjectsSeq;
  try {
    const res = await fetch('/api/health');
    if (!res.ok) return null;
    const data = await res.json();
    if (data.active_room) {
      updateRoomPills(data.active_room);
    }
    updateTvDisplay(data.active_tv, data.active_presence_room);
    if (data.routing_mode) {
      syncRoutingRadios(data.routing_mode);
    }
    if (data.voice_status) {
      voiceStatus = data.voice_status;
    }
    if (typeof data.voice_muted === 'boolean') {
      updateMuteButton(data.voice_muted);
    }
    if (data.voice_name || data.available_voices) {
      updateVoiceSelect(data.voice_name, data.available_voices);
    }
    if (
      (data.active_project_id !== undefined || data.active_project_agent !== undefined) &&
      selectSeqAtStart === selectRequestSeq &&
      fetchSeqAtStart === fetchProjectsSeq
    ) {
      if (data.active_project_id === null) {
        projectsState.active_project_id = null;
        projectsState.explicit_none = true;
        projectsState.pending_project_id = null;
        projectsState.active_project_agent = null;
        if (data.active_agent_id !== undefined) {
          projectsState.active_agent_id = data.active_agent_id;
        }
      } else {
        const prevProjectId = projectsState.active_project_id;
        const prevAgentId = projectsState.active_agent_id;
        if (data.active_project_id) {
          projectsState.active_project_id = data.active_project_id;
          projectsState.explicit_none = false;
          const inCache = (projectsState.projects || []).some(
            (p) => p && p.id === data.active_project_id
          );
          projectsState.pending_project_id = inCache ? null : data.active_project_id;
        }
        projectsState.active_agent_id = data.active_agent_id || projectsState.active_agent_id;
        if (data.active_project_agent) {
          projectsState.active_project_agent = data.active_project_agent;
        } else if (
          (data.active_project_id && data.active_project_id !== prevProjectId) ||
          (data.active_agent_id && data.active_agent_id !== prevAgentId)
        ) {
          const matchedProj =
            (projectsState.projects || []).find(
              (p) => p && p.id === projectsState.active_project_id
            ) || null;
          const targetAgentId =
            projectsState.active_agent_id ||
            (matchedProj && matchedProj.active_agent_id) ||
            '101';
          const matchedAgent = matchedProj
            ? (matchedProj.agents || []).find((a) => a && a.agent_id === targetAgentId) || null
            : null;
          projectsState.active_project_agent =
            (matchedAgent && matchedAgent.project_agent) ||
            (matchedProj && matchedProj.short_name
              ? `project ${matchedProj.short_name}, agent ${targetAgentId}`
              : null);
        }
      }
      normalizeActiveProjects();
      renderProjectsNodeView();
    }
    return data;
  } catch (_) {
    return null;
  }
}

async function hydrateBridgeState() {
  await Promise.allSettled([fetchHealth(), fetchSpeakers(false), fetchProjects()]);
}

async function selectProjectTarget({ project = null, agent = null, project_agent = null }) {
  const reqSeq = ++selectRequestSeq;
  try {
    const res = await fetch('/api/projects/select', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ project, agent, project_agent }),
    });
    const data = await res.json();
    if (reqSeq !== selectRequestSeq) {
      return;
    }
    if (!res.ok) {
      appendMessage('system', `Unable to switch project: ${data.detail || res.statusText}`);
      return;
    }
    projectsState.active_project_id = data.project_id;
    projectsState.active_agent_id = data.agent_id;
    projectsState.active_project_agent = data.project_agent;
    projectsState.explicit_none = data.project_id === null;
    projectsState.pending_project_id =
      data.project_id && !(projectsState.projects || []).some((p) => p && p.id === data.project_id)
        ? data.project_id
        : null;
    normalizeActiveProjects();
    for (const p of projectsState.projects || []) {
      if (!p) continue;
      if (p.active) {
        const ag = (p.agents || []).find((a) => a && a.agent_id === data.agent_id);
        if (ag && data.conversation_id) {
          ag.conversation_id = data.conversation_id;
        }
      }
    }
    renderProjectsNodeView();
    appendMessage('system', `Voice routing switched to ${data.project_agent}`);
    fetchProjects();
  } catch (err) {
    if (reqSeq !== selectRequestSeq) {
      return;
    }
    appendMessage('system', `Project selection error: ${err.message}`);
  }
}

function renderCastHudFeedback(data) {
  if (!castHudFeedback) return;
  castHudFeedback.hidden = false;
  castHudFeedback.dataset.status = data.status || 'info';

  const summary = document.createElement('span');
  const tvPart = data.tv ? `${data.tv}` : 'Room TV';
  const protoPart = data.protocol ? ` (${data.protocol})` : '';
  summary.textContent = `${tvPart}${protoPart}: ${data.message || data.status || 'Updated'}`;

  const nodes = [summary];
  if (data.hud_url) {
    let safeHref = null;
    try {
      const baseOrigin =
        (window.location && window.location.origin && window.location.origin !== 'null')
          ? window.location.origin
          : `${(window.location && window.location.protocol) || 'http:'}//${(window.location && window.location.host) || 'localhost'}`;
      const parsed = new URL(String(data.hud_url), baseOrigin);
      if (parsed.protocol === 'http:' || parsed.protocol === 'https:') {
        safeHref = parsed.href;
      }
    } catch (_) {
      safeHref = null;
    }
    if (safeHref) {
      const link = document.createElement('a');
      link.href = safeHref;
      link.target = '_blank';
      link.rel = 'noopener';
      link.textContent = `Open ${data.hud_url}`;
      nodes.push(link);
    } else {
      const plainUrl = document.createElement('span');
      plainUrl.className = 'node-meta-mono';
      plainUrl.textContent = String(data.hud_url);
      nodes.push(plainUrl);
    }
  }
  castHudFeedback.replaceChildren(...nodes);
}

async function castHudToRoomTv() {
  if (!castHudBtn) return;
  castHudBtn.disabled = true;
  castHudBtn.setAttribute('aria-busy', 'true');
  try {
    const payload = {};
    if (activePresenceRoom) payload.room = activePresenceRoom;
    if (activeTv) payload.tv = activeTv;
    const res = await fetch('/api/cast/hud', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    renderCastHudFeedback(data);
    appendMessage('system', `Cast HUD (${data.tv || activePresenceRoom || 'TV'}): ${data.message || data.status}`);
  } catch (err) {
    renderCastHudFeedback({ status: 'error', message: err.message });
  } finally {
    castHudBtn.disabled = false;
    castHudBtn.removeAttribute('aria-busy');
  }
}

function syncRoutingRadios(mode) {
  routingMode = mode;
  for (const radio of document.querySelectorAll('input[name="routingMode"]')) {
    radio.checked = radio.value === mode;
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

function handleEventHubMessage(data) {
  const scope = data.project_agent || data.short_name || data.project_id || '';
  if (data.type === 'hud_snapshot') {
    ++fetchProjectsSeq;
    if (data.active_room) updateRoomPills(data.active_room);
    updateTvDisplay(data.active_tv, data.active_presence_room);
    if (data.voice_status) voiceStatus = data.voice_status;
    if (data.active_project_id !== undefined) {
      projectsState.active_project_id = data.active_project_id;
    }
    if (data.active_agent_id !== undefined) {
      projectsState.active_agent_id = data.active_agent_id;
    }
    if (data.active_project_agent !== undefined) {
      projectsState.active_project_agent = data.active_project_agent;
    }
    if (Array.isArray(data.projects)) {
      projectsState.projects = data.projects;
    }
    if (data.active_project_id === null) {
      projectsState.explicit_none = true;
      projectsState.pending_project_id = null;
      for (const p of projectsState.projects || []) {
        if (p) p.active = false;
      }
      if (data.active_project_agent === undefined) {
        projectsState.active_project_agent = null;
      }
    } else if (data.active_project_id) {
      projectsState.explicit_none = false;
      projectsState.pending_project_id = null;
    }
    if (
      Array.isArray(data.projects) ||
      data.active_project_id !== undefined ||
      data.active_agent_id !== undefined ||
      data.active_project_agent !== undefined
    ) {
      normalizeActiveProjects();
      renderProjectsNodeView();
    }
    return true;
  }

  if (data.type === 'project_changed') {
    ++fetchProjectsSeq;
    const nextProjectId =
      data.project_id !== undefined ? data.project_id : projectsState.active_project_id;
    const projectChanged = nextProjectId !== projectsState.active_project_id;
    projectsState.active_project_id = nextProjectId;
    if (data.project_id === null) {
      projectsState.explicit_none = true;
      projectsState.pending_project_id = null;
    } else if (nextProjectId) {
      projectsState.explicit_none = false;
      const inCache = (projectsState.projects || []).some((p) => p && p.id === nextProjectId);
      projectsState.pending_project_id = inCache ? null : nextProjectId;
    }
    if (data.agent_id !== undefined) {
      projectsState.active_agent_id = data.agent_id;
    } else if (projectChanged) {
      const matchedTarget =
        (projectsState.projects || []).find((p) => p && p.id === nextProjectId) || null;
      projectsState.active_agent_id = (matchedTarget && matchedTarget.active_agent_id) || '101';
    }
    if (data.project_agent !== undefined) {
      projectsState.active_project_agent = data.project_agent;
    } else {
      const matchedProj =
        (projectsState.projects || []).find((p) => p && p.id === projectsState.active_project_id) ||
        null;
      const targetAgentId =
        projectsState.active_agent_id || (matchedProj && matchedProj.active_agent_id) || '101';
      const matchedAgent = matchedProj
        ? (matchedProj.agents || []).find((a) => a && a.agent_id === targetAgentId) || null
        : null;
      projectsState.active_project_agent =
        (matchedAgent && matchedAgent.project_agent) ||
        (data.short_name ? `project ${data.short_name}, agent ${targetAgentId}` : null) ||
        (data.project_name ? `project ${data.project_name}, agent ${targetAgentId}` : null) ||
        (matchedProj && matchedProj.short_name
          ? `project ${matchedProj.short_name}, agent ${targetAgentId}`
          : null);
    }
    normalizeActiveProjects();
    renderProjectsNodeView();
    appendNodeActivity(
      'project_changed',
      scope || projectsState.active_project_agent || '',
      `Switched active target to ${projectsState.active_project_agent || data.project_id || 'project'}`
    );
    fetchProjects();
    return true;
  }

  if (data.type === 'room_changed') {
    if (data.active_room) {
      updateRoomPills(data.active_room);
    }
    updateTvDisplay(data.active_tv, data.active_presence_room || data.presence_room);
    statusSubtitle.textContent = `Connected · Speaker: ${activeRoom || 'Whole Home'}`;
    appendNodeActivity(
      'room_changed',
      data.active_presence_room || data.presence_room || 'presence',
      `Speaker: ${data.active_room || activeRoom || 'none'} · TV: ${data.active_tv || 'none'}`
    );
    return true;
  }

  if (data.type === 'task_started') {
    const { agent } = findProjectAgent(data.project_id, data.agent_id);
    if (agent) {
      agent.status = 'running';
      agent.current_task = data.prompt || 'Running task…';
      if (data.conversation_id) agent.conversation_id = data.conversation_id;
      renderProjectsNodeView();
    }
    appendNodeActivity('task_started', scope, data.prompt || 'Task started');
    return true;
  }

  if (data.type === 'thought') {
    appendNodeActivity('thought', scope, data.text || '');
    return true;
  }

  if (data.type === 'tool_result') {
    let resultStr = 'ok';
    if (data.result !== undefined && data.result !== null && data.result !== '') {
      resultStr = typeof data.result === 'object' ? JSON.stringify(data.result) : String(data.result);
    }
    const hasError = data.error !== undefined && data.error !== null && data.error !== '';
    const errStr = hasError
      ? (typeof data.error === 'object' ? JSON.stringify(data.error) : String(data.error))
      : '';
    const body = hasError
      ? `✖ ${data.name || 'tool'}: ${errStr}`
      : `✓ ${data.name || 'tool'}: ${resultStr}`;
    appendNodeActivity('tool_result', scope, body);
    if (data.name === 'start_subagent') {
      fetchProjects();
    }
    return true;
  }

  if (data.type === 'token') {
    coalesceNodeToken(data);
    return true;
  }

  if (data.type === 'task_completed') {
    finalizeTokenPreview(data);
    const { agent } = findProjectAgent(data.project_id, data.agent_id);
    if (agent) {
      agent.status = 'idle';
      agent.current_task = null;
      agent.last_summary = data.summary || data.text || agent.last_summary;
      if (data.conversation_id) agent.conversation_id = data.conversation_id;
      renderProjectsNodeView();
    }
    appendNodeActivity('task_completed', scope, data.summary || data.text || 'Completed');
    fetchProjects();
    return true;
  }

  if (data.type === 'task_failed') {
    finalizeTokenPreview(data);
    const { agent } = findProjectAgent(data.project_id, data.agent_id);
    if (agent) {
      agent.status = 'error';
      agent.current_task = null;
      agent.last_summary = data.error || 'Task failed';
      renderProjectsNodeView();
    }
    appendNodeActivity('task_failed', scope, data.error || 'Task failed');
    return true;
  }

  if (data.type === 'transcript') {
    appendTranscript(data.role === 'user' ? 'user' : 'agent', data.text, Boolean(data.partial));
    return true;
  }

  if (data.type === 'transcript_audio') {
    if (data.role === 'user' && data.audio_url) {
      attachPromptAudio(data.audio_url);
    }
    return true;
  }

  if (data.type === 'voice_changed') {
    if (data.voice_name) {
      updateVoiceSelect(data.voice_name);
    }
    return true;
  }

  if (data.type === 'voice_status') {
    voiceStatus = data.voice_status || data.status || voiceStatus;
    if (typeof data.voice_muted === 'boolean') {
      updateMuteButton(data.voice_muted);
    } else if (voiceStatus === 'muted') {
      updateMuteButton(true);
    } else if (voiceStatus === 'listening' || voiceStatus === 'speaking' || voiceStatus === 'thinking') {
      updateMuteButton(false);
    }
    if (data.active_room) updateRoomPills(data.active_room);
    return true;
  }

  return false;
}

// WebSocket Connection to Proxmox Antigravity Live Server
function connectWebSocket() {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${window.location.host}/ws/live`);

  ws.onopen = () => {
    connDot.classList.add('online');
    statusSubtitle.textContent = `Connected · Following in ${activeRoom || 'Whole Home'}`;
  };

  ws.onmessage = (event) => {
    let data;
    try {
      data = JSON.parse(event.data);
    } catch (_) {
      return;
    }

    if (data.type === 'status') {
      if (data.mode) {
        serverMode = data.mode;
      }
      if (data.active_room || Array.isArray(data.speakers)) {
        updateRoomPills(data.active_room, data.speakers || null);
      }
      if (data.active_tv !== undefined || data.active_presence_room) {
        updateTvDisplay(data.active_tv, data.active_presence_room, data.tvs || null);
      }
      if (data.routing_mode) {
        syncRoutingRadios(data.routing_mode);
      }
      if (data.state) {
        voiceStatus = data.state;
      }
      if (typeof data.voice_muted === 'boolean') {
        updateMuteButton(data.voice_muted);
      }
      if (data.voice_name) {
        updateVoiceSelect(data.voice_name);
      }
      statusSubtitle.textContent = `Live (${data.state || 'ready'}) · Speaker: ${activeRoom || 'Whole Home'}`;
      if (data.state === 'thinking') {
        setOrbState('thinking', 'Thinking…');
      } else if (data.state === 'listening' && isLiveMicActive) {
        setOrbState('listening', 'Listening Live');
      }
    } else if (data.type === 'audio_chunk') {
      playPcmChunk(data.pcm_b64, data.sample_rate || 24000);
    } else if (data.type === 'transcript') {
      appendTranscript(data.role === 'user' ? 'user' : 'agent', data.text, Boolean(data.partial));
      if (data.role === 'agent' && data.speak_direct) {
        speakDirectBrowser(data.text);
      }
    } else if (data.type === 'tool_call') {
      const argsStr = JSON.stringify(data.args || {});
      if (data.task_id || data.project_id || data.result === undefined) {
        const scope = data.project_agent || data.short_name || data.project_id || '';
        appendNodeActivity('tool_call', scope, `⚡ ${data.name}(${argsStr})`);
      } else {
        appendMessage('tool', `⚡ ${data.name}(${argsStr}) → ${data.result}`);
        if (data.active_room) {
          updateRoomPills(data.active_room);
        }
        if (data.name === 'switch_project') {
          fetchProjects();
        }
      }
    } else if (data.type === 'interrupted') {
      setOrbState(isLiveMicActive ? 'listening' : 'idle', isLiveMicActive ? 'Listening Live' : 'Tap to Go Live');
    } else {
      handleEventHubMessage(data);
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
      `Mobile browsers require HTTPS for microphone access. Tap the secure HTTPS link above or go to ${httpsTargetUrl}`
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
      },
    });

    // Capture at the device's native rate (iOS often ignores 16 kHz requests and would
    // hand us 48 kHz audio mislabeled as 16 kHz). We downsample to 16 kHz ourselves.
    captureCtx = new (window.AudioContext || window.webkitAudioContext)();
    const source = captureCtx.createMediaStreamSource(micStream);
    analyserNode = captureCtx.createAnalyser();
    analyserNode.fftSize = 256;
    source.connect(analyserNode);

    const handleFrame = (int16, rms) => {
      if (!isLiveMicActive || !ws || ws.readyState !== WebSocket.OPEN) return;
      // Local barge-in only matters when audio plays on this device
      if (rms > 0.045 && (activeSources.length > 0 || (window.speechSynthesis && window.speechSynthesis.speaking))) {
        triggerBargeIn();
      }
      ws.send(JSON.stringify({ type: 'audio_chunk', pcm_b64: int16ToBase64(int16) }));
    };

    if (captureCtx.audioWorklet) {
      await captureCtx.audioWorklet.addModule('/static/mic-worklet.js');
      processorNode = new AudioWorkletNode(captureCtx, 'mic-capture');
      processorNode.port.onmessage = (e) => handleFrame(new Int16Array(e.data.pcm), e.data.rms);
      source.connect(processorNode);
    } else {
      // Fallback for browsers without AudioWorklet: ScriptProcessor + JS downsampling
      processorNode = captureCtx.createScriptProcessor(1024, 1, 1);
      source.connect(processorNode);
      processorNode.connect(captureCtx.destination);
      processorNode.onaudioprocess = (e) => {
        const down = downsampleTo16k(e.inputBuffer.getChannelData(0), captureCtx.sampleRate);
        let sumSq = 0;
        const int16 = new Int16Array(down.length);
        for (let i = 0; i < down.length; i++) {
          const s = Math.max(-1, Math.min(1, down[i]));
          sumSq += s * s;
          int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        handleFrame(int16, Math.sqrt(sumSq / Math.max(1, down.length)));
      };
    }

    // Web Speech is only used when the server is in turn-based fallback mode. With native
    // Gemini Live audio the server already hears the mic stream, so sending recognized text
    // too would make Gemini answer the same request twice.
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRec && serverMode !== 'gemini-live-native-audio') {
      speechRecognition = new SpeechRec();
      speechRecognition.continuous = true;
      speechRecognition.interimResults = false;
      speechRecognition.lang = 'en-US';
      speechRecognition.onresult = (evt) => {
        if (serverMode === 'gemini-live-native-audio') return;
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
    if (processorNode.port) processorNode.port.onmessage = null;
    processorNode = null;
  }
  if (captureCtx) {
    captureCtx.close().catch(() => {});
    captureCtx = null;
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
    const data = await fetchSpeakers(true);
    if (data) {
      const speakersCount = Array.isArray(data)
        ? data.filter((d) => !isTvDevice(d)).length
        : (data.speakers || []).filter((d) => !isTvDevice(d)).length;
      appendMessage('system', `Found ${speakersCount} Google Home speaker(s) on LAN.`);
    }
  } finally {
    refreshSpeakersBtn.textContent = 'Scan LAN';
  }
});

if (castHudBtn) {
  castHudBtn.addEventListener('click', castHudToRoomTv);
}

if (refreshProjectsBtn) {
  refreshProjectsBtn.addEventListener('click', async () => {
    refreshProjectsBtn.textContent = 'Refreshing…';
    try {
      await fetchProjects();
    } finally {
      refreshProjectsBtn.textContent = 'Refresh';
    }
  });
}

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

hydrateBridgeState();
connectWebSocket();
