/**
 * Antigravity JARVIS HUD · Strictly Read-Only 10-Foot TV Dashboard Client
 */

const hudConnDot = document.getElementById('hudConnDot');
const hudConnStatus = document.getElementById('hudConnStatus');
const hudVoiceStatus = document.getElementById('hudVoiceStatus');
const hudVoiceIcon = document.getElementById('hudVoiceIcon');
const hudVoiceLabel = document.getElementById('hudVoiceLabel');
const hudClock = document.getElementById('hudClock');
const hudActiveRoom = document.getElementById('hudActiveRoom');
const hudActiveSpeaker = document.getElementById('hudActiveSpeaker');
const hudActiveTv = document.getElementById('hudActiveTv');
const hudActiveProjectAgent = document.getElementById('hudActiveProjectAgent');
const hudActiveConversation = document.getElementById('hudActiveConversation');
const hudActiveProjectMeta = document.getElementById('hudActiveProjectMeta');
const hudProjectCount = document.getElementById('hudProjectCount');
const hudNodeTree = document.getElementById('hudNodeTree');
const hudStreamCount = document.getElementById('hudStreamCount');
const hudActivityStream = document.getElementById('hudActivityStream');

const MAX_STREAM_ITEMS = 120;
const hudState = {
  active_room: 'Whole Home',
  active_presence_room: 'whole_home',
  active_speaker: 'Whole Home',
  active_tv: null,
  voice_status: 'idle',
  active_project_id: null,
  active_agent_id: '101',
  active_project_agent: null,
  explicit_none: false,
  pending_project_id: null,
  projects: [],
  knownTvs: [],
};

const activeTokenRows = new Map();
let streamEventCount = 0;
let ws = null;
let reconnectDelayMs = 1000;
let refreshTimer = null;
let wsSnapshotReceived = false;
let hudProjectsSeq = 0;

function updateClock() {
  if (!hudClock) return;
  hudClock.textContent = new Date().toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  });
}

function setVoiceStatus(statusText) {
  const normalized = String(statusText || 'idle').toLowerCase();
  hudState.voice_status = normalized;
  const iconMap = {
    idle: '●',
    listening: '◉',
    thinking: '◈',
    speaking: '▶',
  };
  if (hudVoiceStatus) {
    hudVoiceStatus.dataset.voiceStatus = normalized;
  }
  if (hudVoiceIcon) {
    hudVoiceIcon.textContent = iconMap[normalized] || '●';
  }
  if (hudVoiceLabel) {
    hudVoiceLabel.textContent = `Voice: ${normalized.toUpperCase()}`;
  }
}

function renderLocationHeader() {
  if (hudActiveRoom) {
    const roomName = hudState.active_presence_room || hudState.active_room || 'whole home';
    hudActiveRoom.textContent = String(roomName).replace(/_/g, ' ');
  }
  if (hudActiveSpeaker) {
    hudActiveSpeaker.textContent = hudState.active_speaker || hudState.active_room || 'Whole Home';
  }
  if (hudActiveTv) {
    if (hudState.active_tv) {
      const tvMeta = hudState.knownTvs.find(
        (t) => t && t.name && t.name.toLowerCase() === hudState.active_tv.toLowerCase()
      );
      const proto = tvMeta && tvMeta.protocol ? ` (${tvMeta.protocol})` : '';
      hudActiveTv.textContent = `${hudState.active_tv}${proto}`;
    } else {
      hudActiveTv.textContent = 'No TV mapped';
    }
  }
}

function findActiveProjectAndAgent() {
  const projects = Array.isArray(hudState.projects) ? hudState.projects : [];
  const rawActiveId = hudState.active_project_id;
  const matchedById = rawActiveId
    ? projects.find((p) => p && p.id === rawActiveId) || null
    : null;
  const isPendingUncached = Boolean(
    rawActiveId && !matchedById && hudState.pending_project_id === rawActiveId
  );
  const allowFallback = !hudState.explicit_none && !isPendingUncached;
  const proj =
    matchedById ||
    (allowFallback ? (projects.find((p) => p && p.active) || projects[0] || null) : null);
  if (!proj) return { proj: null, agent: null };
  const agentId = hudState.active_agent_id || proj.active_agent_id || '101';
  const agent =
    (proj.agents || []).find((a) => a && a.agent_id === agentId) ||
    (proj.agents || [])[0] ||
    null;
  return { proj, agent };
}

function renderActiveTargetBanner() {
  const { proj, agent } = findActiveProjectAndAgent();
  const label =
    (agent && agent.project_agent) ||
    hudState.active_project_agent ||
    (proj && (proj.short_name || proj.name)) ||
    (hudState.active_project_id && hudState.pending_project_id === hudState.active_project_id
      ? hudState.active_project_id
      : null) ||
    'No project selected';

  if (hudActiveProjectAgent) {
    hudActiveProjectAgent.textContent = label;
  }
  if (hudActiveConversation) {
    const convId = (agent && agent.conversation_id) || 'new thread';
    hudActiveConversation.textContent = `thread: ${convId}`;
  }
  if (hudActiveProjectMeta) {
    if (proj) {
      const branch = proj.git_branch || 'no-git';
      const dirty = Number(proj.uncommitted_changes || 0);
      hudActiveProjectMeta.textContent = `branch: ${branch} · ${dirty} dirty`;
    } else {
      hudActiveProjectMeta.textContent = 'branch: --';
    }
  }
}

function renderNodeTree() {
  if (!hudNodeTree) return;
  const projects = Array.isArray(hudState.projects) ? hudState.projects : [];
  if (hudProjectCount) {
    hudProjectCount.textContent = `${projects.length} project${projects.length === 1 ? '' : 's'}`;
  }

  if (!projects.length) {
    const empty = document.createElement('p');
    empty.className = 'hud-empty';
    empty.textContent = 'No workspace projects discovered.';
    hudNodeTree.replaceChildren(empty);
    return;
  }

  const { proj: activeProj, agent: activeAgent } = findActiveProjectAndAgent();
  const cards = [];
  for (const proj of projects) {
    if (!proj) continue;
    const isProjActive = Boolean(activeProj && proj.id === activeProj.id);

    const card = document.createElement('article');
    card.className = 'hud-project-card';
    card.dataset.active = isProjActive ? 'true' : 'false';

    const head = document.createElement('div');
    head.className = 'hud-project-head';

    const titleWrap = document.createElement('div');
    const shortEl = document.createElement('h3');
    shortEl.className = 'hud-project-short';
    shortEl.textContent = proj.short_name || proj.name || proj.id;
    titleWrap.appendChild(shortEl);

    if (proj.name && proj.short_name && proj.name !== proj.short_name) {
      const rawEl = document.createElement('div');
      rawEl.className = 'hud-project-raw';
      rawEl.textContent = proj.name;
      titleWrap.appendChild(rawEl);
    }
    head.appendChild(titleWrap);

    const badges = document.createElement('div');
    badges.className = 'hud-badge-row';

    if (isProjActive) {
      const activeBadge = document.createElement('span');
      activeBadge.className = 'hud-badge active';
      activeBadge.textContent = 'ACTIVE';
      badges.appendChild(activeBadge);
    }

    const branchBadge = document.createElement('span');
    branchBadge.className = 'hud-badge';
    branchBadge.textContent = proj.git_branch ? `git:${proj.git_branch}` : 'no-git';
    badges.appendChild(branchBadge);

    const dirty = Number(proj.uncommitted_changes || 0);
    const dirtyBadge = document.createElement('span');
    dirtyBadge.className = `hud-badge${dirty > 0 ? ' dirty' : ''}`;
    dirtyBadge.textContent = dirty > 0 ? `${dirty} dirty` : 'clean';
    badges.appendChild(dirtyBadge);

    head.appendChild(badges);
    card.appendChild(head);

    const agentList = document.createElement('ul');
    agentList.className = 'hud-agent-list';
    agentList.setAttribute('role', 'list');

    const agents = Array.isArray(proj.agents) ? proj.agents : [];
    for (const agent of agents) {
      if (!agent) continue;
      const isAgentActive = Boolean(
        isProjActive &&
          activeAgent &&
          (agent === activeAgent ||
            (activeAgent.agent_id && agent.agent_id === activeAgent.agent_id))
      );

      const li = document.createElement('li');
      li.className = 'hud-agent-node';
      li.dataset.active = isAgentActive ? 'true' : 'false';

      const top = document.createElement('div');
      top.className = 'hud-agent-top';

      const nameEl = document.createElement('span');
      nameEl.className = 'hud-agent-name';
      nameEl.textContent = agent.project_agent || `agent ${agent.agent_id}`;
      top.appendChild(nameEl);

      const statusBadge = document.createElement('span');
      const st = agent.status || 'idle';
      statusBadge.className = 'hud-badge';
      statusBadge.dataset.status = st;
      statusBadge.textContent = st.toUpperCase();
      top.appendChild(statusBadge);
      li.appendChild(top);

      const threadEl = document.createElement('div');
      threadEl.className = 'hud-agent-thread';
      threadEl.textContent = `thread: ${agent.conversation_id || 'new thread'}`;
      li.appendChild(threadEl);

      const detail = agent.current_task || agent.last_summary;
      if (detail) {
        const sumEl = document.createElement('p');
        sumEl.className = 'hud-agent-summary';
        sumEl.textContent = agent.current_task ? `Running: ${agent.current_task}` : agent.last_summary;
        li.appendChild(sumEl);
      }

      const swarmNodes = Array.isArray(agent.swarm_nodes) ? agent.swarm_nodes : [];
      if (swarmNodes.length) {
        const swarmList = document.createElement('ul');
        swarmList.className = 'hud-swarm-list';
        swarmList.setAttribute('role', 'list');
        for (const node of swarmNodes) {
          if (!node) continue;
          const chip = document.createElement('li');
          chip.className = 'hud-swarm-chip';
          const label = node.label || node.node_id || 'swarm';
          chip.textContent = `d${node.depth || 1} · ${label}: ${node.status || 'idle'}`;
          swarmList.appendChild(chip);
        }
        li.appendChild(swarmList);
      }

      agentList.appendChild(li);
    }

    card.appendChild(agentList);
    cards.push(card);
  }

  hudNodeTree.replaceChildren(...cards);
}

function appendStreamItem(evType, scope, bodyText, timestamp = null, tokenKey = null) {
  if (!hudActivityStream) return null;
  const emptyEl = hudActivityStream.querySelector('.hud-empty');
  if (emptyEl) emptyEl.remove();

  const li = document.createElement('li');
  li.className = 'hud-stream-item';
  li.dataset.type = evType;
  if (tokenKey) li.dataset.tokenKey = tokenKey;

  const meta = document.createElement('div');
  meta.className = 'hud-stream-meta';

  const typeSpan = document.createElement('span');
  typeSpan.className = 'hud-stream-type';
  typeSpan.textContent = evType.replace(/_/g, ' ');
  meta.appendChild(typeSpan);

  if (scope) {
    const scopeSpan = document.createElement('span');
    scopeSpan.textContent = scope;
    meta.appendChild(scopeSpan);
  }

  const timeSpan = document.createElement('span');
  const parsedTime = timestamp ? new Date(timestamp) : new Date();
  timeSpan.textContent = Number.isNaN(parsedTime.getTime())
    ? ''
    : parsedTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  meta.appendChild(timeSpan);

  const body = document.createElement('p');
  body.className = 'hud-stream-body';
  body.textContent = bodyText;

  li.appendChild(meta);
  li.appendChild(body);
  hudActivityStream.appendChild(li);

  streamEventCount += 1;
  if (hudStreamCount) {
    hudStreamCount.textContent = `${streamEventCount} events`;
  }

  while (hudActivityStream.children.length > MAX_STREAM_ITEMS) {
    const first = hudActivityStream.firstElementChild;
    if (!first) break;
    if (first.dataset.tokenKey) {
      activeTokenRows.delete(first.dataset.tokenKey);
    }
    first.remove();
  }
  hudActivityStream.scrollTop = hudActivityStream.scrollHeight;
  return li;
}

function coalesceHudToken(ev) {
  const key = `${ev.project_id || 'default'}:${ev.agent_id || '101'}:${ev.task_id || 'stream'}`;
  const chunk = ev.text || '';
  if (!chunk) return;

  let entry = activeTokenRows.get(key);
  if (!entry || !entry.el.isConnected) {
    const scope = ev.project_agent || ev.short_name || ev.project_id || 'agent';
    const li = appendStreamItem('token', scope, chunk, ev.timestamp, key);
    if (li) {
      entry = { el: li, bodyEl: li.querySelector('.hud-stream-body'), text: chunk };
      activeTokenRows.set(key, entry);
    }
    return;
  }
  entry.text += chunk;
  if (entry.text.length > 1400) {
    entry.text = `…${entry.text.slice(-1300)}`;
  }
  entry.bodyEl.textContent = entry.text;
  hudActivityStream.scrollTop = hudActivityStream.scrollHeight;
}

function updateAgentNodeState(projectId, agentId, updater) {
  const proj = (hudState.projects || []).find((p) => p && p.id === projectId);
  if (!proj) return false;
  const targetId = agentId || proj.active_agent_id || '101';
  const agent = (proj.agents || []).find((a) => a && a.agent_id === targetId);
  if (!agent) return false;
  updater(agent, proj);
  return true;
}

function scheduleProjectsRefresh() {
  if (refreshTimer) return;
  refreshTimer = setTimeout(async () => {
    refreshTimer = null;
    await fetchProjectsHydration();
  }, 250);
}

function handleHudEvent(ev, replay = false) {
  if (!ev || !ev.type) return;
  const scope = ev.project_agent || ev.short_name || ev.project_id || '';

  switch (ev.type) {
    case 'hud_snapshot': {
      if (replay) break;
      wsSnapshotReceived = true;
      ++hudProjectsSeq;
      hudState.active_room =
        ev.active_room !== undefined ? ev.active_room : hudState.active_room;
      hudState.active_presence_room =
        ev.active_presence_room !== undefined
          ? ev.active_presence_room
          : hudState.active_presence_room;
      hudState.active_speaker =
        ev.active_speaker !== undefined
          ? ev.active_speaker
          : ev.active_room !== undefined
            ? ev.active_room
            : hudState.active_speaker;
      hudState.active_tv = ev.active_tv !== undefined ? ev.active_tv : hudState.active_tv;
      hudState.active_project_id =
        ev.active_project_id !== undefined ? ev.active_project_id : hudState.active_project_id;
      hudState.active_agent_id =
        ev.active_agent_id !== undefined ? ev.active_agent_id : hudState.active_agent_id;
      hudState.active_project_agent =
        ev.active_project_agent !== undefined
          ? ev.active_project_agent
          : hudState.active_project_agent;
      if (Array.isArray(ev.projects)) {
        hudState.projects = ev.projects;
      }
      if (hudState.active_project_id === null) {
        hudState.explicit_none = ev.active_project_id === null || hudState.explicit_none;
        hudState.pending_project_id = null;
        for (const p of hudState.projects || []) {
          if (p) p.active = false;
        }
        if (ev.active_project_agent === undefined) {
          hudState.active_project_agent = null;
        }
      } else if (hudState.active_project_id) {
        hudState.explicit_none = false;
        hudState.pending_project_id = null;
        for (const p of hudState.projects || []) {
          if (!p) continue;
          p.active = p.id === hudState.active_project_id;
          if (p.active && hudState.active_agent_id) {
            p.active_agent_id = hudState.active_agent_id;
          }
        }
      } else {
        hudState.pending_project_id = null;
      }
      setVoiceStatus(ev.voice_status || 'idle');
      renderLocationHeader();
      renderActiveTargetBanner();
      renderNodeTree();

      if (Array.isArray(ev.recent_events) && ev.recent_events.length) {
        activeTokenRows.clear();
        streamEventCount = 0;
        hudActivityStream.replaceChildren();
        for (const item of ev.recent_events.slice(-60)) {
          handleHudEvent(item, true);
        }
      }
      break;
    }

    case 'voice_status': {
      if (!replay) {
        const nextStatus = ev.voice_status || ev.status || 'idle';
        setVoiceStatus(nextStatus);
        if (ev.active_room) {
          hudState.active_room = ev.active_room;
          hudState.active_speaker = ev.active_room;
          renderLocationHeader();
        }
        appendStreamItem('voice_status', ev.active_room || 'voice', `Voice status: ${nextStatus}`, ev.timestamp);
      }
      break;
    }

    case 'room_changed': {
      const eventPresence = ev.active_presence_room || ev.presence_room;
      const eventSpeaker = ev.active_speaker || ev.active_room;
      const eventTv = ev.active_tv;
      if (!replay) {
        hudState.active_room = ev.active_room || hudState.active_room;
        hudState.active_speaker = eventSpeaker || hudState.active_speaker;
        hudState.active_presence_room = eventPresence || hudState.active_presence_room;
        hudState.active_tv = eventTv !== undefined ? eventTv : hudState.active_tv;
        renderLocationHeader();
      }
      const streamSpeaker = eventSpeaker || (replay ? 'none' : hudState.active_speaker) || 'none';
      const streamTv =
        eventTv !== undefined ? (eventTv || 'none') : (replay ? 'none' : hudState.active_tv || 'none');
      appendStreamItem(
        'room_changed',
        eventPresence || (replay ? 'presence' : hudState.active_presence_room) || 'presence',
        `Speaker: ${streamSpeaker} · TV: ${streamTv}`,
        ev.timestamp
      );
      break;
    }

    case 'project_changed': {
      if (!replay) {
        ++hudProjectsSeq;
        const nextProjectId =
          ev.project_id !== undefined ? ev.project_id : hudState.active_project_id;
        const projectChanged = nextProjectId !== hudState.active_project_id;
        hudState.active_project_id = nextProjectId;
        if (ev.project_id === null) {
          hudState.explicit_none = true;
          hudState.pending_project_id = null;
        } else if (nextProjectId) {
          hudState.explicit_none = false;
          const inCache = (hudState.projects || []).some((p) => p && p.id === nextProjectId);
          hudState.pending_project_id = inCache ? null : nextProjectId;
        }
        if (ev.agent_id !== undefined) {
          hudState.active_agent_id = ev.agent_id;
        } else if (projectChanged) {
          const matchedTarget =
            (hudState.projects || []).find((p) => p && p.id === nextProjectId) || null;
          hudState.active_agent_id = (matchedTarget && matchedTarget.active_agent_id) || '101';
        }
        for (const p of hudState.projects || []) {
          if (!p) continue;
          p.active = p.id === hudState.active_project_id;
          if (p.active && hudState.active_agent_id) {
            p.active_agent_id = hudState.active_agent_id;
          }
        }
        if (ev.project_agent !== undefined) {
          hudState.active_project_agent = ev.project_agent;
        } else {
          const matchedProj =
            (hudState.projects || []).find((p) => p && p.id === hudState.active_project_id) ||
            null;
          const targetAgentId =
            hudState.active_agent_id || (matchedProj && matchedProj.active_agent_id) || '101';
          const matchedAgent = matchedProj
            ? (matchedProj.agents || []).find((a) => a && a.agent_id === targetAgentId) || null
            : null;
          hudState.active_project_agent =
            (matchedAgent && matchedAgent.project_agent) ||
            (ev.short_name ? `project ${ev.short_name}, agent ${targetAgentId}` : null) ||
            (ev.project_name ? `project ${ev.project_name}, agent ${targetAgentId}` : null) ||
            (matchedProj && matchedProj.short_name
              ? `project ${matchedProj.short_name}, agent ${targetAgentId}`
              : null);
        }
        renderActiveTargetBanner();
        renderNodeTree();
        scheduleProjectsRefresh();
      }
      appendStreamItem(
        'project_changed',
        scope,
        `Active target switched to ${ev.project_agent || ev.project_id || 'project'}`,
        ev.timestamp
      );
      break;
    }

    case 'task_started': {
      if (!replay) {
        updateAgentNodeState(ev.project_id, ev.agent_id, (ag) => {
          ag.status = 'running';
          ag.current_task = ev.prompt || 'Running task…';
          if (ev.conversation_id) ag.conversation_id = ev.conversation_id;
        });
        renderActiveTargetBanner();
        renderNodeTree();
      }
      appendStreamItem('task_started', scope, ev.prompt || 'Background task started', ev.timestamp);
      break;
    }

    case 'thought': {
      appendStreamItem('thought', scope, ev.text || '', ev.timestamp);
      break;
    }

    case 'tool_call': {
      const argsStr = JSON.stringify(ev.args || {});
      appendStreamItem('tool_call', scope, `${ev.name || 'tool'}(${argsStr})`, ev.timestamp);
      break;
    }

    case 'tool_result': {
      let resultStr = 'ok';
      if (ev.result !== undefined && ev.result !== null && ev.result !== '') {
        resultStr = typeof ev.result === 'object' ? JSON.stringify(ev.result) : String(ev.result);
      }
      const hasError = ev.error !== undefined && ev.error !== null && ev.error !== '';
      const errStr = hasError
        ? (typeof ev.error === 'object' ? JSON.stringify(ev.error) : String(ev.error))
        : '';
      const body = hasError
        ? `ERROR ${ev.name || 'tool'}: ${errStr}`
        : `${ev.name || 'tool'} → ${resultStr}`;
      appendStreamItem('tool_result', scope, body, ev.timestamp);
      if (!replay && ev.name === 'start_subagent') {
        scheduleProjectsRefresh();
      }
      break;
    }

    case 'token': {
      coalesceHudToken(ev);
      break;
    }

    case 'task_completed': {
      const key = `${ev.project_id || 'default'}:${ev.agent_id || '101'}:${ev.task_id || 'stream'}`;
      activeTokenRows.delete(key);
      if (!replay) {
        updateAgentNodeState(ev.project_id, ev.agent_id, (ag) => {
          ag.status = 'idle';
          ag.current_task = null;
          ag.last_summary = ev.summary || ev.text || ag.last_summary;
          if (ev.conversation_id) ag.conversation_id = ev.conversation_id;
        });
        renderActiveTargetBanner();
        renderNodeTree();
        scheduleProjectsRefresh();
      }
      appendStreamItem('task_completed', scope, ev.summary || ev.text || 'Completed', ev.timestamp);
      break;
    }

    case 'task_failed': {
      const key = `${ev.project_id || 'default'}:${ev.agent_id || '101'}:${ev.task_id || 'stream'}`;
      activeTokenRows.delete(key);
      if (!replay) {
        updateAgentNodeState(ev.project_id, ev.agent_id, (ag) => {
          ag.status = 'error';
          ag.current_task = null;
          ag.last_summary = ev.error || 'Task failed';
        });
        renderActiveTargetBanner();
        renderNodeTree();
        scheduleProjectsRefresh();
      }
      appendStreamItem('task_failed', scope, ev.error || 'Task failed', ev.timestamp);
      break;
    }

    default:
      break;
  }
}

async function fetchProjectsHydration() {
  const reqSeq = ++hudProjectsSeq;
  try {
    const res = await fetch('/api/projects');
    if (!res.ok) return;
    const data = await res.json();
    if (reqSeq !== hudProjectsSeq) return;
    const prevProjectId = hudState.active_project_id;
    hudState.active_project_id =
      data.active_project_id !== undefined ? data.active_project_id : hudState.active_project_id;
    hudState.active_agent_id =
      data.active_agent_id !== undefined ? data.active_agent_id : hudState.active_agent_id;
    if (data.active_project_agent !== undefined) {
      hudState.active_project_agent = data.active_project_agent;
    } else if (data.active_project_id !== undefined && data.active_project_id !== prevProjectId) {
      hudState.active_project_agent = null;
    }
    if (Array.isArray(data.projects)) {
      hudState.projects = data.projects;
    }
    if (data.active_project_id === null) {
      hudState.explicit_none = true;
      hudState.pending_project_id = null;
      hudState.active_project_agent = null;
      for (const p of hudState.projects || []) {
        if (p) p.active = false;
      }
    } else if (hudState.active_project_id) {
      hudState.explicit_none = false;
      const inCache = (hudState.projects || []).some(
        (p) => p && p.id === hudState.active_project_id
      );
      hudState.pending_project_id = inCache ? null : hudState.active_project_id;
    } else {
      hudState.pending_project_id = null;
    }
    renderActiveTargetBanner();
    renderNodeTree();
  } catch (_) {}
}

async function hydrateHudInitial() {
  const [projRes, healthRes, spRes] = await Promise.allSettled([
    fetch('/api/projects'),
    fetch('/api/health'),
    fetch('/api/speakers'),
  ]);

  if (!wsSnapshotReceived && healthRes.status === 'fulfilled' && healthRes.value.ok) {
    try {
      const h = await healthRes.value.json();
      if (!wsSnapshotReceived) {
        hudState.active_room = h.active_room || hudState.active_room;
        hudState.active_speaker = h.active_room || hudState.active_speaker;
        hudState.active_presence_room = h.active_presence_room || hudState.active_presence_room;
        hudState.active_tv = h.active_tv !== undefined ? h.active_tv : hudState.active_tv;
        if (h.active_project_id === null) {
          hudState.active_project_id = null;
          hudState.explicit_none = true;
          hudState.pending_project_id = null;
          hudState.active_project_agent = null;
        } else if (h.active_project_id) {
          const prevProjectId = hudState.active_project_id;
          hudState.active_project_id = h.active_project_id;
          hudState.explicit_none = false;
          const inCache = (hudState.projects || []).some((p) => p && p.id === h.active_project_id);
          hudState.pending_project_id = inCache ? null : h.active_project_id;
          if (!h.active_project_agent && h.active_project_id !== prevProjectId) {
            hudState.active_project_agent = null;
          }
        }
        hudState.active_agent_id =
          h.active_agent_id !== undefined && h.active_agent_id !== null
            ? h.active_agent_id
            : hudState.active_agent_id;
        if (h.active_project_id !== null && h.active_project_agent) {
          hudState.active_project_agent = h.active_project_agent;
        }
        if (h.voice_status) setVoiceStatus(h.voice_status);
      }
    } catch (_) {}
  }

  if (spRes.status === 'fulfilled' && spRes.value.ok) {
    try {
      const s = await spRes.value.json();
      if (Array.isArray(s.tvs)) {
        hudState.knownTvs = s.tvs;
      }
      if (!wsSnapshotReceived) {
        if (s.active_tv !== undefined) hudState.active_tv = s.active_tv;
        if (s.active_presence_room) hudState.active_presence_room = s.active_presence_room;
        if (s.active_room) {
          hudState.active_room = s.active_room;
          hudState.active_speaker = s.active_room;
        }
      }
    } catch (_) {}
  }

  if (!wsSnapshotReceived && projRes.status === 'fulfilled' && projRes.value.ok) {
    try {
      const p = await projRes.value.json();
      if (!wsSnapshotReceived) {
        const prevProjectId = hudState.active_project_id;
        hudState.active_project_id =
          p.active_project_id !== undefined ? p.active_project_id : hudState.active_project_id;
        hudState.active_agent_id =
          p.active_agent_id !== undefined ? p.active_agent_id : hudState.active_agent_id;
        if (p.active_project_agent !== undefined) {
          hudState.active_project_agent = p.active_project_agent;
        } else if (p.active_project_id !== undefined && p.active_project_id !== prevProjectId) {
          hudState.active_project_agent = null;
        }
        if (Array.isArray(p.projects)) hudState.projects = p.projects;
        if (p.active_project_id === null) {
          hudState.explicit_none = true;
          hudState.pending_project_id = null;
          hudState.active_project_agent = null;
          for (const proj of hudState.projects || []) {
            if (proj) proj.active = false;
          }
        } else if (p.active_project_id) {
          hudState.explicit_none = false;
          const inCache = (hudState.projects || []).some(
            (proj) => proj && proj.id === p.active_project_id
          );
          hudState.pending_project_id = inCache ? null : p.active_project_id;
        }
      }
    } catch (_) {}
  }

  renderLocationHeader();
  if (!wsSnapshotReceived) {
    renderActiveTargetBanner();
    renderNodeTree();
  }
}

function connectHudWebSocket() {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${window.location.host}/ws/hud`);

  ws.onopen = () => {
    reconnectDelayMs = 1000;
    if (hudConnDot) hudConnDot.classList.add('online');
    if (hudConnStatus) hudConnStatus.textContent = 'Live Telemetry Connected · /ws/hud';
  };

  ws.onmessage = (event) => {
    let payload;
    try {
      payload = JSON.parse(event.data);
    } catch (_) {
      return;
    }
    handleHudEvent(payload, false);
  };

  ws.onclose = () => {
    if (hudConnDot) hudConnDot.classList.remove('online');
    if (hudConnStatus) hudConnStatus.textContent = 'Reconnecting to /ws/hud…';
    const delay = reconnectDelayMs;
    reconnectDelayMs = Math.min(reconnectDelayMs * 2, 10000);
    setTimeout(connectHudWebSocket, delay);
  };
}

updateClock();
setInterval(updateClock, 15000);
hydrateHudInitial();
connectHudWebSocket();
