/* ============================================================
   J.A.R.V.I.S. — клиентская логика (v4)
   Оранжевая тема, мягкий орб, боковые панели, кнопка выхода.
   + Секция «Ресурсы модели» (GPU / контекст / приоритет).
   ============================================================ */

// ============================================================
// SIMPLEX NOISE + THREE.JS ORB
// ============================================================
const noiseGLSL = `
    vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
    vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
    vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
    vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
    float snoise(vec3 v) {
        const vec2 C = vec2(1.0/6.0, 1.0/3.0);
        const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
        vec3 i  = floor(v + dot(v, C.yyy));
        vec3 x0 = v - i + dot(i, C.xxx);
        vec3 g = step(x0.yzx, x0.xyz);
        vec3 l = 1.0 - g;
        vec3 i1 = min(g.xyz, l.zxy);
        vec3 i2 = max(g.xyz, l.zxy);
        vec3 x1 = x0 - i1 + C.xxx;
        vec3 x2 = x0 - i2 + C.yyy;
        vec3 x3 = x0 - D.yyy;
        i = mod289(i);
        vec4 p = permute(permute(permute(
            i.z + vec4(0.0, i1.z, i2.z, 1.0))
          + i.y + vec4(0.0, i1.y, i2.y, 1.0))
          + i.x + vec4(0.0, i1.x, i2.x, 1.0));
        float n_ = 0.142857142857;
        vec3 ns = n_ * D.wyz - D.xzx;
        vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
        vec4 x_ = floor(j * ns.z);
        vec4 y_ = floor(j - 7.0 * x_);
        vec4 x = x_ * ns.x + ns.yyyy;
        vec4 y = y_ * ns.x + ns.yyyy;
        vec4 h = 1.0 - abs(x) - abs(y);
        vec4 b0 = vec4(x.xy, y.xy);
        vec4 b1 = vec4(x.zw, y.zw);
        vec4 s0 = floor(b0)*2.0 + 1.0;
        vec4 s1 = floor(b1)*2.0 + 1.0;
        vec4 sh = -step(h, vec4(0.0));
        vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
        vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
        vec3 p0 = vec3(a0.xy, h.x);
        vec3 p1 = vec3(a0.zw, h.y);
        vec3 p2 = vec3(a1.xy, h.z);
        vec3 p3 = vec3(a1.zw, h.w);
        vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
        p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
        vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
        m = m * m;
        return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
    }
`;

const vertexShader = `
    uniform float uTime;
    uniform float uAudioLow;
    uniform float uAudioMid;
    uniform float uAudioHigh;
    varying vec3 vNormal;
    varying vec3 vPosition;
    ${noiseGLSL}
    void main() {
        float noiseScale = 1.2 + uAudioLow * 0.5;
        float displacement  = snoise(position * noiseScale + uTime * 0.55) * (0.16 + uAudioLow * 0.42);
        displacement       += snoise(position * 2.5 + uTime * 0.95) * (0.06 + uAudioMid * 0.18);
        displacement       += snoise(position * 5.0 + uTime * 1.35) * (0.02 + uAudioHigh * 0.10);
        vec3 newPosition = position + normal * displacement;
        vNormal = normalize(normalMatrix * normal);
        vPosition = newPosition;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(newPosition, 1.0);
    }
`;

const fragmentShader = `
    uniform float uTime;
    uniform float uAudioLow;
    uniform float uAudioMid;
    uniform float uAudioHigh;
    uniform vec3  uColorPrimary;
    uniform vec3  uColorSecondary;
    uniform vec3  uColorDeep;
    varying vec3 vNormal;
    varying vec3 vPosition;

    void main() {
        float edge = 1.0 - abs(dot(vNormal, vec3(0.0, 0.0, 1.0)));
        float fresnel = pow(edge, 2.4);

        vec3 baseColor = mix(uColorDeep, uColorPrimary, 0.35 + uAudioLow * 0.5);
        baseColor = mix(baseColor, uColorSecondary, fresnel * (0.55 + uAudioMid * 0.55));

        float rim = fresnel * (0.9 + uAudioLow * 1.6 + uAudioHigh * 0.7);
        float core = 0.72 + uAudioLow * 0.85 + uAudioMid * 0.35;

        vec3 finalColor = baseColor * core + uColorSecondary * rim * 0.55;

        float alpha = 1.0 - smoothstep(0.55, 1.0, edge);

        gl_FragColor = vec4(finalColor, alpha * 0.97);
    }
`;

// ============================================================
// ORB — глобальное состояние
// ============================================================
let renderer, scene, camera, uniforms, mesh, clock;
let shaderTime = 0;
let lastFrameTime = 0;

function initOrb() {
    const canvas = document.getElementById('orb-canvas');
    if (!canvas || renderer) return;

    renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    renderer.setSize(500, 500);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
    camera.position.z = 3.5;

    const geometry = new THREE.SphereGeometry(1, 128, 128);

    uniforms = {
        uTime:       { value: 0 },
        uAudioLow:   { value: 0 },
        uAudioMid:   { value: 0 },
        uAudioHigh:  { value: 0 },
        uColorPrimary:   { value: new THREE.Color('#FF8A1F') },
        uColorSecondary: { value: new THREE.Color('#FFD070') },
        uColorDeep:      { value: new THREE.Color('#1A0A00') },
    };

    const material = new THREE.ShaderMaterial({
        vertexShader, fragmentShader, uniforms, transparent: true,
    });

    mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);

    clock = new THREE.Clock();
    shaderTime = 0;
    lastFrameTime = performance.now();
    animateOrb();
}

let ttsAmplitude = 0;
let smoothedAmplitude = 0;
let ttsTarget = 0;
let currentVoiceState = 'idle';
let ttsSequenceToken = 0;

function animateOrb() {
    if (!renderer) return;
    requestAnimationFrame(animateOrb);

    const now = performance.now();
    let dt = (now - lastFrameTime) / 1000;
    lastFrameTime = now;
    if (dt > 0.1) dt = 0.1;
    if (dt < 0)   dt = 0;

    const rt = now / 1000;
    const state = currentVoiceState;
    let low, mid, high;

    if (state === 'speaking') {
        ttsAmplitude     += (ttsTarget - ttsAmplitude) * 0.12;
        smoothedAmplitude += (ttsAmplitude - smoothedAmplitude) * 0.10;

        low  = smoothedAmplitude * 0.90;
        mid  = smoothedAmplitude * 0.55;
        high = smoothedAmplitude * 0.25;

        ttsTarget *= 0.97;
        if (ttsTarget < 0.005) ttsTarget = 0;

    } else {
        low  = 0.07 + 0.055 * Math.sin(rt * 1.15);
        mid  = 0.04 + 0.030 * Math.sin(rt * 1.75 + 1.0);
        high = 0.018 + 0.014 * Math.sin(rt * 2.30 + 2.0);
    }

    let timeSpeed;
    if (state === 'speaking')       timeSpeed = 1.30;
    else if (state === 'listening') timeSpeed = 0.80;
    else                            timeSpeed = 0.55;

    shaderTime += dt * timeSpeed;
    uniforms.uTime.value = shaderTime;

    const lerpFactor = (state === 'speaking') ? 0.10 : 0.16;
    uniforms.uAudioLow.value  += (low  - uniforms.uAudioLow.value)  * lerpFactor;
    uniforms.uAudioMid.value  += (mid  - uniforms.uAudioMid.value)  * lerpFactor;
    uniforms.uAudioHigh.value += (high - uniforms.uAudioHigh.value) * lerpFactor;

    if (mesh) {
        const rotSpeed = (state === 'speaking') ? 1.35
                       : (state === 'listening' ? 0.75 : 0.45);
        mesh.rotation.y += dt * 0.55 * rotSpeed;
        mesh.rotation.x += dt * 0.17 * rotSpeed;
        if (state === 'speaking') {
            mesh.rotation.z += dt * smoothedAmplitude * 0.35;
        }
    }

    renderer.render(scene, camera);
}

// ============================================================
// УТИЛИТЫ
// ============================================================
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text == null ? '' : String(text);
    return div.innerHTML;
}

function renderMarkdown(text) {
    if (!text) return '';
    try {
        if (window.marked && window.DOMPurify) {
            const raw = marked.parse(String(text));
            return DOMPurify.sanitize(raw);
        }
    } catch (e) {
        console.warn('markdown render error:', e);
    }
    return escapeHtml(text).replace(/\n/g, '<br>');
}

// ============================================================
// ПАНЕЛИ
// ============================================================
function togglePanel(name) {
    const panel = document.getElementById('panel-' + name);
    const btn = document.getElementById('btn-' + name);
    if (!panel) return;
    panel.classList.toggle('open');
    if (btn) btn.classList.toggle('active');
}

// ============================================================
// ЧАТ
// ============================================================
const chatView = document.getElementById('chat-view');
let chatInitialized = false;

function clearChatPlaceholder() {
    if (chatInitialized) return;
    if (chatView) chatView.innerHTML = '';
    chatInitialized = true;
}

function addMessage(kind, text, opts) {
    if (!chatView) return null;
    opts = opts || {};
    clearChatPlaceholder();
    const div = document.createElement('div');
    div.className = 'message ' + kind;
    const sender = kind === 'user' ? 'Вы'
                 : kind === 'jarvis' ? 'J.A.R.V.I.S.'
                 : 'Система';
    if (kind === 'system' || opts.noSender) {
        div.innerHTML = `<div class="md-content">${renderMarkdown(text)}</div>`;
    } else {
        div.innerHTML = `<div class="sender">${sender}</div>`
                      + `<div class="md-content">${renderMarkdown(text)}</div>`;
    }
    chatView.appendChild(div);
    chatView.scrollTop = chatView.scrollHeight;
    return div;
}

function addSystemLine(text) {
    if (!text) return;
    addMessage('system', text, { noSender: true });
}

// ============================================================
// WEBSOCKET
// ============================================================
let ws = null;
let reconnectTimer = null;
const connDot  = document.getElementById('conn-dot');
const connText = document.getElementById('conn-text');

function setConnection(online) {
    if (!connDot || !connText) return;
    if (online) {
        connDot.classList.add('online');
        connText.textContent = 'Подключено';
    } else {
        connDot.classList.remove('online');
        connText.textContent = 'Отключено';
    }
}

function connectWS() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const url = `${proto}://${location.host}/ws`;
    try {
        ws = new WebSocket(url);
    } catch (e) {
        console.error('WS create error:', e);
        scheduleReconnect();
        return;
    }
    ws.onopen = () => setConnection(true);
    ws.onclose = () => {
        setConnection(false);
        scheduleReconnect();
    };
    ws.onerror = (e) => console.warn('WS error', e);
    ws.onmessage = (ev) => {
        try {
            const msg = JSON.parse(ev.data);
            handleEvent(msg.type, msg.data || {});
        } catch (e) {
            console.error('WS parse error', e);
        }
    };
}

function scheduleReconnect() {
    if (reconnectTimer) return;
    reconnectTimer = setTimeout(() => {
        reconnectTimer = null;
        connectWS();
    }, 2000);
}

function sendWS(payload) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(payload));
        return true;
    }
    return false;
}

// ============================================================
// ОБРАБОТКА СОБЫТИЙ ЯДРА
// ============================================================
function handleEvent(type, data) {
    switch (type) {
        case 'initial_state':
            applyInitialState(data);
            break;
        case 'voice_state':
            currentVoiceState = data.state || 'idle';
            setOrbState(currentVoiceState);
            break;
        case 'tts_amplitude':
            handleTTSAmplitude(data);
            break;
        case 'status':
            showHud(data.text || '', 'busy');
            break;
        case 'step':
            showHud(((data.icon || '') + ' ' + (data.text || '')).trim(),
                    'busy');
            break;
        case 'success':
            showHud('✓ ' + (data.text || ''), 'ok');
            break;
        case 'user_spoke':
            addMessage('user', data.text || '');
            showOverlayText('user', data.text || '');
            break;
        case 'assistant_replied':
            hideHud();
            addMessage('jarvis', data.text || '');
            showOverlayText('jarvis', data.text || '');
            break;
        case 'error':
            hideHud();
            setOrbState('idle');
            addSystemLine('✗ ' + (data.text || 'Ошибка'));
            break;
        case 'warning':
            addSystemLine('⚠ ' + (data.text || ''));
            break;
        case 'command_requested':
            addSystemLine('⚙ ' + (data.text || ''));
            break;
        case 'command_executed':
            if (!data.success) {
                addSystemLine('✗ Команда не выполнена: ' + (data.text || ''));
            }
            break;
        case 'command_cancelled':
            addSystemLine('Команда отменена: ' + (data.text || ''));
            break;
        case 'tool_call': {
            const icon = data.kind === 'search' ? '🌐' : '📄';
            addSystemLine(`${icon} ${data.kind}: ${data.arg || ''}`);
            break;
        }
        case 'confirmation_request':
            showConfirmation(data.message || 'Опасная команда');
            break;
        case 'shutdown':
            addSystemLine('Система остановлена.');
            break;
        default:
            break;
    }
}

// ============================================================
// ТЕКСТ ПОВЕРХ ОРБА
// ============================================================
let overlayHideTimer = null;

function showOverlayText(role, text) {
    const overlay = document.getElementById('overlayText');
    if (!overlay) return;
    const roleEl = overlay.querySelector('.role');
    const bodyEl = overlay.querySelector('.body');
    if (!roleEl || !bodyEl) return;

    roleEl.textContent = role === 'user' ? 'Вы' : 'J.A.R.V.I.S.';
    bodyEl.textContent = text;
    overlay.style.opacity = '0.55';

    if (overlayHideTimer) clearTimeout(overlayHideTimer);
    overlayHideTimer = setTimeout(() => {
        overlay.style.opacity = '0.28';
    }, 6000);
}

// ============================================================
// СОСТОЯНИЕ ОРБА (класс на .orb-wrap) — кольца «думания» и пр.
// ============================================================
// Подписи процесса ПОД орбом
const ORB_STATUS_LABELS = {
    'thinking':  'ДУМАЮ',
    'listening': 'СЛУШАЮ',
    'speaking':  'ГОВОРЮ',
    'idle':      '',
};

function setOrbState(state) {
    const wrap = document.getElementById('orbWrap');
    if (!wrap) return;
    const s = state || 'idle';
    wrap.classList.toggle('thinking',  s === 'thinking');
    wrap.classList.toggle('listening', s === 'listening');
    wrap.classList.toggle('speaking',  s === 'speaking');
    wrap.classList.toggle('idle',      s === 'idle');

    const status = document.getElementById('orbStatus');
    if (status) status.textContent = ORB_STATUS_LABELS[s] || '';
}

// ============================================================
// HUD-СТАТУС (транзиентная строка под индикатором связи)
// ============================================================
let hudTimer = null;

function showHud(text, kind) {
    const chip = document.getElementById('hudChip');
    if (!chip || !text) return;
    chip.textContent = text;
    chip.className = 'hud-chip show' + (kind ? ' ' + kind : '');
    if (hudTimer) clearTimeout(hudTimer);
    // «busy» висит до явного hideHud(); остальные исчезают сами.
    if (kind !== 'busy') {
        hudTimer = setTimeout(() => { chip.classList.remove('show'); }, 3000);
    }
}

function hideHud() {
    const chip = document.getElementById('hudChip');
    if (chip) chip.classList.remove('show');
    if (hudTimer) { clearTimeout(hudTimer); hudTimer = null; }
}

// ============================================================
// TTS АМПЛИТУДЫ
// ============================================================
function handleTTSAmplitude(data) {
    const amps = data.amplitudes || [];
    const chunkSec = data.chunk_sec || 0.04;
    if (!amps.length) return;

    ttsSequenceToken += 1;
    const myToken = ttsSequenceToken;

    let i = 0;
    function step() {
        if (myToken !== ttsSequenceToken) return;
        if (i >= amps.length) return;
        ttsTarget = Math.max(ttsTarget, amps[i]);
        i += 1;
        setTimeout(step, Math.max(20, chunkSec * 1000));
    }
    step();
}

// ============================================================
// НАЧАЛЬНОЕ СОСТОЯНИЕ
// ============================================================
function applyInitialState(s) {
    if (!s) return;
    const models = s.models || {};

    const mem = s.memory || {};
    const memInfo = `${mem.entries || 0} записей · top-${mem.top_k || 0} · ${models.embedding || '—'}`;

    const connTextEl = document.getElementById('conn-text');
    if (connTextEl && models.llm) {
        connTextEl.textContent = `${models.llm} · ${s.tts || ''} · ${s.stt || ''}`;
    }

    const overlay = document.getElementById('overlayText');
    if (overlay) {
        const body = overlay.querySelector('.body');
        if (body) body.textContent =
            `Система готова. Память: ${memInfo}`;
    }
}

// ============================================================
// ПОДТВЕРЖДЕНИЕ
// ============================================================
const confirmModal = document.getElementById('confirmModal');

function showConfirmation(_message) {
    if (confirmModal) confirmModal.classList.add('active');
}

function answerConfirmation(ok) {
    if (confirmModal) confirmModal.classList.remove('active');
    sendWS({ type: 'confirmation_response', ok: !!ok });
}

// ============================================================
// ВВОД ТЕКСТА
// ============================================================
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

function sendMessage() {
    if (!userInput) return;
    const text = userInput.value.trim();
    if (!text) return;
    if (!sendWS({ type: 'user_message', text })) {
        addSystemLine('Нет соединения с сервером.');
        return;
    }
    userInput.value = '';
}

if (sendBtn) sendBtn.addEventListener('click', sendMessage);
if (userInput) {
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            sendMessage();
        }
    });
}

// ============================================================
// НАСТРОЙКИ
// ============================================================
const settingsPanel  = document.getElementById('panel-settings');
const settingsStatus = document.getElementById('settings-status');
const memoryStatus   = document.getElementById('memory-status');
const llmModelStatus = document.getElementById('llm-model-status');

function hasPywebviewApi() {
    return !!(window.pywebview && window.pywebview.api);
}

function openSettings() {
    if (!settingsPanel) return;
    if (settingsStatus) settingsStatus.textContent = '';
    if (memoryStatus) memoryStatus.textContent = '';
    if (llmModelStatus) llmModelStatus.textContent = '';
    const pstat = document.getElementById('prompt-status');
    if (pstat) pstat.textContent = '';
    settingsPanel.classList.add('open');
    const btn = document.getElementById('btn-settings');
    if (btn) btn.classList.add('active');

    if (hasPywebviewApi() && window.pywebview.api.get_settings) {
        window.pywebview.api.get_settings()
            .then((s) => applySettingsToUI(s))
            .catch((e) => {
                console.warn('get_settings error', e);
                if (settingsStatus)
                    settingsStatus.textContent =
                        'Не удалось получить текущие настройки.';
            });
        reloadLLMModels();
        reloadEmbeddingModels();
        reloadSystemPrompt();
    } else {
        if (settingsStatus)
            settingsStatus.textContent =
                'Просмотр/сохранение доступны только в desktop-версии.';
    }
}

function closeSettings() {
    if (!settingsPanel) return;
    settingsPanel.classList.remove('open');
    const btn = document.getElementById('btn-settings');
    if (btn) btn.classList.remove('active');
    if (settingsStatus) settingsStatus.textContent = '';
    if (memoryStatus) memoryStatus.textContent = '';
    if (llmModelStatus) llmModelStatus.textContent = '';
}

function applySettingsToUI(s) {
    if (!s) return;
    const set = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.value = val;
    };
    const chk = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.checked = !!val;
    };

    set('set-wake-word', s.wake_word || '');
    chk('set-require-wake', s.require_wake_word);
    set('set-vad-threshold', s.vad_threshold != null ? s.vad_threshold : 300);
    set('set-tts-engine', s.tts_engine || 'silero');
    set('set-silero-speaker', s.silero_speaker || 'eugene');
    set('set-piper-model-path', s.piper_model_path || '');
    set('set-stt-engine', s.stt_engine || 'kairos-asr');
    set('set-embedding-model-manual', s.embedding_model_key || '');
    set('set-embedding-instruction', s.embedding_instruction || '');
    set('set-semantic-top-k', String(s.semantic_top_k != null ? s.semantic_top_k : 5));
    set('set-search-engine', s.search_engine_priority || 'searxng-first');
    set('set-web-search-max', s.web_search_max_results != null ? s.web_search_max_results : 5);
    set('set-web-fetch-max', s.web_fetch_max_chars != null ? s.web_fetch_max_chars : 4000);
    set('set-max-tool-iterations', s.max_tool_iterations != null ? s.max_tool_iterations : 4);
    chk('set-searxng-enabled', s.searxng_enabled);
    chk('set-searxng-auto-start', s.searxng_auto_start);
    chk('set-searxng-auto-stop', s.searxng_auto_stop);
    set('set-searxng-stop-mode', s.searxng_stop_mode || 'container');
    set('set-lm-url', s.lm_studio_url || 'http://localhost:1234/v1');
    chk('set-lm-auto-start', s.lm_studio_auto_start);
    set('set-lm-cli-path', s.lm_studio_cli_path || '');
    chk('set-lm-auto-stop', s.lm_studio_auto_stop);
    set('set-lm-idle-stop', s.lm_studio_idle_stop_seconds != null ? s.lm_studio_idle_stop_seconds : 600);
    set('set-lm-ttl', s.lm_studio_ttl_seconds != null ? s.lm_studio_ttl_seconds : 180);
    set('set-history-file', s.history_file || '');
    set('set-embeddings-file', s.embeddings_file || '');
    set('set-log-file', s.log_file || '');
    chk('set-verbose-logging', s.verbose_logging);

    // --- Ресурсы LLM ---
    chk('set-llm-limits-enabled', s.llm_limits_enabled);
    set('set-llm-gpu-ratio',
        s.llm_gpu_ratio != null ? s.llm_gpu_ratio : 1.0);
    set('set-llm-context-length',
        s.llm_context_length != null ? s.llm_context_length : 0);
    set('set-llm-priority', s.llm_process_priority || 'normal');

    const embSelect = document.getElementById('set-embedding-model');
    const targetKey = s.embedding_model_key || '';
    if (embSelect) {
        for (const opt of embSelect.options) {
            if (opt.value === targetKey) {
                embSelect.value = targetKey;
                break;
            }
        }
    }
}

function saveSettings() {
    const gv = (id) => { const el = document.getElementById(id); return el ? el.value : ''; };
    const gc = (id) => { const el = document.getElementById(id); return el ? el.checked : false; };

    const manualEmb = gv('set-embedding-model-manual').trim();
    const selectedEmb = gv('set-embedding-model');
    const finalEmb = manualEmb || selectedEmb || '';

    const data = {
        wake_word: gv('set-wake-word').trim(),
        require_wake_word: gc('set-require-wake'),
        vad_threshold: parseInt(gv('set-vad-threshold'), 10) || 300,
        tts_engine: gv('set-tts-engine'),
        silero_speaker: gv('set-silero-speaker').trim(),
        piper_model_path: gv('set-piper-model-path').trim(),
        stt_engine: gv('set-stt-engine'),
        embedding_model_key: finalEmb,
        embedding_instruction: gv('set-embedding-instruction').trim(),
        semantic_top_k: parseInt(gv('set-semantic-top-k'), 10) || 0,
        search_engine_priority: gv('set-search-engine'),
        web_search_max_results: parseInt(gv('set-web-search-max'), 10) || 5,
        web_fetch_max_chars: parseInt(gv('set-web-fetch-max'), 10) || 4000,
        max_tool_iterations: parseInt(gv('set-max-tool-iterations'), 10) || 4,
        searxng_enabled: gc('set-searxng-enabled'),
        searxng_auto_start: gc('set-searxng-auto-start'),
        searxng_auto_stop: gc('set-searxng-auto-stop'),
        searxng_stop_mode: gv('set-searxng-stop-mode'),
        lm_studio_url: gv('set-lm-url').trim(),
        lm_studio_auto_start: gc('set-lm-auto-start'),
        lm_studio_cli_path: gv('set-lm-cli-path').trim(),
        lm_studio_auto_stop: gc('set-lm-auto-stop'),
        lm_studio_idle_stop_seconds: parseInt(gv('set-lm-idle-stop'), 10) || 0,
        lm_studio_ttl_seconds: parseInt(gv('set-lm-ttl'), 10) || 180,
        history_file: gv('set-history-file').trim(),
        embeddings_file: gv('set-embeddings-file').trim(),
        log_file: gv('set-log-file').trim(),
        verbose_logging: gc('set-verbose-logging'),

        // --- Ресурсы LLM ---
        llm_limits_enabled: gc('set-llm-limits-enabled'),
        llm_gpu_ratio: parseFloat(gv('set-llm-gpu-ratio')) || 0,
        llm_context_length: parseInt(gv('set-llm-context-length'), 10) || 0,
        llm_process_priority: gv('set-llm-priority') || 'normal',
    };

    if (!hasPywebviewApi() || !window.pywebview.api.save_settings) {
        if (settingsStatus)
            settingsStatus.textContent = 'Сохранение доступно только в desktop-версии.';
        return;
    }
    if (settingsStatus) settingsStatus.textContent = 'Сохранение...';
    window.pywebview.api.save_settings(data)
        .then((res) => {
            if (settingsStatus) {
                settingsStatus.textContent = (res && res.message) ||
                    (res && res.ok ? 'Сохранено.' : 'Ошибка.');
            }
            if (window.pywebview.api && window.pywebview.api.get_settings) {
                window.pywebview.api.get_settings().then(applySettingsToUI);
            }
        })
        .catch((e) => {
            if (settingsStatus) settingsStatus.textContent = 'Ошибка: ' + e;
        });
}

// ---------- Системный промпт ----------
function reloadSystemPrompt() {
    const status = document.getElementById('prompt-status');
    const area = document.getElementById('set-system-prompt');
    if (!area) return;

    if (!hasPywebviewApi() || !window.pywebview.api.get_system_prompt) {
        if (status) status.textContent = 'Доступно только в desktop-версии.';
        return;
    }
    if (status) status.textContent = 'Загрузка...';
    window.pywebview.api.get_system_prompt()
        .then((text) => {
            area.value = text || '';
            if (status) status.textContent = '';
        })
        .catch((e) => {
            if (status) status.textContent = 'Ошибка: ' + e;
        });
}

function saveSystemPrompt() {
    const status = document.getElementById('prompt-status');
    const area = document.getElementById('set-system-prompt');
    if (!area) return;

    if (!hasPywebviewApi() || !window.pywebview.api.save_system_prompt) {
        if (status) status.textContent = 'Доступно только в desktop-версии.';
        return;
    }
    if (status) status.textContent = 'Сохранение...';
    window.pywebview.api.save_system_prompt(area.value)
        .then((res) => {
            if (status) status.textContent = (res && res.message) || 'Готово.';
        })
        .catch((e) => {
            if (status) status.textContent = 'Ошибка: ' + e;
        });
}

// ---------- Списки моделей ----------
function reloadLLMModels() {
    if (!hasPywebviewApi() || !window.pywebview.api.list_llm_models) return;
    if (llmModelStatus) llmModelStatus.textContent = 'Загрузка списка LLM...';
    window.pywebview.api.list_llm_models()
        .then((models) => {
            const sel = document.getElementById('set-llm-model');
            if (!sel) return;
            sel.innerHTML = '';
            if (!models || !models.length) {
                sel.innerHTML = '<option value="">— нет LLM —</option>';
                if (llmModelStatus)
                    llmModelStatus.textContent = 'LM Studio не вернул ни одной LLM.';
                return;
            }
            for (const m of models) {
                const opt = document.createElement('option');
                opt.value = m.key;
                opt.textContent = m.display_name
                                + (m.params_string ? ' ' + m.params_string : '')
                                + (m.active ? ' (активна)' : '');
                if (m.active) opt.selected = true;
                sel.appendChild(opt);
            }
            if (llmModelStatus) llmModelStatus.textContent = '';
        })
        .catch((e) => {
            if (llmModelStatus) llmModelStatus.textContent = 'Ошибка: ' + e;
        });
}

function reloadEmbeddingModels() {
    if (!hasPywebviewApi() || !window.pywebview.api.list_embedding_models) return;
    window.pywebview.api.list_embedding_models()
        .then((models) => {
            const sel = document.getElementById('set-embedding-model');
            if (!sel) return;
            sel.innerHTML = '<option value="">— не выбрана —</option>';
            if (!models) return;
            for (const m of models) {
                const opt = document.createElement('option');
                opt.value = m.key;
                opt.textContent = m.display_name + (m.active ? ' (активна)' : '');
                if (m.active) opt.selected = true;
                sel.appendChild(opt);
            }
        })
        .catch((e) => { console.warn('list_embedding_models', e); });
}

// ---------- Смена LLM на лету ----------
document.addEventListener('DOMContentLoaded', () => {
    const llmSelect = document.getElementById('set-llm-model');
    if (llmSelect) {
        llmSelect.addEventListener('change', (ev) => {
            const key = ev.target.value;
            if (!key) return;
            if (!hasPywebviewApi() || !window.pywebview.api.set_llm_model) {
                if (llmModelStatus)
                    llmModelStatus.textContent = 'Доступно только в desktop-версии.';
                return;
            }
            if (llmModelStatus)
                llmModelStatus.textContent = `Загрузка ${key}...`;
            window.pywebview.api.set_llm_model(key)
                .then((res) => {
                    if (llmModelStatus)
                        llmModelStatus.textContent = (res && res.message) || '';
                    if (res && res.ok && window.pywebview.api.get_settings) {
                        window.pywebview.api.get_settings().then(applySettingsToUI);
                    }
                })
                .catch((e) => {
                    if (llmModelStatus) llmModelStatus.textContent = 'Ошибка: ' + e;
                });
        });
    }
});

// ---------- Память ----------
function clearHistory(mode) {
    if (!hasPywebviewApi() || !window.pywebview.api.clear_history) {
        if (memoryStatus)
            memoryStatus.textContent = 'Доступно только в desktop-версии.';
        return;
    }
    if (memoryStatus) memoryStatus.textContent = 'Работаю...';
    window.pywebview.api.clear_history(mode)
        .then((res) => {
            if (memoryStatus) memoryStatus.textContent = (res && res.message) || '';
        })
        .catch((e) => {
            if (memoryStatus) memoryStatus.textContent = 'Ошибка: ' + e;
        });
}

// ============================================================
// ОСТАНОВКА ПРИЛОЖЕНИЯ
// ============================================================
function quitApp() {
    if (!hasPywebviewApi() || !window.pywebview.api.quit) {
        if (!confirm('Остановить J.A.R.V.I.S.? Это закроет приложение.')) return;
        try { window.close(); } catch (e) {}
        return;
    }
    if (!confirm('Остановить J.A.R.V.I.S. и закрыть приложение?')) return;
    window.pywebview.api.quit()
        .catch((e) => console.warn('quit error:', e));
}

// ============================================================
// СТАРТ
// ============================================================
window.addEventListener('load', () => {
    connectWS();
    initOrb();
    setOrbState('idle');
    if (typeof THREE === 'undefined') {
        console.warn('THREE.js не загружен — орб не будет работать');
    }
});

// Экспорт наружу для inline-обработчиков в HTML
window.togglePanel     = togglePanel;
window.openSettings    = openSettings;
window.closeSettings   = closeSettings;
window.saveSettings    = saveSettings;
window.reloadLLMModels = reloadLLMModels;
window.reloadEmbeddingModels = reloadEmbeddingModels;
window.reloadSystemPrompt    = reloadSystemPrompt;
window.saveSystemPrompt      = saveSystemPrompt;
window.clearHistory    = clearHistory;
window.answerConfirmation = answerConfirmation;
window.showConfirmation   = showConfirmation;
window.quitApp         = quitApp;