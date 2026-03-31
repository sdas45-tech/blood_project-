/**
 * Blood Health Advisor — Client Application v3.0
 * Handles all UI interactions, API calls, and dynamic rendering.
 */

// ─── API BASE URL ───────────────────────────────
// Option 1: Local Development (FastAPI on port 8000)
//const BACKEND_PORT = 8022;
//const BASE_URL = window.location.port == BACKEND_PORT ? '' : `http://127.0.0.1:${BACKEND_PORT}`;

// Option 2: Remote Production (UEM Server)
const BASE_URL = "https://server.uemcseaiml.org:8022/blood";

// ─── STATE ──────────────────────────────────────
const state = {
    lat: null,
    lon: null,
    locationReady: false,
    selectedFile: null,
    menstrualData: null,
    currentTime: null,
    hospitalMap: null,
};

// ─── DOM REFS ───────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ─── TOAST NOTIFICATIONS ────────────────────────
function showToast(msg, type = 'info', duration = 4000) {
    const container = $('#toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    const icons = { success: '✅', error: '❌', info: 'ℹ️' };
    toast.innerHTML = `<span>${icons[type] || 'ℹ️'}</span><span>${msg}</span>`;
    container.appendChild(toast);
    setTimeout(() => {
        toast.classList.add('fade-out');
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// ─── TIME WIDGET ────────────────────────────────
async function updateTime() {
    try {
        const res = await fetch(`${BASE_URL}/api/current-time`);
        const data = await res.json();
        state.currentTime = data;
        $('#timeValue').textContent = data.time.slice(0, 5);
        $('#timeDate').textContent = `${data.day}, ${data.date}`;
        $('#timeGreeting').textContent = data.greeting;
    } catch {
        const now = new Date();
        $('#timeValue').textContent = now.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' });
        $('#timeDate').textContent = now.toLocaleDateString('en-IN', { weekday: 'long', year: 'numeric', month: 'short', day: 'numeric' });
    }
}

// ─── GEOLOCATION ────────────────────────────────
async function fallbackLocation(statusEl, errMsg) {
    statusEl.className = 'location-status warning';
    statusEl.innerHTML = `⚠️ GPS unavailable: ${errMsg}.<br>Estimating via IP...`;
    showToast('GPS disabled. Estimating location...', 'warning');

    try {
        const res = await fetch("https://ipapi.co/json/");
        if (!res.ok) throw new Error("IP Geolocation failed");
        const data = await res.json();
        if (data.latitude && data.longitude) {
            state.lat = data.latitude;
            state.lon = data.longitude;
            state.locationReady = true;
            statusEl.className = 'location-status active';
            statusEl.textContent = `📍 Estimated Location: ${data.city || 'Unknown'}, ${data.region || ''} (${state.lat.toFixed(2)}°, ${state.lon.toFixed(2)}°)`;
            showToast(`Location set to ${data.city}`, 'success');
        } else {
            throw new Error("Invalid IP response");
        }
    } catch (e) {
        statusEl.className = 'location-status error';
        statusEl.textContent = `❌ Location completely unavailable.`;
        showToast('Could not detect location. Using default testing mode.', 'error');
        // Fallback to Kolkata if everything fails completely
        state.lat = 22.5726;
        state.lon = 88.3639;
        state.locationReady = true;
    }
}

function getLocation() {
    const statusEl = $('#locationStatus');
    statusEl.className = 'location-status';
    statusEl.textContent = '📍 Detecting your location natively...';

    if (!navigator.geolocation) {
        fallbackLocation(statusEl, 'Not supported by your browser');
        return;
    }

    navigator.geolocation.getCurrentPosition(
        (pos) => {
            state.lat = pos.coords.latitude;
            state.lon = pos.coords.longitude;
            state.locationReady = true;
            statusEl.className = 'location-status active';
            statusEl.textContent = `📍 Location: ${state.lat.toFixed(4)}°N, ${state.lon.toFixed(4)}°E`;
            showToast('Location detected successfully', 'success');
        },
        (err) => {
            fallbackLocation(statusEl, err.message);
        },
        { enableHighAccuracy: true, timeout: 10000 }
    );
}

// ─── FILE UPLOAD ────────────────────────────────
function initUpload() {
    const zone = $('#uploadZone');
    const input = $('#fileInput');
    const browseBtn = $('#browseBtn');
    const preview = $('#previewContainer');
    const previewImg = $('#previewImage');
    const fileInfo = $('#fileInfo');
    const analyzeBtn = $('#analyzeBtn');
    const clearBtn = $('#clearBtn');

    browseBtn.addEventListener('click', (e) => { e.stopPropagation(); input.click(); });
    zone.addEventListener('click', () => input.click());

    // Drag and drop
    zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });

    input.addEventListener('change', () => { if (input.files.length) handleFile(input.files[0]); });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            showToast('Please upload an image file', 'error');
            return;
        }
        state.selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            fileInfo.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
            preview.classList.add('visible');
            analyzeBtn.disabled = false;
            zone.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    clearBtn.addEventListener('click', () => {
        state.selectedFile = null;
        preview.classList.remove('visible');
        zone.style.display = '';
        analyzeBtn.disabled = true;
        input.value = '';
        $('#resultsSection').classList.remove('visible');
    });

    analyzeBtn.addEventListener('click', analyzeImage);
}

// ─── ANALYZE IMAGE ──────────────────────────────
async function analyzeImage() {
    if (!state.selectedFile) return;

    const overlay = $('#loadingOverlay');
    overlay.classList.add('visible');
    $('#loadingText').textContent = 'Analyzing blood cell image...';

    const formData = new FormData();
    formData.append('file', state.selectedFile);

    let url = `${BASE_URL}/predict`;
    const params = [];
    if (state.lat && state.lon) {
        params.push(`lat=${state.lat}`, `lon=${state.lon}`);
    }
    if (params.length) url += '?' + params.join('&');

    try {
        const res = await fetch(url, { method: 'POST', body: formData });
        if (!res.ok) throw new Error(`Server error: ${res.status}`);
        const data = await res.json();
        renderResults(data);
        showToast('Analysis complete!', 'success');
    } catch (err) {
        showToast(`Analysis failed: ${err.message}`, 'error');
    } finally {
        overlay.classList.remove('visible');
    }
}

// ─── RENDER RESULTS ─────────────────────────────
function renderResults(data) {
    const section = $('#resultsSection');
    section.classList.add('visible');

    // Prediction
    const isHealthy = !data.predicted_class.toLowerCase().includes('unhealthy');
    const iconEl = $('#resultIcon');
    iconEl.className = `result-icon ${isHealthy ? 'success' : 'danger'}`;
    iconEl.textContent = isHealthy ? '✅' : '⚠️';

    $('#resultClass').textContent = data.predicted_class;
    const confPct = (data.confidence * 100).toFixed(1);
    $('#resultConfidence').textContent = `Confidence: ${confPct}%`;

    // Confidence bar
    setTimeout(() => {
        $('#confidenceBar').style.width = `${confPct}%`;
    }, 100);

    // Probabilities
    const probGrid = $('#probGrid');
    probGrid.innerHTML = '';
    if (data.probabilities) {
        for (const [cls, prob] of Object.entries(data.probabilities)) {
            const pct = (prob * 100).toFixed(1);
            probGrid.innerHTML += `
                <div class="prob-item">
                    <div class="prob-name">${cls}</div>
                    <div class="prob-value">${pct}%</div>
                </div>
            `;
        }
    }

    // Steps
    const stepsList = $('#stepsList');
    stepsList.innerHTML = '';
    (data.preliminary_steps || []).forEach((step, i) => {
        stepsList.innerHTML += `
            <li>
                <span class="step-num">${i + 1}</span>
                <span>${step}</span>
            </li>
        `;
    });

    // Nearest hospital
    const hospCard = $('#hospitalResultCard');
    if (data.nearest_hospital) {
        hospCard.style.display = 'block';
        const h = data.nearest_hospital;

        const doctorsHtml = (h.available_doctors || []).map((d) => `
            <span class="doctor-chip">
                <span class="avail-dot ${d.available ? 'online' : (d.nearly ? 'warning' : 'offline')}"></span>
                ${d.name} · ${d.timing} · 📞 ${d.phone}
            </span>
        `).join('');

        $('#nearestHospitalInfo').innerHTML = `
            <div style="display:flex; gap:12px; align-items:center; margin-bottom:12px">
                <span style="font-size:1.8rem">🏥</span>
                <div>
                    <strong style="font-size:1.1rem">${h.name}</strong>
                    <div style="font-size:0.85rem; color:var(--text-secondary)">${h.distance} km away</div>
                </div>
            </div>
            <div style="font-size:0.88rem; color:var(--text-secondary)">
                📍 ${h.address || 'Address not available'}<br>
                📞 ${h.phone || 'N/A'}<br>
                ${h.emergency ? '🚨 <span style="color:var(--accent-danger)">Emergency Services Available</span>' : ''}
            </div>
            
            <a href="https://www.google.com/maps/dir/?api=1&destination=${h.lat},${h.lon}" target="_blank" class="btn btn-secondary btn-sm" style="width: 100%; margin: 12px 0; justify-content: center;">
                🧭 Navigate Here
            </a>
            
            ${doctorsHtml ? `<div class="doctors-inner"><h5>Available Doctors</h5>${doctorsHtml}</div>` : ''}
        `;
    } else {
        hospCard.style.display = 'none';
    }

    // Scroll to results
    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ─── HOSPITAL MAP ───────────────────────────────
function renderHospitalMap(hospitals) {
    const mapContainer = $('#hospitalMap');
    mapContainer.style.display = 'block';

    // Destroy previous map if exists
    if (state.hospitalMap) {
        state.hospitalMap.remove();
        state.hospitalMap = null;
    }

    const map = L.map('hospitalMap').setView([state.lat, state.lon], 13);
    state.hospitalMap = map;

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);

    // User marker (blue)
    L.marker([state.lat, state.lon], {
        icon: L.divIcon({
            className: 'user-marker',
            html: '<div style="background:#3b82f6;color:#fff;border-radius:50%;width:32px;height:32px;display:flex;align-items:center;justify-content:center;font-size:16px;border:3px solid #fff;box-shadow:0 2px 8px rgba(0,0,0,0.3);">📍</div>',
            iconSize: [32, 32],
            iconAnchor: [16, 16]
        })
    }).addTo(map).bindPopup('<strong>📍 You are here</strong>');

    // Hospital markers (red)
    const bounds = L.latLngBounds([[state.lat, state.lon]]);
    hospitals.forEach((h) => {
        if (!h.lat || !h.lon) return;
        const markerColor = h.emergency ? '#ef4444' : '#10b981';
        const marker = L.marker([h.lat, h.lon], {
            icon: L.divIcon({
                className: 'hospital-marker',
                html: `<div style="background:${markerColor};color:#fff;border-radius:50%;width:30px;height:30px;display:flex;align-items:center;justify-content:center;font-size:14px;border:2px solid #fff;box-shadow:0 2px 6px rgba(0,0,0,0.3);">🏥</div>`,
                iconSize: [30, 30],
                iconAnchor: [15, 15]
            })
        }).addTo(map);

        const doctors = (h.available_doctors || []).map(d => `${d.name} (${d.timing})`).join('<br>');
        marker.bindPopup(`
            <div style="min-width:180px">
                <strong>${h.name}</strong><br>
                <small>📍 ${h.distance} km away</small><br>
                <small>📞 ${h.phone || 'N/A'}</small>
                ${h.emergency ? '<br><small style="color:#ef4444">🚨 Emergency Available</small>' : ''}
                ${doctors ? '<br><hr style="margin:4px 0"><small><strong>Doctors:</strong><br>' + doctors + '</small>' : ''}
            </div>
        `);
        bounds.extend([h.lat, h.lon]);
    });

    // Fit map to show all markers
    if (hospitals.length > 0) {
        map.fitBounds(bounds, { padding: [40, 40] });
    }
}

// ─── HOSPITALS ──────────────────────────────────
async function findHospitals() {
    if (!state.locationReady) {
        showToast('Wait for location detection or refresh', 'error');
        return;
    }

    const grid = $('#hospitalGrid');
    grid.innerHTML = `
        <div class="skeleton skeleton-card"></div>
        <div class="skeleton skeleton-card"></div>
        <div class="skeleton skeleton-card"></div>
    `;

    try {
        const radiusElem = $('#hospRadius');
        const radius = radiusElem ? parseInt(radiusElem.value, 10) * 1000 : 5000;
        const res = await fetch(`${BASE_URL}/api/hospitals-doctors?lat=${state.lat}&lon=${state.lon}&radius=${radius}`);
        const data = await res.json();

        if (!data.success || !data.hospitals?.length) {
            grid.innerHTML = `
                <div class="empty-state" style="grid-column:1/-1">
                    <div class="empty-icon">🏥</div>
                    <h4>No hospitals found nearby</h4>
                    <p>Try increasing the search radius</p>
                </div>
            `;
            return;
        }


        grid.innerHTML = '';
        data.hospitals.forEach((h) => {
            const doctorsHtml = (h.available_doctors || []).map((d) => `
                <span class="doctor-chip">
                    <span class="avail-dot ${d.available ? 'online' : (d.nearly ? 'warning' : 'offline')}"></span>
                    ${d.name} · ${d.timing} · 📞 ${d.phone}
                </span>
            `).join('');

            grid.innerHTML += `
                <div class="hospital-card">
                    ${h.emergency
                    ? '<span class="emergency-badge">🚨 Emergency</span>'
                    : h.doctor_available_now
                        ? '<span class="open-badge">✅ Open Now</span>'
                        : ''}
                    <h4>${h.name}</h4>
                    <div class="meta">
                        <div class="meta-item">
                            <span class="icon">📍</span>
                            <span class="distance">${h.distance} km</span> — ${h.address || 'Address N/A'}
                        </div>
                        <div class="meta-item">
                            <span class="icon">📞</span>
                            ${h.phone || 'N/A'}
                        </div>
                        ${h.rating ? `<div class="meta-item"><span class="icon">⭐</span> ${h.rating}/5</div>` : ''}
                        ${h.departments?.length ? `<div class="meta-item"><span class="icon">🏢</span> ${h.departments.join(', ')}</div>` : ''}
                    </div>
                    
                    <a href="https://www.google.com/maps/dir/?api=1&destination=${h.lat},${h.lon}" target="_blank" class="btn btn-secondary btn-sm" style="width: 100%; margin: 12px 0 16px 0; justify-content: center;">
                        🧭 Navigate Here
                    </a>
                    
                    ${doctorsHtml ? `
                        <div class="doctors-inner">
                            <h5>Doctors</h5>
                            ${doctorsHtml}
                        </div>
                    ` : ''}
                </div>
            `;
        });

        showToast(`Found ${data.hospitals.length} hospitals nearby`, 'success');

        // Render map
        renderHospitalMap(data.hospitals);
    } catch (err) {
        grid.innerHTML = `
            <div class="empty-state" style="grid-column:1/-1">
                <div class="empty-icon">❌</div>
                <h4>Failed to load hospitals</h4>
                <p>${err.message}</p>
            </div>
        `;
        showToast('Failed to load hospitals', 'error');
    }
}

// ─── DOCTORS ────────────────────────────────────
async function searchDoctors() {
    if (!state.locationReady) {
        showToast('Wait for location detection', 'error');
        return;
    }

    const specialty = $('#specialtyFilter').value;
    const availableOnly = $('#availableOnlyToggle').checked;
    const grid = $('#doctorGrid');

    grid.innerHTML = `
        <div class="skeleton skeleton-card"></div>
        <div class="skeleton skeleton-card"></div>
    `;

    try {
        const radiusElem = $('#docRadius');
        const radius = radiusElem ? parseInt(radiusElem.value, 10) * 1000 : 10000;
        let url = `${BASE_URL}/api/doctors?lat=${state.lat}&lon=${state.lon}&radius=${radius}`;
        if (specialty) url += `&specialty=${encodeURIComponent(specialty)}`;
        if (availableOnly) url += `&available_only=true`;

        const res = await fetch(url);
        const data = await res.json();

        if (!data.success || !data.doctors?.length) {
            grid.innerHTML = `
                <div class="empty-state" style="grid-column:1/-1">
                    <div class="empty-icon">👨‍⚕️</div>
                    <h4>No doctors found</h4>
                    <p>Try changing filters or increasing search area</p>
                </div>
            `;
            return;
        }

        grid.innerHTML = '';
        data.doctors.forEach((d) => {
            const initials = d.doctor_name.replace('Dr. ', '').split(' ').map(n => n[0]).join('').slice(0, 2);
            grid.innerHTML += `
                <div class="doctor-card">
                    <div class="doc-header">
                        <div class="doc-avatar">${initials}</div>
                        <div>
                            <div class="doc-name">${d.doctor_name}</div>
                            <div class="doc-spec">${d.specialization}</div>
                        </div>
                        <span class="availability-badge ${d.available_now ? 'available' : (d.nearly_available ? 'warning' : 'unavailable')}" style="margin-left:auto">
                            <span class="avail-dot" style="width:6px;height:6px;border-radius:50%;display:inline-block;background:${d.available_now ? 'var(--accent-success)' : (d.nearly_available ? 'var(--accent-warning)' : 'var(--accent-danger)')}"></span>
                            ${d.available_now ? 'Available Now' : (d.nearly_available ? 'Available Soon' : 'Unavailable')}
                        </span>
                    </div>
                    <div class="doc-meta">
                        <div class="doc-meta-item">🕒 ${d.timing}</div>
                        <div class="doc-meta-item">📱 ${d.doctor_phone} (Direct)</div>
                        <div class="doc-meta-item">🏥 ${d.hospital_name} (${d.hospital_distance_km} km)</div>
                        <div class="doc-meta-item">📍 ${d.hospital_address}</div>
                        <div class="doc-meta-item">📞 ${d.hospital_phone} (Reception)</div>
                        ${d.hospital_emergency ? '<div class="doc-meta-item">🚨 Emergency services available</div>' : ''}
                    </div>
                    <a href="https://www.google.com/maps/dir/?api=1&destination=${d.hospital_lat},${d.hospital_lon}" target="_blank" class="btn btn-secondary btn-sm" style="width: 100%; margin-top: 16px; justify-content: center;">
                        🧭 Navigate to Hospital
                    </a>
                </div>
            `;
        });

        showToast(`Found ${data.doctors.length} doctors`, 'success');
    } catch (err) {
        grid.innerHTML = `
            <div class="empty-state" style="grid-column:1/-1">
                <div class="empty-icon">❌</div>
                <h4>Failed to load doctors</h4>
                <p>${err.message}</p>
            </div>
        `;
        showToast('Failed to load doctors', 'error');
    }
}

// ─── MENSTRUAL HEALTH ───────────────────────────
async function loadMenstrualHealth() {
    try {
        const res = await fetch(`${BASE_URL}/api/menstrual-health`);
        const data = await res.json();
        if (!data.success) throw new Error('API error');
        state.menstrualData = data;
        renderMenstrualHealth(data);
    } catch (err) {
        showToast('Failed to load menstrual health data', 'error');
    }
}

function renderMenstrualHealth(data) {
    // ── Foods to Eat ──
    const eatContainer = $('#foodsToEatContent');
    eatContainer.innerHTML = '';
    (data.foods_to_eat || []).forEach((cat) => {
        const itemsHtml = cat.items.map((item) => `
            <div class="food-item">
                <div class="food-name">${item.name}</div>
                <div class="food-benefit">${item.benefit}</div>
            </div>
        `).join('');

        eatContainer.innerHTML += `
            <div class="food-category">
                <div class="food-category-header">
                    <div class="cat-icon">${cat.icon}</div>
                    <div>
                        <h4>${cat.category}</h4>
                        <div class="why">${cat.why}</div>
                    </div>
                </div>
                <div class="food-items-grid">${itemsHtml}</div>
            </div>
        `;
    });

    // ── Foods to Avoid ──
    const avoidContainer = $('#foodsToAvoidContent');
    avoidContainer.innerHTML = '';
    (data.foods_to_avoid || []).forEach((cat) => {
        const chips = cat.items.map((i) => `<span class="avoid-chip">${i}</span>`).join('');
        avoidContainer.innerHTML += `
            <div class="avoid-card">
                <div class="avoid-header">
                    <span class="icon">${cat.icon}</span>
                    <h4>${cat.category}</h4>
                </div>
                <div class="reason">${cat.reason}</div>
                <div class="avoid-items">${chips}</div>
                <div class="tip">💡 ${cat.tip}</div>
            </div>
        `;
    });

    // ── Cycle Phases ──
    const phasesContainer = $('#cyclePhasesContent');
    phasesContainer.innerHTML = '<div class="cycle-timeline">';
    (data.cycle_phases || []).forEach((phase) => {
        const focusTags = phase.focus.map((f) => `<span class="focus-tag">${f}</span>`).join('');
        const bestFoods = phase.best_foods.map((f) => `<div style="font-size:0.82rem;color:var(--text-secondary);margin:2px 0">• ${f}</div>`).join('');

        phasesContainer.innerHTML += `
            <div class="phase-card">
                <div class="phase-icon">${phase.icon}</div>
                <div class="phase-name">${phase.phase}</div>
                <div class="phase-days">${phase.days}</div>
                <div class="phase-desc">${phase.description}</div>
                <div class="focus-tags">${focusTags}</div>
                ${bestFoods}
                <div class="exercise-note">🏃 ${phase.exercise}</div>
            </div>
        `;
    });
    phasesContainer.innerHTML += '</div>';

    // ── Supplements ──
    const suppContainer = $('#supplementsContent');
    suppContainer.innerHTML = '<div class="supplements-grid">';
    (data.supplements || []).forEach((s) => {
        suppContainer.innerHTML += `
            <div class="supplement-card">
                <div class="supp-name">💊 ${s.name}</div>
                <div class="supp-dosage">${s.dosage}</div>
                <div class="supp-when">📅 ${s.when}</div>
                <div class="supp-note">${s.note}</div>
            </div>
        `;
    });
    suppContainer.innerHTML += '</div>';

    // ── Warning Signs ──
    const warnContainer = $('#warningsContent');
    warnContainer.innerHTML = '<div class="warning-list">';
    (data.warning_signs || []).forEach((w) => {
        warnContainer.innerHTML += `
            <div class="warning-item">
                <div class="warn-icon">⚠️</div>
                <div>
                    <div class="warn-sign">${w.sign}</div>
                    <div class="warn-action">→ ${w.action}</div>
                </div>
            </div>
        `;
    });
    warnContainer.innerHTML += '</div>';

    // ── Daily Tips ──
    const tipsContainer = $('#dailyTipsContent');
    tipsContainer.innerHTML = '<div class="tips-grid">';
    (data.daily_tips || []).forEach((tip) => {
        tipsContainer.innerHTML += `<div class="tip-card">${tip}</div>`;
    });
    tipsContainer.innerHTML += '</div>';
}

// ─── TAB SWITCHING ──────────────────────────────
function initTabs() {
    const tabs = $$('.health-tab');
    tabs.forEach((tab) => {
        tab.addEventListener('click', () => {
            // Remove active from all
            tabs.forEach((t) => t.classList.remove('active'));
            $$('.health-panel').forEach((p) => p.classList.remove('active'));

            tab.classList.add('active');
            const panelId = `panel-${tab.dataset.tab}`;
            const panel = $(`#${panelId}`);
            if (panel) panel.classList.add('active');
        });
    });
}

// ─── NAVBAR ─────────────────────────────────────
function initNavbar() {
    const navbar = $('#navbar');
    const hamburger = $('#hamburger');
    const navLinks = $('#navLinks');

    // Scroll effect
    window.addEventListener('scroll', () => {
        navbar.classList.toggle('scrolled', window.scrollY > 50);
    });

    // Mobile toggle
    hamburger.addEventListener('click', () => {
        navLinks.classList.toggle('mobile-open');
    });

    // Active link on scroll
    const sections = $$('section[id]');
    const navItems = $$('.navbar-links a');

    window.addEventListener('scroll', () => {
        let current = '';
        sections.forEach((section) => {
            const top = section.offsetTop - 100;
            if (window.scrollY >= top) current = section.id;
        });

        navItems.forEach((a) => {
            a.classList.toggle('active', a.getAttribute('href') === `#${current}`);
        });
    });

    // Close mobile nav on link click
    navItems.forEach((a) => {
        a.addEventListener('click', () => navLinks.classList.remove('mobile-open'));
    });
}

// ─── INTERSECTION OBSERVER (Scroll Animations) ─
function initAnimations() {
    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                }
            });
        },
        { threshold: 0.1 }
    );

    $$('.animate-in').forEach((el) => observer.observe(el));
}

// ─── INIT ───────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initNavbar();
    initUpload();
    initTabs();
    initAnimations();

    // Get location
    getLocation();

    // Time
    updateTime();
    setInterval(updateTime, 30000);

    // Load menstrual health data
    loadMenstrualHealth();

    // Range slider handlers
    const hospRadius = $('#hospRadius');
    if (hospRadius) hospRadius.addEventListener('input', (e) => $('#hospRadiusVal').textContent = e.target.value);

    const docRadius = $('#docRadius');
    if (docRadius) docRadius.addEventListener('input', (e) => $('#docRadiusVal').textContent = e.target.value);

    // Button handlers
    $('#findHospitalsBtn').addEventListener('click', findHospitals);
    $('#refreshLocationBtn').addEventListener('click', getLocation);
    $('#searchDoctorsBtn').addEventListener('click', searchDoctors);
});