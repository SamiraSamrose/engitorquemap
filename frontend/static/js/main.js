// **File: frontend/static/js/main.js** (Main Application Logic)

/**
 * Main Application Logic for EngiTorqueMap
 * Coordinates all UI interactions and data flow
 */

class EngiTorqueMapApp {
    constructor() {
        this.wsClient = new WebSocketClient();
        this.flowGrid = null;
        this.currentTrack = null;
        this.currentDriver = null;
        this.trackMap = null;
        
        this.init();
    }
    
    async init() {
        console.log('Initializing EngiTorqueMap...');
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Initialize views
        this.setupViews();
        
        // Load initial data
        await this.loadInitialData();
        
        // Setup WebSocket
        this.setupWebSocket();
        
        // Check health
        await this.checkHealth();
        
        console.log('EngiTorqueMap initialized');
    }
    
    setupEventListeners() {
        // Navigation tabs
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                this.switchView(e.target.dataset.view);
            });
        });
        
        // Track selection
        const trackSelect = document.getElementById('track-select');
        if (trackSelect) {
            trackSelect.addEventListener('change', (e) => {
                this.loadTrack(e.target.value);
            });
        }
        
        // Driver selection
        const driverSelect = document.getElementById('driver-select');
        if (driverSelect) {
            driverSelect.addEventListener('change', (e) => {
                this.loadDriverSignature(e.target.value);
            });
        }
        
        // FlowGrid controls
        const startBtn = document.getElementById('start-streaming');
        const stopBtn = document.getElementById('stop-streaming');
        
        if (startBtn) {
            startBtn.addEventListener('click', () => this.startFlowGrid());
        }
        if (stopBtn) {
            stopBtn.addEventListener('click', () => this.stopFlowGrid());
        }
        
        // Time-Shift predictor
        const genSuggestionsBtn = document.getElementById('generate-suggestions');
        if (genSuggestionsBtn) {
            genSuggestionsBtn.addEventListener('click', () => this.generateTimeSh

iftSuggestions());
        }
        
        // Forensic replay
        const loadForensicBtn = document.getElementById('load-forensic');
        if (loadForensicBtn) {
            loadForensicBtn.addEventListener('click', () => this.loadForensicReplay());
        }
        
        // 3D Hologram controls
        const playHologramBtn = document.getElementById('play-hologram');
        if (playHologramBtn) {
            playHologramBtn.addEventListener('click', () => this.playHologram());
        }
    }
    
    setupViews() {
        // Initialize FlowGrid
        this.flowGrid = new FlowGrid('flowgrid-canvas');
    }

    async loadInitialData() {
    try {
        // Load available tracks
        const tracksData = await API.getTracks();
        this.populateTrackSelect(tracksData.tracks);
        
        // Load data status
        const dataStatus = await API.getDataStatus();
        console.log('Data status:', dataStatus);
        
        // Load default track if available
        if (tracksData.tracks && tracksData.tracks.length > 0) {
            this.loadTrack(tracksData.tracks[0]);
        }
        
    } catch (error) {
        console.error('Error loading initial data:', error);
        this.showNotification('Error loading initial data', 'error');
    }
}

populateTrackSelect(tracks) {
    const trackSelects = [
        document.getElementById('track-select'),
        document.getElementById('ts-track-select')
    ];
    
    trackSelects.forEach(select => {
        if (select) {
            select.innerHTML = '<option value="">Select Track...</option>';
            tracks.forEach(track => {
                const option = document.createElement('option');
                option.value = track;
                option.textContent = track.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                select.appendChild(option);
            });
        }
    });
}

async loadTrack(trackName) {
    if (!trackName) return;
    
    this.currentTrack = trackName;
    this.showLoading('track-info');
    
    try {
        // Load track geometry
        const geometry = await API.getTrackGeometry(trackName);
        
        // Display track info
        this.displayTrackInfo(geometry);
        
        // Initialize track map
        if (this.trackMap) {
            this.trackMap.remove();
        }
        this.trackMap = Visualization.initTrackMap('energy-grid-map', geometry);
        
        // Load energy grid
        const energyGrid = await API.getEnergyGrid(trackName);
        this.displayEnergyGrid(energyGrid);
        
    } catch (error) {
        console.error('Error loading track:', error);
        this.showNotification('Error loading track data', 'error');
    }
}

displayTrackInfo(geometry) {
    const container = document.getElementById('track-info');
    if (!container) return;
    
    container.innerHTML = `
        <div class="track-details">
            <p><strong>Track:</strong> ${geometry.track_name}</p>
            <p><strong>Length:</strong> ${geometry.length_meters.toFixed(0)} m</p>
            <p><strong>Sectors:</strong> ${geometry.sectors.length}</p>
            <p><strong>Corners:</strong> ${geometry.corners.length}</p>
        </div>
    `;
}

displayEnergyGrid(energyGrid) {
    console.log('Energy grid loaded:', energyGrid.grid_points, 'points');
    // Energy grid is visualized on the map
}

async loadDriverSignature(driverId) {
    if (!driverId) return;
    
    this.currentDriver = driverId;
    this.showLoading('driver-signature');
    
    try {
        const signature = await API.getDriverSignature(driverId);
        this.displayDriverSignature(signature);
        
    } catch (error) {
        console.error('Error loading driver signature:', error);
        this.showNotification('Error loading driver signature', 'error');
    }
}

displayDriverSignature(signature) {
    const container = document.getElementById('driver-signature');
    if (!container) return;
    
    container.innerHTML = `
        <div class="driver-details">
            <h4>Style: ${signature.style}</h4>
            <div id="driver-radar-chart" style="height: 300px;"></div>
            <div class="recommendations">
                <h5>Recommendations:</h5>
                <ul>
                    ${signature.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                </ul>
            </div>
        </div>
    `;
    
    // Create radar chart
    Visualization.createDriverSignatureChart('driver-radar-chart', signature);
}

setupWebSocket() {
    // Connect WebSocket for real-time updates
    this.wsClient.on('connected', () => {
        console.log('WebSocket connected');
        this.updateConnectionStatus(true);
    });
    
    this.wsClient.on('disconnected', () => {
        console.log('WebSocket disconnected');
        this.updateConnectionStatus(false);
    });
    
    this.wsClient.on('energy_update', (data) => {
        this.handleEnergyUpdate(data);
    });
    
    this.wsClient.connect();
}

updateConnectionStatus(connected) {
    const statusIndicator = document.getElementById('connection-status');
    const statusText = document.getElementById('status-text');
    
    if (statusIndicator) {
        statusIndicator.className = `status-indicator ${connected ? 'connected' : 'disconnected'}`;
    }
    
    if (statusText) {
        statusText.textContent = connected ? 'Connected' : 'Disconnected';
    }
}

handleEnergyUpdate(data) {
    // Update FlowGrid
    if (this.flowGrid && this.flowGrid.isRunning) {
        this.flowGrid.updateEnergyData(data);
    }
    
    // Update real-time metrics
    this.updateRealTimeMetrics(data);
}

updateRealTimeMetrics(data) {
    const speedEl = document.getElementById('metric-speed');
    const efficiencyEl = document.getElementById('metric-efficiency');
    const brakeEl = document.getElementById('metric-brake');
    
    if (speedEl) speedEl.textContent = `${data.speed?.toFixed(1) || '--'} m/s`;
    if (efficiencyEl) efficiencyEl.textContent = `${(data.energy_efficiency * 100)?.toFixed(0) || '--'}%`;
    if (brakeEl) brakeEl.textContent = `${(data.braking_power / 1000)?.toFixed(1) || '--'} kW`;
}

async startFlowGrid() {
    try {
        const config = {
            session_id: 'session_' + Date.now(),
            track: this.currentTrack
        };
        
        await API.startFlowGridStream(config);
        
        this.flowGrid.start();
        
        document.getElementById('start-streaming').disabled = true;
        document.getElementById('stop-streaming').disabled = false;
        document.getElementById('stream-status').textContent = 'Streaming...';
        
        this.showNotification('FlowGrid streaming started', 'success');
        
    } catch (error) {
        console.error('Error starting FlowGrid:', error);
        this.showNotification('Error starting stream', 'error');
    }
}

async stopFlowGrid() {
    try {
        await API.stopFlowGridStream();
        
        this.flowGrid.stop();
        
        document.getElementById('start-streaming').disabled = false;
        document.getElementById('stop-streaming').disabled = true;
        document.getElementById('stream-status').textContent = 'Stopped';
        
        this.showNotification('FlowGrid streaming stopped', 'success');
        
    } catch (error) {
        console.error('Error stopping FlowGrid:', error);
    }
}

async generateTimeShiftSuggestions() {
    const driverId = document.getElementById('ts-driver-id').value;
    const trackName = document.getElementById('ts-track-select').value;
    
    if (!driverId || !trackName) {
        this.showNotification('Please select driver and track', 'warning');
        return;
    }
    
    this.showLoading('suggestions-container');
    
    try {
        const suggestions = await API.getImprovementSuggestions(driverId, trackName);
        this.displayTimeShiftSuggestions(suggestions);
        
    } catch (error) {
        console.error('Error generating suggestions:', error);
        this.showNotification('Error generating suggestions', 'error');
    }
}

displayTimeShiftSuggestions(suggestions) {
    const container = document.getElementById('suggestions-container');
    if (!container) return;
    
    if (!suggestions || suggestions.length === 0) {
        container.innerHTML = '<p>No suggestions available</p>';
        return;
    }
    
    container.innerHTML = suggestions.map(suggestion => {
        const gainClass = suggestion.estimated_gain > 0.1 ? 'high-gain' : 'medium-gain';
        
        return `
            <div class="suggestion-card ${gainClass}">
                <div class="suggestion-header">
                    <span class="suggestion-type">${suggestion.type}</span>
                    <span class="suggestion-gain">+${suggestion.estimated_gain.toFixed(3)}s</span>
                </div>
                <p class="suggestion-description">${suggestion.description}</p>
                <div class="suggestion-details">
                    <div class="detail-row">
                        <span class="detail-label">Sector:</span>
                        <span class="detail-value">${suggestion.sector}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Difficulty:</span>
                        <span class="detail-value">${suggestion.difficulty}</span>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

async loadForensicReplay() {
    const sessionId = document.getElementById('forensic-session-id').value;
    
    if (!sessionId) {
        this.showNotification('Please enter session ID', 'warning');
        return;
    }
    
    this.showLoading('drift-patterns-chart');
    
    try {
        const forensicData = await API.getForensicReplay(sessionId);
        this.displayForensicData(forensicData);
        
    } catch (error) {
        console.error('Error loading forensic replay:', error);
        this.showNotification('Error loading forensic data', 'error');
    }
}

displayForensicData(forensicData) {
    // Display drift patterns
    if (forensicData.drift_patterns) {
        Visualization.createDriftPatternsChart('drift-patterns-chart', forensicData.drift_patterns);
    }
    
    // Display grip failures
    const gripContainer = document.getElementById('grip-failures');
    if (gripContainer && forensicData.grip_failures) {
        gripContainer.innerHTML = forensicData.grip_failures.map(failure => `
            <div class="failure-item">
                <strong>Index ${failure.index}:</strong> 
                Efficiency drop: ${(failure.efficiency_drop * 100).toFixed(1)}%
            </div>
        `).join('');
    }
    
    // Display tire stress
    if (forensicData.tire_stress_accumulation) {
        Visualization.createTireStressChart('tire-stress-chart', forensicData.tire_stress_accumulation);
    }
    
    // Display efficiency timeline
    if (forensicData.energy_timeline) {
        Visualization.createEnergyTimeline('efficiency-timeline', forensicData.energy_timeline);
    }
}

async playHologram() {
    if (!this.currentTrack) {
        this.showNotification('Please select a track first', 'warning');
        return;
    }
    
    try {
        const hologramData = await API.getHologramData(this.currentTrack);
        this.initializeHologram(hologramData);
        
    } catch (error) {
        console.error('Error loading hologram:', error);
        this.showNotification('Error loading hologram', 'error');
    }
}

initializeHologram(hologramData) {
    const container = document.getElementById('threejs-container');
    if (!container) return;
    
    // Clear existing content
    container.innerHTML = '';
    
    // Create Three.js scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f172a);
    
    const camera = new THREE.PerspectiveCamera(
        75,
        container.clientWidth / container.clientHeight,
        0.1,
        1000
    );
    
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);
    
    // Add lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 10, 10);
    scene.add(directionalLight);
    
    // Create track path
    const points = hologramData.points_3d.map(p => 
        new THREE.Vector3(p[0] * 1000, p[2] * 10, p[1] * 1000)
    );
    
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({ color: 0x2563eb, linewidth: 3 });
    const line = new THREE.Line(geometry, material);
    scene.add(line);
    
    // Add energy flow vectors
    hologramData.flow_vectors.forEach(vector => {
        const origin = new THREE.Vector3(
            vector.position[0] * 1000,
            vector.position[2] * 10,
            vector.position[1] * 1000
        );
        
        const direction = new THREE.Vector3(
            vector.direction[0],
            vector.direction[2],
            vector.direction[1]
        ).normalize();
        
        const length = vector.magnitude * 5;
        const color = vector.magnitude > 0.5 ? 0x10b981 : 0xef4444;
        
        const arrowHelper = new THREE.ArrowHelper(direction, origin, length, color);
        scene.add(arrowHelper);
    });
    
    // Position camera
    const bounds = hologramData.bounds;
    const centerX = (bounds.min[0] + bounds.max[0]) / 2 * 1000;
    const centerZ = (bounds.min[1] + bounds.max[1]) / 2 * 1000;
    
    camera.position.set(centerX, 200, centerZ + 300);
    camera.lookAt(centerX, 0, centerZ);
    
    // Animation loop
    let animationId;
    function animate() {
        animationId = requestAnimationFrame(animate);
        
        // Rotate camera around track
        const time = Date.now() * 0.0001;
        camera.position.x = centerX + Math.cos(time) * 300;
        camera.position.z = centerZ + Math.sin(time) * 300;
        camera.lookAt(centerX, 0, centerZ);
        
        renderer.render(scene, camera);
    }
    
    animate();
    
    // Store animation ID for cleanup
    this.hologramAnimationId = animationId;
}

switchView(viewName) {
    // Hide all views
    document.querySelectorAll('.view').forEach(view => {
        view.classList.remove('active');
    });
    
    // Show selected view
    const selectedView = document.getElementById(`${viewName}-view`);
    if (selectedView) {
        selectedView.classList.add('active');
    }
    
    // Update active tab
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.classList.remove('active');
        if (tab.dataset.view === viewName) {
            tab.classList.add('active');
        }
    });
}

async checkHealth() {
    try {
        const health = await API.healthCheck();
        console.log('Health check:', health);
    } catch (error) {
        console.error('Health check failed:', error);
    }
}

showLoading(containerId) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = '<div class="loading"></div>';
    }
}

showNotification(message, type = 'info') {
    console.log(`[${type.toUpperCase()}] ${message}`);
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 80px;
        right: 20px;
        padding: 1rem 1.5rem;
        background: ${type === 'error' ? '#ef4444' : type === 'success' ? '#10b981' : '#2563eb'};
        color: white;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        z-index: 1000;
        animation: slideIn 0.3s ease-out;
    `;
    
    document.body.appendChild(notification);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}
}
// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
window.app = new EngiTorqueMapApp();
});