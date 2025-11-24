/** frontend/static/js/api.js (API Client)
 * API Client for EngiTorqueMap
 * Handles all HTTP requests to backend API
 */

const API = {
    baseURL: '/api/v1',
    
    /**
     * Generic request handler
     */
    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
        };
        
        const config = { ...defaultOptions, ...options };
        
        try {
            const response = await fetch(url, config);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error(`API Error [${endpoint}]:`, error);
            throw error;
        }
    },
    
    /**
     * GET request
     */
    async get(endpoint) {
        return this.request(endpoint, { method: 'GET' });
    },
    
    /**
     * POST request
     */
    async post(endpoint, data) {
        return this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data),
        });
    },
    
    // Track Geometry API (STEP 1)
    async getTracks() {
        return this.get('/tracks');
    },
    
    async getTrackGeometry(trackName) {
        return this.get(`/tracks/${trackName}/geometry`);
    },
    
    async getTrackSectors(trackName) {
        return this.get(`/tracks/${trackName}/sectors`);
    },
    
    async getEnergyGrid(trackName) {
        return this.get(`/tracks/${trackName}/energy-grid`);
    },
    
    // Energy Vectors API (STEP 2)
    async computeEnergyVectors(telemetryData) {
        return this.post('/energy/compute', telemetryData);
    },
    
    async getSessionEnergy(sessionId) {
        return this.get(`/energy/session/${sessionId}`);
    },
    
    // Driver Signature API (STEP 3)
    async getDriverSignature(driverId) {
        return this.get(`/drivers/${driverId}/signature`);
    },
    
    async getDriverClusters() {
        return this.get('/drivers/clusters');
    },
    
    // Time-Shift Prediction API (STEP 4)
    async predictTimeDelta(predictionRequest) {
        return this.post('/timeshift/predict', predictionRequest);
    },
    
    async getImprovementSuggestions(driverId, trackName) {
        return this.get(/timeshift/suggestions/${driverId}/${trackName});
    },

    // Forecast API (STEP 5)
async forecastEvent(eventData) {
    return this.post('/forecast/event', eventData);
},

async getWeatherForecast(trackName, date) {
    return this.get(`/forecast/weather/${trackName}/${date}`);
},

// FlowGrid API (STEP 6)
async startFlowGridStream(config) {
    return this.post('/flowgrid/stream/start', config);
},

async stopFlowGridStream() {
    return this.post('/flowgrid/stream/stop', {});
},

// Forensic Analysis API (STEP 7)
async getForensicReplay(sessionId) {
    return this.get(`/forensic/replay/${sessionId}`);
},

async getDriftPatterns(sessionId) {
    return this.get(`/forensic/drift-patterns/${sessionId}`);
},

// 3D Visualization API (STEP 8)
async getHologramData(trackName) {
    return this.get(`/visualization/hologram/${trackName}`);
},

async getTimeShiftVisualization(lapId) {
    return this.get(`/visualization/timeshift/${lapId}`);
},

// Strategy API (STEP 9)
async getStrategyAlerts() {
    return this.get('/strategy/alerts');
},

async analyzeOpponent(opponentId) {
    return this.get(`/strategy/opponent-analysis/${opponentId}`);
},

async getStrategyRecommendation(raceState) {
    return this.post('/strategy/recommendation', raceState);
},

// Multi-Agent System API (STEP 10)
async queryAgentSystem(query) {
    return this.post('/agents/query', query);
},

async getAgentStatus() {
    return this.get('/agents/status');
},

// Data Management API
async downloadTrackData(trackName) {
    return this.post('/data/download', { track_name: trackName });
},

async getDataStatus() {
    return this.get('/data/status');
},

// Network/Community Detection
async getCommunityNetwork(trackName) {
    return this.get(`/network/community/${trackName}`);
},

// Health Check
async healthCheck() {
    return this.get('/health');
}
};
/**

WebSocket Client for Real-Time Updates
*/
class WebSocketClient {
constructor() {
this.ws = null;
this.reconnectAttempts = 0;
this.maxReconnectAttempts = 5;
this.reconnectDelay = 1000;
this.listeners = {};
}
connect(endpoint = '/ws/flowgrid') {
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const wsUrl = ${protocol}//${window.location.host}${endpoint};
console.log('Connecting to WebSocket:', wsUrl);
 
 this.ws = new WebSocket(wsUrl);
 
 this.ws.onopen = () => {
     console.log('WebSocket connected');
     this.reconnectAttempts = 0;
     this.emit('connected');
 };
 
 this.ws.onmessage = (event) => {
     try {
         const data = JSON.parse(event.data);
         this.emit('message', data);
         
         // Emit specific event types
         if (data.type) {
             this.emit(data.type, data.data);
         }
     } catch (error) {
         console.error('WebSocket message parse error:', error);
     }
 };
 
 this.ws.onerror = (error) => {
     console.error('WebSocket error:', error);
     this.emit('error', error);
 };
 
 this.ws.onclose = () => {
     console.log('WebSocket disconnected');
     this.emit('disconnected');
     this.attemptReconnect();
 };
}
send(data) {
if (this.ws && this.ws.readyState === WebSocket.OPEN) {
this.ws.send(JSON.stringify(data));
} else {
console.warn('WebSocket not connected');
}
}
on(event, callback) {
if (!this.listeners[event]) {
this.listeners[event] = [];
}
this.listeners[event].push(callback);
}
off(event, callback) {
if (this.listeners[event]) {
this.listeners[event] = this.listeners[event].filter(cb => cb !== callback);
}
}
emit(event, data) {
if (this.listeners[event]) {
this.listeners[event].forEach(callback => callback(data));
}
}
attemptReconnect() {
if (this.reconnectAttempts < this.maxReconnectAttempts) {
this.reconnectAttempts++;
console.log(Reconnect attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts});
setTimeout(() => {
         this.connect();
     }, this.reconnectDelay * this.reconnectAttempts);
 } else {
     console.error('Max reconnect attempts reached');
 }
 }
disconnect() {
if (this.ws) {
this.ws.close();
this.ws = null;
}
}
}

// Export for use in other modules
window.API = API;
window.WebSocketClient = WebSocketClient;
