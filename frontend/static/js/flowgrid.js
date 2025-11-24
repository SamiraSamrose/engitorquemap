// **File: frontend/static/js/flowgrid.js** (FlowGrid Real-Time Visualization)

/**
 * FlowGrid Real-Time Energy Visualization
 * STEP 6: Real-Time FlowGrid Engine
 */

class FlowGrid {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.canvas = null;
        this.ctx = null;
        this.energyGrid = [];
        this.animationFrame = null;
        this.isRunning = false;
        
        this.init();
    }
    
    init() {
        // Create canvas
        this.canvas = document.createElement('canvas');
        this.canvas.width = this.container.clientWidth;
        this.canvas.height = this.container.clientHeight;
        this.container.appendChild(this.canvas);
        
        this.ctx = this.canvas.getContext('2d');
        
        // Handle resize
        window.addEventListener('resize', () => this.handleResize());
    }
    
    handleResize() {
        this.canvas.width = this.container.clientWidth;
        this.canvas.height = this.container.clientHeight;
    }
    
    start() {
        this.isRunning = true;
        this.animate();
    }
    
    stop() {
        this.isRunning = false;
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }
    }
    
    updateEnergyData(energyVectors) {
        // Add new energy data point
        this.energyGrid.push({
            timestamp: Date.now(),
            ...energyVectors
        });
        
        // Keep only last 500 points
        if (this.energyGrid.length > 500) {
            this.energyGrid.shift();
        }
    }
    
    animate() {
        if (!this.isRunning) return;
        
        this.draw();
        this.animationFrame = requestAnimationFrame(() => this.animate());
    }
    
    draw() {
        const { width, height } = this.canvas;
        
        // Clear canvas
        this.ctx.fillStyle = '#0f172a';
        this.ctx.fillRect(0, 0, width, height);
        
        if (this.energyGrid.length === 0) {
            // Draw placeholder
            this.ctx.fillStyle = '#475569';
            this.ctx.font = '20px sans-serif';
            this.ctx.textAlign = 'center';
            this.ctx.fillText('Waiting for telemetry data...', width / 2, height / 2);
            return;
        }
        
        // Draw energy flow field
        this.drawEnergyField();
        
        // Draw energy vectors
        this.drawEnergyVectors();
        
        // Draw current state
        this.drawCurrentState();
    }
    
    drawEnergyField() {
        const { width, height } = this.canvas;
        const gridSize = 20;
        
        for (let y = 0; y < height; y += gridSize) {
            for (let x = 0; x < width; x += gridSize) {
                // Get nearest energy data
                const dataIndex = Math.floor((x / width) * this.energyGrid.length);
                const energyData = this.energyGrid[dataIndex] || this.energyGrid[this.energyGrid.length - 1];
                
                // Calculate energy intensity
                const intensity = this.calculateEnergyIntensity(energyData);
                
                // Draw energy cell
                this.ctx.fillStyle = this.getEnergyColor(intensity, energyData);
                this.ctx.globalAlpha = 0.3;
                this.ctx.fillRect(x, y, gridSize - 1, gridSize - 1);
                this.ctx.globalAlpha = 1.0;
            }
        }
    }
    
    drawEnergyVectors() {
        const { width, height } = this.canvas;
        const step = Math.max(1, Math.floor(this.energyGrid.length / 50));
        
        this.ctx.strokeStyle = '#2563eb';
        this.ctx.lineWidth = 2;
        
        for (let i = 0; i < this.energyGrid.length; i += step) {
            const x = (i / this.energyGrid.length) * width;
            const energyData = this.energyGrid[i];
            
            // Draw vector arrow
            const vectorLength = this.calculateVectorLength(energyData);
            const vectorAngle = this.calculateVectorAngle(energyData);
            
            this.drawArrow(x, height / 2, vectorLength, vectorAngle);
        }
    }
    
    drawCurrentState() {
        if (this.energyGrid.length === 0) return;
        
        const current = this.energyGrid[this.energyGrid.length - 1];
        const { width, height } = this.canvas;
        
        // Draw current energy indicator
        const x = width - 100;
        const y = 50;
        
        this.ctx.fillStyle = '#1e293b';
        this.ctx.fillRect(x - 10, y - 10, 110, 120);
        
        this.ctx.fillStyle = '#f1f5f9';
        this.ctx.font = '14px sans-serif';
        this.ctx.textAlign = 'left';
        
        this.ctx.fillText(`Speed: ${current.speed?.toFixed(1) || 0} m/s`, x, y);
        this.ctx.fillText(`Brake: ${(current.braking_power / 1000)?.toFixed(1) || 0} kW`, x, y + 20);
        this.ctx.fillText(`Accel: ${(current.acceleration_power / 1000)?.toFixed(1) || 0} kW`, x, y + 40);
        this.ctx.fillText(`Efficiency: ${(current.energy_efficiency * 100)?.toFixed(0) || 0}%`, x, y + 60);
    }
    
    calculateEnergyIntensity(energyData) {
        const totalPower = Math.abs(energyData.braking_power || 0) + 
                          Math.abs(energyData.acceleration_power || 0);
        return Math.min(1.0, totalPower / 200000); // Normalize to 0-1
    }
    
    getEnergyColor(intensity, energyData) {
        // Red for braking, green for acceleration, blue for neutral
        if (energyData.braking_power > energyData.acceleration_power) {
            return `rgba(239, 68, 68, ${intensity})`;
        } else if (energyData.acceleration_power > energyData.braking_power) {
            return `rgba(16, 185, 129, ${intensity})`;
        } else {
            return `rgba(37, 99, 235, ${intensity})`;
        }
    }
    
    calculateVectorLength(energyData) {
        const netPower = Math.abs(energyData.net_power || 0);
        return Math.min(50, netPower / 2000);
    }
    
    calculateVectorAngle(energyData) {
        // Angle based on energy direction
        if (energyData.net_power > 0) {
            return 0; // Right (acceleration)
        } else {
            return Math.PI; // Left (braking)
        }
    }
    
    drawArrow(x, y, length, angle) {
        this.ctx.save();
        this.ctx.translate(x, y);
        this.ctx.rotate(angle);
        
        // Arrow shaft
        this.ctx.beginPath();
        this.ctx.moveTo(0, 0);
        this.ctx.lineTo(length, 0);
        this.ctx.stroke();
        
        // Arrow head
        this.ctx.beginPath();
        this.ctx.moveTo(length, 0);
        this.ctx.lineTo(length - 5, -3);
        this.ctx.lineTo(length - 5, 3);
        this.ctx.closePath();
        this.ctx.fill();
        
        this.ctx.restore();
    }
}

window.FlowGrid = FlowGrid;
