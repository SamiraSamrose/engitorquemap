// **File: frontend/static/js/visualization.js** (Visualization Functions)

/**
 * Visualization Functions for EngiTorqueMap
 * Handles D3.js, Plotly, and Leaflet visualizations
 */

const Visualization = {
    /**
     * Initialize track map with Leaflet
     */
    initTrackMap(containerId, trackGeometry) {
        const container = document.getElementById(containerId);
        
        // Clear existing map
        container.innerHTML = '';
        
        // Create Leaflet map
        const map = L.map(containerId, {
            zoomControl: true,
            attributionControl: false
        });
        
        // Add tile layer (dark theme)
        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            maxZoom: 19
        }).addTo(map);
        
        // Add track center line
        if (trackGeometry.center_line) {
            const centerLine = L.polyline(trackGeometry.center_line, {
                color: '#2563eb',
                weight: 3,
                opacity: 0.8
            }).addTo(map);
            
            // Fit map to track bounds
            map.fitBounds(centerLine.getBounds());
        }
        
        // Add track boundaries
        if (trackGeometry.boundaries) {
            L.polyline(trackGeometry.boundaries.inner, {
                color: '#ef4444',
                weight: 2,
                opacity: 0.6,
                dashArray: '5, 5'
            }).addTo(map);
            
            L.polyline(trackGeometry.boundaries.outer, {
                color: '#ef4444',
                weight: 2,
                opacity: 0.6,
                dashArray: '5, 5'
            }).addTo(map);
        }
        
        // Add corner markers
        if (trackGeometry.corners) {
            trackGeometry.corners.forEach((corner, idx) => {
                const position = trackGeometry.center_line[corner.apex_index];
                
                L.circleMarker(position, {
                    color: this.getCornerColor(corner.type),
                    fillColor: this.getCornerColor(corner.type),
                    fillOpacity: 0.7,
                    radius: 8
                }).bindPopup(`
                    <strong>Corner ${idx + 1}</strong><br>
                    Type: ${corner.type}<br>
                    Curvature: ${corner.max_curvature.toFixed(4)}
                `).addTo(map);
            });
        }
        
        return map;
    },
    
    getCornerColor(cornerType) {
        const colors = {
            'hairpin': '#ef4444',
            'slow': '#f59e0b',
            'medium': '#eab308',
            'fast': '#10b981',
            'straight': '#6b7280'
        };
        return colors[cornerType] || '#6b7280';
    },
    
    /**
     * Create energy timeline chart with Plotly
     */
    createEnergyTimeline(containerId, energyData) {
        const timestamps = energyData.map(e => e.timestamp);
        const brakingPower = energyData.map(e => e.braking_power);
        const accelPower = energyData.map(e => e.acceleration_power);
        const netPower = energyData.map(e => e.net_power);
        
        const traces = [
            {
                x: timestamps,
                y: brakingPower,
                name: 'Braking Power',
                type: 'scatter',
                mode: 'lines',
                line: { color: '#ef4444', width: 2 }
            },
            {
                x: timestamps,
                y: accelPower,
                name: 'Acceleration Power',
                type: 'scatter',
                mode: 'lines',
                line: { color: '#10b981', width: 2 }
            },
            {
                x: timestamps,
                y: netPower,
                name: 'Net Power',
                type: 'scatter',
                mode: 'lines',
                line: { color: '#2563eb', width: 2 }
            }
        ];
        
        const layout = {
            title: 'Energy Flow Timeline',
            xaxis: { title: 'Time (s)', color: '#94a3b8' },
            yaxis: { title: 'Power (W)', color: '#94a3b8' },
            paper_bgcolor: '#1e293b',
            plot_bgcolor: '#0f172a',
            font: { color: '#f1f5f9' },
            legend: { orientation: 'h', y: -0.2 }
        };
        
        Plotly.newPlot(containerId, traces, layout, { responsive: true });
    },
    
    /**
     * Create sector comparison chart
     */
    createSectorChart(containerId, sectorData) {
        const sectors = sectorData.map(s => `Sector ${s.sector_id}`);
        const avgPower = sectorData.map(s => s.average_power);
        const efficiency = sectorData.map(s => s.average_efficiency * 100);
        
        const trace1 = {
            x: sectors,
            y: avgPower,
            name: 'Average Power',
            type: 'bar',
            marker: { color: '#2563eb' }
        };
        
        const trace2 = {
            x: sectors,
            y: efficiency,
            name: 'Efficiency %',
            type: 'bar',
            yaxis: 'y2',
            marker: { color: '#10b981' }
        };
        
        const layout = {
            title: 'Sector Analysis',
            xaxis: { title: 'Sector', color: '#94a3b8' },
            yaxis: { 
                title: 'Power (W)', 
                color: '#2563eb',
                side: 'left'
            },
            yaxis2: {
                title: 'Efficiency (%)',
                color: '#10b981',
                overlaying: 'y',
                side: 'right'
            },
            paper_bgcolor: '#1e293b',
            plot_bgcolor: '#0f172a',
            font: { color: '#f1f5f9' },
            barmode: 'group'
        };
        
        Plotly.newPlot(containerId, [trace1, trace2], layout, { responsive: true });
    },
    
    /**
     * Create driver signature radar chart
     */
    createDriverSignatureChart(containerId, signature) {
        const features = signature.features;
        
        const categories = [
            'Brake Aggression',
            'Throttle Aggression',
            'Steering Smoothness',
            'Cornering Aggression',
            'Speed Management'
        ];
        
        const values = [
            features.brake_aggression / 30 * 100,
            features.throttle_aggression / 20 * 100,
            features.steering_smoothness * 100,
            features.cornering_aggression / 1.5 * 100,
            (1 - features.speed_variance / 20) * 100
        ];
        
        const trace = {
            type: 'scatterpolar',
            r: values,
            theta: categories,
            fill: 'toself',
            line: { color: '#2563eb' },
            fillcolor: 'rgba(37, 99, 235, 0.3)'
        };
        
        const layout = {
            polar: {
                radialaxis: {
                    visible: true,
                    range: [0, 100],
                    color: '#94a3b8'
                },
                angularaxis: { color: '#94a3b8' }
            },
            paper_bgcolor: '#1e293b',
            plot_bgcolor: '#1e293b',
            font: { color: '#f1f5f9' }
        };
        
        Plotly.newPlot(containerId, [trace], layout, { responsive: true });
    },
    
    /**
     * Create drift patterns chart
     */
    createDriftPatternsChart(containerId, driftData) {
        const indices = driftData.map(d => d.index);
        const driftRates = driftData.map(d => d.drift_rate);
        const efficiencies = driftData.map(d => d.efficiency * 100);
        
        const trace1 = {
            x: indices,
            y: driftRates,
            name: 'Drift Rate',
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: '#ef4444', width: 2 }
        };
        
        const trace2 = {
            x: indices,
            y: efficiencies,
            name: 'Efficiency %',
            type: 'scatter',
            mode: 'lines',
            yaxis: 'y2',
            line: { color: '#2563eb', width: 2 }
        };
        
        const layout = {
            title: 'Energy Drift Patterns',
            xaxis: { title: 'Sample Index', color: '#94a3b8' },
            yaxis: { 
                title: 'Drift Rate',
                color: '#ef4444'
            },
            yaxis2: {
                title: 'Efficiency %',
                color: '#2563eb',
                overlaying: 'y',
                side: 'right'
            },
            paper_bgcolor: '#1e293b',
            plot_bgcolor: '#0f172a',
            font: { color: '#f1f5f9' }
        };
        
        Plotly.newPlot(containerId, [trace1, trace2], layout, { responsive: true });
    },
    
    /**
     * Create tire stress accumulation chart
     */
    createTireStressChart(containerId, stressData) {
        const trace = {
            y: stressData.stress_timeline,
            type: 'scatter',
            mode: 'lines',
            fill: 'tozeroy',
            line: { color: '#f59e0b', width: 2 },
            fillcolor: 'rgba(245, 158, 11, 0.3)'
        };
        
        const layout = {
            title: 'Cumulative Tire Stress',
            xaxis: { title: 'Sample Index', color: '#94a3b8' },
            yaxis: { title: 'Cumulative Stress', color: '#94a3b8' },
            paper_bgcolor: '#1e293b',
            plot_bgcolor: '#0f172a',
            font: { color: '#f1f5f9' },
            annotations: [{
                x: stressData.stress_timeline.length,
                y: stressData.total_stress,
                text: `Total: ${stressData.total_stress.toFixed(0)}`,
                showarrow: true,
                arrowhead: 2,
                ax: -40,
                ay: -40,
                font: { color: '#f59e0b' }
            }]
        };
        
        Plotly.newPlot(containerId, [trace], layout, { responsive: true });
    },
    
    /**
     * Create network community visualization with D3
     */
    createCommunityNetwork(containerId, networkData) {
        const container = document.getElementById(containerId);
        container.innerHTML = '';
        
        const width = container.clientWidth;
        const height = 600;
        
        const svg = d3.select(`#${containerId}`)
            .append('svg')
            .attr('width', width)
            .attr('height', height);
        
        // Create force simulation
        const simulation = d3.forceSimulation(networkData.nodes)
            .force('link', d3.forceLink(networkData.edges)
                .id(d => d.id)
                .distance(100))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(width / 2, height / 2));
        
        // Draw edges
        const link = svg.append('g')
            .selectAll('line')
            .data(networkData.edges)
            .enter()
            .append('line')
            .attr('stroke', '#475569')
            .attr('stroke-width', d => Math.sqrt(d.weight));
        
        // Draw nodes
        const node = svg.append('g')
            .selectAll('circle')
            .data(networkData.nodes)
            .enter()
            .append('circle')
            .attr('r', d => 5 + d.degree * 2)
            .attr('fill', '#2563eb')
            .call(d3.drag()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended));
        
        // Add labels
        const label = svg.append('g')
            .selectAll('text')
            .data(networkData.nodes)
            .enter()
            .append('text')
            .text(d => d.label)
            .attr('font-size', 10)
            .attr('fill', '#f1f5f9')
            .attr('dx', 10)
            .attr('dy', 4);
        
        // Update positions on tick
        simulation.on('tick', () => {
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            
            node
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);
            
            label
                .attr('x', d => d.x)
                .attr('y', d => d.y);
        });
        
        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }
        
        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }
        
        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }
    }
};

window.Visualization = Visualization;
