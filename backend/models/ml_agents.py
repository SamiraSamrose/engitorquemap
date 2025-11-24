## **File: backend/models/ml_agents.py** (STEP 10: Multi-Agent System)

"""
Multi-Agent System - STEP 10
Coordinates 4 agents: Energy Field, Driver Time-Shift, Strategy, Opponent Dynamics
Hybrid ML (LSTMs, XGBoost, Bayesian, CNN) + AI reasoning (LLM)
"""
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import asyncio

from config import settings
from services.llm_rag_service import LLMRAGService
from services.energy_vectors import EnergyVectorService
from models.timeshift_predictor import TimeShiftPredictor

logger = logging.getLogger(__name__)


class BaseAgent:
    """Base class for all agents"""
    
    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.status = "initialized"
        self.last_update = None
    
    async def process(self, data: Dict) -> Dict:
        """Process input data and return results"""
        raise NotImplementedError
    
    def get_status(self) -> Dict:
        """Get agent status"""
        return {
            "agent_id": self.agent_id,
            "type": self.agent_type,
            "status": self.status,
            "last_update": self.last_update
        }


class EnergyFieldAgent(BaseAgent):
    """
    Agent responsible for updating FlowGrid energy fields
    Monitors energy consumption and identifies hotspots
    """
    
    def __init__(self, energy_service: EnergyVectorService):
        super().__init__("energy_field_agent", "energy_monitoring")
        self.energy_service = energy_service
    
    async def process(self, data: Dict) -> Dict:
        """Update energy field based on telemetry"""
        self.status = "processing"
        
        try:
            # Compute energy vectors from telemetry
            energy_vectors = await self.energy_service.compute_vectors(data['telemetry'])
            
            # Identify energy anomalies
            anomalies = self._detect_anomalies(energy_vectors)
            
            # Calculate efficiency metrics
            efficiency = self._calculate_efficiency_metrics(energy_vectors)
            
            self.status = "active"
            return {
                "agent": self.agent_id,
                "energy_vectors": energy_vectors,
                "anomalies": anomalies,
                "efficiency": efficiency,
                "recommendations": self._generate_energy_recommendations(anomalies)
            }
        except Exception as e:
            logger.error(f"Energy Field Agent error: {e}")
            self.status = "error"
            return {"agent": self.agent_id, "error": str(e)}
    
    def _detect_anomalies(self, energy_vectors: Dict) -> List[Dict]:
        """Detect unusual energy patterns"""
        anomalies = []
        
        # High braking power
        if energy_vectors['braking_power'] > 150000:  # 150kW
            anomalies.append({
                "type": "excessive_braking",
                "severity": "high",
                "value": energy_vectors['braking_power']
            })
        
        # Low efficiency
        if energy_vectors['energy_efficiency'] < 0.5:
            anomalies.append({
                "type": "low_efficiency",
                "severity": "medium",
                "value": energy_vectors['energy_efficiency']
            })
        
        return anomalies
    
    def _calculate_efficiency_metrics(self, energy_vectors: Dict) -> Dict:
        """Calculate comprehensive efficiency metrics"""
        return {
            "overall_efficiency": energy_vectors['energy_efficiency'],
            "brake_efficiency": 1.0 - (energy_vectors['braking_power'] / 
                                      (energy_vectors['acceleration_power'] + 1.0)),
            "aero_loss_percent": (energy_vectors['aerodynamic_drag'] / 
                                 (energy_vectors['acceleration_power'] + 1.0)) * 100
        }
    
    def _generate_energy_recommendations(self, anomalies: List[Dict]) -> List[str]:
        """Generate recommendations based on anomalies"""
        recommendations = []
        
        for anomaly in anomalies:
            if anomaly['type'] == 'excessive_braking':
                recommendations.append(
                    "Excessive braking detected. Consider earlier brake application with less pressure."
                )
            elif anomaly['type'] == 'low_efficiency':
                recommendations.append(
                    "Low energy efficiency. Review throttle and brake overlap zones."
                )
        
        return recommendations


class DriverTimeShiftAgent(BaseAgent):
    """
    Agent that generates time-shift insights
    "Brake 12m later → +0.17s" predictions
    """
    
    def __init__(self, timeshift_model: TimeShiftPredictor):
        super().__init__("driver_timeshift_agent", "performance_optimization")
        self.timeshift_model = timeshift_model
    
    async def process(self, data: Dict) -> Dict:
        """Generate time-shift suggestions"""
        self.status = "processing"
        
        try:
            driver_id = data.get('driver_id')
            track_name = data.get('track_name')
            current_lap = data.get('current_lap_data')
            
            # Generate optimization suggestions
            suggestions = await self.timeshift_model.generate_suggestions(
                driver_id, track_name
            )
            
            # Prioritize suggestions by potential gain
            top_suggestions = sorted(
                suggestions, 
                key=lambda x: x.get('estimated_gain', 0), 
                reverse=True
            )[:5]
            
            self.status = "active"
            return {
                "agent": self.agent_id,
                "driver_id": driver_id,
                "suggestions": top_suggestions,
                "total_potential_gain": sum(s.get('estimated_gain', 0) for s in top_suggestions)
            }
        except Exception as e:
            logger.error(f"Time-Shift Agent error: {e}")
            self.status = "error"
            return {"agent": self.agent_id, "error": str(e)}


class StrategyAgent(BaseAgent):
    """
    Agent for pit/pace recommendations
    Analyzes race situation and provides strategic guidance
    """
    
    def __init__(self):
        super().__init__("strategy_agent", "race_strategy")
        self.race_state = {}
    
    async def process(self, data: Dict) -> Dict:
        """Generate strategy recommendations"""
        self.status = "processing"
        
        try:
            race_state = data.get('race_state', {})
            self.race_state = race_state
            
            # Analyze pit window
            pit_recommendation = self._analyze_pit_window(race_state)
            
            # Analyze pace strategy
            pace_recommendation = self._analyze_pace_strategy(race_state)
            
            # Analyze tire strategy
            tire_recommendation = self._analyze_tire_strategy(race_state)
            
            # Risk assessment
            risk_assessment = self._assess_risks(race_state)
            
            self.status = "active"
            return {
                "agent": self.agent_id,
                "pit_recommendation": pit_recommendation,
                "pace_recommendation": pace_recommendation,
                "tire_recommendation": tire_recommendation,
                "risk_assessment": risk_assessment
            }
        except Exception as e:
            logger.error(f"Strategy Agent error: {e}")
            self.status = "error"
            return {"agent": self.agent_id, "error": str(e)}
    
    def _analyze_pit_window(self, race_state: Dict) -> Dict:
        """Analyze optimal pit window"""
        current_lap = race_state.get('current_lap', 0)
        total_laps = race_state.get('total_laps', 60)
        fuel_level = race_state.get('fuel_percent', 100)
        tire_age = race_state.get('tire_age_laps', 0)
        
        # Simple pit window logic
        optimal_pit_lap = int(total_laps * 0.5)  # Mid-race
        
        # Adjust for tire degradation
        if tire_age > 20:
            recommendation = "pit_soon"
            urgency = "high"
        elif tire_age > 15:
            recommendation = "pit_window_open"
            urgency = "medium"
        else:
            recommendation = "stay_out"
            urgency = "low"
        
        return {
            "recommendation": recommendation,
            "optimal_lap": optimal_pit_lap,
            "urgency": urgency,
            "reason": f"Tire age: {tire_age} laps"
        }
    
    def _analyze_pace_strategy(self, race_state: Dict) -> Dict:
        """Analyze pace strategy"""
        position = race_state.get('position', 10)
        gap_ahead = race_state.get('gap_ahead', 2.0)
        gap_behind = race_state.get('gap_behind', 3.0)
        
        if gap_ahead < 1.0:
            pace = "attack"
            target_delta = -0.3
        elif gap_behind < 1.5:
            pace = "defend"
            target_delta = 0.0
        else:
            pace = "manage"
            target_delta = 0.1
        
        return {
            "pace_mode": pace,
            "target_delta_per_lap": target_delta,
            "priority": "position_defense" if gap_behind < 1.5 else "position_gain"
        }
    
    def _analyze_tire_strategy(self, race_state: Dict) -> Dict:
        """Analyze tire strategy"""
        tire_age = race_state.get('tire_age_laps', 0)
        tire_temp = race_state.get('tire_temp', 90)
        
        if tire_temp > 110:
            recommendation = "cool_tires"
            action = "Reduce pace to cool tires"
        elif tire_temp < 80:
            recommendation = "warm_tires"
            action = "Increase pace to build temperature"
        else:
            recommendation = "optimal"
            action = "Maintain current pace"
        
        return {
            "recommendation": recommendation,
            "action": action,
            "tire_age_laps": tire_age,
            "estimated_remaining_life": max(0, 25 - tire_age)
        }
    
    def _assess_risks(self, race_state: Dict) -> Dict:
        """Assess strategic risks"""
        risks = []
        
        fuel_level = race_state.get('fuel_percent', 100)
        if fuel_level < 15:
            risks.append({
                "type": "fuel_critical",
                "severity": "high",
                "recommendation": "Fuel save mode immediately"
            })
        
        tire_age = race_state.get('tire_age_laps', 0)
        if tire_age > 22:
            risks.append({
                "type": "tire_degradation",
                "severity": "medium",
                "recommendation": "Consider pit stop"
            })
        
        return {
            "risk_level": "high" if len(risks) > 0 else "low",
            "risks": risks
        }


class OpponentDynamicsAgent(BaseAgent):
    """
    Agent for opponent analysis
    Predicts rival energy-use tendencies and strategies
    """
    
    def __init__(self):
        super().__init__("opponent_dynamics_agent", "opponent_analysis")
        self.opponent_profiles = {}
    
    async def process(self, data: Dict) -> Dict:
        """Analyze opponent dynamics"""
        self.status = "processing"
        
        try:
            opponent_id = data.get('opponent_id')
            opponent_telemetry = data.get('opponent_telemetry', {})
            
            # Build/update opponent profile
            profile = self._build_opponent_profile(opponent_id, opponent_telemetry)
            
            # Predict opponent strategy
            strategy_prediction = self._predict_opponent_strategy(profile)
            
            # Identify weaknesses
            weaknesses = self._identify_weaknesses(profile)
            
            # Generate counter-strategy
            counter_strategy = self._generate_counter_strategy(weaknesses)
            
            self.status = "active"
            return {
                "agent": self.agent_id,
                "opponent_id": opponent_id,
                "profile": profile,
                "strategy_prediction": strategy_prediction,
                "weaknesses": weaknesses,
                "counter_strategy": counter_strategy
            }
        except Exception as e:
            logger.error(f"Opponent Dynamics Agent error: {e}")
            self.status = "error"
            return {"agent": self.agent_id, "error": str(e)}
    
    def _build_opponent_profile(self, opponent_id: str, telemetry: Dict) -> Dict:
        """Build opponent energy/style profile"""
        if opponent_id not in self.opponent_profiles:
            self.opponent_profiles[opponent_id] = {
                "energy_usage": [],
                "brake_points": [],
                "lap_times": []
            }
        
        profile = self.opponent_profiles[opponent_id]
        
        # Update profile with new data
        if telemetry:
            profile['energy_usage'].append(telemetry.get('energy_consumption', 0))
            profile['brake_points'].append(telemetry.get('brake_point', 0))
            profile['lap_times'].append(telemetry.get('lap_time', 0))
        
        # Calculate statistics
        return {
            "opponent_id": opponent_id,
            "avg_energy_usage": np.mean(profile['energy_usage']) if profile['energy_usage'] else 0,
            "consistency": 1.0 / (np.std(profile['lap_times']) + 0.01) if profile['lap_times'] else 0,
            "aggression_level": self._calculate_aggression(profile)
        }
    
    def _calculate_aggression(self, profile: Dict) -> float:
        """Calculate opponent aggression level"""
        # Simplified aggression metric
        energy_usage = profile.get('energy_usage', [])
        if not energy_usage:
            return 0.5
        
        avg_energy = np.mean(energy_usage)
        # High energy usage = aggressive
        return min(1.0, avg_energy / 1000.0)
    
    def _predict_opponent_strategy(self, profile: Dict) -> Dict:
        """Predict opponent's likely strategy"""
        aggression = profile.get('aggression_level', 0.5)
        consistency = profile.get('consistency', 0.5)
        
        if aggression > 0.7:
            strategy = "aggressive_attack"
            pit_timing = "early"
        elif aggression < 0.3:
            strategy = "conservative_management"
            pit_timing = "late"
        else:
            strategy = "balanced"
            pit_timing = "mid_race"
        
        return {
            "predicted_strategy": strategy,
            "predicted_pit_timing": pit_timing,
            "confidence": float(consistency)
        }
    
    def _identify_weaknesses(self, profile: Dict) -> List[Dict]:
        """Identify opponent weaknesses"""
        weaknesses = []
        
        consistency = profile.get('consistency', 0.5)
        if consistency < 0.5:
            weaknesses.append({
                "type": "inconsistent_pace",
                "severity": "medium",
                "description": "Opponent shows inconsistent lap times"
            })
        
        aggression = profile.get('aggression_level', 0.5)
        if aggression > 0.8:
            weaknesses.append({
                "type": "high_tire_wear",
                "severity": "high",
                "description": "Aggressive style likely causes high tire degradation"
            })
        
        return weaknesses
    
    def _generate_counter_strategy(self, weaknesses: List[Dict]) -> Dict:
        """Generate counter-strategy based on weaknesses"""
        if not weaknesses:
            return {
                "recommendation": "standard_race_pace",
                "details": "No significant weaknesses identified"
            }
        
        # Counter aggressive opponents
        if any(w['type'] == 'high_tire_wear' for w in weaknesses):
            return {
                "recommendation": "tire_conservation",
                "details": "Conserve tires early, attack in final stint when opponent degrades"
            }
        
        # Counter inconsistent opponents
        if any(w['type'] == 'inconsistent_pace' for w in weaknesses):
            return {
                "recommendation": "consistent_pressure",
                "details": "Maintain consistent pressure to force errors"
            }
        
        return {
            "recommendation": "monitor_and_adapt",
            "details": "Continue monitoring opponent behavior"
        }


class MultiAgentSystem:
    """
    Coordinates all agents and provides unified interface
    Hybrid ML + AI reasoning layer
    """
    
    def __init__(self, energy_service: EnergyVectorService, 
                 timeshift_model: TimeShiftPredictor,
                 llm_service: LLMRAGService):
        self.energy_agent = EnergyFieldAgent(energy_service)
        self.timeshift_agent = DriverTimeShiftAgent(timeshift_model)
        self.strategy_agent = StrategyAgent()
        self.opponent_agent = OpponentDynamicsAgent()
        self.llm_service = llm_service
        
        self.agents = [
            self.energy_agent,
            self.timeshift_agent,
            self.strategy_agent,
            self.opponent_agent
        ]
        
        logger.info("Multi-Agent System initialized with 4 agents")
    
    async def process_query(self, query: Dict) -> Dict:
        """
        Process query through multi-agent system
        
        Input:
        {
            "type": "performance_analysis" | "strategy_recommendation" | "opponent_analysis",
            "data": {...}
        }
        """
        query_type = query.get('type')
        data = query.get('data', {})
        
        # Route to appropriate agents
        if query_type == 'performance_analysis':
            results = await self._performance_analysis(data)
        elif query_type == 'strategy_recommendation':
            results = await self._strategy_recommendation(data)
        elif query_type == 'opponent_analysis':
            results = await self._opponent_analysis(data)
        else:
            results = await self._comprehensive_analysis(data)
        
        # Use LLM to synthesize results and provide explanation
        synthesis = await self.llm_service.synthesize_agent_results(results)
        
        return {
            "query_type": query_type,
            "agent_results": results,
            "synthesis": synthesis,
            "timestamp": data.get('timestamp')
        }
    
    async def _performance_analysis(self, data: Dict) -> Dict:
        """Run performance analysis using energy and time-shift agents"""
        energy_result = await self.energy_agent.process(data)
        timeshift_result = await self.timeshift_agent.process(data)
        
        return {
            "energy_analysis": energy_result,
            "timeshift_analysis": timeshift_result
        }
    
    async def _strategy_recommendation(self, data: Dict) -> Dict:
        """Run strategy analysis"""
        strategy_result = await self.strategy_agent.process(data)
        
        return {
            "strategy_analysis": strategy_result
        }
    
    async def _opponent_analysis(self, data: Dict) -> Dict:
        """Run opponent analysis"""
        opponent_result = await self.opponent_agent.process(data)
        
        return {
            "opponent_analysis": opponent_result
        }
    
    async def _comprehensive_analysis(self, data: Dict) -> Dict:
        """Run all agents for comprehensive analysis"""
        results = await asyncio.gather(
            self.energy_agent.process(data),
            self.timeshift_agent.process(data),
            self.strategy_agent.process(data),
            self.opponent_agent.process(data),
            return_exceptions=True
        )
        
        return {
            "energy_analysis": results[0] if not isinstance(results[0], Exception) else {"error": str(results[0])},
            "timeshift_analysis": results[1] if not isinstance(results[1], Exception) else {"error": str(results[1])},
            "strategy_analysis": results[2] if not isinstance(results[2], Exception) else {"error": str(results[2])},
            "opponent_analysis": results[3] if not isinstance(results[3], Exception) else {"error": str(results[3])}
        }
    
    async def get_status(self) -> List[Dict]:
        """Get status of all agents"""
        return [agent.get_status() for agent in self.agents]
    
    async def get_strategy_alerts(self) -> List[Dict]:
        """
        Get real-time strategy alerts
        STEP 9: IoT Strategy Command Center
        """
        alerts = []
        
        # Check each agent for alerts
        for agent in self.agents:
            if agent.status == "error":
                alerts.append({
                    "severity": "high",
                    "source": agent.agent_id,
                    "message": f"{agent.agent_type} agent encountered an error",
                    "timestamp": agent.last_update
                })
        
        # Add synthetic alerts for demonstration
        alerts.extend([
            {
                "severity": "medium",
                "source": "energy_field_agent",
                "message": "Energy risk detected in Sector 2",
                "timestamp": "now"
            },
            {
                "severity": "low",
                "source": "strategy_agent",
                "message": "Optimal pit window opening in 3 laps",
                "timestamp": "now"
            }
        ])
        
        return alerts
    
    async def analyze_opponent(self, opponent_id: str) -> Dict:
        """Analyze specific opponent"""
        data = {"opponent_id": opponent_id, "opponent_telemetry": {}}
        return await self.opponent_agent.process(data)
    
    async def get_recommendation(self, race_state: Dict) -> Dict:
        """Get AI-powered race recommendations"""
        data = {"race_state": race_state}
        
        # Get strategy recommendation
        strategy_result = await self.strategy_agent.process(data)
        
        # Use LLM to enhance recommendation
        enhanced = await self.llm_service.enhance_recommendation(strategy_result)
        
        return enhanced
