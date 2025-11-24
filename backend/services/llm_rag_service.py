## **File: backend/services/llm_rag_service.py** (LLM + RAG Integration)

"""
LLM + RAG Service
Provides AI reasoning layer using LLM with RAG
Generates explanations, recommendations, and insights
"""
from typing import Dict, List, Optional
import logging
import json

try:
    from langchain.chat_models import ChatOpenAI, ChatAnthropic
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema import HumanMessage, SystemMessage
    import chromadb
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("langchain/chromadb not installed. LLM features limited.")

from config import settings
from services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class LLMRAGService:
    """
    LLM + RAG service for AI reasoning and explanation generation
    """
    
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.llm = None
        self.vector_db = None
        
        if LANGCHAIN_AVAILABLE:
            self._initialize_llm()
            self._initialize_vector_db()
    
    def _initialize_llm(self):
        """Initialize LLM (OpenAI or Anthropic)"""
        try:
            if settings.OPENAI_API_KEY:
                self.llm = ChatOpenAI(
                    model=settings.LLM_MODEL,
                    api_key=settings.OPENAI_API_KEY,
                    temperature=0.7
                )
                logger.info(f"Initialized OpenAI LLM: {settings.LLM_MODEL}")
            elif settings.ANTHROPIC_API_KEY:
                self.llm = ChatAnthropic(
                    model="claude-3-sonnet-20240229",
                    api_key=settings.ANTHROPIC_API_KEY,
                    temperature=0.7
                )
                logger.info("Initialized Anthropic LLM")
            else:
                logger.warning("No API key provided for LLM")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
    
    def _initialize_vector_db(self):
        """Initialize ChromaDB for RAG"""
        try:
            self.vector_db = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)
            
            # Create collection if doesn't exist
            try:
                self.collection = self.vector_db.get_collection("engitorquemap_knowledge")
            except:
                self.collection = self.vector_db.create_collection(
                    name="engitorquemap_knowledge",
                    metadata={"description": "Racing telemetry and strategy knowledge"}
                )
            
            logger.info("Initialized ChromaDB vector database")
        except Exception as e:
            logger.error(f"Failed to initialize vector DB: {e}")
    
    async def synthesize_agent_results(self, agent_results: Dict) -> Dict:
        """
        Synthesize results from multiple agents using LLM
        Provides coherent explanation and recommendations
        """
        if not self.llm or not LANGCHAIN_AVAILABLE:
            return {
                "synthesis": "LLM not available",
                "recommendations": []
            }
        
        try:
            # Create prompt with agent results
            prompt = self._create_synthesis_prompt(agent_results)
            
            # Query LLM
            messages = [
                SystemMessage(content="You are an expert racing engineer analyzing telemetry and strategy data."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm(messages)
            
            # Parse response
            synthesis = {
                "explanation": response.content,
                "key_insights": self._extract_key_insights(response.content),
                "recommendations": self._extract_recommendations(response.content)
            }
            
            return synthesis
            
        except Exception as e:
            logger.error(f"Failed to synthesize agent results: {e}")
            return {
                "synthesis": "Error in synthesis",
                "error": str(e)
            }
    
    def _create_synthesis_prompt(self, agent_results: Dict) -> str:
        """Create prompt for LLM synthesis"""
        prompt_parts = ["Analyze the following racing data and provide insights:\n"]
        
        if 'energy_analysis' in agent_results:
            energy = agent_results['energy_analysis']
            prompt_parts.append(f"\nEnergy Analysis:")
            prompt_parts.append(f"- Energy efficiency: {energy.get('efficiency', {})}")
            prompt_parts.append(f"- Anomalies detected: {len(energy.get('anomalies', []))}")
        
        if 'timeshift_analysis' in agent_results:
            timeshift = agent_results['timeshift_analysis']
            prompt_parts.append(f"\nPerformance Optimization:")
            prompt_parts.append(f"- Suggestions available: {len(timeshift.get('suggestions', []))}")
            if timeshift.get('suggestions'):
                top_suggestion = timeshift['suggestions'][0]
                prompt_parts.append(f"- Top opportunity: {top_suggestion.get('description')}")
        
        if 'strategy_analysis' in agent_results:
            strategy = agent_results['strategy_analysis']
            prompt_parts.append(f"\nStrategy Recommendations:")
            prompt_parts.append(f"- Pit recommendation: {strategy.get('pit_recommendation', {}).get('recommendation')}")
            prompt_parts.append(f"- Pace mode: {strategy.get('pace_recommendation', {}).get('pace_mode')}")
        
        prompt_parts.append("\nProvide a concise summary with key insights and actionable recommendations.")
        
        return "\n".join(prompt_parts)
    
    def _extract_key_insights(self, llm_response: str) -> List[str]:
        """Extract key insights from LLM response"""
        # Simple extraction - look for bullet points or numbered lists
        insights = []
        lines = llm_response.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('-') or line.startswith('•') or line[0:2].replace('.', '').isdigit():
                insight = line.lstrip('-•0123456789. ')
                if len(insight) > 10:
                    insights.append(insight)
        
        return insights[:5]  # Return top 5
    
    def _extract_recommendations(self, llm_response: str) -> List[str]:
        """Extract recommendations from LLM response"""
        # Look for recommendation keywords
        recommendations = []
        lines = llm_response.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ['recommend', 'should', 'consider', 'suggest']):
                rec = line.strip()
                if len(rec) > 10:
                    recommendations.append(rec)
        
        return recommendations[:3]  # Return top 3
    
    async def enhance_recommendation(self, strategy_result: Dict) -> Dict:
        """Enhance strategy recommendation with LLM explanation"""
        if not self.llm or not LANGCHAIN_AVAILABLE:
            return strategy_result
        
        try:
            prompt = f"""
            Given this strategy recommendation:
            Pit: {strategy_result.get('pit_recommendation')}
            Pace: {strategy_result.get('pace_recommendation')}
            Tire: {strategy_result.get('tire_recommendation')}
            
            Provide a clear, concise explanation for the driver and team.
            """
            
            messages = [
                SystemMessage(content="You are a racing strategist."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm(messages)
            
            strategy_result['enhanced_explanation'] = response.content
            return strategy_result
            
        except Exception as e:
            logger.error(f"Failed to enhance recommendation: {e}")
            return strategy_result
    
    async def query_knowledge_base(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Query vector database for relevant knowledge
        RAG retrieval step
        """
        if not self.vector_db:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.generate_embedding(query)
            
            # Query vector DB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            
            # Format results
            documents = []
            if results and 'documents' in results:
                for i, doc in enumerate(results['documents'][0]):
                    documents.append({
                        "content": doc,
                        "metadata": results['metadatas'][0][i] if 'metadatas' in results else {},
                        "distance": results['distances'][0][i] if 'distances' in results else 0.0
                    })
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to query knowledge base: {e}")
            return []
    
    async def add_to_knowledge_base(self, document: str, metadata: Dict = None):
        """Add document to vector database"""
        if not self.vector_db:
            return
        
        try:
            # Generate embedding
            embedding = self.embedding_service.generate_embedding(document)
            
            # Add to collection
            self.collection.add(
                embeddings=[embedding.tolist()],
                documents=[document],
                metadatas=[metadata or {}],
                ids=[f"doc_{hash(document)}"]
            )
            
            logger.info("Added document to knowledge base")
            
        except Exception as e:
            logger.error(f"Failed to add to knowledge base: {e}")
    
    async def generate_prompt_engineering_query(self, user_query: str, context: Dict) -> str:
        """
        Generate structured query using prompt engineering
        Optimizes query for better LLM responses
        """
        structured_query = f"""
        Context:
        - Track: {context.get('track_name', 'unknown')}
        - Driver: {context.get('driver_id', 'unknown')}
        - Session: {context.get('session_type', 'unknown')}
        
        User Query: {user_query}
        
        Provide a detailed, data-driven response considering:
        1. Current telemetry and energy usage
        2. Historical performance patterns
        3. Track-specific characteristics
        4. Strategic implications
        """
        
        return structured_query
