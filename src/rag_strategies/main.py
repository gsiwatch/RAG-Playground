import re
import random
import uuid

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum

# from rag_strategies.ingestion.processor import DocumentProcessor
from rag_strategies.retrieval.rag_system import RAGSystem
from rag_strategies.utils.logger import setup_logger
from rag_strategies.utils.ssl_utils import setup_ssl_certificates

logger = setup_logger(__name__)

class MessageType(Enum):
    GREETING = "greeting"
    POLICY = "policy"
    GENERAL = "general"

class Channel(Enum):
    SERVICING = "Servicing"
    ORIGINATION = "Origination"
    ORIGINATION_RETAIL = "Origination-Retail"

class MessageRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    channel: Optional[Channel] = None
    metadata: Optional[Dict] = Field(default_factory=dict)

    @validator('message')
    def message_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Message cannot be empty')
        return v.strip()

class MessageResponse(BaseModel):
    response: str
    conversation_id: str
    message_type: MessageType
    sources: Optional[List[Dict]] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict] = None

class ConversationManager:
    def __init__(self):
        self.conversations = {}
        self.rag_system = RAGSystem()
        self.max_history = 10
        self.greeting_responses = [
            "Hello! How can I help you with policy questions today?",
            "Hi! I'm here to help with policy-related questions.",
            "Welcome! Ask me anything about our policies."
        ]
        self.general_responses = [
            "I can help you with policy-related questions. Could you please be more specific?",
            "I'm specialized in policy information. What would you like to know?",
            "Please ask me about specific policies or procedures."
        ]

    async def process_message(self, request: MessageRequest) -> MessageResponse:
        """Process incoming message and generate response"""
        conversation_id = request.conversation_id or str(uuid.uuid4())
        history = self.conversations.get(conversation_id, [])

        try:
            message_type = await self._determine_message_type(request.message)
            
            response = await self._handle_message(
                message_type=message_type,
                request=request,
                history=history
            )

            self._update_history(conversation_id, request.message, response)

            return MessageResponse(
                response=response["answer"],
                conversation_id=conversation_id,
                message_type=message_type,
                sources=response.get("citations", []),
                confidence=response.get("confidence"),
                metadata=response.get("metadata")
            )

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _determine_message_type(self, message: str) -> MessageType:
        """Determine the type of message"""
        # Greeting patterns
        greeting_patterns = [
            r"^\s*(?:hi|hello|hey|greetings|good morning|good afternoon|good evening)\s*$",
            r"^\s*(?:hi|hello|hey)\s+(?:there|everyone|all)\s*$"
        ]
        
        # Check for greetings
        if any(re.match(pattern, message.lower()) for pattern in greeting_patterns):
            return MessageType.GREETING
            
        # Policy-related keywords
        policy_keywords = [
            'policy', 'requirement', 'procedure', 'pace', 'lien',
            'modification', 'short sale', 'deed', 'loan', 'mortgage',
            'guidelines', 'rules', 'process'
        ]
        
        # Check for policy-related content
        if any(keyword in message.lower() for keyword in policy_keywords):
            return MessageType.POLICY
            
        return MessageType.GENERAL

    async def _handle_message(
        self,
        message_type: MessageType,
        request: MessageRequest,
        history: List[Dict]
    ) -> Dict:
        """Route message to appropriate handler"""
        if message_type == MessageType.GREETING:
            return await self._handle_greeting(request)
        elif message_type == MessageType.POLICY and request.channel:
            return await self._handle_policy_query(request, history)
        else:
            return await self._handle_general_message(request)

    async def _handle_greeting(self, request: MessageRequest) -> Dict:
        """Handle greeting messages"""
        return {
            "answer": random.choice(self.greeting_responses),
            "confidence": 1.0,
            "metadata": {
                "message_type": "greeting",
                "timestamp": datetime.utcnow().isoformat()
            }
        }

    async def _handle_policy_query(
        self, 
        request: MessageRequest, 
        history: List[Dict]
    ) -> Dict:
        """Handle policy-related queries using RAG system"""
        try:
            response = await self.rag_system.process_query(
                query=request.message,
                channel=request.channel.value,
                metadata={
                    'conversation_id': request.conversation_id,
                    'history': history,
                    **request.metadata
                }
            )
            
            # Enhance response metadata
            response['metadata'].update({
                'channel': request.channel.value,
                'timestamp': datetime.utcnow().isoformat(),
                'conversation_id': request.conversation_id
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing policy query: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Error processing policy query"
            )

    async def _handle_general_message(self, request: MessageRequest) -> Dict:
        """Handle general messages"""
        return {
            "answer": random.choice(self.general_responses),
            "confidence": 1.0,
            "metadata": {
                "message_type": "general",
                "timestamp": datetime.utcnow().isoformat()
            }
        }

    def _update_history(
        self,
        conversation_id: str,
        message: str,
        response: Dict
    ):
        """Update conversation history"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        self.conversations[conversation_id].append({
            "message": message,
            "response": response,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep only last N messages
        self.conversations[conversation_id] = \
            self.conversations[conversation_id][-self.max_history:]

app = FastAPI(
    title="Policy Chat API",
    description="API for policy-related chat interactions",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Initialize API dependencies"""
    setup_ssl_certificates()
    logger.info("API starting up...")

conversation_manager = ConversationManager()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/message", response_model=MessageResponse)
async def process_message(request: MessageRequest):
    """Process chat message and return response"""
    if request.message_type == MessageType.POLICY and not request.channel:
        raise HTTPException(
            status_code=400,
            detail="Channel is required for policy queries"
        )
    return await conversation_manager.process_message(request)

@app.get("/health")
async def health_check():
    """Check API health status"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
