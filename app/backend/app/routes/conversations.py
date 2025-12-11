from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import desc, func, select

from app.database import AsyncSessionLocal, Conversation, Message

router = APIRouter()

# TODO: break update_conversation into smaller pieces, e.g.:
# - update title
# - add message
# - delete message

class ConversationCreate(BaseModel):
    session_id: str
    title: str | None = None
    dataset_ids: str | None = None

class ConversationUpdate(BaseModel):
    title: str | None = None

class MessageCreate(BaseModel):
    conversation_id: int
    role: str
    content: str
    csv_file: str | None = None
    row_count: int | None = None

class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    csv_file: str | None = None
    row_count: int | None = None
    created_at: datetime

    class Config:
        from_attributes = True

class ConversationResponse(BaseModel):
    id: int
    session_id: str
    title: str | None
    dataset_ids: str | None
    message_count: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class ConversationDetailResponse(BaseModel):
    id: int
    session_id: str
    title: str | None
    dataset_ids: str | None
    created_at: datetime
    updated_at: datetime
    messages: list[MessageResponse]

    class Config:
        from_attributes = True

@router.get("/", response_model=list[ConversationResponse])
async def list_conversations():
    """Get all conversations with message counts"""
    async with AsyncSessionLocal() as db:
        # Get conversations with message counts
        result = await db.execute(
            select(
                Conversation,
                func.count(Message.id).label("message_count")
            )
            .outerjoin(Message, Conversation.id == Message.conversation_id)
            .group_by(Conversation.id)
            .order_by(desc(Conversation.updated_at))
        )

        conversations = []
        for conversation, message_count in result:
            conversations.append(ConversationResponse(
                id=conversation.id,
                session_id=conversation.session_id,
                title=conversation.title,
                dataset_ids=conversation.dataset_ids,
                message_count=message_count,
                created_at=conversation.created_at,
                updated_at=conversation.updated_at
            ))

        return conversations

@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(conversation_id: int):
    """Get a conversation with all its messages"""
    async with AsyncSessionLocal() as db:
        # Get conversation
        result = await db.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        conversation = result.scalar_one_or_none()

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Get messages
        messages_result = await db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
        )
        messages = messages_result.scalars().all()

        return ConversationDetailResponse(
            id=conversation.id,
            session_id=conversation.session_id,
            title=conversation.title,
            dataset_ids=conversation.dataset_ids,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            messages=[MessageResponse.model_validate(msg) for msg in messages]
        )

@router.post("/", response_model=ConversationResponse)
async def create_conversation(conversation: ConversationCreate):
    """Create a new conversation"""
    async with AsyncSessionLocal() as db:
        # Check if conversation with this session_id already exists
        result = await db.execute(
            select(Conversation).where(Conversation.session_id == conversation.session_id)
        )
        existing = result.scalar_one_or_none()

        if existing:
            return ConversationResponse(
                id=existing.id,
                session_id=existing.session_id,
                title=existing.title,
                dataset_ids=existing.dataset_ids,
                message_count=0,
                created_at=existing.created_at,
                updated_at=existing.updated_at
            )

        # Create new conversation
        new_conversation = Conversation(
            session_id=conversation.session_id,
            title=conversation.title,
            dataset_ids=conversation.dataset_ids
        )
        db.add(new_conversation)
        await db.commit()
        await db.refresh(new_conversation)

        return ConversationResponse(
            id=new_conversation.id,
            session_id=new_conversation.session_id,
            title=new_conversation.title,
            dataset_ids=new_conversation.dataset_ids,
            message_count=0,
            created_at=new_conversation.created_at,
            updated_at=new_conversation.updated_at
        )

@router.put("/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(conversation_id: int, update: ConversationUpdate):
    """Update a conversation (e.g., change title)"""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        conversation = result.scalar_one_or_none()

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        if update.title is not None:
            conversation.title = update.title

        conversation.updated_at = datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(conversation)

        # Get message count
        count_result = await db.execute(
            select(func.count(Message.id)).where(Message.conversation_id == conversation_id)
        )
        message_count = count_result.scalar()

        return ConversationResponse(
            id=conversation.id,
            session_id=conversation.session_id,
            title=conversation.title,
            dataset_ids=conversation.dataset_ids,
            message_count=message_count,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at
        )

@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: int):
    """Delete a conversation and all its messages"""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        conversation = result.scalar_one_or_none()

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        await db.delete(conversation)
        await db.commit()

        return {"message": "Conversation deleted successfully"}

@router.post("/message", response_model=MessageResponse)
async def create_message(message: MessageCreate):
    """Add a message to a conversation"""
    async with AsyncSessionLocal() as db:
        # Verify conversation exists
        result = await db.execute(
            select(Conversation).where(Conversation.id == message.conversation_id)
        )
        conversation = result.scalar_one_or_none()

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Create message
        new_message = Message(
            conversation_id=message.conversation_id,
            role=message.role,
            content=message.content,
            csv_file=message.csv_file,
            row_count=message.row_count
        )
        db.add(new_message)

        # Update conversation timestamp
        conversation.updated_at = datetime.now(timezone.utc)

        await db.commit()
        await db.refresh(new_message)

        return MessageResponse.model_validate(new_message)
