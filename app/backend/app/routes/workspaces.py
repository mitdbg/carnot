"""Workspace CRUD routes.

Workspaces are the top-level entity in the sidebar.  Each workspace
contains one or more conversations (strictly one today) and zero or
more notebooks.

``POST /workspaces/`` creates a workspace **and** its first
conversation atomically so the frontend never has to issue two
requests.
"""

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.database import Conversation, Message, Notebook, QueryEvent, Workspace, get_db

router = APIRouter()


# ── Request / Response Models ───────────────────────────────────────────

class WorkspaceCreate(BaseModel):
    title: str = "Untitled Workspace"
    dataset_ids: str | None = None


class WorkspaceUpdate(BaseModel):
    title: str | None = None
    dataset_ids: str | None = None


class ConversationSummary(BaseModel):
    id: int
    session_id: str
    title: str | None
    message_count: int
    is_query_active: bool = False
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class NotebookSummary(BaseModel):
    id: int
    notebook_uuid: str
    label: str
    query: str
    cell_count: int
    created_at: datetime

    class Config:
        from_attributes = True


class WorkspaceResponse(BaseModel):
    """Returned by ``GET /workspaces/`` (list view)."""
    id: int
    session_id: str
    title: str
    dataset_ids: str | None
    created_at: datetime
    updated_at: datetime
    conversation_count: int
    message_count: int

    class Config:
        from_attributes = True


class WorkspaceDetailResponse(BaseModel):
    """Returned by ``GET /workspaces/{id}`` (detail view)."""
    id: int
    session_id: str
    title: str
    dataset_ids: str | None
    created_at: datetime
    updated_at: datetime
    conversations: list[ConversationSummary]
    notebooks: list[NotebookSummary]
    total_cost_usd: float | None = None

    class Config:
        from_attributes = True


# ── Endpoints ───────────────────────────────────────────────────────────

@router.get("/", response_model=list[WorkspaceResponse])
async def list_workspaces(
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all workspaces for the current user (sidebar)."""
    result = await db.execute(
        select(
            Workspace,
            func.count(Conversation.id.distinct()).label("conversation_count"),
            func.count(Message.id).label("message_count"),
        )
        .where(Workspace.user_id == user_id)
        .outerjoin(Conversation, Workspace.id == Conversation.workspace_id)
        .outerjoin(Message, Conversation.id == Message.conversation_id)
        .group_by(Workspace.id)
        .order_by(desc(Workspace.updated_at))
    )

    workspaces = []
    for workspace, conversation_count, message_count in result:
        workspaces.append(WorkspaceResponse(
            id=workspace.id,
            session_id=workspace.session_id,
            title=workspace.title,
            dataset_ids=workspace.dataset_ids,
            created_at=workspace.created_at,
            updated_at=workspace.updated_at,
            conversation_count=conversation_count,
            message_count=message_count,
        ))

    return workspaces


@router.get("/{workspace_id}", response_model=WorkspaceDetailResponse)
async def get_workspace(
    workspace_id: int,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a workspace with its conversations and notebooks."""
    # Fetch workspace
    result = await db.execute(
        select(Workspace).where(
            Workspace.id == workspace_id,
            Workspace.user_id == user_id,
        )
    )
    workspace = result.scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    # Fetch conversations with message counts
    conv_result = await db.execute(
        select(
            Conversation,
            func.count(Message.id).label("message_count"),
        )
        .where(Conversation.workspace_id == workspace_id)
        .outerjoin(Message, Conversation.id == Message.conversation_id)
        .group_by(Conversation.id)
        .order_by(Conversation.created_at)
    )
    conversations = [
        ConversationSummary(
            id=conv.id,
            session_id=conv.session_id,
            title=conv.title,
            message_count=msg_count,
            is_query_active=conv.is_query_active,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
        )
        for conv, msg_count in conv_result
    ]

    # Fetch notebooks
    nb_result = await db.execute(
        select(Notebook)
        .where(Notebook.workspace_id == workspace_id)
        .order_by(Notebook.created_at)
    )
    notebooks_rows = nb_result.scalars().all()
    notebooks = [
        NotebookSummary(
            id=nb.id,
            notebook_uuid=nb.notebook_uuid,
            label=nb.label,
            query=nb.query,
            cell_count=len(nb.cells_json) if nb.cells_json else 0,
            created_at=nb.created_at,
        )
        for nb in notebooks_rows
    ]

    # Sum per-step costs from query_events across all conversations
    # in this workspace to derive the total workspace cost.
    conv_ids = [c.id for c in conversations]
    total_cost: float | None = None
    if conv_ids:
        cost_result = await db.execute(
            select(func.sum(QueryEvent.step_cost_usd))
            .where(QueryEvent.conversation_id.in_(conv_ids))
        )
        total_cost = cost_result.scalar()

    return WorkspaceDetailResponse(
        id=workspace.id,
        session_id=workspace.session_id,
        title=workspace.title,
        dataset_ids=workspace.dataset_ids,
        created_at=workspace.created_at,
        updated_at=workspace.updated_at,
        conversations=conversations,
        notebooks=notebooks,
        total_cost_usd=total_cost,
    )


@router.post("/", response_model=WorkspaceDetailResponse)
async def create_workspace(
    body: WorkspaceCreate,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new workspace with its first conversation.

    The backend generates ``session_id`` for both the workspace and its
    first conversation (they share the same value in the 1:1 case).
    """
    session_id = str(uuid4())

    workspace = Workspace(
        user_id=user_id,
        session_id=session_id,
        title=body.title,
        dataset_ids=body.dataset_ids,
    )
    db.add(workspace)
    await db.flush()  # get workspace.id without committing

    conversation = Conversation(
        workspace_id=workspace.id,
        user_id=user_id,
        session_id=session_id,  # same as workspace in 1:1 case
    )
    db.add(conversation)
    await db.commit()
    await db.refresh(workspace)
    await db.refresh(conversation)

    return WorkspaceDetailResponse(
        id=workspace.id,
        session_id=workspace.session_id,
        title=workspace.title,
        dataset_ids=workspace.dataset_ids,
        created_at=workspace.created_at,
        updated_at=workspace.updated_at,
        conversations=[
            ConversationSummary(
                id=conversation.id,
                session_id=conversation.session_id,
                title=conversation.title,
                message_count=0,
                created_at=conversation.created_at,
                updated_at=conversation.updated_at,
            )
        ],
        notebooks=[],
    )


@router.put("/{workspace_id}", response_model=WorkspaceResponse)
async def update_workspace(
    workspace_id: int,
    update: WorkspaceUpdate,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update a workspace's title or dataset_ids."""
    result = await db.execute(
        select(Workspace).where(
            Workspace.id == workspace_id,
            Workspace.user_id == user_id,
        )
    )
    workspace = result.scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    if update.title is not None:
        workspace.title = update.title
    if update.dataset_ids is not None:
        workspace.dataset_ids = update.dataset_ids

    workspace.updated_at = datetime.now(timezone.utc)  # noqa: UP017
    await db.commit()
    await db.refresh(workspace)

    # Get conversation count and total message count
    count_result = await db.execute(
        select(
            func.count(Conversation.id.distinct()),
            func.count(Message.id),
        )
        .outerjoin(Message, Conversation.id == Message.conversation_id)
        .where(Conversation.workspace_id == workspace_id)
    )
    conversation_count, message_count = count_result.one()

    return WorkspaceResponse(
        id=workspace.id,
        session_id=workspace.session_id,
        title=workspace.title,
        dataset_ids=workspace.dataset_ids,
        created_at=workspace.created_at,
        updated_at=workspace.updated_at,
        conversation_count=conversation_count,
        message_count=message_count,
    )


@router.delete("/{workspace_id}")
async def delete_workspace(
    workspace_id: int,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a workspace and cascade to conversations, messages, notebooks, stats."""
    result = await db.execute(
        select(Workspace).where(
            Workspace.id == workspace_id,
            Workspace.user_id == user_id,
        )
    )
    workspace = result.scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    await db.delete(workspace)
    await db.commit()

    return {"message": "Workspace deleted successfully"}


@router.get("/{workspace_id}/notebooks", response_model=list[NotebookSummary])
async def list_workspace_notebooks(
    workspace_id: int,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List notebooks for a workspace."""
    # Verify workspace ownership
    ws_result = await db.execute(
        select(Workspace).where(
            Workspace.id == workspace_id,
            Workspace.user_id == user_id,
        )
    )
    if not ws_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Workspace not found")

    nb_result = await db.execute(
        select(Notebook)
        .where(Notebook.workspace_id == workspace_id)
        .order_by(Notebook.created_at)
    )
    notebooks = nb_result.scalars().all()

    return [
        NotebookSummary(
            id=nb.id,
            notebook_uuid=nb.notebook_uuid,
            label=nb.label,
            query=nb.query,
            cell_count=len(nb.cells_json) if nb.cells_json else 0,
            created_at=nb.created_at,
        )
        for nb in notebooks
    ]
