from fastapi import APIRouter

router = APIRouter(tags=["tools"], prefix="/tools")

@router.get("")
def list_tools():
    return {"tools": []}