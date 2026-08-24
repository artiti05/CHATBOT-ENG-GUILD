import shutil
from pathlib import Path
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.api.dependencies import verify_admin_key
from src.cache_db.document_registry import DocumentRegistry
from src.config import NEW_UPLOADS_DIR, PARSED_OUTPUT_DIR, PDFS_DIR, TEXTS_DIR
from src.kb_ingestor.ingestion import IngestionPipeline

router = APIRouter()
pipeline = IngestionPipeline()
registry = DocumentRegistry()

class ReingestRequest(BaseModel):
    files: List[str]
    reingest_all: bool = False

def process_file_background(file_path: Path):
    try:
        print(f"[Admin Background] Starting processing for {file_path}")
        pipeline.process_file(file_path)
        print(f"[Admin Background] Finished processing for {file_path}")
    except Exception as e:
        print(f"[Admin Background Error] Failed processing {file_path}: {e}")

@router.post("/files/upload", dependencies=[Depends(verify_admin_key)], status_code=202)
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    clean_filename = Path(file.filename).name
    ext = clean_filename.lower().split('.')[-1]
    if ext not in ['pdf', 'txt', 'md']:
        raise HTTPException(status_code=400, detail="Unsupported file format. Only PDF, TXT, MD allowed.")

    save_dir = NEW_UPLOADS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / clean_filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")

    background_tasks.add_task(process_file_background, file_path)
    return {"status": "processing_started", "filename": clean_filename, "message": "File uploaded and background processing started."}


@router.delete("/files/{filename}", dependencies=[Depends(verify_admin_key)])
async def delete_file(filename: str):
    try:
        docs = registry.list_documents()
        doc = next((d for d in docs if d["source_id"] == filename), None)
        if not doc:
            raise HTTPException(status_code=404, detail="File not found in knowledge base.")

        registry.exclude_document(filename)
        return {"status": "success", "message": f"File '{filename}' successfully removed from the knowledge base."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {e}")

@router.post("/files/re-ingest", dependencies=[Depends(verify_admin_key)], status_code=202)
async def reingest_files(req: ReingestRequest, background_tasks: BackgroundTasks):
    directories_to_scan = [PDFS_DIR, TEXTS_DIR, PARSED_OUTPUT_DIR, NEW_UPLOADS_DIR]
    files_to_process = []

    for d in directories_to_scan:
        if d.exists():
            for filepath in d.iterdir():
                if filepath.is_file() and (req.reingest_all or filepath.name in req.files):
                    files_to_process.append(filepath)

    if not files_to_process:
        raise HTTPException(status_code=404, detail="No matching files found in data directories.")

    for filepath in files_to_process:
        try:
            registry.exclude_document(filepath.name)
        except Exception as e:
            print(f"Warning: Could not exclude {filepath.name} before reingest: {e}")

    for filepath in files_to_process:
        background_tasks.add_task(process_file_background, filepath)

    return {
        "status": "processing_started",
        "files_queued": len(files_to_process),
        "filenames": [f.name for f in files_to_process]
    }

@router.get("/files", dependencies=[Depends(verify_admin_key)])
async def list_files():
    try:
        docs = registry.list_documents()
        return {"documents": docs, "count": len(docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {e}")
