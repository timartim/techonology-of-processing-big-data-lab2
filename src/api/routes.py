from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from fastapi.params import Query

from src.api.schemas import PredictResponse, PredictionRecord

router = APIRouter()


def get_service(request: Request):
    return request.app.state.prediction_service


def get_predict_lock(request: Request):
    return request.app.state.predict_lock


@router.post("/predict", response_model=PredictResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    lock = get_predict_lock(request)
    if lock.locked():
        raise HTTPException(
            status_code=409,
            detail="Prediction is already in progress. Please try again later.",
        )

    service = get_service(request)

    async with lock:
        result = await service.predict_and_save(
            file_bytes=data,
            file_name=file.filename or "unknown",
        )

    return PredictResponse(**result.model_dump())


@router.get("/predictions", response_model=list[PredictionRecord])
async def get_last_predictions(
    request: Request,
    limit: int = Query(10, ge=1, le=100),
):
    service = get_service(request)
    return await service.get_last_predictions(limit)