from fastapi import APIRouter, File, UploadFile, HTTPException, Request

from src.api.schemas import PredictResponse

router = APIRouter()


def get_service(request: Request):
    return request.app.state.prediction_service


@router.post("/predict", response_model=PredictResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    service = get_service(request)
    result = await service.predict_and_save(
        file_bytes=data,
        file_name=file.filename or "unknown",
    )

    return PredictResponse(**result.model_dump())