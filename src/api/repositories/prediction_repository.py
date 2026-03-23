from redis.asyncio import Redis
from src.api.schemas import PredictionRecord


class PredictionRepository:
    def __init__(self, redis: Redis) -> None:
        self.redis = redis

    async def save(self, prediction: PredictionRecord) -> None:
        key = f"prediction:{prediction.predictionId}"

        await self.redis.hset(
            key,
            mapping={
                "predictionId": prediction.predictionId,
                "fileName": prediction.fileName,
                "createdAt": prediction.createdAt.isoformat(),
                "dogProbability": str(prediction.dogProbability),
                "predictedLabel": prediction.predictedLabel,
                "modelVersion": prediction.modelVersion,
            },
        )

        await self.redis.zadd(
            "predictions:by_time",
            {prediction.predictionId: prediction.createdAt.timestamp()},
        )