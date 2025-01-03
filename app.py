from fastapi import FastAPI
from src.routes import base, data, generator, generator_agent
import uvicorn

app = FastAPI()

app.include_router(base.base_router)
app.include_router(data.uploader_router)
app.include_router(data.processor_router)
app.include_router(generator_agent.generator_router)

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5000,
        reload=True
    )
