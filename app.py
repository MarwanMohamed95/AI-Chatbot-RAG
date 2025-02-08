from fastapi import FastAPI
from src.routes import base, data, generator
import uvicorn

app = FastAPI()

app.include_router(base.base_router)
app.include_router(data.upload_process_router)
app.include_router(generator.generator_router)

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5000,
        reload=True
    )
