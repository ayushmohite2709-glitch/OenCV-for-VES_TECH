import asyncio
import json
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initial fallback state
latest_data = {
    "timestamp": time.time(),
    "frame_number": 0,
    "fps": 0.0,
    "capacity": {
        "total_registered": 0,
        "present": 0,
        "not_in_room": 0,
        "extra_people": 0
    },
    "traffic": {
        "entry_count": 0,
        "exit_count": 0
    },
    "status_info": {
        "status": "INITIALIZING",
        "alert_type": "OK",
        "alert_message": "WAITING FOR CAMERA FEED"
    }
}

def update_room_data(payload: dict):
    global latest_data
    latest_data = payload

@app.get("/api/stream")
async def stream_room_data():
    async def event_generator():
        while True:
            yield {"data": json.dumps(latest_data)}
            await asyncio.sleep(0.1)

    return EventSourceResponse(event_generator())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)