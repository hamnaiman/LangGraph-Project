from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from groq import APIConnectionError, APIStatusError, APITimeoutError
from pydantic import BaseModel, Field

from main import app as task_graph


BASE = Path(__file__).resolve().parent
app = FastAPI(title="AI Task Organizer")


class TaskRequest(BaseModel):
    message: str = Field(min_length=1, max_length=5000)


@app.get("/")
def home():
    return FileResponse(BASE / "index.html")


@app.post("/organize")
def organize(body: TaskRequest):
    message = body.message.strip()

    if not message:
        raise HTTPException(400, "Please enter your tasks.")

    try:
        output = task_graph.invoke({
            "user_input": message,
            "tasks": "",
            "priorities": "",
            "result": ""
        })

        return {
            "tasks": output["tasks"],
            "priorities": output["priorities"],
            "result": output["result"]
        }

    except APITimeoutError:
        raise HTTPException(
            504,
            "Groq took too long. Please try again."
        )

    except APIConnectionError:
        raise HTTPException(
            502,
            "Cannot connect to Groq. Check your internet."
        )

    except APIStatusError as error:
        messages = {
            400: "Groq rejected the request. Check model settings.",
            401: "Invalid Groq API key. Check .env and restart.",
            403: "Your account cannot access this model.",
            404: "Model not found. Check GROQ_MODEL in .env.",
            429: "Groq usage limit reached. Please try later."
        }

        raise HTTPException(
            502,
            messages.get(
                error.status_code,
                "Groq is unavailable. Please try again later."
            )
        )

    except Exception:
        raise HTTPException(
            500,
            "Could not organize your tasks. Please try again."
        )