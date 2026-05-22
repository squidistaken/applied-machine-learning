from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


# TODO get to retrieve model
# TODO post to request to train a model

# TODO post to clasify an image: returns result with metrics
# TODO get model metrics (confusion matrix and model history)
