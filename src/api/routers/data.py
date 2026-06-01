from fastapi import (
    APIRouter,
    BackgroundTasks,
    Query,
    Path as APIPath,
    HTTPException,
)
from fastapi.responses import FileResponse
from src.utils.api_utils import get_data_files, download_task, preprocess_task
from src.api.schema import (
    DataType,
    SplitType,
    DownloadRequest,
    PreprocessRequest,
    DataMetadata,
    PaginatedDataResponse,
    BackgroundJobResponse,
)

router = APIRouter(prefix="/data", tags=["Data Management"])


@router.post(
    "/download",
    response_model=BackgroundJobResponse,
    summary="Download Kaggle Dataset",
    description="Initiate a background task to download the chest X-ray dataset from Kaggle. "
    "You may provide Kaggle API credentials in the payload if they are not already set globally.",
    response_description="Confirmation that the background download task has been dispatched.",
    responses={
        200: {"description": "Download task initiated successfully."},
        500: {"description": "Internal server error failing to initiate task."},
    },
)
async def download_data(
    request: DownloadRequest, background_tasks: BackgroundTasks
) -> BackgroundJobResponse:
    background_tasks.add_task(
        download_task,
        force_download=request.force_download,
        username=request.kaggle_username,
        key=request.kaggle_key,
    )
    return BackgroundJobResponse(
        message="Dataset download initiated in the background."
    )


@router.post(
    "/preprocess",
    response_model=BackgroundJobResponse,
    summary="Preprocess Raw Dataset",
    description="Initiate a background task to preprocess the downloaded raw dataset.",
    response_description="Confirmation that the background preprocessing task has been dispatched.",
    responses={
        200: {"description": "Preprocessing task initiated successfully."}
    },
)
async def preprocess_data(
    request: PreprocessRequest, background_tasks: BackgroundTasks
) -> BackgroundJobResponse:
    background_tasks.add_task(
        preprocess_task, pipeline=request.pipeline, lgb_size=request.lgb_size
    )
    return BackgroundJobResponse(
        message=f"Preprocessing pipeline '{request.pipeline.value}' initiated in the background."
    )


@router.get(
    "",
    response_model=PaginatedDataResponse,
    summary="List Data File Metadata",
    description="Retrieve a list of file metadata for the specified dataset split and data type.",
    response_description="A paginated response object containing image metadata.",
    responses={
        200: {"description": "Successfully retrieved list of image metadata."},
        400: {"description": "Invalid pagination parameters."},
    },
)
async def get_all_data(
    data_type: DataType = Query(
        DataType.RAW, description="Type of data to retrieve (raw or processed)"
    ),
    split: SplitType = Query(
        SplitType.TRAIN, description="Dataset split (train or test)"
    ),
    page: int = Query(1, ge=1, description="Page number (starts at 1)"),
    limit: int = Query(
        20, ge=1, le=100, description="Number of items to return per page"
    ),
) -> PaginatedDataResponse:
    files = get_data_files(data_type, split)
    total = len(files)
    total_pages = max(1, (total + limit - 1) // limit)

    start = (page - 1) * limit
    end = start + limit
    paginated = files[start:end]

    items = []
    for i, f in enumerate(paginated):
        items.append(
            DataMetadata(
                index=start + i,
                filename=f["filename"],
                label=f["label"],
                split=f["split"],
                data_type=f["data_type"],
            )
        )

    return PaginatedDataResponse(
        total_items=total,
        page=page,
        limit=limit,
        total_pages=total_pages,
        items=items,
    )


@router.get(
    "/{data_type}/{split}/{index}",
    response_class=FileResponse,
    summary="Download Data File by Index",
    description="Retrieve the actual image file.",
    response_description="A streamed image file (JPEG or PGM).",
    responses={
        200: {"description": "Image file successfully streamed."},
        404: {"description": "Index is out of bounds or file does not exist."},
    },
)
async def get_data(
    data_type: DataType = APIPath(
        ..., description="Type of data (raw or processed)"
    ),
    split: SplitType = APIPath(
        ..., description="Dataset split (train or test)"
    ),
    index: int = APIPath(
        ..., ge=0, description="Index of the image in the dataset mapping"
    ),
) -> FileResponse:
    files = get_data_files(data_type, split)
    if index < 0 or index >= len(files):
        raise HTTPException(
            status_code=404,
            detail=f"Index {index} is out of bounds. Total items for this query: {len(files)}.",
        )

    file_path = files[index]["path"]
    return FileResponse(path=file_path, filename=files[index]["filename"])
