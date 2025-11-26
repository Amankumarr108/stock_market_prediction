from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, conint
import datetime
import uvicorn
import os
import pandas as pd
import logging
from typing import List, Optional
import threading

# ------------------------------------------------------------
# Logging & Configuration
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger("stock_prediction_app")

BASE_DIR = os.path.dirname(__file__)

# ------------------------------------------------------------
# FastAPI App Setup
# ------------------------------------------------------------
app = FastAPI(title="Stock Prediction App", description="Predict future prices using ML model.")

# Templates and static
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Allow cross-origin access for APIs (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supported companies (extendable)
COMPANIES = ["SBUX"]

# Model cache (lazy loaded) and a lock to prevent race conditions
_model_lock = threading.Lock()
_model_cache = {"model": None, "scaler": None, "q_weights": None}

MODEL_DIR = BASE_DIR
CSV_SUFFIX = ".csv"

# ------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------
class StockData(BaseModel):
    recent_closes: List[float] = Field(..., description="List of recent closing prices")
    days: conint(ge=1, le=365) = Field(7, description="Number of days to predict (1-365)")

class PredictResponse(BaseModel):
    predictions: List[float]
    company: str
    days: int
    generated_at: datetime.datetime

# ------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------
def _load_model_lazy():
    """
    Lazy load model, scaler and quantum weights using a local helper module.
    Designed to be thread-safe.
    """
    with _model_lock:
        if _model_cache["model"] is not None and _model_cache["scaler"] is not None and _model_cache["q_weights"] is not None:
            logger.debug("Model already loaded in cache.")
            return _model_cache["model"], _model_cache["scaler"], _model_cache["q_weights"]

        try:
            # Import inside function to avoid startup-time dependencies
            from model_utils import load_model_and_scaler  # local module expected
        except Exception as e:
            logger.error("Failed to import model utilities: %s", e)
            raise RuntimeError("Model utilities not available.") from e

        try:
            model, scaler, q_weights = load_model_and_scaler()
        except Exception as e:
            logger.error("Failed to load model/scaler/weights: %s", e)
            raise RuntimeError("Model load failed.") from e

        _model_cache["model"] = model
        _model_cache["scaler"] = scaler
        _model_cache["q_weights"] = q_weights
        logger.info("Model, scaler, and weights loaded and cached.")
        return model, scaler, q_weights

def _read_company_csv(company: str) -> pd.DataFrame:
    """
    Read CSV for a company from the application folder. Ensures 'Close' column exists and index is datetime.
    """
    if company not in COMPANIES:
        raise ValueError("Unsupported company.")

    csv_path = os.path.join(BASE_DIR, f"{company}{CSV_SUFFIX}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df.ffill(inplace=True)
    if 'Close' not in df.columns:
        raise ValueError("CSV missing required 'Close' column.")
    df = df[['Close']].dropna()
    if df.empty:
        raise ValueError("CSV contains no valid 'Close' values after cleaning.")
    return df

# ------------------------------------------------------------
# Exception handlers
# ------------------------------------------------------------
@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError):
    logger.warning("File not found: %s", exc)
    return templates.TemplateResponse("index.html", {"request": request, "error": str(exc), "companies": COMPANIES, "company": COMPANIES[0], "year": datetime.datetime.now().year}, status_code=404)

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    logger.warning("Value error: %s", exc)
    return templates.TemplateResponse("index.html", {"request": request, "error": str(exc), "companies": COMPANIES, "company": COMPANIES[0], "year": datetime.datetime.now().year}, status_code=400)

# ------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------
@app.get("/health")
async def health_check():
    """Basic health endpoint to confirm service is running."""
    return {"status": "ok", "time": datetime.datetime.utcnow().isoformat()}

@app.get("/")
async def home(request: Request, company: Optional[str] = "SBUX"):
    """
    Render the index.html template.
    If requested company is unsupported, fallback to the first supported company.
    """
    if company not in COMPANIES:
        company = COMPANIES[0]
    return templates.TemplateResponse("index.html", {"request": request, "companies": COMPANIES, "company": company, "year": datetime.datetime.now().year})

@app.post("/predict_form")
async def predict_form(request: Request):
    """
    Endpoint for form submissions from the web UI.
    Reads form values, validates, and returns rendered template with results or errors.
    """
    form = await request.form()
    company = form.get("company", COMPANIES[0]).strip()
    days_raw = form.get("days", "7").strip()
    try:
        days = int(days_raw)
    except Exception:
        days = 7

    if company not in COMPANIES:
        return templates.TemplateResponse("index.html", {"request": request, "error": "Invalid company selected.", "companies": COMPANIES, "company": COMPANIES[0], "year": datetime.datetime.now().year})

    try:
        df = _read_company_csv(company)
        closes = df["Close"].dropna().tolist()
        window_size = 4
        if len(closes) < window_size:
            raise ValueError(f"Insufficient data: need at least {window_size} closing prices.")
        recent_data = closes[-window_size:]

        # Lazy load model & scaler
        model, scaler, q_weights = _load_model_lazy()

        # Import predict function lazily
        from model_utils import predict_n_days
        preds = predict_n_days(model=model, scaler=scaler, q_weights=q_weights,
                               recent_data=recent_data, window_size=window_size, n_days=days)
        preds_rounded = [f"{float(x):.2f}" for x in preds]

        return templates.TemplateResponse("index.html", {"request": request, "predictions": preds_rounded, "companies": COMPANIES, "company": company, "year": datetime.datetime.now().year})
    except Exception as e:
        logger.exception("Prediction failed for company=%s: %s", company, e)
        return templates.TemplateResponse("index.html", {"request": request, "error": str(e), "companies": COMPANIES, "company": company, "year": datetime.datetime.now().year})

@app.post("/api/predict", response_model=PredictResponse)
async def predict_api(payload: StockData, company: Optional[str] = "SBUX"):
    """
    JSON API endpoint to obtain predictions programmatically.
    Expects a JSON body with recent_closes (list of floats) and optional days value.
    """
    if company not in COMPANIES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported company.")

    recent = payload.recent_closes
    days = payload.days

    if not isinstance(recent, list) or len(recent) < 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="recent_closes must be a non-empty list.")

    try:
        model, scaler, q_weights = _load_model_lazy()
        from model_utils import predict_n_days
        preds = predict_n_days(model=model, scaler=scaler, q_weights=q_weights,
                               recent_data=recent, window_size=len(recent), n_days=days)
        preds_floats = [float(x) for x in preds]
        return PredictResponse(predictions=preds_floats, company=company, days=days, generated_at=datetime.datetime.utcnow())
    except RuntimeError as e:
        logger.error("Model not available: %s", e)
        raise HTTPException(status_code=503, detail="Model not available.")
    except Exception as e:
        logger.exception("API prediction failed: %s", e)
        raise HTTPException(status_code=500, detail="Internal prediction error.")

# ------------------------------------------------------------
# Startup / Shutdown events (optional)
# ------------------------------------------------------------
@app.on_event("startup")
async def on_startup():
    logger.info("Application startup complete. Ready to accept requests.")

@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Application shutdown initiated.")

# ------------------------------------------------------------
# Run if executed directly
# ------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info("Starting application at http://127.0.0.1:%d", port)
    uvicorn.run(app, host="127.0.0.1", port=port)
