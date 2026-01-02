from pathlib import Path

# Import your existing files WITHOUT changing them
from src.data_preprocessing import process_data
from src.model_training import train_model
from src.control_optimization import optimize

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "turbine_data_clean.csv"

### DATA PROCESSING
X, y = process_data(DATA_PATH)

####️⃣ TRAIN MODEL
model, metrics = train_model(X, y)

### OPTIMIZATION
result = optimize(model, X)

print("\nPipeline completed successfully")