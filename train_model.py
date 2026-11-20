from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression


project_dir = Path(__file__).resolve().parent
data = pd.read_csv(project_dir / "training_data.csv")

model = LinearRegression()
model.fit(data[["YearsExperience"]], data["Salary"])

result = (
    f"Training rows: {len(data)}\n"
    f"Coefficient: {model.coef_[0]:.2f}\n"
    f"Intercept: {model.intercept_:.2f}\n"
)

(project_dir / "linear_model.txt").write_text(result, encoding="utf-8")
print(result)
