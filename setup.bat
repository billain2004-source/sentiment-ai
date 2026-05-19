@echo off
echo ============================================================
echo  SentimentAI - Setup
echo ============================================================

echo.
echo [1/3] Installing Python packages...
venv\Scripts\pip install -r requirements.txt

echo.
echo [2/3] Downloading spaCy English model...
venv\Scripts\python -m spacy download en_core_web_sm

echo.
echo [3/3] Setup complete!
echo.
echo To start the app:  venv\Scripts\python app.py
echo To train model:    venv\Scripts\python train.py
echo.
pause
