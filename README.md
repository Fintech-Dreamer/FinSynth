# Fintech Dreamer

五个终端

```powershell
git clone 
cd 
```



```powershell
nvm install 22.13.0
nvm use 22.13.0
npm install
npm run dev
```



```powershell
conda create -n FinSynth python=3.11
conda activate FinSynth
```



```powershell
cd backend
pip install -r requirements.txt -U
uvicorn open_webui.main:app --host 0.0.0.0 --port 8080 --reload
```



```powershell
cd model
pip install -r requirements.txt -U
uvicorn model_chatbot:app --host 0.0.0.0 --port 8000 --reload
```



```powershell
cd model
uvicorn model_fraud:app --host 0.0.0.0 --port 8001 --reload
```



```powershell
cd model
uvicorn model_compliance:app --host 0.0.0.0 --port 8002 --reload
```

