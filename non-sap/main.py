# Non-SAP backend — ai-trading-platform
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok", "message": "Non-SAP backend running"}

@app.get("/health")
def health():
    return {"healthy": True}
```

5. Click **Commit changes → Commit changes**

---

## Step 3 — Create requirements.txt

Still inside `non-sap/`:

1. Click **Add file → Create new file**
2. Filename: `requirements.txt`
3. Paste:
```
fastapi
uvicorn
