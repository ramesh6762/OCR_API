from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import easyocr
from PIL import Image
import numpy as np
import io
import os
import uvicorn


app = FastAPI(title="OCR API", version="1.0.0")
reader = easyocr.Reader(['en'], gpu=False)  # Load once

@app.post("/ocr/")
async def extract_text(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Resize to reduce memory usage
        image = image.resize((1000, int(image.height * 1000 / image.width)))
        
        image_np = np.array(image)
        results = reader.readtext(image_np)

        response = [{"text": text, "confidence": float(conf), "box": bbox}
                    for (bbox, text, conf) in results]

        return JSONResponse(content={"status": "success", "data": response})

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

# Required for Render to detect and run properly
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
