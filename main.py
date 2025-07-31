from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import easyocr
from PIL import Image
import numpy as np
import io

app = FastAPI(title="OCR API", version="1.0.0")
reader = easyocr.Reader(['en'], gpu=False)

@app.post("/ocr/")
async def extract_text(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)

        results = reader.readtext(image_np)

        response = []
        for (bbox, text, conf) in results:
            response.append({
                "text": text,
                "confidence": float(conf),
                "box": bbox
            })

        return JSONResponse(content={"status": "success", "data": response})

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
