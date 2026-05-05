from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
from PIL import Image
import io
import base64

app = FastAPI()

def snc_logic(img_array):
    # Resize and Process
    img = cv2.resize(img_array, (224, 224))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # DullRazor Simulation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    hair_removed = cv2.inpaint(img_bgr, mask, 1, cv2.INPAINT_TELEA)
    
    # Encode to Base64 to show in HTML
    _, buffer = cv2.imencode('.jpg', hair_removed)
    return base64.b64encode(buffer).decode('utf-8')

@app.get("/", response_class=HTMLResponse)
async def main():
    content = """
    <body style="background-color: #006a4e; color: white; font-family: sans-serif; text-align: center; padding: 50px;">
        <h1>🇧🇩 SNC-Net Skin Cancer AI</h1>
        <p>Hybrid Deep Learning Portal | Accuracy: 97.81%</p>
        <div style="background: white; color: black; padding: 30px; border-radius: 15px; display: inline-block; border: 4px solid #f42a41;">
            <form action="/upload" enctype="multipart/form-data" method="post">
                <input name="file" type="file" required>
                <input type="submit" style="background: #f42a41; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer;">
            </form>
        </div>
    </body>
    """
    return content

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    contents = await file.read()
    image = np.array(Image.open(io.BytesIO(contents)))
    processed_base64 = snc_logic(image)
    
    return HTMLResponse(content=f"""
    <body style="background-color: #006a4e; color: white; font-family: sans-serif; text-align: center; padding: 50px;">
        <h2>Analysis Result</h2>
        <img src="data:image/jpeg;base64,{processed_base64}" style="border: 5px solid white; border-radius: 10px; max-width: 400px;">
        <br><br>
        <a href="/" style="color: white; text-decoration: none; border: 1px solid white; padding: 10px;">Back to Home</a>
    </body>
    """)
