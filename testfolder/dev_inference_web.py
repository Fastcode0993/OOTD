"""
dev_inference_web.py
────────────────────
Flask 웹 인터페이스로 가상 이미지를 생성하고 추론하는 개발 도구.

실행:
  python dev_inference_web.py
  
브라우저: http://localhost:5000
"""

import sys
from pathlib import Path
import json
import io
from datetime import datetime

_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np
from PIL import Image, ImageDraw
from flask import Flask, render_string, request, jsonify, send_file

from project.ai.infer import run_inference
from project.ai.model_loader import ModelLoader

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# 모델 글로벌 로드
loader = ModelLoader()
model_path = str(_ROOT / "project" / "ai" / "models" / "personal_color.pt")
loader.load(model_path)


# ========================================================================
# 이미지 생성 함수
# ========================================================================
def create_dummy_image(tone="warm_light", width=640, height=480):
    """단색 피부톤 이미지"""
    skin_colors = {
        "warm_light":   (180, 160, 140),
        "warm_deep":    (120, 100, 80),
        "cool_light":   (160, 140, 140),
        "cool_deep":    (100, 80, 100),
    }
    color = skin_colors.get(tone, (160, 140, 120))
    img = np.full((height, width, 3), color, dtype=np.uint8)
    cv2.circle(img, (width // 2, height // 2), 150, color, -1)
    noise = np.random.randint(-30, 30, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def create_pil_image(tone="warm_light", width=640, height=480):
    """PIL로 얼굴 형태"""
    skin_colors_rgb = {
        "warm_light":   (217, 180, 105),
        "warm_deep":    (139, 110, 65),
        "cool_light":   (198, 166, 143),
        "cool_deep":    (138, 110, 125),
    }
    color = skin_colors_rgb.get(tone, (200, 150, 100))
    img = Image.new('RGB', (width, height), (220, 220, 220))
    draw = ImageDraw.Draw(img)
    center_x, center_y, radius = width // 2, height // 2, 150
    bbox = [center_x - radius, center_y - radius, center_x + radius, center_y + radius]
    draw.ellipse(bbox, fill=color)
    eye_y = center_y - 60
    draw.ellipse([center_x - 80, eye_y - 20, center_x - 50, eye_y + 20], fill=(50, 50, 50))
    draw.ellipse([center_x + 50, eye_y - 20, center_x + 80, eye_y + 20], fill=(50, 50, 50))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ========================================================================
# Flask 라우트
# ========================================================================
@app.route('/')
def index():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>퍼스널 컬러 추론 테스트</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Noto Sans KR', sans-serif; background: #1A1512; color: #F5EFE6; padding: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            h1 { text-align: center; color: #C9A96E; margin-bottom: 30px; font-size: 32px; }
            .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }
            .card { background: #231E1A; padding: 20px; border-radius: 12px; border: 1px solid #4A4030; }
            .card h2 { color: #C9A96E; margin-bottom: 15px; font-size: 18px; }
            select, button { background: #C9A96E; color: #1A1512; border: none; padding: 10px 16px; 
                border-radius: 6px; font-size: 14px; font-weight: bold; cursor: pointer; margin: 5px 0; width: 100%; }
            button:hover { background: #D4B47E; }
            select { appearance: none; padding-right: 30px; }
            
            .result { margin-top: 20px; padding: 15px; background: #0D0B09; border-radius: 8px; 
                border-left: 3px solid #C9A96E; }
            .result.success { border-left-color: #7EC8A4; }
            .result.error { border-left-color: #E86C6C; color: #FF9999; }
            
            .top3 { margin-top: 10px; }
            .top3-item { padding: 8px; background: #2C2520; margin: 5px 0; border-radius: 6px; }
            .palette { display: flex; gap: 8px; margin-top: 10px; }
            .palette-swatch { width: 40px; height: 40px; border-radius: 6px; border: 1px solid #555; }
            
            .preview { margin-top: 15px; }
            .preview img { width: 100%; border-radius: 8px; border: 1px solid #4A4030; }
            
            .stats { color: #999; font-size: 13px; margin-top: 10px; }
            @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>✨ 퍼스널 컬러 추론 테스트</h1>
            
            <div class="grid">
                <!-- 단색 배경 모드 -->
                <div class="card">
                    <h2>1️⃣ 단색 배경</h2>
                    <p style="color: #999; font-size: 13px; margin-bottom: 10px;">피부톤의 색상으로 이미지 생성</p>
                    <select id="tone-dummy">
                        <option value="warm_light">따뜻한 밝은 톤 (봄)</option>
                        <option value="warm_deep">따뜻한 어두운 톤 (가을)</option>
                        <option value="cool_light">차가운 밝은 톤 (여름)</option>
                        <option value="cool_deep">차가운 어두운 톤 (겨울)</option>
                    </select>
                    <button onclick="testDummy()">추론하기</button>
                    <div id="result-dummy" class="result" style="display:none;"></div>
                    <div class="preview"><img id="preview-dummy" style="display:none;"></div>
                </div>
                
                <!-- PIL 얼굴 모드 -->
                <div class="card">
                    <h2>2️⃣ PIL 얼굴</h2>
                    <p style="color: #999; font-size: 13px; margin-bottom: 10px;">눈·입 등이 있는 얼굴 형태</p>
                    <select id="tone-pil">
                        <option value="warm_light">따뜻한 밝은 톤</option>
                        <option value="warm_deep">따뜻한 어두운 톤</option>
                        <option value="cool_light">차가운 밝은 톤</option>
                        <option value="cool_deep">차가운 어두운 톤</option>
                    </select>
                    <button onclick="testPIL()">추론하기</button>
                    <div id="result-pil" class="result" style="display:none;"></div>
                    <div class="preview"><img id="preview-pil" style="display:none;"></div>
                </div>
                
                <!-- 일괄 테스트 -->
                <div class="card">
                    <h2>3️⃣ 일괄 테스트</h2>
                    <p style="color: #999; font-size: 13px; margin-bottom: 10px;">모든 피부톤 조합 테스트</p>
                    <button onclick="testAll()" style="background: #7EC8A4; color: white;">🚀 모든 조합 테스트</button>
                    <div id="result-all" class="result" style="display:none;"></div>
                </div>
                
                <!-- 파일 업로드 -->
                <div class="card">
                    <h2>4️⃣ 얼굴 이미지 업로드</h2>
                    <p style="color: #999; font-size: 13px; margin-bottom: 10px;">실제 얼굴 사진으로 테스트</p>
                    <input type="file" id="file-input" accept="image/*" style="padding: 8px; cursor: pointer;">
                    <button onclick="testUpload()">업로드 & 추론</button>
                    <div id="result-file" class="result" style="display:none;"></div>
                    <div class="preview"><img id="preview-file" style="display:none;"></div>
                </div>
            </div>
        </div>
        
        <script>
            const API_BASE = '/api';
            
            function showResult(elemId, data, imageSrc=null) {
                const elem = document.getElementById(elemId);
                if (data.success) {
                    elem.className = 'result success';
                    elem.innerHTML = \`
                        <strong style="color: #7EC8A4;">✓ \${data.label_ko}</strong> 
                        (신뢰도: \${(data.confidence * 100).toFixed(1)}%)<br>
                        <small>\${data.label_en} | 추론: \${data.inference_ms.toFixed(2)}ms</small>
                        <div class="top3">
                            <strong>TOP 3:</strong>
                            \${data.top3.map((x, i) => \`
                                <div class="top3-item">\${i+1}. \${x.label_ko} <span style="float:right;">\${(x.confidence*100).toFixed(1)}%</span></div>
                            \`).join('')}
                        </div>
                        <div class="palette">
                            \${data.recommended_colors.map(c => \`<div class="palette-swatch" style="background: \${c};"></div>\`).join('')}
                        </div>
                    \`;
                } else {
                    elem.className = 'result error';
                    elem.innerHTML = \`<strong>✗ 실패:</strong> \${data.message}\`;
                }
                elem.style.display = 'block';
                if (imageSrc) document.getElementById(imageSrc).src = data.image_b64;
            }
            
            async function testDummy() {
                const tone = document.getElementById('tone-dummy').value;
                const res = await fetch(\`\${API_BASE}/infer-dummy\`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({tone})
                });
                const data = await res.json();
                showResult('result-dummy', data, 'preview-dummy');
            }
            
            async function testPIL() {
                const tone = document.getElementById('tone-pil').value;
                const res = await fetch(\`\${API_BASE}/infer-pil\`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({tone})
                });
                const data = await res.json();
                showResult('result-pil', data, 'preview-pil');
            }
            
            async function testAll() {
                const elem = document.getElementById('result-all');
                elem.innerHTML = '⏳ 테스트 진행 중...';
                elem.style.display = 'block';
                const res = await fetch(\`\${API_BASE}/infer-all\`);
                const data = await res.json();
                elem.innerHTML = \`
                    <strong>✓ 모든 테스트 완료</strong><br>
                    <pre style="margin-top: 10px; font-size: 12px; overflow-x: auto;">
                    \${JSON.stringify(data, null, 2)}
                    </pre>
                \`;
            }
            
            async function testUpload() {
                const file = document.getElementById('file-input').files[0];
                if (!file) { alert('파일을 선택하세요'); return; }
                const formData = new FormData();
                formData.append('file', file);
                const res = await fetch(\`\${API_BASE}/infer-file\`, {method: 'POST', body: formData});
                const data = await res.json();
                showResult('result-file', data, 'preview-file');
            }
        </script>
    </body>
    </html>
    """
    return render_string(html)


@app.route('/api/infer-dummy', methods=['POST'])
def infer_dummy():
    tone = request.json.get('tone', 'warm_light')
    img = create_dummy_image(tone)
    result = run_inference(img)
    result['image_b64'] = get_image_b64(img)
    return jsonify(result)


@app.route('/api/infer-pil', methods=['POST'])
def infer_pil():
    tone = request.json.get('tone', 'warm_light')
    img = create_pil_image(tone)
    result = run_inference(img)
    result['image_b64'] = get_image_b64(img)
    return jsonify(result)


@app.route('/api/infer-file', methods=['POST'])
def infer_file():
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "파일 없음"})
    file = request.files['file']
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"success": False, "message": "이미지 디코딩 실패"})
    result = run_inference(img)
    result['image_b64'] = get_image_b64(img)
    return jsonify(result)


@app.route('/api/infer-all', methods=['GET'])
def infer_all():
    results = {}
    for tone in ['warm_light', 'warm_deep', 'cool_light', 'cool_deep']:
        img = create_dummy_image(tone)
        result = run_inference(img)
        results[tone] = {
            'label_ko': result['label_ko'],
            'confidence': result['confidence']
        }
    return jsonify(results)


def get_image_b64(img):
    _, buf = cv2.imencode('.jpg', img)
    import base64
    return f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌐 웹 기반 추론 테스트 도구")
    print("="*60)
    print("📍 브라우저에서 접속: http://localhost:5000")
    print("   (Ctrl+C로 종료)\n")
    app.run(debug=True, port=5000, use_reloader=False)
