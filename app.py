import os, uuid
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import onnxruntime as ort
from flask import Flask, request, render_template_string

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

NAMES = ["aircraft", "fish", "other", "shipwreck"]
IMG_SIZE = 640
CONF = 0.50
IOU = 0.45

session = ort.InferenceSession(
    os.getenv("MODEL_PATH", "best.onnx"),
    providers=["CPUExecutionProvider"]
)
INPUT_NAME = session.get_inputs()[0].name

HTML = """<!doctype html>
<html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sonar AI Detector</title>
<style>
body{margin:0;background:#071827;color:#fff;font-family:Arial}.wrap{max-width:900px;margin:auto;padding:25px 16px}
.card{background:#10283c;border-radius:18px;padding:24px}.sub{opacity:.8}
form{margin-top:20px;border:2px dashed #55738a;padding:25px;text-align:center;border-radius:14px}
button{margin-top:14px;padding:12px 24px;border:0;border-radius:9px;font-weight:bold}
img{max-width:100%;border-radius:10px;margin-top:18px}.result{margin-top:18px;padding:16px;background:#0b1e2e;border-radius:12px}
.item{padding:10px 0;border-bottom:1px solid #ffffff22}.badge{font-weight:bold}
.err{margin-top:18px;padding:12px;background:#4b1f25;border-radius:9px}
</style></head><body><div class="wrap"><div class="card">
<h1>AI Underwater Sonar Detection</h1><div class="sub">Aircraft • Fish • Other • Shipwreck</div>
<form method="post" enctype="multipart/form-data"><input type="file" name="image" accept="image/*" required>
<br><button>Detect Objects</button></form>
{% if image %}<img src="data:image/jpeg;base64,{{image}}">{% endif %}
{% if results %}<div class="result"><h2>Results</h2>
{% for r in results %}<div class="item"><span class="badge">{{r.label}}</span> — {{r.conf}}%</div>{% endfor %}
</div>{% endif %}{% if error %}<div class="err">{{error}}</div>{% endif %}
</div></div></body></html>"""

def letterbox(im):
    im = im.convert("RGB")
    w,h = im.size
    scale = min(IMG_SIZE/w, IMG_SIZE/h)
    nw,nh = int(round(w*scale)), int(round(h*scale))
    resized = im.resize((nw,nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB",(IMG_SIZE,IMG_SIZE),(114,114,114))
    px=(IMG_SIZE-nw)//2; py=(IMG_SIZE-nh)//2
    canvas.paste(resized,(px,py))
    arr=np.asarray(canvas,dtype=np.float32)/255.0
    return arr.transpose(2,0,1)[None], scale, px, py

def iou_one(box, boxes):
    x1=np.maximum(box[0],boxes[:,0]); y1=np.maximum(box[1],boxes[:,1])
    x2=np.minimum(box[2],boxes[:,2]); y2=np.minimum(box[3],boxes[:,3])
    inter=np.maximum(0,x2-x1)*np.maximum(0,y2-y1)
    area=(box[2]-box[0])*(box[3]-box[1])
    areas=(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    return inter/(area+areas-inter+1e-6)

def nms(boxes,scores,threshold):
    order=np.argsort(scores)[::-1]; keep=[]
    while len(order):
        i=order[0]; keep.append(i)
        if len(order)==1: break
        overlaps=iou_one(boxes[i],boxes[order[1:]])
        order=order[1:][overlaps<=threshold]
    return keep

def detect(im):
    x,scale,pad_x,pad_y=letterbox(im)
    out=session.run(None,{INPUT_NAME:x})[0]
    # YOLO export is normally [1, 4+classes, 8400].
    pred=out[0]
    if pred.shape[0] < pred.shape[1]:
        pred=pred.T
    boxes_xywh=pred[:,:4]
    class_scores=pred[:,4:]
    cls=np.argmax(class_scores,axis=1)
    scores=class_scores[np.arange(len(cls)),cls]
    mask=scores>=CONF
    boxes_xywh=boxes_xywh[mask]; scores=scores[mask]; cls=cls[mask]
    if not len(scores): return []

    xyxy=np.empty_like(boxes_xywh)
    xyxy[:,0]=boxes_xywh[:,0]-boxes_xywh[:,2]/2
    xyxy[:,1]=boxes_xywh[:,1]-boxes_xywh[:,3]/2
    xyxy[:,2]=boxes_xywh[:,0]+boxes_xywh[:,2]/2
    xyxy[:,3]=boxes_xywh[:,1]+boxes_xywh[:,3]/2

    # Class-wise NMS
    keep=[]
    for c in np.unique(cls):
        idx=np.where(cls==c)[0]
        keep.extend(idx[nms(xyxy[idx],scores[idx],IOU)])
    keep=sorted(keep,key=lambda i:scores[i],reverse=True)

    results=[]
    for i in keep:
        b=xyxy[i].copy()
        b[[0,2]]=(b[[0,2]]-pad_x)/scale
        b[[1,3]]=(b[[1,3]]-pad_y)/scale
        b[0]=max(0,min(im.width,b[0])); b[2]=max(0,min(im.width,b[2]))
        b[1]=max(0,min(im.height,b[1])); b[3]=max(0,min(im.height,b[3]))
        results.append({"label":NAMES[int(cls[i])],"confidence":float(scores[i]),"box":b.tolist()})
    return results

def draw(im, results):
    out=im.convert("RGB").copy()
    d=ImageDraw.Draw(out)
    for r in results:
        x1,y1,x2,y2=r["box"]
        d.rectangle((x1,y1,x2,y2),outline="red",width=max(2,int(min(im.size)/200)))
        text=f'{r["label"]} {r["confidence"]*100:.1f}%'
        d.text((x1,max(0,y1-18)),text,fill="white")
    return out

def jpeg_b64(im):
    import io, base64
    b=io.BytesIO(); im.save(b,"JPEG",quality=88)
    return base64.b64encode(b.getvalue()).decode()

@app.route("/",methods=["GET","POST"])
def index():
    results=[]; error=None; preview=None
    if request.method=="POST":
        f=request.files.get("image")
        if not f: error="Please select an image."
        else:
            try:
                im=Image.open(f.stream).convert("RGB")
                results=detect(im)
                preview=jpeg_b64(draw(im,results))
                if not results: error="No confident object detected (threshold 50%)."
            except Exception as e:
                error="Detection error: "+str(e)
    shown=[{"label":r["label"],"conf":f'{r["confidence"]*100:.1f}'} for r in results]
    return render_template_string(HTML,results=shown,error=error,image=preview)

if __name__=="__main__":
    app.run(host="0.0.0.0",port=int(os.getenv("PORT","10000")))
