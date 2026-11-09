"""Pack the scan into a single self-contained WebGL page."""
import base64, json
import numpy as np

import os
OUT = os.environ.get("OUT", "out"); NAME = os.environ.get("NAME", "pear")
with open(f"{OUT}/{NAME}-scan.ply", "rb") as f:
    data = f.read()
i = data.index(b"end_header\n") + len(b"end_header\n")
rec = np.frombuffer(data[i:], dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("i", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
rng = np.random.default_rng(1)
N = min(len(rec), int(os.environ.get("NPTS", 420000)))
rec = rec[rng.choice(len(rec), N, replace=False)]
x, y, z = rec["x"], rec["y"], rec["z"]
lo = np.array([x.min(), y.min(), z.min()]); hi = np.array([x.max(), y.max(), z.max()])
q = lambda v, a, b: np.round((v - a) / (b - a) * 65535).astype("<u2")
xyz = np.stack([q(x, lo[0], hi[0]), q(y, lo[1], hi[1]), q(z, lo[2], hi[2])], 1).ravel()
inten = np.clip(rec["i"] / np.percentile(rec["i"], 99.5), 0, 1)
attr = np.stack([rec["r"], rec["g"], rec["b"], (inten * 255).astype("u1")], 1).ravel().astype("u1")
stats = json.load(open(f"{OUT}/{NAME}-scan-stats.json"))
meta = dict(n=N, lo=lo.tolist(), hi=hi.tolist(), stats=stats)
B = base64.b64encode(xyz.tobytes()).decode(); A = base64.b64encode(attr.tobytes()).decode()

html = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#0f0f10;--ink:#e8e4dc;--dim:#9a958c;--acc:#e3b062}
html,body{margin:0;height:100%;background:var(--bg);color:var(--ink);font:14px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
#c{position:fixed;inset:0;width:100%;height:100%;touch-action:none;cursor:grab}
#c:active{cursor:grabbing}
.panel{position:fixed;left:12px;top:12px;max-width:min(380px,calc(100% - 24px));background:rgba(20,20,22,.82);backdrop-filter:blur(6px);
 border:1px solid #2b2b2e;border-radius:10px;padding:12px 14px}
h1{font-size:15px;margin:0 0 4px;font-weight:600}
p{margin:4px 0 8px;color:var(--dim);font-size:12.5px}
.row{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}
button{background:#232326;color:var(--ink);border:1px solid #3a3a3e;border-radius:6px;padding:5px 9px;font:inherit;font-size:12.5px;cursor:pointer}
button.on{background:var(--acc);color:#1a1408;border-color:var(--acc)}
label{font-size:12.5px;color:var(--dim);display:flex;align-items:center;gap:8px}
input[type=range]{flex:1}
.stats{font:11.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--dim);margin-top:6px;white-space:pre}
.hint{position:fixed;right:12px;bottom:10px;color:var(--dim);font-size:11.5px}
#toggle{display:none}
@media (max-width:600px){.panel{font-size:12px}.stats{display:none}}
</style></head><body>
<canvas id="c"></canvas>
<div class="panel">
 <h1>__TITLE__</h1>
 <p>__BLURB__</p>
 <div class="row" id="modes">
  <button data-m="0" class="on">camera colour</button><button data-m="1">1550 nm intensity</button><button data-m="2">height</button>
 </div>
 <label>height &times;<span id="ev">8</span><input id="ex" type="range" min="1" max="40" value="8"></label>
 <label>point size<input id="ps" type="range" min="1" max="6" value="2" step="0.5"></label>
 <div class="row"><button id="top">top</button><button id="rake">raking</button><button id="spin">spin</button></div>
 <div class="stats" id="st"></div>
</div>
<div class="hint">drag to orbit &middot; scroll / pinch to zoom &middot; shift-drag to pan</div>
<script>
const META = __META__;
function b64(s){const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return u}
const P=new Uint16Array(b64("__XYZ__").buffer), A=b64("__ATTR__");
const cv=document.getElementById('c'), gl=cv.getContext('webgl',{antialias:false});
const S=META.stats;
document.getElementById('st').textContent=
 `points        ${S.points.toLocaleString()} (showing ${META.n.toLocaleString()})\n`+
 `height RMSE   ${S.height_rmse_um.toFixed(1)} µm vs ground truth\n`+
 `blind spots   ${(100*S.shadowed_left).toFixed(2)}% left, ${(100*S.shadowed_right).toFixed(2)}% right, ${(100*S.unseen_merged).toFixed(2)}% both\n`+
 `thickest paint ${S.paint_max_mm.toFixed(2)} mm`;
const vs=`attribute vec3 p;attribute vec4 a;uniform mat4 M;uniform vec3 lo,hi;uniform float ex,ps;uniform int mode;varying vec3 col;
void main(){vec3 w=lo+p/65535.0*(hi-lo);float zr=(w.z-lo.z)/(hi.z-lo.z);w.z=(w.z-lo.z)*ex;
 vec3 c=vec3(a.rgb);float i=a.a;
 if(mode==0)col=c*(0.6+0.55*i);else if(mode==1)col=vec3(i);
 else{float t=clamp(zr,0.,1.);col=mix(mix(vec3(.05,.03,.2),vec3(.8,.25,.35),smoothstep(0.,.55,t)),vec3(1.,.9,.55),smoothstep(.45,1.,t));}
 gl_Position=M*vec4(w,1.);gl_PointSize=ps;}`;
const fs=`precision mediump float;varying vec3 col;void main(){gl_FragColor=vec4(col,1.);}`;
function sh(t,s){const o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);return o}
const pr=gl.createProgram();gl.attachShader(pr,sh(gl.VERTEX_SHADER,vs));gl.attachShader(pr,sh(gl.FRAGMENT_SHADER,fs));gl.linkProgram(pr);gl.useProgram(pr);
function buf(d,loc,n,type,norm){const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,d,gl.STATIC_DRAW);
 const l=gl.getAttribLocation(pr,loc);gl.enableVertexAttribArray(l);gl.vertexAttribPointer(l,n,type,norm,0,0)}
buf(P,'p',3,gl.UNSIGNED_SHORT,false);buf(A,'a',4,gl.UNSIGNED_BYTE,true);
const U=n=>gl.getUniformLocation(pr,n);
gl.uniform3fv(U('lo'),META.lo);gl.uniform3fv(U('hi'),META.hi);
const D0=1.15*Math.max(META.hi[0]-META.lo[0],META.hi[1]-META.lo[1]);
let mode=0,ex=8,ps=2,yaw=-0.5,pitch=0.75,dist=D0,tx=0,ty=0,spin=false;
const cx=(META.lo[0]+META.hi[0])/2, cy=(META.lo[1]+META.hi[1])/2;
function mat(){const w=cv.width,h=cv.height,f=1/Math.tan(0.35),asp=w/h,n=1,fa=1000;
 const Pm=[f/asp,0,0,0, 0,f,0,0, 0,0,(fa+n)/(n-fa),-1, 0,0,2*fa*n/(n-fa),0];
 const cyw=Math.cos(yaw),syw=Math.sin(yaw),cp=Math.cos(pitch),sp=-Math.sin(pitch);
 // view: translate to centre, rotate yaw about z, tilt, push back
 const T=[1,0,0,0,0,1,0,0,0,0,1,0,-cx-tx,-cy-ty,0,1];
 const Rz=[cyw,syw,0,0,-syw,cyw,0,0,0,0,1,0,0,0,0,1];
 const Rx=[1,0,0,0,0,cp,sp,0,0,-sp,cp,0,0,0,0,1];
 const Tz=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,-dist,1];
 return [Tz,Rx,Rz,T].reduce((A,B)=>mul(A,B),Pm)}
function mul(a,b){const o=new Array(16).fill(0);for(let c=0;c<4;c++)for(let r=0;r<4;r++)for(let k=0;k<4;k++)o[c*4+r]+=a[k*4+r]*b[c*4+k];return o}
function draw(){const d=Math.min(devicePixelRatio,2);const w=innerWidth*d|0,h=innerHeight*d|0;if(cv.width!==w||cv.height!==h){cv.width=w;cv.height=h}
 gl.viewport(0,0,w,h);gl.clearColor(.06,.06,.065,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);
 gl.uniformMatrix4fv(U('M'),false,new Float32Array(mat()));gl.uniform1f(U('ex'),ex);gl.uniform1f(U('ps'),ps*d);gl.uniform1i(U('mode'),mode);
 gl.drawArrays(gl.POINTS,0,META.n)}
function loop(){if(spin)yaw+=0.004;draw();requestAnimationFrame(loop)}loop();
document.querySelectorAll('#modes button').forEach(b=>b.onclick=()=>{mode=+b.dataset.m;document.querySelectorAll('#modes button').forEach(x=>x.classList.toggle('on',x===b))});
document.getElementById('ex').oninput=e=>{ex=+e.target.value;document.getElementById('ev').textContent=ex};
document.getElementById('ps').oninput=e=>{ps=+e.target.value};
document.getElementById('top').onclick=()=>{yaw=0;pitch=0.001;dist=D0;tx=ty=0};
document.getElementById('rake').onclick=()=>{yaw=-0.2;pitch=1.25;dist=0.75*D0};
const sb=document.getElementById('spin');sb.onclick=()=>{spin=!spin;sb.classList.toggle('on',spin)};
let ptrs=new Map(),last=null,pinch=null;
cv.addEventListener('pointerdown',e=>{cv.setPointerCapture(e.pointerId);ptrs.set(e.pointerId,[e.clientX,e.clientY]);last=[e.clientX,e.clientY,e.shiftKey]});
cv.addEventListener('pointermove',e=>{if(!ptrs.has(e.pointerId))return;ptrs.set(e.pointerId,[e.clientX,e.clientY]);
 if(ptrs.size===2){const [a,b]=[...ptrs.values()];const d=Math.hypot(a[0]-b[0],a[1]-b[1]);if(pinch)dist=Math.max(0.15*D0,Math.min(4*D0,dist*pinch/d));pinch=d;return}
 if(!last)return;const dx=e.clientX-last[0],dy=e.clientY-last[1];
 if(e.shiftKey){const s=dist/innerHeight;tx-=(dx*Math.cos(yaw)+dy*Math.sin(yaw))*s;ty-=(-dx*Math.sin(yaw)+dy*Math.cos(yaw))*s*-1}
 else{yaw+=dx*0.006;pitch=Math.max(0.001,Math.min(1.5,pitch+dy*0.005))}
 last=[e.clientX,e.clientY]});
const up=e=>{ptrs.delete(e.pointerId);if(ptrs.size<2)pinch=null;if(!ptrs.size)last=null};
cv.addEventListener('pointerup',up);cv.addEventListener('pointercancel',up);
cv.addEventListener('wheel',e=>{e.preventDefault();dist=Math.max(0.15*D0,Math.min(4*D0,dist*Math.exp(e.deltaY*0.001)))},{passive:false});
</script></body></html>"""
html = html.replace("__TITLE__", os.environ.get("TITLE", "Scan")).replace("__BLURB__", os.environ.get("BLURB", ""))
html = html.replace("__META__", json.dumps(meta)).replace("__XYZ__", B).replace("__ATTR__", A)
open(f"{OUT}/{NAME}-scan-viewer.html", "w").write(html)
print(f"{N} points, {len(html) / 1e6:.1f} MB")
