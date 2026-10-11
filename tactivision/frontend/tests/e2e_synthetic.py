import json, math, random
from playwright.sync_api import sync_playwright
random.seed(1)
Hp=[760,60,260, 0,420,100, 0,-.15,1]
def proj(x,y,dx=0):
    w=Hp[6]*x+Hp[7]*y+Hp[8]; return [(Hp[0]*x+Hp[1]*y+Hp[2])/w+dx,(Hp[3]*x+Hp[4]*y+Hp[5])/w,w]
base=[]
for i in range(11): base.append(("TEAM_A",f"P{i+1:04d}",.15+.07*(i%5)+random.random()*.05,.12+.075*(i//5*3+i%3)+random.random()*.05))
for i in range(11): base.append(("TEAM_B",f"P{i+12:04d}",.55+.07*(i%5)+random.random()*.05,.12+.075*(i//5*3+i%3)+random.random()*.05))
pos=lambda i,t:[base[i][2]+.02*math.sin(t/2+i),base[i][3]+.02*math.cos(t/3+i)]
holders=[0,3,6,4,9,2]; frames=[]; truth={}
dxf=lambda t:-15*t if t<9 else 0
for k in range(0,300):
    t=k*0.04; seg=0 if t<9 else 1; dx=dxf(t)
    cam=[1,0,-dx,0,1,0,0,0,1] if seg==0 else [1,0,0,0,1,0,0,0,1]
    players=[]; sg=int(t//1.6); ph=t-sg*1.6; a=holders[sg%6]; b=holders[(sg+1)%6]
    for i,(team,pid,_,_) in enumerate(base):
        px,py=pos(i,t); gx,gy,w=proj(px,py,dx); h=55/w
        players.append({"player_id":pid,"team_id":team,"status":"visible","bbox":[gx-9,gy-h,gx+9,gy],"ground":[gx,gy],"predicted":False,"det_conf":.9,"identity_conf":.9,"team_conf":.8})
        truth[(k,pid)]=(px,py)
    pa=pos(a,t);pb=pos(b,t); bp=pa if ph<.8 else [pa[0]+(pb[0]-pa[0])*(ph-.8)/.8,pa[1]+(pb[1]-pa[1])*(ph-.8)/.8]
    bx,by,_=proj(*bp,dx)
    frames.append({"t_ms":k*40,"players":players,"ball":{"image":[bx,by-3],"conf":.7},"calibration_conf":0,"camera_cut":k==225,"seg":seg,"cam":cam})
state={"meta":{"fps":25},"teams":{"TEAM_A":{"name":None},"TEAM_B":{"name":None}},"frames":frames,"passes":[],"audit":[],"metrics":{},"heatmaps":{}}
LM=[[0,0],[1,0],[1,1],[0,1],[.5,0],[.5,1],[.5,.5],[.1571,.2035],[.8429,.7965],[.1571,.7965]]
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1500,"height":1000}); errs=[]
    pg.on("pageerror",lambda e:errs.append(str(e)))
    pg.goto("file:///home/claude/tactivision/frontend/index.html")
    pg.evaluate("""()=>{const v=$('vid');let ct=0;Object.defineProperty(v,'videoWidth',{get:()=>1280});Object.defineProperty(v,'videoHeight',{get:()=>720});
      Object.defineProperty(v,'duration',{get:()=>12});Object.defineProperty(v,'currentTime',{get:()=>ct,set:x=>{ct=x}});}""")
    pg.evaluate("s=>loadState(s)",state)
    # calibrate at t=6 s, mid-pan, using that frame's true projection
    pg.evaluate("()=>{$('vid').currentTime=6.0;$('cal').click()}")
    dx=dxf(6.0)
    pg.evaluate("a=>{const [lm,H,dx]=a;PAIRS=lm.map((l,i)=>{const w=H[6]*l[0]+H[7]*l[1]+H[8];return{img:[(H[0]*l[0]+H[1]*l[1]+H[2])/w+dx+0.4,(H[3]*l[0]+H[4]*l[1]+H[5])/w-0.4],pt:l,lm:i}})}",[LM,Hp,dx])
    pg.evaluate("()=>$('calok').click()"); pg.wait_for_timeout(300)
    print("calres:",pg.inner_text("#calres"))
    pg.evaluate("()=>{$('calx').click();$('vid').currentTime=3.0}"); pg.select_option("#atk","r"); pg.wait_for_timeout(500)
    # accuracy of pitch mapping across the whole pan, shot 1 only
    res=pg.evaluate("()=>S.frames.filter(f=>f.seg===0).flatMap(f=>f.players.map(p=>({k:f.t_ms/40,id:p.player_id,q:pitchOf(p.ground,f)})))")
    errs_m=[];miss=0
    for r in res:
        if r["q"] is None: miss+=1; continue
        tx,ty=truth[(int(r["k"]),r["id"])]; errs_m.append(math.hypot((r["q"][0]-tx)*105,(r["q"][1]-ty)*68))
    print("shot1 obs",len(res),"unmapped",miss,"mean err m %.3f max %.3f"%(sum(errs_m)/len(errs_m),max(errs_m)))
    res2=pg.evaluate("()=>S.frames.filter(f=>f.seg===1).flatMap(f=>f.players.map(p=>pitchOf(p.ground,f)))")
    print("shot2 (uncalibrated) mapped:",sum(1 for r in res2 if r),"of",len(res2))
    print("summary:",pg.inner_text("#msg")); print("metrics:",pg.evaluate("()=>METRICS")); print("passes",pg.evaluate("()=>PASSES.length"))
    pg.screenshot(path="shot2.png",full_page=True); print("errors:",errs); b.close()
