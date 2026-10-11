// node frontend/tests/passes.test.js
const fs=require('fs'),assert=require('assert');
const h=fs.readFileSync(__dirname+'/../index.html','utf8');
const code=h.split('/*PASS*/')[1].split('/*ENDPASS*/')[0];
const {detectPasses,buildMetrics,PC}=new Function(code+';return{detectPasses,buildMetrics,PC}')();
const pl=(id,team,x,y=500)=>({player_id:id,team_id:team,predicted:false,bbox:[x-15,y-100,x+15,y],ground:[x,y],det_conf:.9});
const noPitch=()=>null;
// ball held by `holder` x-pos for frames; builds a timeline
function play(steps){const frames=[];let t=0;for(const s of steps){for(let i=0;i<s.n;i++){
  frames.push({t_ms:t,players:s.players,ball:s.ball===null?null:{image:s.ball(i/s.n),conf:s.interp?.2:.8,interp:!!s.interp}});t+=40}}return frames}
const A1=pl('P0001','TEAM_A',200),A2=pl('P0002','TEAM_A',700),B1=pl('P0003','TEAM_B',700);
const at=(x,y=495)=>()=>[x,y], lerp=(x0,x1)=>f=>[x0+(x1-x0)*f,495];
const base=[A1,A2,B1];

// 1. completed pass A1 -> A2 (ball travels 500px ~ 5 player heights)
let r=detectPasses(play([{n:8,players:[A1,A2],ball:at(205)},{n:15,players:[A1,A2],ball:lerp(215,690)},{n:8,players:[A1,A2],ball:at(695)}]),noPitch,PC);
assert.equal(r.passes.length,1);assert.equal(r.passes[0].outcome,'completed');assert.equal(r.passes[0].from,'P0001');assert.equal(r.passes[0].to,'P0002');
let m=buildMetrics(r,null);assert.equal(m.pass_accuracy.TEAM_A,100);assert.equal(m.pass_accuracy.TEAM_B,null);assert.equal(m.progressive_passes,undefined);
// 2. pass lost to opponent
r=detectPasses(play([{n:8,players:[A1,B1],ball:at(205)},{n:15,players:[A1,B1],ball:lerp(215,690)},{n:8,players:[A1,B1],ball:at(695)}]),noPitch,PC);
assert.equal(r.passes.length,1);assert.equal(r.passes[0].outcome,'failed');assert.equal(r.recov.TEAM_B,1);
// 3. no ball detections => nothing fabricated
r=detectPasses(play([{n:40,players:base,ball:null}]),noPitch,PC);
assert.equal(r.passes.length,0);assert.equal(buildMetrics(r,null).possession,undefined);assert.equal(buildMetrics(r,null).pass_accuracy,undefined);
// 4. two players side by side trading the ball (tiny displacement) is not a pass
const C=pl('P0004','TEAM_A',230);
r=detectPasses(play([{n:8,players:[A1,C],ball:at(205)},{n:8,players:[A1,C],ball:at(228)}]),noPitch,PC);
assert.equal(r.passes.length,0);
// 5. contested ball between opponents is not given to anyone
const B2=pl('P0005','TEAM_B',215);
r=detectPasses(play([{n:20,players:[A1,B2],ball:at(207)}]),noPitch,PC);
assert.equal(r.segs.length,0);
// 6. long gap => uncertain, excluded from accuracy
r=detectPasses(play([{n:8,players:[A1,A2],ball:at(205)},{n:95,players:[A1,A2],ball:lerp(215,690)},{n:8,players:[A1,A2],ball:at(695)}]),noPitch,PC);
assert.equal(r.passes[0].outcome,'uncertain');assert.equal(buildMetrics(r,null).pass_accuracy,undefined);
// 7. progressive passes need pitch coords + attack direction (A1 -> A2 forward, A2 -> A1 backward)
const pp=(g)=>g[0]<400?[.25,.5]:[.7,.5];
r=detectPasses(play([{n:8,players:[A1,A2],ball:at(205)},{n:15,players:[A1,A2],ball:lerp(215,690)},{n:8,players:[A1,A2],ball:at(695)},
  {n:15,players:[A1,A2],ball:lerp(690,215)},{n:8,players:[A1,A2],ball:at(205)}]),(g,t)=>pp(g),PC);
assert.equal(r.passes.length,2);
assert.equal(buildMetrics(r,true).progressive_passes.TEAM_A,1);   // attacking right: only the forward pass counts
assert.equal(buildMetrics(r,false).progressive_passes.TEAM_A,1);  // attacking left: only the backward-in-x pass counts
assert.equal(buildMetrics(r,null).progressive_passes,undefined);  // direction unknown => unavailable
// 8. pressure: opponent near the carrier is counted for the pressing team only
const B3=pl('P0006','TEAM_B',350);
r=detectPasses(play([{n:20,players:[A1,B3],ball:at(205)}]),noPitch,PC);
m=buildMetrics(r,null);assert.equal(m.pressure_index.TEAM_B,1);assert.equal(m.pressure_index.TEAM_A,null);
r=detectPasses(play([{n:40,players:[A1,B3],ball:null}]),noPitch,PC);assert.equal(buildMetrics(r,null).pressure_index,undefined);
// 9. a pass whose end ball is mostly interpolated is only "uncertain", never "completed"/"failed"
r=detectPasses(play([{n:8,players:[A1,A2],ball:at(205)},{n:15,players:[A1,A2],ball:lerp(215,690)},{n:8,players:[A1,A2],ball:at(695),interp:true}]),noPitch,PC);
assert.equal(r.passes.length,1);assert.equal(r.passes[0].outcome,'uncertain');assert.equal(buildMetrics(r,null).pass_accuracy,undefined);
// 10. a fast ball flying past a bystander is not a touch (no phantom interception)
const BY=pl('P0009','TEAM_B',450);
r=detectPasses(play([{n:8,players:[A1,A2,BY],ball:at(205)},{n:15,players:[A1,A2,BY],ball:lerp(215,690)},{n:8,players:[A1,A2,BY],ball:at(695)}]),noPitch,PC);
assert.equal(r.passes.length,1);assert.equal(r.passes[0].outcome,'completed');assert.equal(r.recov.TEAM_B,0);
// 11. a camera pan must not make a resting ball look fast
const panned=play([{n:8,players:[A1],ball:at(205)}]).map((f,i)=>({...f,cam:[1,0,i*40,0,1,0,0,0,1]}));
panned.forEach((f,i)=>{f.players=[pl('P0001','TEAM_A',200-i*40)];f.ball.image=[205-i*40,495]});   // everything shifts left by 40 px/frame
r=detectPasses(panned,noPitch,PC);assert.equal(r.segs.length,1);
console.log('all pass tests passed');
