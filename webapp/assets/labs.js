/* Original interactive teaching models. These do not run Ray or predict performance. */
(() => {
  const id = document.body.dataset.lab;
  if (!id) return;
  const lab = document.getElementById('lab'), root = document.getElementById('lab-root');
  const meta = {
    map:['点开一层，看看它负责什么','从编排走到计算；同一个名字里的 actor 可能代表不同层次。'],
    refs:['提交不等于完成','用四个步骤观察 ObjectRef 与具体结果；时间推进是教学模拟。'],
    actors:['同一 Actor 与两个 Actor','切换提交目标，比较默认同步 Actor 的排队方式。'],
    placement:['四张卡，在哪里等待','改变资源与部署方式，查看 _get_placement_group_layout 的普通分支。'],
    data:['两层引用，两个取值位置','切换 driver 和 DP worker，检查此时持有的是引用还是数据。'],
    timeline:['沿 train.py 单步前进','固定无 critic、无 offload 的默认生成路线，观察每一步的完成边界。']
  };
  lab.hidden = false;
  lab.querySelector('.lab-title').textContent = '交互实验 · ' + meta[id][0];
  lab.querySelector('.lab-intro').textContent = meta[id][1];
  const banner = document.getElementById('lab-banner'); banner.className = 'lab-banner';
  const link = document.createElement('a'); link.href = '#lab'; link.textContent = `本章实验：${meta[id][0]} ↓`; banner.append(link);
  const note = document.createElement('p'); note.className='lab-note'; note.textContent='教学简化：在浏览器中演示关系与顺序，不启动 Ray，也不代表 GPU 实测。';
  const flow = labels => '<div class="flow">' + labels.map((x,i)=>`<div class="flow-node" data-node="${i}">${x}</div>`).join('') + '</div>';
  const output = () => root.querySelector('.lab-output');
  const activate = index => root.querySelectorAll('.flow-node').forEach((n,i) => n.classList.toggle('active',i===index));
  function stepper(steps, labels) {
    let step = 0;
    root.innerHTML = flow(labels) + '<div class="lab-controls"><button class="secondary" data-action="reset">重置</button><button class="secondary" data-action="prev">上一步</button><button data-action="next">下一步</button><span class="step-count"></span></div><div class="lab-output" role="status" aria-live="polite"></div>';
    const render = () => {
      output().innerHTML = steps[step][1]; activate(steps[step][0]);
      root.querySelector('.step-count').textContent = `${step+1} / ${steps.length}`;
      root.querySelector('[data-action=prev]').disabled = step===0;
      root.querySelector('[data-action=next]').disabled = step===steps.length-1;
    };
    root.querySelector('[data-action=reset]').onclick=()=>{step=0;render();};
    root.querySelector('[data-action=prev]').onclick=()=>{step=Math.max(0,step-1);render();};
    root.querySelector('[data-action=next]').onclick=()=>{step=Math.min(steps.length-1,step+1);render();};
    render();
  }
  if (id==='map') {
    const entries = [
      ['Driver','train.py','按顺序申请资源、创建模型、提交生成和训练，并在 get 处等待。RayTrainGroup 也位于这个进程。'],
      ['Ray','资源与远程对象','安排 Actor 所在节点与逻辑资源，传递远程调用及其结果引用。它不计算 GRPO loss，也不替代 NCCL all-reduce。'],
      ['Megatron','训练 worker','保存训练模型和 optimizer，按 TP / PP / DP 等配置执行数值训练。'],
      ['SGLang','推理服务','接收生成请求，执行模型推理，维护推理侧的模型权重和 KV cache。']
    ];
    root.innerHTML='<div class="lab-controls">'+entries.map((x,i)=>`<button data-layer="${i}" class="secondary">${x[0]}</button>`).join('')+'</div><div class="lab-output" role="status"></div>';
    const show=i=>{output().innerHTML=`<strong>${entries[i][0]} · ${entries[i][1]}</strong><br>${entries[i][2]}`;root.querySelectorAll('button').forEach((b,j)=>b.setAttribute('aria-pressed',String(i===j)));};
    root.querySelectorAll('button').forEach((b,i)=>b.onclick=()=>show(i));show(0);
  }
  if (id==='refs') stepper([
    [0,'<strong>还没提交</strong><br>driver 持有输入 3。worker 还没有这项任务。'],
    [0,'<strong>square.remote(3) 已返回 ObjectRef</strong><br>driver 可以继续提交其他工作。该任务可能仍在等待资源。'],
    [1,'<strong>worker 完成计算，结果是 9</strong><br>ObjectRef 现在可以取得结果。driver 手里的 ref 仍是引用，不会自己变成整数。'],
    [2,'<strong>ray.get(ref) 返回 9</strong><br>driver 获取具体值。若 worker 失败，这个位置也可能抛异常。']
  ],['<b>driver</b>提交任务','<b>worker</b>执行 square','<b>driver</b>获取结果']);
  if (id==='actors') {
    root.innerHTML='<div class="lab-controls"><label>提交目标 <select id="actor-mode"><option value="one">一个默认同步 Actor</option><option value="two">两个独立 Actor</option></select></label></div><div id="actor-lanes"></div><div class="lab-output" role="status"></div>';
    const render=()=>{const one=root.querySelector('select').value==='one';root.querySelector('#actor-lanes').innerHTML=one?'<div class="lane"><span>Actor A</span><div class="bars"><span class="bar">add(2)</span><span class="bar">add(3)</span></div></div>':'<div class="lane"><span>Actor A</span><div class="bars"><span class="bar">add(2)</span><span class="bar wait">空闲</span></div></div><div class="lane"><span>Actor B</span><div class="bars"><span class="bar">add(3)</span><span class="bar wait">空闲</span></div></div>';output().innerHTML=one?'<strong>结果依次为 2、5</strong><br>同一调用方的两次方法调用共享一个 self.value，按顺序处理。':'<strong>结果分别为 2、3</strong><br>两个 Actor 各有自己的 self.value。资源足够时可并行，但状态不会自动合并。';};root.querySelector('select').onchange=render;render();
  }
  if (id==='placement') {
    root.innerHTML='<div class="lab-controls"><label>训练 GPU <select id="train-gpus"><option>2</option><option>4</option></select></label><label>rollout GPU <select id="rollout-gpus"><option>2</option><option>4</option></select></label><label>可用 GPU <select id="available-gpus"><option>3</option><option>4</option><option>6</option><option>8</option></select></label><label>布局 <select id="placement-mode"><option value="separate">分卡</option><option value="colocate">colocate</option></select></label></div><div class="gpu-grid"></div><div class="lab-output" role="status"></div>';
    const render=()=>{let t=+root.querySelector('#train-gpus').value,r=+root.querySelector('#rollout-gpus').value,a=+root.querySelector('#available-gpus').value,c=root.querySelector('#placement-mode').value==='colocate',n=c?Math.max(t,r):t+r,off=c?0:t;root.querySelector('.gpu-grid').innerHTML=Array.from({length:n},(_,i)=>{const tr=i<t,rr=i>=off&&i<off+r,cl=tr&&rr?'shared':tr?'train':'rollout';return `<div class="gpu ${cl}">逻辑槽 ${i}<br>${tr&&rr?'训练 + rollout':tr?'训练':'rollout'}</div>`;}).join('');output().innerHTML=`申请 <strong>${n}</strong> 个 1 GPU + 1 CPU bundles；rollout offset = <strong>${off}</strong>。<br>`+(a<n?`<strong>等待 PG</strong>：只有 ${a} 张可用 GPU，不足以满足整组。尚未进入创建业务模型阶段。`:'<strong>GPU 数量满足必要条件</strong>：还要检查 CPU、节点内 bundle 可放置性与实际调度。即使 PG ready，模型也还没初始化。');};root.querySelectorAll('select').forEach(s=>s.onchange=render);render();
  }
  if (id==='data') {
    root.innerHTML=flow(['<b>RolloutManager</b>DP0: [0,2]<br>DP1: [1,3]','<b>driver</b>list[Box]','<b>训练 worker</b>按 dp_rank 取数'])+'<div class="lab-controls"><label>观察位置 <select id="data-location"><option value="outer">driver：generate 刚提交</option><option value="driver">driver：外层 get 完成</option><option value="dp0">worker：DP rank 0</option><option value="dp1">worker：DP rank 1</option></select></label></div><div class="lab-output" role="status"></div>';
    const states={outer:[1,'<strong>ObjectRef（外层）</strong><br>代表 generate 方法结果；尚未得到分片引用清单。'],driver:[1,'<strong>[Box(ref0), Box(ref1)]</strong><br>外层 get 完成，得到包装列表。Box 内仍是 ObjectRef，driver 没有显式 get 这两个分片。'],dp0:[2,'<strong>ray.get(boxes[0].inner) → [0,2]</strong><br>DP rank 0 在训练进程里获取本地需要的数据分片。'],dp1:[2,'<strong>ray.get(boxes[1].inner) → [1,3]</strong><br>DP rank 1 获取另一个分片。TP 成员不应该用 global rank 直接索引这个 DP 列表。']};const render=()=>{const s=states[root.querySelector('select').value];activate(s[0]);output().innerHTML=s[1];};root.querySelector('select').onchange=render;render();
  }
  if (id==='timeline') stepper([
    [0,'<strong>0 · 初始同步</strong><br>创建模型后先 update_weights；确保第一次生成使用训练侧加载的策略。'],
    [1,'<strong>1 · 提交并等待生成</strong><br>driver 在 ray.get(manager.generate.remote(k)) 等待；默认下一批还没提交。'],
    [0,'<strong>2 · 拿到 list[Box]</strong><br>生成完成、数据已打包。driver 持有每个 DP 分片的引用包装。'],
    [2,'<strong>3 · 提交所有训练 worker</strong><br>async_train 返回多个 ObjectRef；随后 driver 的 ray.get 等全部 worker 完成。'],
    [2,'<strong>4 · 后端训练结束</strong><br>本轮可包含多个 optimizer step。Ray 等远程方法结束，数值计算由 Megatron 执行。'],
    [0,'<strong>5 · 保存与清理</strong><br>满足周期条件时保存；异步 checkpoint 是否持久完成仍需看 force_sync。清理对应 rollout 的专用数据。'],
    [1,'<strong>6 · 更新推理权重</strong><br>普通 update_weights 内部等待远程调用，控制消息与参数张量走各自路径。'],
    [0,'<strong>7 · 下一轮 / 评估</strong><br>经过可选 evaluation，循环才进入下一次默认 generate。不能仅靠 async_train 的名字推断重叠。']
  ],['<b>driver</b>编排 / 等待','<b>rollout</b>生成 / 接收权重','<b>训练 workers</b>取数 / 训练 / 发送权重']);
  root.append(note);
})();
