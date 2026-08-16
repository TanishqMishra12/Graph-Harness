import { motion } from 'framer-motion';
import { Network, Zap, CheckCircle2, Terminal, ShieldAlert, Cpu } from 'lucide-react';
import './index.css';

const fadeIn = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.6 }
};

const stagger = {
  animate: {
    transition: {
      staggerChildren: 0.1
    }
  }
};

function App() {
  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Abstract Background Shapes */}
      <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-primary/20 rounded-full blur-[120px] mix-blend-screen pointer-events-none" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-purple-500/20 rounded-full blur-[120px] mix-blend-screen pointer-events-none" />

      <main className="container mx-auto px-6 py-16 relative z-10 max-w-6xl">
        
        {/* Hero Section */}
        <motion.header 
          className="text-center mb-24 mt-12"
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8 }}
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-card text-sm font-medium mb-8">
            <span className="flex h-2 w-2 rounded-full bg-primary animate-pulse" />
            Project Complete — NVIDIA NIM + SGH
          </div>
          <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight mb-6">
            Structured Graph <br />
            <span className="animated-gradient glow-text">Harness (SGH)</span>
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto leading-relaxed">
            From Agent Loops to Structured Graphs. Decoupling the execution scheduler from the LLM context loop to eliminate unbounded retries, enable true concurrency, and create immutable audit trails.
          </p>
        </motion.header>

        {/* Core Achievements / Metrics Grid */}
        <motion.section 
          className="grid md:grid-cols-3 gap-6 mb-24"
          variants={stagger}
          initial="initial"
          whileInView="animate"
          viewport={{ once: true }}
        >
          <motion.div variants={fadeIn} className="glass-card p-8 rounded-2xl relative overflow-hidden group">
            <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
              <Zap size={64} />
            </div>
            <div className="text-primary mb-4"><Zap size={32} /></div>
            <h3 className="text-4xl font-bold mb-2">10.1x</h3>
            <p className="text-lg font-medium text-foreground mb-1">Latency Speedup</p>
            <p className="text-sm text-muted-foreground">Concurrent execution (0.50s) vs Sequential Agent Loops (5.06s)</p>
          </motion.div>

          <motion.div variants={fadeIn} className="glass-card p-8 rounded-2xl relative overflow-hidden group">
            <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
              <Network size={64} />
            </div>
            <div className="text-purple-400 mb-4"><Network size={32} /></div>
            <h3 className="text-4xl font-bold mb-2">|U| &gt; 1</h3>
            <p className="text-lg font-medium text-foreground mb-1">Max Concurrency</p>
            <p className="text-sm text-muted-foreground">Topologically sorts and parallelizes independent DAG nodes natively.</p>
          </motion.div>

          <motion.div variants={fadeIn} className="glass-card p-8 rounded-2xl relative overflow-hidden group">
            <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
              <ShieldAlert size={64} />
            </div>
            <div className="text-rose-400 mb-4"><ShieldAlert size={32} /></div>
            <h3 className="text-4xl font-bold mb-2">3-Tier</h3>
            <p className="text-lg font-medium text-foreground mb-1">Deterministic Recovery</p>
            <p className="text-sm text-muted-foreground">Local retry → Semantic patch → Global request_replan limits failure cascades.</p>
          </motion.div>
        </motion.section>

        {/* Demo Execution Results */}
        <motion.section 
          className="mb-24"
          initial="initial"
          whileInView="animate"
          viewport={{ once: true }}
          variants={fadeIn}
        >
          <h2 className="text-3xl font-bold mb-8 flex items-center gap-3">
            <Cpu className="text-primary" /> Live Demo Execution
          </h2>
          
          <div className="grid lg:grid-cols-5 gap-8">
            <div className="lg:col-span-2 space-y-6">
              <div className="glass-card p-6 rounded-2xl">
                <h4 className="font-semibold text-lg mb-4 text-foreground/80">Topological Graph</h4>
                <div className="space-y-4 relative before:absolute before:inset-0 before:ml-5 before:-translate-x-px md:before:mx-auto md:before:translate-x-0 before:h-full before:w-0.5 before:bg-gradient-to-b before:from-transparent before:via-border before:to-transparent">
                  
                  <div className="relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group is-active">
                    <div className="flex items-center justify-center w-10 h-10 rounded-full border-4 border-background bg-primary text-primary-foreground shrink-0 md:order-1 md:group-odd:-translate-x-1/2 md:group-even:translate-x-1/2 shadow-lg z-10">
                      <CheckCircle2 size={20} />
                    </div>
                    <div className="w-[calc(100%-4rem)] md:w-[calc(50%-2.5rem)] p-4 rounded-xl glass-card">
                      <div className="text-sm font-bold text-primary mb-1">Round 1 (LLM)</div>
                      <div className="text-foreground font-medium">write_code</div>
                      <div className="text-xs text-muted-foreground mt-1">Llama-3.1-70B generated syntax</div>
                    </div>
                  </div>
                  
                  <div className="relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group is-active">
                    <div className="flex items-center justify-center w-10 h-10 rounded-full border-4 border-background bg-purple-500 text-white shrink-0 md:order-1 md:group-odd:-translate-x-1/2 md:group-even:translate-x-1/2 shadow-lg z-10">
                      <CheckCircle2 size={20} />
                    </div>
                    <div className="w-[calc(100%-4rem)] md:w-[calc(50%-2.5rem)] p-4 rounded-xl glass-card border-purple-500/20">
                      <div className="text-sm font-bold text-purple-400 mb-1">Round 2 (Parallel)</div>
                      <div className="flex gap-2">
                        <span className="px-2 py-1 bg-muted rounded-md text-xs font-mono">lint_code</span>
                        <span className="px-2 py-1 bg-muted rounded-md text-xs font-mono">test_code</span>
                      </div>
                      <div className="text-xs text-muted-foreground mt-2">Tools executed concurrently</div>
                    </div>
                  </div>

                  <div className="relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group is-active">
                    <div className="flex items-center justify-center w-10 h-10 rounded-full border-4 border-background bg-emerald-500 text-white shrink-0 md:order-1 md:group-odd:-translate-x-1/2 md:group-even:translate-x-1/2 shadow-lg z-10">
                      <CheckCircle2 size={20} />
                    </div>
                    <div className="w-[calc(100%-4rem)] md:w-[calc(50%-2.5rem)] p-4 rounded-xl glass-card border-emerald-500/20">
                      <div className="text-sm font-bold text-emerald-400 mb-1">Round 3 (Human)</div>
                      <div className="text-foreground font-medium">deploy_code</div>
                      <div className="text-xs text-muted-foreground mt-1">Mocked human approval</div>
                    </div>
                  </div>

                </div>
              </div>
            </div>

            <div className="lg:col-span-3">
              <div className="glass-card rounded-2xl overflow-hidden h-full flex flex-col shadow-2xl border-white/5">
                <div className="bg-black/40 border-b border-white/5 px-4 py-3 flex items-center gap-2">
                  <div className="flex gap-1.5">
                    <div className="w-3 h-3 rounded-full bg-red-500/80"></div>
                    <div className="w-3 h-3 rounded-full bg-yellow-500/80"></div>
                    <div className="w-3 h-3 rounded-full bg-green-500/80"></div>
                  </div>
                  <div className="text-xs text-muted-foreground ml-2 font-mono flex items-center gap-2">
                    <Terminal size={14} /> stdout — SGH Execution
                  </div>
                </div>
                <div className="p-6 font-mono text-sm leading-relaxed overflow-x-auto text-gray-300">
                  <div className="text-primary mb-2">INFO: Dispatcher starting plan '5b2fd7b3-f846-4f3e-9b7a-2780c2d30579' v1 (4 nodes, 4 edges)</div>
                  <div className="mb-2">INFO: Round 1: dispatch 1 node(s) (1 fresh, 0 recovery)</div>
                  <div className="text-purple-400 mb-4">LiteLLM completion() model= meta/llama-3.1-70b-instruct; provider = nvidia_nim</div>
                  
                  <div className="mb-2">INFO: Round 2: dispatch 2 node(s) (2 fresh, 0 recovery)</div>
                  <div className="text-emerald-400 mb-1">INFO: Linting code:</div>
                  <div className="pl-4 border-l-2 border-emerald-500/30 mb-4 text-emerald-200">
                    def hello_world():<br/>
                    &nbsp;&nbsp;&nbsp;&nbsp;return 'hello'
                  </div>
                  
                  <div className="text-emerald-400 mb-1">INFO: Testing code:</div>
                  <div className="pl-4 border-l-2 border-emerald-500/30 mb-4 text-emerald-200">
                    def hello_world():<br/>
                    &nbsp;&nbsp;&nbsp;&nbsp;return 'hello'
                  </div>

                  <div className="mb-4">INFO: Round 3: dispatch 1 node(s) (1 fresh, 0 recovery)</div>
                  
                  <div className="text-white font-bold mb-4">
                    INFO: Plan '5b2fd7b3-f846-4f3e-9b7a-2780c2d30579' v1 complete in 16.95s | succeeded=True | rounds=3
                  </div>

                  <div className="text-gray-400">
                    INFO: Terminal states: {'{'}<br/>
                    &nbsp;&nbsp;'write_code': &lt;NodeState.EXECUTED: 'executed'&gt;,<br/>
                    &nbsp;&nbsp;'lint_code': &lt;NodeState.EXECUTED: 'executed'&gt;,<br/>
                    &nbsp;&nbsp;'test_code': &lt;NodeState.EXECUTED: 'executed'&gt;,<br/>
                    &nbsp;&nbsp;'deploy_code': &lt;NodeState.EXECUTED: 'executed'&gt;<br/>
                    {'}'}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </motion.section>

      </main>
    </div>
  );
}

export default App;
