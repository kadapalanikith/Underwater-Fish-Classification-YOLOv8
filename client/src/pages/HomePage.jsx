import React from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import {
  Fish, Activity, ChevronRight, Database,
  Cpu, Award, Eye, BookOpen, Code, AlertTriangle
} from 'lucide-react';

const CLASSES = [
  'AngelFish','BlueTang','ButterflyFish','ClownFish','GoldFish',
  'Gourami','MorishIdol','PlatyFish','RibbonedSweetlips',
  'ThreeStripedDamselfish','YellowCichlid','YellowTang','ZebraFish'
];

const STATS = [
  { val: '0.91', key: 'mAP@50' },
  { val: '0.86', key: 'Precision' },
  { val: '0.87', key: 'Recall' },
  { val: '13',   key: 'Fish Classes' },
  { val: '≤45ms', key: 'Latency' },
  { val: '640px', key: 'Input Size' },
];

const FEATURES = [
  { icon: <Eye size={22} />,      title: 'Real-Time Detection',     desc: 'YOLOv8 processes each uploaded image in milliseconds, marking every identified fish with a bounding box.' },
  { icon: <Database size={22} />, title: 'Open Dataset Trained',    desc: 'Model trained on curated Roboflow/Kaggle underwater frames — 2,000+ labeled instances across 13 classes.' },
  { icon: <Cpu size={22} />,      title: 'Edge-Ready Architecture', desc: 'YOLOv8 Nano backbone allows deployment on low-power CPUs and ARM-based research hardware.' },
  { icon: <Code size={22} />,     title: 'Open API',                desc: 'A fully documented FastAPI backend with a /api/predict endpoint you can integrate with any client.' },
  { icon: <Award size={22} />,    title: 'Academic Research',       desc: 'Developed as a minor academic project, targeting IEEE publication in marine AI.' },
  { icon: <Activity size={22} />, title: 'Confidence Filtering',    desc: 'Adjust detection confidence threshold (default 25%) to trade off precision vs recall for your use case.' },
];

const HOW_TO = [
  { step: '01', title: 'Open the Demo', desc: 'Navigate to the Demo tab and you will see the AI inference pipeline.' },
  { step: '02', title: 'Upload an Image', desc: 'Click the upload zone or drag-and-drop any JPEG/PNG underwater fish photo. Max 10 MB.' },
  { step: '03', title: 'Run Inference', desc: 'Click "Process with YOLOv8". The pipeline animates through Input → Preprocessing → YOLOv8 Model → Detection & Output.' },
  { step: '04', title: 'Inspect Results', desc: 'The annotated image with bounding boxes appears on the right, with species names and confidence percentages.' },
];

const fadeIn = {
  hidden: { opacity: 0, y: 24 },
  show:   { opacity: 1, y: 0, transition: { duration: 0.55 } }
};

export default function HomePage() {
  return (
    <div style={{ paddingTop: '70px' }}>

      {/* ── Hero ───────────────────────────────────────── */}
      <section style={{ padding: '7rem 0 5rem', background: 'linear-gradient(135deg, #F9F7F7 0%, #DBE2EF 100%)' }}>
        <div className="container" style={{ maxWidth: '900px', textAlign: 'center' }}>
          <motion.div initial="hidden" animate="show" variants={fadeIn}>
            <div className="badge">Academic Research Project · YOLOv8</div>
            <h1 style={{ fontSize: 'clamp(2.5rem, 6vw, 4.2rem)', color: '#112D4E', marginBottom: '1.5rem' }}>
              Intelligent Underwater<br />Fish Classification
            </h1>
            <p style={{ fontSize: '1.2rem', color: '#6b7f9e', maxWidth: '600px', margin: '0 auto 2.5rem', lineHeight: 1.7 }}>
              Upload any underwater photo and our production-grade YOLOv8 model instantly detects and classifies 13 marine fish species with bounding-box precision.
            </p>
            <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap' }}>
              <Link to="/demo" className="btn btn-primary" style={{ padding: '1rem 2.25rem', fontSize: '1.05rem' }}>
                Try the Model <ChevronRight size={20} />
              </Link>
              <Link to="/research" className="btn btn-outline">
                View Research  <BookOpen size={18} />
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* ── Stats ──────────────────────────────────────── */}
      <section className="section" style={{ background: '#fff' }}>
        <div className="container">
          <motion.div initial="hidden" whileInView="show" viewport={{ once: true }} variants={fadeIn}>
            <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
              <div className="badge">Performance Metrics</div>
              <h2 style={{ fontSize: 'clamp(1.6rem, 3vw, 2.4rem)' }}>Benchmark Highlights</h2>
            </div>
            <div className="stat-grid">
              {STATS.map(s => (
                <div className="stat-card" key={s.key}>
                  <div className="stat-val">{s.val}</div>
                  <div className="stat-key">{s.key}</div>
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* ── About Project ─────────────────────────────── */}
      <section className="section" style={{ background: '#F9F7F7' }}>
        <div className="container">
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4rem', alignItems: 'center' }}>
            <motion.div initial="hidden" whileInView="show" viewport={{ once: true }} variants={fadeIn}>
              <div className="badge">About the Project</div>
              <h2 style={{ fontSize: 'clamp(1.6rem, 3vw, 2.4rem)', marginBottom: '1.25rem' }}>
                Why DeepSea AI?
              </h2>
              <p style={{ color: '#6b7f9e', lineHeight: 1.8, marginBottom: '1.25rem' }}>
                Traditional marine biodiversity surveys are time-consuming, expensive, and require
                domain expertise. DeepSea AI automates species identification from a single image,
                enabling scalable monitoring for researchers, aquaculture operators, and conservationists.
              </p>
              <p style={{ color: '#6b7f9e', lineHeight: 1.8, marginBottom: '2rem' }}>
                Our custom-trained YOLOv8 model achieves state-of-the-art accuracy on 13 tropical
                freshwater and marine species while remaining lightweight enough to run on CPU.
              </p>
              <Link to="/research" className="btn btn-outline">Read the Research →</Link>
            </motion.div>

            <motion.div initial="hidden" whileInView="show" viewport={{ once: true }} variants={{ ...fadeIn, hidden: { ...fadeIn.hidden, x: 24 }, show: { ...fadeIn.show, x: 0 } }}>
              <div className="card" style={{ background: '#DBE2EF', border: 'none' }}>
                <div style={{ fontWeight: 700, marginBottom: '1rem', color: '#112D4E' }}>
                  <Fish size={18} style={{ display: 'inline', marginRight: 6 }} />
                  Classifiable Species (13 classes)
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                  {CLASSES.map(c => (
                    <span key={c} className="species-tag">{c}</span>
                  ))}
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* ── Features ──────────────────────────────────── */}
      <section className="section" style={{ background: '#fff' }}>
        <div className="container">
          <motion.div initial="hidden" whileInView="show" viewport={{ once: true }} variants={fadeIn} style={{ textAlign: 'center', marginBottom: '3rem' }}>
            <div className="badge">What's Inside</div>
            <h2 style={{ fontSize: 'clamp(1.6rem, 3vw, 2.4rem)' }}>Core Features</h2>
          </motion.div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem' }}>
            {FEATURES.map((f, i) => (
              <motion.div key={i} initial="hidden" whileInView="show" viewport={{ once: true }} variants={fadeIn}
                className="card" style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                <div style={{ width: 44, height: 44, borderRadius: 12, background: 'rgba(63,114,175,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#3F72AF' }}>
                  {f.icon}
                </div>
                <h4 style={{ fontSize: '1rem', fontWeight: 700, color: '#112D4E' }}>{f.title}</h4>
                <p style={{ fontSize: '0.875rem', color: '#6b7f9e', lineHeight: 1.7 }}>{f.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ── How To Use ────────────────────────────────── */}
      <section className="section" style={{ background: '#F9F7F7' }}>
        <div className="container" style={{ maxWidth: '800px', margin: '0 auto' }}>
          <motion.div initial="hidden" whileInView="show" viewport={{ once: true }} variants={fadeIn} style={{ textAlign: 'center', marginBottom: '3rem' }}>
            <div className="badge">Getting Started</div>
            <h2 style={{ fontSize: 'clamp(1.6rem, 3vw, 2.4rem)' }}>How to Use the Model</h2>
          </motion.div>
          <div className="timeline">
            {HOW_TO.map((step, i) => (
              <motion.div key={i} initial="hidden" whileInView="show" viewport={{ once: true }} variants={fadeIn} className="timeline-item">
                <div style={{ display: 'flex', alignItems: 'flex-start', gap: '1rem' }}>
                  <div style={{ fontSize: '2rem', fontWeight: 900, color: '#DBE2EF', lineHeight: 1, minWidth: 44 }}>{step.step}</div>
                  <div>
                    <h4 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: '0.35rem', color: '#112D4E' }}>{step.title}</h4>
                    <p style={{ color: '#6b7f9e', fontSize: '0.9rem', lineHeight: 1.7 }}>{step.desc}</p>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
          <div style={{ textAlign: 'center', marginTop: '3rem' }}>
            <Link to="/demo" className="btn btn-primary" style={{ padding: '1rem 2.5rem' }}>
              Launch Demo →
            </Link>
          </div>
        </div>
      </section>

      {/* ── Tech Stack ────────────────────────────────── */}
      <section className="section" style={{ background: '#112D4E', color: '#DBE2EF' }}>
        <div className="container" style={{ textAlign: 'center' }}>
          <div className="badge" style={{ background: 'rgba(63,114,175,0.3)', color: '#DBE2EF' }}>Technology Stack</div>
          <h2 style={{ fontSize: 'clamp(1.6rem, 3vw, 2.4rem)', color: '#fff', marginBottom: '3rem' }}>Built With Modern AI Tools</h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.5rem' }}>
            {[
              { name: 'YOLOv8', sub: 'Ultralytics (AI Backbone)' },
              { name: 'PyTorch', sub: 'Deep Learning Framework' },
              { name: 'FastAPI', sub: 'Python REST Backend' },
              { name: 'React 19', sub: 'Vite + Frontend' },
              { name: 'OpenCV', sub: 'Image Processing' },
              { name: 'Docker', sub: 'Production Container' },
            ].map((t, i) => (
              <div key={i} style={{ padding: '1.5rem', borderRadius: 18, background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', textAlign: 'left' }}>
                <div style={{ fontWeight: 900, fontSize: '1.1rem', color: '#3F72AF', marginBottom: '0.25rem' }}>{t.name}</div>
                <div style={{ fontSize: '0.8rem', color: '#9aafcc' }}>{t.sub}</div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
