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
  { val: '0.841', key: 'Held-out test mAP@50* (95% CI 0.823–0.880)' },
  { val: '0.636', key: 'Held-out test mAP@50–95* (95% CI 0.615–0.694)' },
  { val: '0.036', key: 'OBSEA zero-shot mAP@50‡ (class-agnostic)' },
  { val: '0.017', key: 'OBSEA zero-shot mAP@50–95‡' },
  { val: '0.908', key: 'Historical validation mAP@50†' },
  { val: '13/13', key: 'Classes present in test labels*' },
];

const FEATURES = [
  { icon: <Eye size={22} />,      title: 'Fast CPU Inference',     desc: 'A 30-image CPU run measured 60.7 ms mean YOLOv8 inference per image at 640 px; the validation protocol reported 44.8 ms. Timings vary with hardware and load and are not accuracy claims.' },
  { icon: <Database size={22} />, title: 'Dataset Baseline',    desc: 'The local dataset has 8,229 images (13 classes) from the Kaggle/Roboflow fish source; the saved baseline was trained on that dataset family. The test split shares exact duplicates and source stems with training data; the audit reports a 399-image leakage-filtered subset.' },
  { icon: <Cpu size={22} />,      title: 'CPU-Friendly Architecture', desc: 'YOLOv8 Nano is small enough for CPU inference. ARM or other edge deployments have not been tested in this workspace.' },
  { icon: <Code size={22} />,     title: 'Open API',                desc: 'A fully documented FastAPI backend with a /api/predict endpoint you can integrate with any client.' },
  { icon: <Award size={22} />,    title: 'Research Status',       desc: 'Held-out test evaluation is recorded (mAP@50 0.841, 95% CI 0.823–0.880 on the leakage-filtered subset). A zero-shot OBSEA check reached class-agnostic mAP@50 0.036, showing substantial domain shift. No publication is verified, and RT-DETR is not yet fish fine-tuned.' },
  { icon: <Activity size={22} />, title: 'Confidence Filtering',    desc: 'Adjust detection confidence threshold (default 25%) to trade off precision vs recall for your use case.' },
];

const HOW_TO = [
  { step: '01', title: 'Open the Demo', desc: 'Navigate to the Demo tab and choose image or video mode.' },
  { step: '02', title: 'Upload a File', desc: 'Click the upload zone or drag-and-drop an image (JPG/PNG, up to 25 MB) or a video (up to 100 MB).' },
  { step: '03', title: 'Run Inference', desc: 'Click "Detect Fish" for images or "Track & Count" for video. The pipeline animates through Input → Enhance → Detect → Output.' },
  { step: '04', title: 'Inspect Results', desc: 'The annotated image or tracked video appears on the right, with species names and confidence values where detections exist.' },
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
              Upload an image for YOLOv8 fish detection. Predictions are model outputs and have not been independently validated on every image or domain.
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
            <p style={{ color: '#6b7f9e', fontSize: '0.8rem', textAlign: 'center', marginTop: '1rem' }}>
              * Re-evaluated saved checkpoint on 399 leakage-filtered test images / 639 instances; polygons were converted to boxes and some classes have low support. Confidence intervals are image-level percentile intervals (2,000 resamples) conditional on all 13 classes being present; they do not fix leakage. † Original notebook validation log; 12 of 13 classes had validation instances. ‡ OBSEA is one external Mediterranean dataset; Diver and Octopus annotations were excluded, all fish taxa were collapsed to one class, and the checkpoint was not tuned on OBSEA. It is not 13-class species accuracy. The local test split still shares duplicates/source overlaps with training data.
            </p>
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
                The saved checkpoint was re-evaluated on the local test split: mAP@50 0.840 on 697 images and 0.841 on a 399-image leakage-filtered subset, with mAP@50–95 of 0.622 and 0.636. Image-level 95% intervals for the filtered subset were 0.823–0.880 and 0.615–0.694. Validation re-evaluation was mAP@50 0.910, close to the historical notebook log. The test split shares duplicates/source stems with training data, so these are not fully independent results; low-support classes and YellowCichlid remain weak. A separate zero-shot OBSEA test reached class-agnostic mAP@50 0.036, which is preliminary evidence of domain shift rather than a species result.
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
