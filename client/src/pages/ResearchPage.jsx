import React from 'react';
import { motion } from 'framer-motion';
import { FileText, Download, BookOpen, ExternalLink } from 'lucide-react';

const SAMPLE_PDF = '/research/RE_Paper.pdf';

const PAPERS = [
  {
    title: 'Underwater Fish Species Classification Using YOLOv8 — A Deep Learning Approach',
    journal: 'IEEE International Conference on Marine AI (Pending Submission)',
    year: '2026',
    abstract: 'This paper presents an automated system for classifying 13 underwater fish species using the YOLOv8 nano architecture. Trained on a custom-labeled dataset with 2,000+ annotated instances, the model achieves a mAP@50 of 0.91. We describe the data collection strategy, transfer learning approach, and deployment as a FastAPI web service with a React frontend.',
    pdf: SAMPLE_PDF,
    status: 'preprint',
  }
];

const METRICS = [
  { label: 'Precision',    val: '0.86', note: 'TP / (TP + FP)' },
  { label: 'Recall',       val: '0.87', note: 'TP / (TP + FN)' },
  { label: 'mAP@50',       val: '0.91', note: 'Mean Average Precision' },
  { label: 'mAP@50–95',    val: '0.66', note: 'Strict accuracy' },
];

const METHODOLOGY = [
  { step: '1', title: 'Dataset Collection', body: 'Sourced labeled underwater fish images from Kaggle and Roboflow. Applied augmentation: horizontal/vertical flip, brightness jitter, mosaic.' },
  { step: '2', title: 'Transfer Learning',  body: 'Fine-tuned YOLOv8n pretrained on COCO. Frozen backbone for 5 epochs, then full fine-tune for 5 epochs at 1e-4 LR.' },
  { step: '3', title: 'Evaluation',         body: 'Evaluated on a held-out test split (15%). Computed precision-recall curves and confusion matrices per species.' },
  { step: '4', title: 'Deployment',         body: 'Packaged as a FastAPI REST API with React frontend, served via Docker container. Supports CPU and CUDA inference.' },
];

const fadeIn = { hidden: { opacity: 0, y: 20 }, show: { opacity: 1, y: 0, transition: { duration: 0.55 } } };

export default function ResearchPage() {
  return (
    <div style={{ paddingTop: '70px' }}>

      {/* Header */}
      <section style={{ padding: '4rem 0 3rem', background: 'linear-gradient(135deg, #F9F7F7 0%, #DBE2EF 100%)' }}>
        <div className="container" style={{ textAlign: 'center' }}>
          <div className="badge">Academic Research</div>
          <h1 style={{ fontSize: 'clamp(2rem, 5vw, 3rem)', marginBottom: '1rem' }}>Research & Publications</h1>
          <p style={{ color: '#6b7f9e', maxWidth: '540px', margin: '0 auto', fontSize: '1.05rem' }}>
            Methodology, benchmarks, and academic outcomes for the DeepSea AI classification project.
          </p>
        </div>
      </section>

      {/* Paper Card */}
      <section className="section" style={{ background: '#fff' }}>
        <div className="container" style={{ maxWidth: '880px' }}>
          <motion.div initial="hidden" whileInView="show" viewport={{ once: true }} variants={fadeIn}>
            <div className="badge">Published Work</div>
            <h2 style={{ fontSize: '1.8rem', marginBottom: '2rem' }}>Paper</h2>
            {PAPERS.map((p, i) => (
              <div key={i} className="card" style={{ marginBottom: '1.5rem' }}>
                <div style={{ display: 'flex', gap: '1.25rem', alignItems: 'flex-start' }}>
                  <div style={{ width: 50, height: 50, flexShrink: 0, borderRadius: 14, background: 'rgba(63,114,175,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#3F72AF' }}>
                    <FileText size={24} />
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.5rem', alignItems: 'center' }}>
                      <span style={{ padding: '0.2rem 0.65rem', background: 'rgba(234,179,8,0.12)', color: '#a16207', borderRadius: 9999, fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase' }}>
                        {p.status}
                      </span>
                      <span style={{ fontSize: '0.82rem', color: '#9aafcc' }}>{p.journal}</span>
                    </div>
                    <h3 style={{ fontSize: '1.05rem', fontWeight: 800, color: '#112D4E', marginBottom: '0.75rem', lineHeight: 1.4 }}>{p.title}</h3>
                    <p style={{ fontSize: '0.875rem', color: '#6b7f9e', lineHeight: 1.7, marginBottom: '1rem' }}>{p.abstract}</p>
                    <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
                      <a href={p.pdf} target="_blank" rel="noreferrer" className="btn btn-primary" style={{ padding: '0.6rem 1.25rem', fontSize: '0.875rem' }}>
                        <FileText size={16} /> View PDF
                      </a>
                      <a href={p.pdf} download="DeepSeaAI_Sample_Paper.pdf" className="btn btn-outline" style={{ padding: '0.6rem 1.25rem', fontSize: '0.875rem' }}>
                        <Download size={16} /> Download
                      </a>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Embedded PDF Viewer */}
      <section className="section" style={{ background: '#F9F7F7' }}>
        <div className="container">
          <motion.div initial="hidden" whileInView="show" viewport={{ once: true }} variants={fadeIn}>
            <div className="badge">Document Preview</div>
            <h2 style={{ fontSize: '1.8rem', marginBottom: '1.5rem' }}>Research Paper Preview</h2>
            <div className="pdf-viewer" style={{ boxShadow: '0 8px 40px rgba(17,45,78,0.1)' }}>
              <object
                data={SAMPLE_PDF}
                type="application/pdf"
                width="100%"
                height="100%"
                style={{ borderRadius: 18, display: 'block' }}
              >
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: '1rem', background: '#f0f4f8', borderRadius: 18 }}>
                  <BookOpen size={64} color="#9aafcc" />
                  <p style={{ color: '#6b7f9e', fontWeight: 600 }}>PDF preview not available in this browser.</p>
                  <a href={SAMPLE_PDF} download className="btn btn-primary" style={{ padding: '0.65rem 1.5rem' }}>
                    <Download size={16} /> Download Paper
                  </a>
                </div>
              </object>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Metrics */}
      <section className="section" style={{ background: '#fff' }}>
        <div className="container" style={{ maxWidth: '880px' }}>
          <motion.div initial="hidden" whileInView="show" viewport={{ once: true }} variants={fadeIn}>
            <div className="badge">Evaluation</div>
            <h2 style={{ fontSize: '1.8rem', marginBottom: '2rem' }}>Performance Metrics</h2>
            <div className="stat-grid">
              {METRICS.map(m => (
                <div key={m.label} className="stat-card" style={{ borderTop: '4px solid #3F72AF' }}>
                  <div className="stat-val">{m.val}</div>
                  <div className="stat-key" style={{ marginBottom: '0.25rem' }}>{m.label}</div>
                  <div style={{ fontSize: '0.72rem', color: '#c0cfe0' }}>{m.note}</div>
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* Methodology */}
      <section className="section" style={{ background: '#F9F7F7' }}>
        <div className="container" style={{ maxWidth: '880px' }}>
          <motion.div initial="hidden" whileInView="show" viewport={{ once: true }} variants={fadeIn}>
            <div className="badge">Methodology</div>
            <h2 style={{ fontSize: '1.8rem', marginBottom: '2.5rem' }}>Training Pipeline</h2>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '1.25rem' }}>
              {METHODOLOGY.map((m, i) => (
                <motion.div key={i} initial="hidden" whileInView="show" viewport={{ once: true }} variants={fadeIn} className="card">
                  <div style={{ fontSize: '2rem', fontWeight: 900, color: '#DBE2EF', marginBottom: '0.5rem' }}>{m.step}</div>
                  <h4 style={{ fontSize: '1rem', fontWeight: 800, marginBottom: '0.5rem', color: '#112D4E' }}>{m.title}</h4>
                  <p style={{ fontSize: '0.875rem', color: '#6b7f9e', lineHeight: 1.7 }}>{m.body}</p>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  );
}
