import React, { useState, useRef } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Upload, X, Fish, Activity, RefreshCw,
  CheckCircle2, AlertCircle, ImagePlus, Zap, ScanLine, BarChart2
} from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Pipeline step definitions
const STEPS = [
  { id: 'input',      label: 'Input',      Icon: ImagePlus },
  { id: 'preprocess', label: 'Preprocess', Icon: ScanLine  },
  { id: 'cnn',        label: 'CNN Extract', Icon: Zap       },
  { id: 'classify',   label: 'Classify',   Icon: BarChart2  },
  { id: 'output',     label: 'Output',     Icon: CheckCircle2 },
];

// State machine: which step is active/done during the pipeline
function getStepState(stepIdx, activeIdx, done) {
  if (done) return stepIdx === 4 ? 'glow-output' : 'done';
  if (stepIdx < activeIdx) return 'done';
  if (stepIdx === activeIdx) return 'active';
  return 'idle';
}

export default function DemoPage() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl]       = useState(null);
  const [result, setResult]               = useState(null);
  const [loading, setLoading]             = useState(false);
  const [error, setError]                 = useState(null);
  const [pipelineStep, setPipelineStep]   = useState(-1); // -1 = not started
  const [pipelineDone, setPipelineDone]   = useState(false);
  const fileInputRef = useRef(null);

  const clearAll = () => {
    setSelectedImage(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setPipelineStep(-1);
    setPipelineDone(false);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    if (file.size > 10 * 1024 * 1024) {
      setError('Image exceeds 10 MB. Please choose a smaller file.');
      return;
    }
    setSelectedImage(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
    setPipelineStep(0); // light up Input step
    setPipelineDone(false);
  };

  const handlePredict = async () => {
    if (!selectedImage) return;
    setLoading(true);
    setError(null);
    setPipelineDone(false);

    // Animated pipeline sequence
    const delay = (ms) => new Promise(r => setTimeout(r, ms));
    const animate = async () => {
      for (let i = 0; i <= 3; i++) {
        setPipelineStep(i);
        await delay(650);
      }
    };

    const [animRes, apiRes] = await Promise.allSettled([
      animate(),
      (async () => {
        const form = new FormData();
        form.append('file', selectedImage);
        return axios.post(`${API_URL}/api/predict`, form, {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 40000,
        });
      })()
    ]);

    // Ensure pipeline shows step 4 (output) regardless
    setPipelineStep(4);
    
    if (apiRes.status === 'fulfilled') {
      setPipelineDone(true);
      setResult(apiRes.value.data);
    } else {
      setError(apiRes.reason?.response?.data?.detail || 'Connection error. Is the backend running?');
      setPipelineDone(false);
      setPipelineStep(-1);
    }
    setLoading(false);
  };

  const lineProgress = pipelineStep >= 0 ? Math.min((pipelineStep / 4) * 100, 100) : 0;

  return (
    <div style={{ paddingTop: '70px' }}>

      {/* Header */}
      <section style={{ padding: '4rem 0 2rem', background: 'linear-gradient(135deg, #F9F7F7 0%, #DBE2EF 100%)' }}>
        <div className="container" style={{ textAlign: 'center' }}>
          <div className="badge">Live Demo</div>
          <h1 style={{ fontSize: 'clamp(2rem, 5vw, 3rem)', marginBottom: '0.75rem' }}>AI Inference Pipeline</h1>
          <p style={{ color: '#6b7f9e', fontSize: '1.05rem', maxWidth: '540px', margin: '0 auto' }}>
            Upload an underwater photo and watch the YOLOv8 model classify each fish species in real time.
          </p>
        </div>
      </section>

      {/* Pipeline Visualization */}
      <section style={{ padding: '2rem 0', background: '#fff', borderBottom: '1px solid #DBE2EF' }}>
        <div className="container" style={{ maxWidth: '750px' }}>
          <div className="pipeline-wrapper">
            {STEPS.map((step, idx) => {
              const state = getStepState(idx, pipelineStep, pipelineDone);
              const StateIcon = step.Icon;
              return (
                <React.Fragment key={step.id}>
                  <div className="pipeline-step">
                    <motion.div
                      className={`pipeline-icon ${state}`}
                      animate={state === 'active' ? { scale: [1, 1.08, 1] } : {}}
                      transition={{ duration: 0.6, repeat: state === 'active' ? Infinity : 0 }}
                    >
                      <StateIcon size={22} />
                    </motion.div>
                    <div className={`pipeline-label ${state}`}>{step.label}</div>
                  </div>
                  {idx < STEPS.length - 1 && (
                    <div className="pipeline-line">
                      <motion.div
                        className="pipeline-line-fill"
                        animate={{ width: pipelineStep > idx ? '100%' : '0%' }}
                        transition={{ duration: 0.5, ease: 'easeInOut' }}
                      />
                    </div>
                  )}
                </React.Fragment>
              );
            })}
          </div>
        </div>
      </section>

      {/* Main Demo Grid */}
      <section style={{ padding: '3rem 0 5rem' }}>
        <div className="container" style={{ maxWidth: '1100px' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', alignItems: 'start' }}>

            {/* ── Left: Upload ─────────────────────────── */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
              <h2 style={{ fontSize: '1.5rem', marginBottom: 0 }}>Upload Image</h2>
              <p style={{ color: '#6b7f9e', marginTop: 0, fontSize: '0.9rem' }}>
                Supports JPEG / PNG up to 10 MB. High-resolution photos give better results.
              </p>

              {/* Upload Zone */}
              <div
                className={`upload-zone ${previewUrl ? 'has-image' : ''} ${!previewUrl && pipelineStep === -1 ? '' : 'active'}`}
                onClick={() => fileInputRef.current?.click()}
              >
                <AnimatePresence mode="wait">
                  {previewUrl ? (
                    <motion.div key="img" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                      style={{ width: '100%', height: '100%', padding: '1rem', display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>
                      <img src={previewUrl} alt="Preview" style={{ maxHeight: '280px', maxWidth: '100%', objectFit: 'contain', borderRadius: 12 }} />
                      <button
                        onClick={(e) => { e.stopPropagation(); clearAll(); }}
                        style={{ position: 'absolute', top: 12, right: 12, padding: '0.4rem', borderRadius: '50%', border: 'none', background: '#fff', boxShadow: '0 2px 8px rgba(0,0,0,0.15)', cursor: 'pointer', color: '#112D4E', display: 'flex' }}
                      >
                        <X size={18} />
                      </button>
                    </motion.div>
                  ) : (
                    <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} style={{ textAlign: 'center', padding: '3rem 2rem' }}>
                      <div style={{ width: 72, height: 72, borderRadius: '50%', background: '#DBE2EF', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 1.25rem', color: '#3F72AF' }}>
                        <Upload size={32} />
                      </div>
                      <div style={{ fontWeight: 700, color: '#112D4E', marginBottom: '0.5rem' }}>Click or Drag & Drop</div>
                      <div style={{ fontSize: '0.85rem', color: '#9aafcc' }}>Upload an underwater fish photo</div>
                    </motion.div>
                  )}
                </AnimatePresence>
                <input type="file" ref={fileInputRef} accept="image/*" onChange={handleImageChange} style={{ display: 'none' }} />
              </div>

              {/* Analyse Button */}
              <motion.button
                className="btn btn-primary"
                onClick={handlePredict}
                disabled={!selectedImage || loading}
                whileTap={{ scale: 0.97 }}
                style={{ width: '100%', padding: '1rem', fontSize: '1rem', gap: '0.6rem' }}
              >
                {loading
                  ? <><RefreshCw size={18} style={{ animation: 'spin 1s linear infinite' }} /> Analyzing…</>
                  : <><Activity size={18} /> Process with YOLOv8</>
                }
              </motion.button>

              <style>{`@keyframes spin { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }`}</style>

              {error && (
                <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
                  style={{ padding: '0.9rem 1.2rem', background: 'rgba(239,68,68,0.07)', border: '1px solid rgba(239,68,68,0.2)',
                    borderRadius: 14, display: 'flex', gap: '0.6rem', alignItems: 'flex-start', color: '#b91c1c', fontSize: '0.875rem' }}>
                  <AlertCircle size={18} style={{ marginTop: 1, flexShrink: 0 }} />{error}
                </motion.div>
              )}
            </div>

            {/* ── Right: Results ────────────────────────── */}
            <div className="card" style={{ minHeight: '460px', display: 'flex', flexDirection: 'column' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
                <h3 style={{ fontSize: '1.2rem', fontWeight: 800 }}>Inference Output</h3>
                {result && <span style={{ background: 'rgba(34,197,94,0.1)', color: '#16a34a', padding: '0.25rem 0.75rem', borderRadius: 9999, fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase' }}>Success</span>}
              </div>

              <AnimatePresence mode="wait">
                {result ? (
                  <motion.div key="result" initial={{ opacity: 0, scale: 0.97 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }}
                    style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                    <div style={{ borderRadius: 14, overflow: 'hidden', border: '1px solid #DBE2EF' }}>
                      <img src={result.annotated_image} alt="Annotated" style={{ width: '100%', display: 'block' }} />
                    </div>
                    <div>
                      <div style={{ fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: '#9aafcc', marginBottom: '0.75rem' }}>
                        Detected Species
                      </div>
                      {result.detections?.length > 0 ? (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
                          {result.detections.map((d, i) => (
                            <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.7rem 1rem', background: '#F9F7F7', borderRadius: 12, border: '1px solid #DBE2EF' }}>
                              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', fontWeight: 700, color: '#112D4E' }}>
                                <Fish size={16} color="#3F72AF" />
                                <span style={{ textTransform: 'capitalize' }}>{d.species}</span>
                              </div>
                              <span style={{ fontFamily: 'monospace', fontSize: '0.9rem', fontWeight: 700, color: '#3F72AF' }}>
                                {(d.confidence * 100).toFixed(1)}%
                              </span>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div style={{ padding: '2rem', textAlign: 'center', background: '#F9F7F7', borderRadius: 14, border: '1px dashed #DBE2EF', color: '#9aafcc' }}>
                          No fish detected at 25% confidence.
                        </div>
                      )}
                    </div>
                  </motion.div>
                ) : (
                  <motion.div key="waiting" initial={{ opacity: 0 }} animate={{ opacity: 0.4 }} exit={{ opacity: 0 }}
                    style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '1rem', color: '#9aafcc', textAlign: 'center' }}>
                    <Activity size={64} />
                    <div style={{ fontWeight: 700, fontSize: '1.1rem' }}>Ready for Inference</div>
                    <div style={{ fontSize: '0.85rem', maxWidth: '240px', lineHeight: 1.6 }}>Upload a photo and press the process button to start detection.</div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
