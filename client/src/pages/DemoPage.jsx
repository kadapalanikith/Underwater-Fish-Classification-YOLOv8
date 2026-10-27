import React, { useState, useRef, useCallback } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Upload, X, Fish, Activity, RefreshCw,
  CheckCircle2, AlertCircle, ImagePlus, Zap, ScanLine, BarChart2,
  Video, Download, Eye, Layers, ChevronDown
} from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080';

// Pipeline step definitions (image mode)
const IMAGE_STEPS = [
  { id: 'input',      label: 'Input',     Icon: ImagePlus },
  { id: 'enhance',    label: 'Enhance',   Icon: ScanLine },
  { id: 'detect',     label: 'Detect',    Icon: Zap },
  { id: 'output',     label: 'Output',    Icon: CheckCircle2 },
];

const ENHANCEMENT_OPTIONS = [
  { value: 'raw',        label: 'Raw (No Enhancement)' },
  { value: 'clahe',      label: 'CLAHE (Contrast Boost)' },
  { value: 'gray_world', label: 'Gray-World (White Balance)' },
  { value: 'fusion',     label: 'Fusion (Combined)' },
];

const MODEL_OPTIONS = [
  { value: 'yolov8n', label: 'YOLOv8n — Fish Baseline' },
  { value: 'rtdetr',  label: 'RT-DETR-L — Pretrained Benchmark' },
];

function PipelineBar({ steps, activeIdx, done }) {
  return (
    <div className="pipeline-wrapper">
      {steps.map((step, idx) => {
        const state = done ? 'done' : idx < activeIdx ? 'done' : idx === activeIdx ? 'active' : 'idle';
        const Icon = step.Icon;
        return (
          <React.Fragment key={step.id}>
            <div className="pipeline-step">
              <motion.div
                className={`pipeline-icon ${state}`}
                animate={state === 'active' ? { scale: [1, 1.1, 1] } : {}}
                transition={{ duration: 0.7, repeat: state === 'active' ? Infinity : 0 }}
              >
                <Icon size={20} />
              </motion.div>
              <div className={`pipeline-label ${state}`}>{step.label}</div>
            </div>
            {idx < steps.length - 1 && (
              <div className="pipeline-line">
                <motion.div
                  className="pipeline-line-fill"
                  animate={{ width: activeIdx > idx ? '100%' : '0%' }}
                  transition={{ duration: 0.4, ease: 'easeInOut' }}
                />
              </div>
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
}

function ConfidenceBar({ value }) {
  const pct = Math.round(value * 100);
  const color = pct >= 70 ? '#22c55e' : pct >= 45 ? '#f59e0b' : '#ef4444';
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.82rem' }}>
      <div style={{ flex: 1, background: '#e2e8f0', borderRadius: 9999, height: 6 }}>
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.6 }}
          style={{ background: color, height: '100%', borderRadius: 9999 }}
        />
      </div>
      <span style={{ fontWeight: 700, color, minWidth: '3.2rem', textAlign: 'right' }}>{pct}%</span>
    </div>
  );
}

export default function DemoPage() {
  const [mode, setMode] = useState('image'); // 'image' | 'video'
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [enhancement, setEnhancement] = useState('raw');
  const [modelName, setModelName] = useState('yolov8n');
  const [conf, setConf] = useState(0.25);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [pipelineStep, setPipelineStep] = useState(-1);
  const [pipelineDone, setPipelineDone] = useState(false);
  const fileInputRef = useRef(null);

  const clearAll = useCallback(() => {
    setSelectedFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setPipelineStep(-1);
    setPipelineDone(false);
    if (fileInputRef.current) fileInputRef.current.value = '';
  }, [previewUrl]);

  const handleFileChange = useCallback((e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const maxSize = mode === 'video' ? 100 : 25;
    if (file.size > maxSize * 1024 * 1024) {
      setError(`File exceeds ${maxSize} MB limit.`);
      return;
    }
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
    setPipelineStep(0);
    setPipelineDone(false);
  }, [mode]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) handleFileChange({ target: { files: [file] } });
  }, [handleFileChange]);

  const handlePredict = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    setPipelineDone(false);

    const delay = (ms) => new Promise(r => setTimeout(r, ms));

    const animate = async () => {
      const total = mode === 'video' ? 4 : IMAGE_STEPS.length;
      for (let i = 1; i < total - 1; i++) {
        setPipelineStep(i);
        await delay(450);
      }
    };

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('enhancement', enhancement);
    formData.append('model_name', modelName);
    formData.append('conf', conf);

    const endpoint = mode === 'video' ? `${API_URL}/api/track` : `${API_URL}/api/predict`;
    if (mode === 'video') {
      formData.append('max_frames', 150);
    }

    const [, apiRes] = await Promise.allSettled([
      animate(),
      axios.post(endpoint, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 300000,
      }),
    ]);

    setPipelineStep(IMAGE_STEPS.length - 1);

    if (apiRes.status === 'fulfilled') {
      setPipelineDone(true);
      setResult(apiRes.value.data);
    } else {
      let msg = 'Connection error. Is the backend running on port 8080?';
      if (apiRes.reason?.code === 'ECONNABORTED') msg = 'Request timed out. Processing took too long.';
      else if (apiRes.reason?.response?.data?.detail) msg = apiRes.reason.response.data.detail;
      setError(msg);
      setPipelineDone(false);
    }
    setLoading(false);
  };

  const downloadResult = () => {
    if (!result) return;
    if (mode === 'image' && result.annotated_image) {
      const a = document.createElement('a');
      a.href = result.annotated_image;
      a.download = 'deepseaai_detection.jpg';
      a.click();
    } else if (mode === 'video' && result.video_base64) {
      const a = document.createElement('a');
      a.href = result.video_base64;
      a.download = 'deepseaai_tracked.mp4';
      a.click();
    }
  };

  const acceptAttr = mode === 'video' ? 'video/*' : 'image/*';

  return (
    <div style={{ paddingTop: '70px' }}>

      {/* ── Header ───────────────────────────────── */}
      <section style={{ padding: '3.5rem 0 2rem', background: 'linear-gradient(135deg, #F9F7F7 0%, #DBE2EF 100%)' }}>
        <div className="container" style={{ textAlign: 'center' }}>
          <div className="badge">Live Demo</div>
          <h1 style={{ fontSize: 'clamp(1.8rem, 4vw, 2.8rem)', marginBottom: '0.75rem' }}>
            AI Detection &amp; Tracking Pipeline
          </h1>
          <p style={{ color: '#6b7f9e', maxWidth: '580px', margin: '0 auto 1.5rem' }}>
            Detect and classify 13 tropical fish species. Choose image or video mode, apply underwater
            enhancement, and inspect model output. RT-DETR is a COCO-pretrained timing baseline, not a
            fish-accurate model. Video tracking has synthetic smoke-test coverage only.
          </p>

          {/* Mode Toggle */}
          <div style={{ display: 'inline-flex', borderRadius: 12, overflow: 'hidden', border: '1.5px solid #DBE2EF', background: '#fff' }}>
            {[
              { id: 'image', label: 'Image', Icon: ImagePlus },
              { id: 'video', label: 'Video + Tracking', Icon: Video },
            ].map(({ id, label, Icon }) => (
              <button
                key={id}
                onClick={() => { setMode(id); clearAll(); }}
                style={{
                  padding: '0.6rem 1.4rem', border: 'none', cursor: 'pointer', display: 'flex',
                  alignItems: 'center', gap: '0.5rem', fontSize: '0.9rem', fontWeight: 600,
                  background: mode === id ? '#3F72AF' : 'transparent',
                  color: mode === id ? '#fff' : '#112D4E',
                  transition: 'all 0.2s'
                }}
              >
                <Icon size={16} />{label}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* ── Pipeline Bar ─────────────────────────── */}
      <section style={{ padding: '1.5rem 0', background: '#fff', borderBottom: '1px solid #DBE2EF' }}>
        <div className="container" style={{ maxWidth: '700px' }}>
          <PipelineBar steps={IMAGE_STEPS} activeIdx={pipelineStep} done={pipelineDone} />
        </div>
      </section>

      {/* ── Main Grid ────────────────────────────── */}
      <section style={{ padding: '2.5rem 0 5rem' }}>
        <div className="container" style={{ maxWidth: '1150px' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', alignItems: 'start' }}>

            {/* ── Left Panel ───────── */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.2rem' }}>
              <h2 style={{ fontSize: '1.4rem', marginBottom: 0 }}>
                {mode === 'image' ? 'Upload Image' : 'Upload Video'}
              </h2>

              {/* Drop Zone */}
              <div
                className={`upload-zone ${previewUrl ? 'has-image' : ''}`}
                onClick={() => fileInputRef.current?.click()}
                onDrop={handleDrop}
                onDragOver={(e) => e.preventDefault()}
              >
                <AnimatePresence mode="wait">
                  {previewUrl ? (
                    <motion.div key="preview" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                      style={{ width: '100%', height: '100%', padding: '1rem', display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>
                      {mode === 'image'
                        ? <img src={previewUrl} alt="Preview" style={{ maxHeight: 260, maxWidth: '100%', objectFit: 'contain', borderRadius: 10 }} />
                        : (
                          <div style={{ textAlign: 'center', color: '#3F72AF' }}>
                            <Video size={48} style={{ marginBottom: '0.5rem' }} />
                            <div style={{ fontWeight: 700 }}>{selectedFile?.name}</div>
                            <div style={{ fontSize: '0.8rem', color: '#9aafcc' }}>{(selectedFile?.size / (1024 * 1024)).toFixed(2)} MB</div>
                          </div>
                        )
                      }
                      <button
                        onClick={(e) => { e.stopPropagation(); clearAll(); }}
                        style={{ position: 'absolute', top: 10, right: 10, padding: '0.35rem', borderRadius: '50%', border: 'none', background: '#fff', boxShadow: '0 2px 8px rgba(0,0,0,0.15)', cursor: 'pointer', color: '#112D4E', display: 'flex' }}
                      >
                        <X size={16} />
                      </button>
                    </motion.div>
                  ) : (
                    <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={{ textAlign: 'center', padding: '2.5rem 1.5rem' }}>
                      <div style={{ width: 64, height: 64, borderRadius: '50%', background: '#DBE2EF', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 1rem', color: '#3F72AF' }}>
                        {mode === 'video' ? <Video size={28} /> : <Upload size={28} />}
                      </div>
                      <div style={{ fontWeight: 700, color: '#112D4E', marginBottom: '0.3rem' }}>Click or Drag &amp; Drop</div>
                      <div style={{ fontSize: '0.82rem', color: '#9aafcc' }}>
                        {mode === 'image' ? 'JPG / PNG up to 25 MB' : 'MP4 / AVI up to 100 MB'}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
                <input type="file" ref={fileInputRef} accept={acceptAttr} onChange={handleFileChange} style={{ display: 'none' }} />
              </div>

              {/* ── Settings Panel ───── */}
              <div className="card" style={{ padding: '1.2rem', gap: '1rem', display: 'flex', flexDirection: 'column' }}>
                <div style={{ fontWeight: 700, fontSize: '0.9rem', color: '#112D4E', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <Layers size={16} /> Detection Settings
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.8rem' }}>
                  {/* Enhancement */}
                  <div>
                    <label style={{ fontSize: '0.78rem', color: '#6b7f9e', fontWeight: 600, display: 'block', marginBottom: '0.3rem' }}>
                      Enhancement
                    </label>
                    <div style={{ position: 'relative' }}>
                      <select
                        value={enhancement}
                        onChange={e => setEnhancement(e.target.value)}
                        style={{ width: '100%', padding: '0.5rem 2rem 0.5rem 0.7rem', border: '1.5px solid #DBE2EF', borderRadius: 8, fontSize: '0.82rem', background: '#fff', appearance: 'none', color: '#112D4E', fontWeight: 600, cursor: 'pointer' }}
                      >
                        {ENHANCEMENT_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                      </select>
                      <ChevronDown size={14} style={{ position: 'absolute', right: 8, top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none', color: '#6b7f9e' }} />
                    </div>
                  </div>

                  {/* Model */}
                  <div>
                    <label style={{ fontSize: '0.78rem', color: '#6b7f9e', fontWeight: 600, display: 'block', marginBottom: '0.3rem' }}>
                      Model
                    </label>
                    <div style={{ position: 'relative' }}>
                      <select
                        value={modelName}
                        onChange={e => setModelName(e.target.value)}
                        style={{ width: '100%', padding: '0.5rem 2rem 0.5rem 0.7rem', border: '1.5px solid #DBE2EF', borderRadius: 8, fontSize: '0.82rem', background: '#fff', appearance: 'none', color: '#112D4E', fontWeight: 600, cursor: 'pointer' }}
                      >
                        {MODEL_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                      </select>
                      <ChevronDown size={14} style={{ position: 'absolute', right: 8, top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none', color: '#6b7f9e' }} />
                    </div>
                  </div>
                </div>

                {/* Confidence Slider */}
                <div>
                  <label style={{ fontSize: '0.78rem', color: '#6b7f9e', fontWeight: 600, display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                    <span>Confidence Threshold</span>
                    <span style={{ color: '#3F72AF', fontWeight: 800 }}>{(conf * 100).toFixed(0)}%</span>
                  </label>
                  <input
                    type="range" min="0.05" max="0.95" step="0.05" value={conf}
                    onChange={e => setConf(parseFloat(e.target.value))}
                    style={{ width: '100%', accentColor: '#3F72AF' }}
                  />
                </div>

                {modelName === 'rtdetr' && (
                  <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'flex-start', fontSize: '0.75rem', color: '#a16207', background: 'rgba(234,179,8,0.08)', border: '1px solid rgba(234,179,8,0.2)', borderRadius: 10, padding: '0.6rem 0.75rem', lineHeight: 1.55 }}>
                    <AlertCircle size={14} style={{ marginTop: 2, flexShrink: 0 }} />
                    <span>RT-DETR-L is a pretrained generic checkpoint, not fish fine-tuned. Use it for timing/resource comparison only, not species-accuracy claims.</span>
                  </div>
                )}
              </div>

              {/* ── Analyse Button ───── */}
              <motion.button
                className="btn btn-primary"
                onClick={handlePredict}
                disabled={!selectedFile || loading}
                whileTap={{ scale: 0.97 }}
                style={{ width: '100%', padding: '0.95rem', fontSize: '1rem', gap: '0.6rem' }}
              >
                {loading
                  ? <><RefreshCw size={18} style={{ animation: 'spin 1s linear infinite' }} /> Processing…</>
                  : <><Activity size={18} /> {mode === 'image' ? 'Detect Fish' : 'Track &amp; Count'}</>
                }
              </motion.button>
              <style>{`@keyframes spin { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }`}</style>

              {/* Error */}
              {error && (
                <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}
                  style={{ padding: '0.85rem 1.1rem', background: 'rgba(239,68,68,0.07)', border: '1px solid rgba(239,68,68,0.2)', borderRadius: 12, display: 'flex', gap: '0.6rem', alignItems: 'flex-start', color: '#b91c1c', fontSize: '0.85rem' }}>
                  <AlertCircle size={17} style={{ marginTop: 1, flexShrink: 0 }} />{error}
                </motion.div>
              )}
            </div>

            {/* ── Right: Results ───── */}
            <div className="card" style={{ minHeight: '480px', display: 'flex', flexDirection: 'column' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.2rem' }}>
                <h3 style={{ fontSize: '1.15rem', fontWeight: 800, margin: 0 }}>
                  {mode === 'image' ? 'Inference Output' : 'Tracking Results'}
                </h3>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  {result && (
                    <span style={{ background: 'rgba(34,197,94,0.1)', color: '#16a34a', padding: '0.2rem 0.65rem', borderRadius: 9999, fontSize: '0.73rem', fontWeight: 700, textTransform: 'uppercase' }}>
                      ✓ Success
                    </span>
                  )}
                  {result && (
                    <button onClick={downloadResult}
                      style={{ padding: '0.3rem 0.7rem', border: '1.5px solid #DBE2EF', borderRadius: 8, background: '#fff', cursor: 'pointer', color: '#3F72AF', display: 'flex', alignItems: 'center', gap: '0.3rem', fontSize: '0.8rem', fontWeight: 600 }}>
                      <Download size={13} /> Download
                    </button>
                  )}
                </div>
              </div>

              <AnimatePresence mode="wait">
                {result ? (
                  <motion.div key="result" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                    style={{ display: 'flex', flexDirection: 'column', gap: '1.2rem', flex: 1 }}>

                    {/* Annotated Image or Video */}
                    {mode === 'image' && result.annotated_image && (
                      <div style={{ borderRadius: 12, overflow: 'hidden', border: '1px solid #DBE2EF' }}>
                        <img src={result.annotated_image} alt="Detected" style={{ width: '100%', display: 'block' }} />
                      </div>
                    )}

                    {mode === 'video' && result.video_base64 && (
                      <video controls style={{ width: '100%', borderRadius: 12, border: '1px solid #DBE2EF' }}>
                        <source src={result.video_base64} type="video/mp4" />
                      </video>
                    )}

                    {/* ── Stats Row ── */}
                    <div style={{ display: 'grid', gridTemplateColumns: mode === 'video' ? '1fr 1fr 1fr' : '1fr 1fr 1fr 1fr', gap: '0.6rem' }}>
                      {mode === 'image' && [
                        { label: 'Detections', value: result.total_detections ?? 0 },
                        { label: 'Enhancement', value: result.enhancement_applied ?? 'raw' },
                        { label: 'Detect ms', value: `${result.detection_latency_ms ?? 0}ms` },
                        { label: 'Total ms', value: `${result.total_latency_ms ?? 0}ms` },
                      ].map(({ label, value }) => (
                        <div key={label} style={{ textAlign: 'center', background: '#f8f9ff', borderRadius: 10, padding: '0.6rem 0.4rem' }}>
                          <div style={{ fontSize: '1rem', fontWeight: 800, color: '#112D4E' }}>{value}</div>
                          <div style={{ fontSize: '0.7rem', color: '#9aafcc', marginTop: 2 }}>{label}</div>
                        </div>
                      ))}
                      {mode === 'video' && [
                        { label: 'Fish Counted', value: result.total_fish_counted ?? 0 },
                        { label: 'Frames', value: result.frames_processed ?? 0 },
                        { label: 'Species', value: Object.keys(result.species_counts || {}).length },
                      ].map(({ label, value }) => (
                        <div key={label} style={{ textAlign: 'center', background: '#f8f9ff', borderRadius: 10, padding: '0.6rem 0.4rem' }}>
                          <div style={{ fontSize: '1.2rem', fontWeight: 800, color: '#3F72AF' }}>{value}</div>
                          <div style={{ fontSize: '0.7rem', color: '#9aafcc', marginTop: 2 }}>{label}</div>
                        </div>
                      ))}
                    </div>

                    <p style={{ fontSize: '0.72rem', color: '#9aafcc', margin: 0 }}>
                      Model: {result.model_used ?? modelName}
                      {result.model_fallback ? ' (fallback: requested model unavailable)' : ''}
                    </p>

                    {/* ── Detections / Species Counts ── */}
                    <div>
                      <div style={{ fontSize: '0.73rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: '#9aafcc', marginBottom: '0.7rem' }}>
                        {mode === 'image' ? 'Detected Species' : 'Species Counts'}
                      </div>

                      {mode === 'image' && result.detections?.length > 0 ? (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.55rem', maxHeight: '240px', overflowY: 'auto' }}>
                          {result.detections.map((d, i) => (
                            <div key={i} style={{ display: 'flex', flexDirection: 'column', gap: '0.2rem', padding: '0.6rem 0.8rem', background: '#f8f9ff', borderRadius: 10 }}>
                              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 3 }}>
                                <span style={{ fontWeight: 700, color: '#112D4E', fontSize: '0.88rem' }}>
                                  <Fish size={13} style={{ verticalAlign: 'middle', marginRight: 4 }} />{d.species}
                                </span>
                              </div>
                              <ConfidenceBar value={d.confidence} />
                            </div>
                          ))}
                        </div>
                      ) : mode === 'image' ? (
                        <p style={{ color: '#9aafcc', fontSize: '0.88rem' }}>No fish detected above confidence threshold.</p>
                      ) : null}

                      {mode === 'video' && result.species_counts && Object.keys(result.species_counts).length > 0 ? (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem' }}>
                          {Object.entries(result.species_counts).map(([sp, count]) => (
                            <div key={sp} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.5rem 0.8rem', background: '#f8f9ff', borderRadius: 10 }}>
                              <span style={{ fontWeight: 700, color: '#112D4E', fontSize: '0.88rem' }}>
                                <Fish size={13} style={{ verticalAlign: 'middle', marginRight: 4 }} />{sp}
                              </span>
                              <span style={{ background: '#3F72AF', color: '#fff', borderRadius: 9999, padding: '0.15rem 0.6rem', fontSize: '0.78rem', fontWeight: 700 }}>{count}</span>
                            </div>
                          ))}
                        </div>
                      ) : mode === 'video' ? (
                        <p style={{ color: '#9aafcc', fontSize: '0.88rem' }}>No fish crossed the counting gate.</p>
                      ) : null}
                    </div>
                  </motion.div>
                ) : (
                  <motion.div key="empty-result" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                    style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', flex: 1, gap: '1rem', color: '#c8d6e8' }}>
                    <Eye size={52} strokeWidth={1.2} />
                    <p style={{ fontSize: '0.9rem', textAlign: 'center', maxWidth: 260, margin: 0 }}>
                      Upload a file and click Detect Fish to see results here.
                    </p>
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
